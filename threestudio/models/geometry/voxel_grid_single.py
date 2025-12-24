from dataclasses import dataclass, field

import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry, 
    BaseImplicitGeometry, 
    contract_to_unisphere,
)
from threestudio.models.networks import get_conv
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

from threestudio.data.binvox_rw import (
    get_vox_from_binvox_512, 
    write_ply_triangle, 
    get_simple_coarse_voxel,
)


@threestudio.register("voxel-grid-single")
class VoxelGrid(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        grid_size: Tuple[int, int, int] = field(default_factory=lambda: (32, 32, 32))
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        regularization: bool = False

        category: str = ""  # voxel category, 00000000 building, 03001627 chair ...
        upsample_rate: int = 4  #  upsample rate
        downsample_rate: int = 16  #  downsample rate, 512 -> 32

        conv_network_config: dict = field(
            default_factory=lambda: {
                # "otype": "DECORGAN",
                "otype": "DECORGANLG",
                "output_activation": "none",
                "n_neurons": 64,
            }
        )

        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']

        # we don't use density_bias, i.e. density blob
        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = "auto"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.grid_size = self.cfg.grid_size
        self.regularization = self.cfg.regularization

        # cache for test so only need to run upsampling network once
        self.test_upsampled_density_grid = None
        self.test_upsampled_feature_grid = None

        data_dir = os.path.join("./threestudio/data/decorgan", self.cfg.category, "train/000129_62b50be1aa5f4c78b183ffd758147360")
        try:
            downsample_rate = self.cfg.downsample_rate
            voxel = get_vox_from_binvox_512(os.path.join(data_dir, "model_depth_fusion.binvox"))
        except:
            try:
                downsample_rate = self.cfg.downsample_rate // 2
                data_dict = h5py.File(os.path.join(data_dir, "model_depth_fusion.hdf5"), 'r')
                voxel = data_dict["voxel"][:]  # (256, 256, 256)
                data_dict.close()
            except:
                downsample_rate = self.cfg.downsample_rate // 8
                data_dict = h5py.File(os.path.join(data_dir, "model_depth_fusion_coarse.hdf5"), 'r')
                voxel = data_dict["voxel"][:]  # (64, 64, 64)
                data_dict.close()
        voxel = np.rot90(voxel, k=1, axes=(1, 0)).copy()
        voxel = np.rot90(voxel, k=1, axes=(2, 1)).copy()
        # voxel = np.flip(voxel, axis=2).copy()
        voxel_tensor = torch.from_numpy(voxel).to(self.device).unsqueeze(0).unsqueeze(0).float()
        coarse_voxel_tensor = F.max_pool3d(voxel_tensor, kernel_size=downsample_rate, stride=downsample_rate)
        coarse_voxel = coarse_voxel_tensor.detach().cpu().numpy()[0, 0]
        coarse_voxel = np.round(coarse_voxel).astype(np.uint8)  # (32, 32, 32)

        self.grid = torch.from_numpy(coarse_voxel).to(self.device).unsqueeze(0).unsqueeze(0).float()
        self.mask = F.interpolate(self.grid, scale_factor=self.cfg.upsample_rate, mode='nearest')
        self.mask = F.max_pool3d(self.mask, kernel_size=3, stride=1, padding=1)

        self.density_network = get_conv(1, 1, self.cfg.conv_network_config)
        self.feature_network = get_conv(1, 3, self.cfg.conv_network_config)
    
    def get_trilinear_feature(
        self, points: Float[Tensor, "*N Di"], grid: Float[Tensor, "1 Df G1 G2 G3"]
    ) -> Float[Tensor, "*N Df"]:
        points_shape = points.shape[:-1]
        df = grid.shape[1]
        di = points.shape[-1]
        out = F.grid_sample(
            grid, points.view(1, 1, 1, -1, di), align_corners=False, mode="bilinear"
        )
        out = out.reshape(df, -1).T.reshape(*points_shape, df)
        return out
    
    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        if self.training:
            self.test_upsampled_density_grid = None
            self.test_upsampled_feature_grid = None
            upsampled_density_grid = self.density_network(self.grid)
            upsampled_feature_grid = self.feature_network(self.grid)
        else:
            if self.test_upsampled_density_grid is None:
                with torch.no_grad():
                    self.test_upsampled_density_grid = self.density_network(self.grid)
                    self.test_upsampled_feature_grid = self.feature_network(self.grid)

            upsampled_density_grid = self.test_upsampled_density_grid
            upsampled_feature_grid = self.test_upsampled_feature_grid

        upsampled_density_grid = get_activation(self.cfg.density_activation)(upsampled_density_grid) * self.mask
        density = self.get_trilinear_feature(points, upsampled_density_grid)
        features = self.get_trilinear_feature(points, upsampled_feature_grid)

        output = {
            "density": density,
            "features": features,
        }

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                eps = 1.0e-3
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})
        
        if self.regularization:
            coarse_density = self.get_trilinear_feature(points, self.mask * 100)
            output.update(
                {
                    "density_grid": upsampled_density_grid, 
                    "coarse_density": coarse_density,
                }
            )

        return output
    
    def forward_density(
            self, points: Float[Tensor, "*N Di"]
    ) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample
        
        if self.training:
            upsampled_density_grid = self.density_network(self.grid)
        else:
            assert self.test_upsampled_density_grid is not None
            upsampled_density_grid = self.test_upsampled_density_grid

        upsampled_density_grid = get_activation(self.cfg.density_activation)(upsampled_density_grid) * self.mask
        density = self.get_trilinear_feature(points, upsampled_density_grid)

        return density
    
    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points)
        return density, None
    
    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)
    
    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample
        
        upsampled_feature_grid = self.feature_network(self.grid)
        features = self.get_trilinear_feature(points, upsampled_feature_grid)

        out.update(
            {
                "features": features,
            }
        )
        return out
    
    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "VoxelGrid":
        if isinstance(other, VoxelGrid):
            instance = VoxelGrid(cfg, **kwargs)
            # load state_dict
