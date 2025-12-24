import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("artdeco-system")
class ARTDECOSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        guidance_out = self.guidance(
            out["comp_rgb"], self.prompt_utils, **batch
        )

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)
        
        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
        
        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
        
        # 2d normal smoothness loss
        if hasattr(self.cfg.loss, "lambda_normal_smoothness_2d") and self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smoothness loss, no comp_normal is found in the output."
                )
            loss_normal_smoothness_2d = (
                out["comp_normal"][:, 1:, :, :] - out["comp_normal"][:, :-1, :, :]
            ).square().mean() + (
                out["comp_normal"][:, :, 1:, :] - out["comp_normal"][:, :, :-1, :]
            ).square().mean()
            self.log("trian/loss_normal_smoothness_2d", loss_normal_smoothness_2d)
            loss += loss_normal_smoothness_2d * self.C(self.cfg.loss.lambda_normal_smoothness_2d)

        # our proposed opacity loss
        if hasattr(self.cfg.loss, "lambda_reg_2d") and self.C(self.cfg.loss.lambda_reg_2d) > 0:
            loss_reg_2d = ((out["opacity"] - out["opacity_coarse"]) ** 2).mean()
            self.log("train/loss_reg_2d", loss_reg_2d)
            loss += loss_reg_2d * self.C(self.cfg.loss.lambda_reg_2d)

        # image condition loss
        if hasattr(self.cfg.loss, "lambda_image") and self.C(self.cfg.loss.lambda_image) > 0:
            elevation, azimuth = batch["elevation"], batch["azimuth"]
        
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.jpg",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": None},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity_coarse"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "opacity_coarse" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.jpg",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": None},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity_coarse"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "opacity_coarse" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
    
    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.jpg",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        
        # save to /scratch/, only for using Compute Canada
        from_path = self.get_save_dir()[:-5]
        to_path = "/scratch/qiminc/MVDream-threestudio/outputs/" + self.get_save_dir()[:-5].split("outputs/")[1]
        command = f"cp -r {from_path} {to_path}"
        os.system(command)
