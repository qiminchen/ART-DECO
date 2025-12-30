import os
import torch

from collections import OrderedDict

ckpt_path = "./path/to/ckpts/last.ckpt"
ckpt = torch.load(ckpt_path)

new_state_dict_density = OrderedDict()
new_state_dict_feature = OrderedDict()
for k, v in ckpt["state_dict"].items():
    if k.startswith("geometry.density_network"):
        new_k = k[25:]
        new_state_dict_density[new_k] = v
    elif k.startswith("geometry.feature_network"):
        new_k = k[25:]
        new_state_dict_feature[new_k] = v

torch.save({
    "density_network_state_dict": new_state_dict_density,
    "feature_network_state_dict": new_state_dict_feature,
}, "./path/to/web-demo/ckpts/last_clean.ckpt")
