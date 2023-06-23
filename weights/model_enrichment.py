import numpy as np
import torch
import os
from pathlib import Path

# its cpu on my computer but cuda on Gabriel's
_DEVICE = 'cpu'
_L_SRC_OBS_DIM = 223
_L_TARGET_OBS_DIM = 225

l_weights_src_fp = Path(os.getcwd()) / 'final_learner_model.pt'
l_weights_src = torch.load(l_weights_src_fp, map_location=torch.device(_DEVICE))

# modify each tensor with entry weight 223 to have entry weight 225
for net in ('model', 'model_target'):
    for key in l_weights_src[net]:
        tensor = l_weights_src[net][key]
        if tensor.dim() == 2 and tensor.shape[1] == _L_SRC_OBS_DIM:
            enrichment = torch.randn((tensor.shape[0],
                                      _L_TARGET_OBS_DIM - _L_SRC_OBS_DIM), 
                                     dtype=tensor.dtype,
                                     device=torch.device(_DEVICE))
            l_weights_src[net][key] = torch.cat((tensor, enrichment), dim=1)


l_weights_target_fp = Path(os.getcwd()) / 'l_enriched.pt'
torch.save(l_weights_src, l_weights_target_fp)

