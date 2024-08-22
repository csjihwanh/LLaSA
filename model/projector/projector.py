import torch
import torch.nn as nn

def seg_projector_builder(config):

    proj_config = config.video_projector_configs

    projector_cls = config.model.video_projector_cls
    hidden_dim = proj_config.hidden_dim
    mlp_depth = proj_config.mlp_depth

    if projector_cls == 'mlp':

        layers = [nn.Linear(hidden_dim, hidden_dim)]

        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        return nn.Sequential(*layers)
    
    elif projector_cls == 'linear':

        return nn.Linear(hidden_dim, hidden_dim)
    

    raise ValueError(f'invalid projector type: {config.projector_type}')