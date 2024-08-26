# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from hydra import initialize_config_module
from config.config import load_configs

hydra.core.global_hydra.GlobalHydra.instance().clear()

cfg = load_configs()
sam2_cfg = cfg.segmentation.cfg_dir
print(sam2_cfg)

initialize_config_module(sam2_cfg, version_base="1.2")