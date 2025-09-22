# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from einops import rearrange

class SoundMultimodalProjectorConfig(PretrainedConfig):
    model_type = "sound_mm_projector"

    def __init__(self, sound_mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.sound_mm_projector_type = sound_mm_projector_type

class AudioDownSampleBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x = self.conv1(x)
        x = rearrange(x, "b c t -> b t c")
        return x

class AudioDownSamplePoolBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x = self.pool(x)
        x = rearrange(x, "b c t -> b t c")
        return x

class AudioDownSampleMaxPoolBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x = self.pool(x)
        x = rearrange(x, "b c t -> b t c")
        return x

class SoundMultimodalProjector(PreTrainedModel):
    config_class = SoundMultimodalProjectorConfig

    def __init__(self, sound_mm_projector_cfg: SoundMultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(sound_mm_projector_cfg)
        # sound_mm_projector_type = sound_mm_projector_cfg.sound_mm_projector_type
        if hasattr(config, "sound_mm_projector"):
            sound_mm_projector_type = config.sound_mm_projector
        else:
            sound_mm_projector_type = sound_mm_projector_cfg.sound_mm_projector_type
        self.sound_mm_projector_type = sound_mm_projector_type
        self.config.sound_mm_projector_type = sound_mm_projector_type

        if hasattr(config, "sound_mm_projector_cfg") and type(config.sound_mm_projector_cfg) == dict:
            config.sound_mm_projector_cfg["sound_mm_projector_type"] = sound_mm_projector_type

        if sound_mm_projector_type == "mlp":
            self.layers = nn.Sequential(
                nn.Linear(config.sound_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif sound_mm_projector_type == "mlp_downsample":
            self.downsample_block = AudioDownSampleBlock(config.sound_hidden_size)
            self.layers = nn.Sequential(
                nn.Linear(config.sound_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif sound_mm_projector_type == "mlp_downsample_pool":
            self.downsample_block = AudioDownSamplePoolBlock(config.sound_hidden_size)
            self.layers = nn.Sequential(
                nn.Linear(config.sound_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif sound_mm_projector_type == "mlp_downsample_pool_max":
            self.downsample_block = AudioDownSampleMaxPoolBlock(config.sound_hidden_size)
            self.layers = nn.Sequential(
                nn.Linear(config.sound_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {sound_mm_projector_type}")


    def forward(self, x, *args, **kwargs):
        if self.sound_mm_projector_type in ["mlp_downsample", "mlp_downsample_pool", "mlp_downsample_pool_max"]:
            x = self.downsample_block(x)
        return self.layers(x)


AutoConfig.register("sound_mm_projector", SoundMultimodalProjectorConfig)
AutoModel.register(SoundMultimodalProjectorConfig, SoundMultimodalProjector)
