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



class SpeechMultimodalProjectorConfig(PretrainedConfig):
    model_type = "speech_mm_projector"

    def __init__(self, speech_mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.speech_mm_projector_type = speech_mm_projector_type


class SpeechMultimodalProjector(PreTrainedModel):
    config_class = SpeechMultimodalProjectorConfig

    def __init__(self, speech_mm_projector_cfg: SpeechMultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(speech_mm_projector_cfg)
        # speech_mm_projector_type = speech_mm_projector_cfg.speech_mm_projector_type

        if hasattr(config, "speech_mm_projector"):
            speech_mm_projector_type = config.speech_mm_projector
        else:
            speech_mm_projector_type = speech_mm_projector_cfg.speech_mm_projector_type

        if speech_mm_projector_type == "mlp":
            # self.conv = nn.Conv1d(config.speech_hidden_size, config.speech_hidden_size, kernel_size=2, stride=2)
            self.layers = nn.Sequential(
                nn.Linear(config.speech_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif speech_mm_projector_type == "mlp_downsample":
            self.downsample_block = AudioDownSampleBlock(config.speech_hidden_size)
            self.layers = nn.Sequential(
                nn.Linear(config.speech_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {speech_mm_projector_type}")

    def forward(self, x, *args, **kwargs):
        # x = self.conv(x.transpose(1,2)).transpose(1,2)
        if self.speech_mm_projector_type == "mlp_downsample":
            x = self.downsample_block(x)
        return self.layers(x)


AutoConfig.register("speech_mm_projector", SpeechMultimodalProjectorConfig)
AutoModel.register(SpeechMultimodalProjectorConfig, SpeechMultimodalProjector)
