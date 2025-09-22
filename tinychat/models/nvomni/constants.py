# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SOUND_TOKEN = "<sound>"
DEFAULT_SPEECH_TOKEN = "<speech>"
SENTINEL_TOKEN = "<vila/sentinel>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


SENTINEL_TOKEN = "<vila/sentinel>"

MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
    "speech": "<speech>",
    "sound": "<sound>",
}

# <image> <vila/video> <vila/sentinel>
# TODO(ligeng): need to discuss with Zhijian for the following tokens for different models.
"""
vila:
    151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151646: AddedToken("[BOS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151647: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151648: AddedToken("<vila/sentinel>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151649: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151650: AddedToken("<vila/video>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),

xvila:
    151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151646: AddedToken("[BOS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151647: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151648: AddedToken("<vila/sentinel>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151649: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151650: AddedToken("<vila/video>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151651: AddedToken("<speech>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151652: AddedToken("<sound>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151653: AddedToken("<|image_bos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151654: AddedToken("<|image_eos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151655: AddedToken("<|video_bos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151656: AddedToken("<|video_eos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151657: AddedToken("<|speech_bos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151658: AddedToken("<|speech_eos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151659: AddedToken("<|sound_bos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    151660: AddedToken("<|sound_eos|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
"""
MM_BOS_EOS_TOKENS = {
    "image": ["<|image_bos|>", "<|image_eos|>"],
    "video": ["<|video_bos|>", "<|video_eos|>"],
    "speech": ["<|speech_bos|>", "<|speech_eos|>"],
    "sound": ["<|sound_bos|>", "<|sound_eos|>"],
}

NUM_EXTRA_TOKENS_VILA = 8
NUM_EXTRA_TOKENS_XVILA = 10
