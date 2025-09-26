import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import whisper
import numpy as np
from itertools import chain
from .nvomni.distributed import all_gather as vila_all_gather
from .nvomni.constants import DEFAULT_IMAGE_TOKEN
from .nvomni.tokenizer_utils import tokenize_conversation
from collections import OrderedDict, defaultdict, deque
from .nvomni.mm_utils import process_image, process_images 
from .nvomni.media import extract_media
from transformers import AutoProcessor, AutoModel, AutoConfig, GenerationConfig, AutoTokenizer, PretrainedConfig, WhisperFeatureExtractor
from .nvomni.modeling_vila import (
    VILAPretrainedModel,
    VILAForCausalLM,
    get_model_config, 
    build_mm_projector, 
    build_vision_tower, 
    build_audio_tower, 
    build_speech_mm_projector, 
    build_sound_mm_projector,
    get_pg_manager,
)
from tinychat.utils.load_quant import load_awq_model
import time
import torch
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from .nvomni.media_encoder import BasicImageEncoder, BasicVideoEncoder, TSPVideoEncoder, BasicSoundEncoder, CacheFeatures
from .nvomni.configuration_vila import VILAConfig
from tinychat.models.qwen2 import Qwen2ForCausalLM
from .nvomni.builder import infer_stop_tokens, MEDIA_TOKENS
from .nvomni.constants import IGNORE_INDEX
import copy
from copy import deepcopy
def build_tokenizer(
    model_name_or_path: str,
    config: PretrainedConfig,
    ):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", use_fast=True, legacy=False)
    # Load chat template if specified.
    if getattr(config, "chat_template", None) is not None:
        print(f"Using chat template: {config.chat_template}")
        fpath = os.path.join(os.path.dirname(__file__), "chat_templates", f"{config.chat_template}.jinja")
        if not os.path.exists(fpath):
            fpath = os.path.join(os.path.dirname(model_name_or_path), f"{config.chat_template}.jinja")
        with open(fpath) as fd:
            chat_template = fd.read()
        tokenizer.chat_template = chat_template.replace("    ", "").replace("\n", "")

    # Set stop tokens for the tokenizer
    tokenizer.stop_tokens = infer_stop_tokens(tokenizer)
    tokenizer.stop_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.stop_tokens)

    # Add media tokens to the tokenizer
    tokenizer.media_tokens = MEDIA_TOKENS
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        if config.speech_tower_cfg is None and name == "speech":
            continue
        if config.sound_tower_cfg is None and name == "sound":
            continue
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)
        tokenizer.media_tokens[name] = token
    return tokenizer


class QuantVILAPretrainedModel(VILAPretrainedModel):
    def __init__(self, config: VILAConfig, quant_path=None, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        cfgs = get_model_config(config)
        self.quant_path=quant_path
        # if len(cfgs) == 3:
        #     llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        # else:
        #     raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")
        if len(cfgs) == 7:
            (
                llm_cfg,
                vision_tower_cfg,
                speech_tower_cfg,
                sound_tower_cfg,
                mm_projector_cfg,
                speech_mm_projector_cfg,
                sound_mm_projector_cfg,
            ) = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `speech_mm_projector_cfg` `sound_mm_projector_cfg` `vision_tower_cfg` `speech_tower_cfg` `sound_tower_cfg` not found in the config."
            )
        self.llm_cfg = AutoConfig.from_pretrained(llm_cfg)
        # loading on auto by default
        device_map = kwargs.get("device_map", "auto")
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)

        # XVILA
        # NUM_EXTRA_TOKENS = NUM_EXTRA_TOKENS_VILA + NUM_EXTRA_TOKENS_XVILA
        # if isinstance(self.config, dict):
        #     self.vocab_size = config.llm_cfg["vocab_size"] + NUM_EXTRA_TOKENS
        # else:
        #     self.vocab_size = self.tokenizer.vocab_size + NUM_EXTRA_TOKENS
        #     logging.info(
        #         f"[XGrammar] config is not a dict, loading vocab size from tokenizer {self.tokenizer.vocab_size} + {NUM_EXTRA_TOKENS} => {self.vocab_size}"
        #     )
        if speech_tower_cfg:
            self.speech_tower = build_audio_tower(speech_tower_cfg, config, encoder_type="speech").half()
            self.speech_mm_projector = build_speech_mm_projector(speech_mm_projector_cfg, config).half()
        if sound_tower_cfg:
            self.sound_tower = build_audio_tower(sound_tower_cfg, config, encoder_type="sound").half()
            self.sound_mm_projector = build_sound_mm_projector(sound_mm_projector_cfg, config).half()


        if device_map in ["auto", "cuda"]:
            self.mm_projector = self.mm_projector.cuda()
            self.vision_tower = self.vision_tower.cuda()
            self.speech_tower = self.speech_tower.cuda() if hasattr(self, "speech_tower") else None
            self.sound_tower = self.sound_tower.cuda() if hasattr(self, "sound_tower") else None
            self.speech_mm_projector = self.speech_mm_projector.cuda() if hasattr(self, "speech_mm_projector") else None
            self.sound_mm_projector = self.sound_mm_projector.cuda() if hasattr(self, "sound_mm_projector") else None
        # set device_map auto can autoamtically shard llm to different devices
        self.tokenizer= build_tokenizer(llm_cfg, config)
        if quant_path is not None:
            from tinychat.modules import (
                make_quant_norm,
                make_quant_attn,
                make_fused_mlp,
                make_fused_vision_attn,
            )
            self.llm = Qwen2ForCausalLM(self.llm_cfg).half()
            self.llm = load_awq_model(self.llm, quant_path, 4, 128, "cuda:0")
            make_quant_attn(self.llm, "cuda:0", True)
            make_quant_norm(self.llm)
            self.llm.cpu()
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.llm=self.llm.cuda()
        else:
            self.llm=Qwen2ForCausalLM.from_pretrained(llm_cfg)

            
        
        self.llm_model_embed_tokens = self.llm.model.embed_tokens

        # NOTE(ligeng): hard code to set padding_side to left
        self.tokenizer.padding_side = "left"

        self.vocab_size = len(self.tokenizer)
        self.update_vocab_size = lambda: setattr(self, "vocab_size", len(self.tokenizer))

        self.encoders = {}
        for name in ["image", "video", "speech", "sound"]:
            encoder_config = getattr(self.config, f"{name}_encoder")
            if isinstance(encoder_config, str):
                encoder_config = json.loads(encoder_config)
            if encoder_config.get("embed_time", False) == "True":
                if "trope_dim" not in encoder_config and encoder_config.get("time_embed_type", "") in ["pixel", "lang"]:
                    encoder_config["trope_dim"] = self.config.hidden_size // 2
                    print(f"Warning: trope_dim not found in config, defaulting to hidden_size // 2: {encoder_config['trope_dim']}")
            
            encoder_config.pop('_target_')  # hw: 删除 _target_
            if name == "video":
                self.encoders[name] = TSPVideoEncoder(parent=self, **encoder_config)
            elif name == "image":
                self.encoders[name] = BasicImageEncoder(self)
            else:
                self.encoders[name] = BasicSoundEncoder(parent=self, **encoder_config)
      

        self.post_config()
        self.is_loaded = True


        self.llm_only_need_embed = kwargs.get("llm_only_need_embed", False)
        if self.llm_only_need_embed:
            print("We only need the embed_tokens in llm.")
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()

        assert (
            self.llm is not None
            or self.vision_tower is not None
            or self.speech_tower is not None
            or self.mm_projector is not None
            or self.speech_mm_projector is not None
        ), "At least one of the components must be instantiated."

class QuantVILAForCausalLM(VILAForCausalLM, QuantVILAPretrainedModel):
    def __init__(self, config: VILAConfig, quant_path=None, *args, **kwargs):
        QuantVILAPretrainedModel.__init__(self, config, quant_path, *args, **kwargs)
    def _VILAForCausalLM__embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        mm_info,
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)

        if self.config.unified_audio_encoder:
            assert len(media["speech"]) == 0

        for name in media:
            _encoder = self.encoders[name]
            if name in ["speech", "sound"] and self.config.unified_audio_encoder:
                _encoder = self.encoders["sound"]

            if self.training:
                 # Gather metainfo of media objects from all ranks
                if name in ["speech", "sound"]:

                    info = []
                    if type(media.get(name, {})) is dict:
                        for _dict in media.get(name, {}):
                            info.append({k: {"shape": v.shape, "dtype": v.dtype} for k, v in _dict.items()})
                    elif type(media.get(name, {})) is list:
                        info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                    else:
                        raise ValueError(f"Unsupported media type: {type(media.get(name, {}))}")

                    # infos = list(chain(*distributed.all_gather(info)))
                    infos_list = vila_all_gather(info)
                    infos = list(chain(*infos_list))

                    # The entire batch does not contain any media objects of this type.
                    if not infos:
                        continue

                    # for audio encoding, we have to ensure the batch size is the same for all ranks. If not, we need to pad the batch with dummy tensors to the max batch size
                    max_batch_size = max(len(_info) for _info in infos_list)
                    missing_batch_size = max_batch_size - len(info)

                    _media = media.get(name, [])

                    _medias = list(chain(vila_all_gather(_media)))
                    if missing_batch_size > 0:
                        for i in range(missing_batch_size):
                            # use one of the media tensors to create a dummy tensor
                            if type(media.get(name, {})) is dict:
                                _dummy = {k: v.clone().to(device=self.device) for k, v in _medias[0].items()}
                            elif type(media.get(name, {})) is list:
                                if type(_medias[0]) is torch.Tensor:
                                    _dummy = _medias[0].clone().to(device=self.device)
                                elif type(_medias[0]) is np.ndarray:
                                    _dummy = _medias[0].copy()
                                else:
                                    raise ValueError(f"Unsupported media type: {type(_medias[0])}")
                            else:
                                raise ValueError(f"Unsupported media type: {type(media.get(name, {}))}")
                            _media.append(_dummy)
                            mm_info["audio_info"].append(["dummy"])
                    # print(f"rank {torch.distributed.get_rank()}: {name}, len of info: {len(info)}, len of infos: {len(infos)}, missing_batch_size: {missing_batch_size}")

                    # we need to also align the length of all audio samples in the batch size
                    cur_batch_max_audio_samples = max(len(_audio) for _audio in _medias)
                    cur_batch_max_audio_samples = int(np.ceil(cur_batch_max_audio_samples  / (self.config.audio_sampling_rate * 30)) * (self.config.audio_sampling_rate * 30)) # should be multiple of 30 seconds
                    cur_batch_max_audio_samples = min(cur_batch_max_audio_samples, self.config.audio_chunk_length * self.config.audio_sampling_rate)
                    cur_batch_max_audio_duration = cur_batch_max_audio_samples // self.config.audio_sampling_rate

                    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
                        "Qwen/Qwen2.5-Omni-7B", chunk_length=cur_batch_max_audio_duration, sampling_rate=self.config.audio_sampling_rate, hop_length=self.config.audio_hop_length
                    )

                    # use WhisperFeatureExtractor in transformers to load
                    new_media = []

                    aud_idx = 0
                    for _batch_idx in range(len(mm_info["audio_info"])):
                        _audio_info = mm_info["audio_info"][_batch_idx]
                        if _audio_info is not None:
                            for _mm_idx in range(len(_audio_info)):
                                _audio = _media[aud_idx]
                                if type(_audio) is torch.Tensor:
                                    device = _audio.device
                                    dtype = _audio.dtype
                                    _audio = _audio.cpu().float()
                                else:
                                    # logger.warning(f"The audio type is not a tensor, which is unexpected. Using the device and dtype of the model. media: {media}, mm_info: {mm_info}")
                                    device = self.device
                                    dtype = self.dtype
                                _audio = whisper.pad_or_trim(_audio, length=cur_batch_max_audio_samples)
                                aud_idx += 1
                                stft_features = whisper_feature_extractor(
                                    _audio,
                                    sampling_rate=self.config.audio_sampling_rate,
                                    return_attention_mask=True,
                                    padding="max_length",
                                    return_tensors="pt",
                                ).to(device, dtype)
                                new_media.append(stft_features)
                                if _audio_info[_mm_idx] != "dummy":
                                    _audio_info[_mm_idx]["new_audio_chunk_length"] = cur_batch_max_audio_duration
                                    _audio_info[_mm_idx]["new_audio_n_samples"] = cur_batch_max_audio_samples
                                    _audio_info[_mm_idx]["audio_end_sample_sec"] = _audio_info[_mm_idx]["audio_start_sec"] + cur_batch_max_audio_duration
                                    _audio_info[_mm_idx]["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]

                    assert aud_idx == len(_media), "The number of audio info does not match the number of audio samples."
                    _media = new_media

                    _fea = _encoder(_media, media_config[name], mm_info)
                    # [751, 1536]
                    # consume dummy features later
                    _dummy_fea = _fea[len(info) :]
                    embeds["dummy"].extend(_dummy_fea)

                    # remove the dummy features
                    _real_fea = _fea[: len(info)]
                    if len(info) > 0:
                        embeds[name] = deque(_real_fea)

                else:
                    # Gather metainfo of media objects from all ranks
                    info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                    infos = list(chain(vila_all_gather(info)))

                    # The entire batch does not contain any media objects of this type.
                    if not infos:
                        continue

                    # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                    if media.get(name) is None or len(media[name]) == 0:
                        dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.device)
                        embeds["dummy"].extend(self.encoders[name]([dummy], media_config[name]))
                        continue
                    embeds[name] = deque(self.encoders[name](media[name], media_config[name]))

            else:
                torch.cuda.synchronize()
                _encoder_time_start = time.time()
                if name == "sound":
                    all_audio_chunk_lengths = []
                    for _sample_idx in range(len(media[name])):
                        for _mm_idx in range(len(mm_info["audio_info"][_sample_idx])):
                            _new_audio_chunk_length = mm_info["audio_info"][_sample_idx][_mm_idx]["new_audio_chunk_length"]
                            all_audio_chunk_lengths.append(_new_audio_chunk_length)
                    cur_batch_max_audio_duration = max(all_audio_chunk_lengths)
                    cur_batch_max_audio_samples = cur_batch_max_audio_duration * self.config.audio_sampling_rate
                    # for qwen omni audio
                    # cur_batch_max_audio_samples = 960000

                    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
                            "Qwen/Qwen2.5-Omni-7B", chunk_length=cur_batch_max_audio_duration, sampling_rate=self.config.audio_sampling_rate, hop_length=self.config.audio_hop_length
                    )                   
                    new_media = []
                    _idx = 0
                    assert len(all_audio_chunk_lengths) == len(media[name]), "The number of audio chunk lengths does not match the number of audio samples."

                    _media = media.get(name, [])
                    aud_idx = 0
                    for _batch_idx in range(len(mm_info["audio_info"])):
                        _audio_info = mm_info["audio_info"][_batch_idx]
                        if _audio_info is not None:
                            for _mm_idx in range(len(_audio_info)):
                                _audio = _media[aud_idx]
                                if type(_audio) is torch.Tensor:
                                    device = _audio.device
                                    dtype = _audio.dtype
                                    _audio = _audio.cpu().float()
                                else:
                                    # logger.warning(f"The audio type is not a tensor, which is unexpected. Using the device and dtype of the model. media: {media}, mm_info: {mm_info}")
                                    device = self.device
                                    dtype = self.dtype
                                _audio = whisper.pad_or_trim(_audio, length=cur_batch_max_audio_samples)
                                aud_idx += 1
                                # print("audio shape: ", _audio.shape, "cur_batch_max_audio_samples: ", cur_batch_max_audio_samples, mm_info["audio_info"])
                                stft_features = whisper_feature_extractor(
                                    _audio,
                                    sampling_rate=self.config.audio_sampling_rate,
                                    return_attention_mask=True,
                                    padding="max_length",
                                    return_tensors="pt",
                                ).to(device, dtype)
                                # log_file = "audio_shapes_log.txt"
                                # # 将信息追加写入文件
                                # shape1 = stft_features["input_features"].shape
                                # shape2 = stft_features["attention_mask"].shape
                                # with open(log_file, "a", encoding="utf-8") as file:
                                #     file.write(
                                #         f"audio shape: {_audio.shape} cur_batch_max_audio_samples: {cur_batch_max_audio_samples} "
                                #         f"{mm_info} {shape1} {shape2}\n"
                                #     )
                                new_media.append(stft_features)
                                if _audio_info[_mm_idx] != "dummy":
                                    _audio_info[_mm_idx]["new_audio_chunk_length"] = cur_batch_max_audio_duration
                                    _audio_info[_mm_idx]["new_audio_n_samples"] = cur_batch_max_audio_samples
                                    _audio_info[_mm_idx]["audio_end_sample_sec"] = _audio_info[_mm_idx]["audio_start_sec"] + cur_batch_max_audio_duration
                                    _audio_info[_mm_idx]["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]
                    media[name] = new_media
                    # print("len new_media: ", len(new_media))
                    
                # print("name: ", name, "media[name]: ", media[name], "len(media[name]): ", len(media[name]))
                # print(media_config)
                if len(media[name]) > 0:
                    # torch.cuda.synchronize()
                    # _encoder_time_end = time.time()
                    # print(name, "encoder time: ", _encoder_time_end - _encoder_time_start)
                    embeds[name] = deque(_encoder(media[name], media_config[name], mm_info))
                    torch.cuda.synchronize()
                    _encoder_time_end = time.time()
                    print(name, "encoder time: ", _encoder_time_end - _encoder_time_start, "shape: ", embeds[name][0].shape)
                # time_list=[]
                # for i in range(10):
                #     torch.cuda.synchronize()
                #     start_time = time.time()
                #     if len(media[name]) > 0:
                #         embeds[name] = deque(_encoder(media[name], media_config[name], mm_info))
                #     torch.cuda.synchronize()
                #     end_time = time.time()
                #     time_list.append(end_time - start_time)
                # print(f"Encoding media '{name}' took {np.mean(time_list):.4f} seconds")
        # print(embeds)
        return embeds
    
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_list = []
        for _ in range(1):
            torch.cuda.synchronize()
            t1=time.time()
            results=VILAForCausalLM._embed(
                self, 
                input_ids,
                media,
                media_config,
                labels,
                attention_mask
            )
            torch.cuda.synchronize()
            t2=time.time()
            time_list.append(t2-t1)
        print(f"embed time: {sum(time_list)/len(time_list)}")
        return results
    
   

    def benchmark(        
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_output_ids_only: bool = True,
        **generation_kwargs,
    ) -> torch.LongTensor:
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)

        output_ids = self.llm.benchmark(inputs_embeds, attention_mask, 128, self.quant_path is not None)

        if return_output_ids_only:
            return_value = output_ids
        else:
            # by default, return the input_ids and output_ids concatenated to keep consistency with the community VLMs like qwen
            generation_config = generation_kwargs.get("generation_config", None)
            if generation_config is not None:
                num_generations = generation_config.num_return_sequences
                repeat_input_ids = input_ids.repeat_interleave(num_generations, dim=0)
                return_value = torch.cat([repeat_input_ids, output_ids], dim=-1)
            else:
                return_value = torch.cat([input_ids, output_ids], dim=-1)

        return return_value
    @torch.inference_mode()
    def benchmark_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format=None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        xgr_logits_processor = None

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(media["image"][0], self.config, None, enable_dynamic_res=True).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(media["image"], self.vision_tower.image_processor, self.config).half()
                media[name] = [image for image in images]
            elif name == "video":
                if self.config.image_aspect_ratio == "dynamic" and self.config.video_max_tiles > 1:
                    media[name] = [
                        process_images(
                            images,
                            self.vision_tower.image_processor,
                            self.config,
                            enable_dynamic_res=True,
                            max_tiles=self.config.video_max_tiles,
                        ).half()
                        for images in media[name]
                    ]
                elif self.config.image_aspect_ratio == "dynamic_s2" and self.config.video_max_tiles > 1:
                    self.config.image_processor = self.vision_tower.image_processor
                    if type(self.config.s2_scales) is str:
                        self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                    media[name] = [
                        torch.cat(
                            [
                                process_image(
                                    image,
                                    self.config,
                                    None,
                                    enable_dynamic_s2=True,
                                    max_tiles=self.config.video_max_tiles,
                                )[0].half()
                                for image in images
                            ]
                        )
                        for images in media[name]
                    ]
                else:
                    media[name] = [
                        process_images(images, self.vision_tower.image_processor, self.config)
                        for images in media[name]
                    ]
            elif name == "speech":
                speeches = media["speech"]
                media[name] = [speech for speech in speeches]
            elif name == "sound":
                # sounds = process_sounds(media["sound"]).half()
                sounds = media["sound"]
                # media[name] = [{k: v.half() for sound in sounds for k, v in sound.items()]
                for sound in sounds:
                    if type(sound) is dict:
                        for k, v in sound.items():
                            sound[k] = v.half()
                media[name] = [sound for sound in sounds]
            elif name == "video_info":
                media[name] = [media["video_info"]]
            elif name == "audio_info":
                media[name] = [media["audio_info"]]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()

        output_ids = self.benchmark(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                logits_processor=xgr_logits_processor,  # structured generation
            )

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return response
    
    
def add_to_sys_path_direct(model_path):
    """Add model path directly to sys.path"""
    if model_path not in sys.path:
        sys.path.insert(0, model_path)  # Insert at beginning for priority
        print(f"✓ Added to sys.path: {model_path}")
    else:
        print(f"Already in sys.path: {model_path}")

class NVOmniVideoInference:
    """A class to handle NVOmni video model inference with improved error handling and flexibility."""
    
    def __init__(self, model_path: str, torch_dtype="torch.float16", device_map="auto"):
        """
        Initialize the NVOmni model for video inference.
        
        Args:
            model_path (str): Path to the model directory
            torch_dtype: PyTorch data type for model weights
            device_map (str): Device mapping strategy for model loading
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.model = None
        self.processor = None
        self.config = None
        self.device = None
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)    
        self.load_model()
        
    def validate_paths(self, model_path: str, video_path: str = None) -> bool:
        """Validate that required paths exist."""
        if not Path(model_path).exists():
            self.logger.error(f"Model path does not exist: {model_path}")
            return False
            
        if video_path and not Path(video_path).exists():
            self.logger.error(f"Video path does not exist: {video_path}")
            return False
            
        return True
    
    def load_model(self) -> bool:
        """Load the model, processor, and config with error handling."""
        if not self.validate_paths(self.model_path):
            return False
            
        # try:
        if True:
            self.logger.info("Loading model configuration...")
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            self.logger.info("Loading model...")
            start_time = time.time()
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True  # More memory efficient loading
            )#.to(eval(self.torch_dtype))
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

            # Set device for single-device setups
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = next(self.model.parameters()).device if self.model.parameters() else torch.device('cpu')
            
            self.logger.info(f"Model successfully loaded on device: {self.device}")
            self._print_model_info()
            return True
            
        # except Exception as e:
        #     self.logger.error(f"Failed to load model: {str(e)}")
        #     return False
    
    def _print_model_info(self):
        """Print useful information about the loaded model."""
        self.logger.info("=" * 50)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 50)
        
        if self.config:
            self.logger.info(f"Model type: {getattr(self.config, 'model_type', 'Unknown')}")
            self.logger.info(f"Hidden size: {getattr(self.config, 'hidden_size', 'Unknown')}")
            
        if self.model and torch.cuda.is_available():
            self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            self.logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def create_conversation(self, video_path: str, text_prompt: str) -> List[Dict[str, Any]]:
        """
        Create a conversation format for the model.
        
        Args:
            video_path (str): Path to the video file
            text_prompt (str): Text prompt for the model
            
        Returns:
            List[Dict]: Conversation in the expected format
        """
        return [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": text_prompt}
            ]
        }]

    @torch.inference_mode()    
    def generate_response(
        self, 
        video_path: str, 
        text_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = None,
        num_video_frames: int = -1,
        load_audio_in_video: bool = True,
        audio_length: Union[int, str] = "max_3600",
    ) -> Optional[str]:
        """
        Generate a response from the model given a video and text prompt.
        
        Args:
            video_path (str): Path to the video file
            text_prompt (str): Text prompt for the model
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            do_sample (bool): Whether to use sampling
            custom_generation_config (GenerationConfig): Custom generation configuration
            
        Returns:
            Optional[str]: Generated response or None if failed
        """
        if not self.model or not self.processor:
            self.logger.error("Model or processor not loaded. Please initialize the model first.")
            return None
            
        if not self.validate_paths(self.model_path, video_path):
            return None
        
        # try:
        if True:
        
            self.logger.info(f"Processing video: {video_path}")
            self.logger.info(f"Text prompt: {text_prompt}")
            
            # Create conversation
            conversation = self.create_conversation(video_path, text_prompt)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            self.logger.info(f"Chat template applied")

            # set model params
            self.model.config.load_audio_in_video = load_audio_in_video
            self.processor.config.load_audio_in_video = load_audio_in_video
            if num_video_frames > 0:
                self.model.config.num_video_frames = num_video_frames
                self.processor.config.num_video_frames = num_video_frames
            if audio_length != -1:
                self.model.config.audio_chunk_length = audio_length
                self.processor.config.audio_chunk_length = audio_length
            self.logger.info(f"Model config - load_audio_in_video: {self.model.config.load_audio_in_video}, num_video_frames: {self.model.config.num_video_frames}, audio_chunk_length: {self.model.config.audio_chunk_length}")
            
            # Process inputs
            start_time = time.time()
            inputs = self.processor([text])
            
            # Move inputs to the correct device if needed
            if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                inputs.input_ids = inputs.input_ids.to(self.device)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Input processing completed in {processing_time:.2f} seconds")
            
            self.logger.info("Generating response...")
            start_time = time.time()

            generation_kwargs = {"max_new_tokens": max_new_tokens, "max_length": 99999999}
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            if do_sample is not None:
                generation_kwargs["do_sample"] = do_sample
            if temperature is not None:
                generation_kwargs["temperature"] = temperature

            generation_config = self.model.default_generation_config
            generation_config.update(**generation_kwargs)

            self.logger.info(f"Generation config: {generation_config.to_dict()}")


            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    media=getattr(inputs, 'media', None),
                    media_config=getattr(inputs, 'media_config', None),
                    generation_config=generation_config,
                )
            
            generation_time = time.time() - start_time
            self.logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Decode response
            response = self.processor.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0]

            # Extract only the new generated text (remove the input prompt)
            # if text in response:
            #     response = response[len(text):].strip()
            
            return response
            
        # except Exception as e:
        #     self.logger.error(f"Error during generation: {str(e)}")
        #     return None
    
    def batch_generate(
        self, 
        video_text_pairs: List[tuple], 
        **generation_kwargs
    ) -> List[Optional[str]]:
        """
        Generate responses for multiple video-text pairs.
        
        Args:
            video_text_pairs (List[tuple]): List of (video_path, text_prompt) tuples
            **generation_kwargs: Arguments passed to generate_response
            
        Returns:
            List[Optional[str]]: List of generated responses
        """
        responses = []
        for i, (video_path, text_prompt) in enumerate(video_text_pairs):
            self.logger.info(f"Processing batch item {i+1}/{len(video_text_pairs)}")
            response = self.generate_response(video_path, text_prompt, **generation_kwargs)
            responses.append(response)
            
            # Clear cache between generations to manage memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return responses
