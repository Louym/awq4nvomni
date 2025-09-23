import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from transformers import AutoProcessor, AutoModel, AutoConfig, GenerationConfig, AutoTokenizer, PretrainedConfig
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
    
    
    # def _embed(
    #     self,
    #     input_ids: torch.Tensor,
    #     media: Dict[str, List[torch.Tensor]],
    #     media_config: Dict[str, Dict[str, Any]],
    #     labels: Optional[torch.Tensor],
    #     attention_mask: Optional[torch.Tensor],
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # NOTE(ligeng): deep copy to avoid modifying the original media and media_config
    #     media = copy.deepcopy(media)
    #     media_config = copy.deepcopy(media_config)

    #     labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
    #     attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

    #     PROCESS_GROUP_MANAGER = get_pg_manager()
    #     if PROCESS_GROUP_MANAGER is not None:
    #         for name in media:
    #             self.encoders[name].end_tokens = None

    #     # Extract text and media embeddings
    #     text_embeds = self.llm_model_embed_tokens(input_ids)

    #     mm_info = {}
    #     if "video_info" in media:
    #         video_info = media["video_info"]
    #         del media["video_info"]
    #         mm_info['video_info'] = video_info
    #     else:
    #         video_info = None

    #     if "audio_info" in media:
    #         audio_info = media["audio_info"]
    #         del media["audio_info"]
    #         mm_info['audio_info'] = audio_info
    #     else:
    #         audio_info = None

    #     if media is not None:
    #         media_embeds = self.__embed_media_tokens(media, media_config, mm_info)
    #     else:
    #         # no media was provided, so we just return an empty dict
    #         media_embeds = {}

    #     if PROCESS_GROUP_MANAGER is not None:
    #         media_embeds_video = []
    #         for i, images in enumerate(media_embeds["video"]):
    #             num_video_frame = media["video"][i].shape[0]
    #             if False:  # self.encoders["video"].pool_sizes:
    #                 pool_size = self.encoders["video"].pool_sizes[0][0]
    #                 num_video_frame = num_video_frame // pool_size * pool_size
    #             media_embeds_video += torch.unbind(images.reshape(num_video_frame, -1, images.shape[-1]))
    #         media_embeds["video"] = deque(media_embeds_video)

    #     # This is a workaround to make sure the dummy embeddings are consumed
    #     while media_embeds.get("dummy"):
    #         dummy_embed = media_embeds["dummy"].popleft()
    #         text_embeds += torch.sum(dummy_embed) * 0

    #     # Based on segment_aud_indices_list and segment_vis_indices_list, get interleaved vis-aud embeddings for video
    #     video_sound_embeds_idx = 0
    #     sep_embed = self.encoders["video"].embed_tokens("\n")
    #     text_embeds = text_embeds.to(self.dtype)
    #     sep_embed = sep_embed.to(text_embeds.dtype)
        
    #     if video_info is not None and self.config.load_audio_in_video and self.config.interleaved_vis_aud_in_video:
    #         assert self.encoders["video"].end_tokens is None, "end_tokens must be None for interleaved vis-aud in video"
    #         new_video_embeds = deque()
    #         video_embeds_idx = 0
    #         for k in range(len(video_info)):
    #             if video_info[k] is None:
    #                 continue
    #             for i in range(len(video_info[k])):
    #                 has_audio = video_info[k][i]["has_audio"]
    #                 if not has_audio:
    #                     new_video_embeds.append(media_embeds["video"][video_embeds_idx])
    #                     video_embeds_idx += 1
    #                     continue

    #                 # Check bounds for sound embeddings
    #                 if video_sound_embeds_idx >= len(media_embeds["sound"]):
    #                     raise ValueError(f"Sound embeddings index {video_sound_embeds_idx} out of bounds for video_info[{k}][{i}]")

    #                 segment_aud_indices_list = video_info[k][i]["segment_aud_indices_list"]
    #                 segment_vis_indices_list = video_info[k][i]["segment_vis_indices_list"]

    #                 # vis_t_pool_size = self.encoders["video"].pool_sizes[0][0]
    #                 vis_fea_len_per_frame =  media_embeds["video"][video_embeds_idx].shape[0] / video_info[k][i]["expected_frame_count"]
    #                 # aud_fea_len_per_sample =  media_embeds["sound"][video_sound_embeds_idx].shape[0] / video_info[k][i]["expected_audio_count"]
    #                 aud_fea_len_per_stft_frame =  media_embeds["sound"][video_sound_embeds_idx].shape[0] / audio_info[k][i]["new_audio_n_stft_frames"]
    #                 vis_end = 0
    #                 aud_end = 0
    #                 _new_video_embed = []
    #                 for j in range(len(segment_vis_indices_list)):
    #                     _vis_aud_fea = []
    #                     if len(segment_vis_indices_list[j]) > 0:
    #                         _new_frames = [int(np.ceil((_frame+1) * vis_fea_len_per_frame)) for _frame in segment_vis_indices_list[j]]
    #                         _vis_fea_end = _new_frames[-1]
    #                         # Ensure we don't exceed the available features
    #                         _vis_fea_end = min(_vis_fea_end, media_embeds["video"][video_embeds_idx].shape[0])
    #                         if j == len(segment_vis_indices_list) - 1 and i == len(video_info) - 1 and k == len(video_info[i]) - 1 and not _vis_fea_end == media_embeds["video"][video_embeds_idx].shape[0]: 
    #                             print(f"Warning: The number of last interleaved video features does not match the video feature length. Expected: {media_embeds['video'][video_embeds_idx].shape[0]}, Got: {_vis_fea_end}")
    #                             _vis_fea_end = media_embeds["video"][video_embeds_idx].shape[0]
    #                         _vis_fea = media_embeds["video"][video_embeds_idx][vis_end:_vis_fea_end]
    #                         vis_end = _vis_fea_end
    #                         _vis_aud_fea.append(_vis_fea)
    #                     _vis_aud_fea.append(sep_embed)
    #                     if len(segment_aud_indices_list[j]) > 0:
    #                         _new_audio_indices = [int(np.ceil(_fea * aud_fea_len_per_stft_frame)) for _fea in segment_aud_indices_list[j]]
    #                         _aud_fea_end = _new_audio_indices[-1]
    #                         # Ensure we don't exceed the available features
    #                         _aud_fea_end = min(_aud_fea_end, media_embeds["sound"][video_sound_embeds_idx].shape[0])
    #                         _aud_fea = media_embeds["sound"][video_sound_embeds_idx][aud_end:_aud_fea_end]
    #                         _vis_aud_fea.append(_aud_fea)
    #                         aud_end = _aud_fea_end
    #                     _vis_aud_fea.append(sep_embed)
    #                     _new_video_embed.append(torch.cat(_vis_aud_fea, dim=0))
    #                 video_sound_embeds_idx += 1
    #                 new_video_embeds.append(torch.cat(_new_video_embed, dim=0))
    #                 video_embeds_idx += 1

    #         assert len(new_video_embeds) == len(media_embeds["video"]), "The number of new video embeddings does not match the number of original video embeddings."
    #         media_embeds["video"] = new_video_embeds
    #     # Remove padding
    #     batch_size = labels.shape[0]
    #     text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
    #     labels = [labels[k][attention_mask[k]] for k in range(batch_size)]
    #     # Build inverse mapping from token ID to media name
    #     media_tokens = {}
    #     for name, token_id in self.tokenizer.media_token_ids.items():
    #         media_tokens[token_id] = name

    #     # Fuse text and media embeddings
    #     inputs_m, labels_m = [], []
    #     sound_embeds_idx = 0
    #     for k in range(batch_size):
    #         inputs_mk, labels_mk = [], []
    #         pos = 0
    #         while pos < len(labels[k]):
    #             if input_ids[k][pos].item() in media_tokens:
    #                 name = media_tokens[input_ids[k][pos].item()] if PROCESS_GROUP_MANAGER is None else "video"
    #                 if input_ids[k][pos].item() == self.tokenizer.media_token_ids["sound"]:
    #                     if self.config.interleaved_vis_aud_in_video:
    #                         if sound_embeds_idx < video_sound_embeds_idx:
    #                             media_embeds[name].popleft()
    #                             sound_embeds_idx += 1
    #                             pos += 1
    #                             continue
    #                     sound_embeds_idx += 1

    #                 end = pos + 1
    #                 input = media_embeds[name].popleft()
    #                 label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
    #             else:
    #                 end = pos
    #                 while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
    #                     end += 1
    #                 input = text_embeds[k][pos:end]
    #                 label = labels[k][pos:end]
                
    #             inputs_mk.append(input)
    #             labels_mk.append(label)
    #             pos = end
    #         inputs_m.append(torch.cat(inputs_mk, dim=0))
    #         labels_m.append(torch.cat(labels_mk, dim=0))
    #     inputs, labels = inputs_m, labels_m

    #     inputs[0] += sep_embed.mean() * 0 # dummy embedding
    #     # Check if all media embeddings are consumed

    #     for name in media_embeds:
    #         if media_embeds[name]:
    #             raise ValueError(f"Not all {name} embeddings are consumed! Still {len(media_embeds[name])} left.")

    #     # Truncate sequences to `model_max_length` as media embeddings are inserted
    #     inputs, labels = self.__truncate_sequence(inputs, labels)

    #     # Pad sequences to the longest one in the batch
    #     return self.__batchify_sequence(inputs, labels)
    
    
    
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_list = []
        for _ in range(5):
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
    
    # def __embed_media_tokens(
    #     self,
    #     media: Dict[str, List[torch.Tensor]],
    #     media_config: Dict[str, Dict[str, Any]],
    #     mm_info,
    # ) -> Dict[str, List[torch.Tensor]]:
    #     time_list = []
    #     for _ in range(20):
    #         torch.cuda.synchronize()
    #         t1=time.time()
    #         results=VILAForCausalLM._VILAForCausalLM__embed_media_tokens(
    #             self,
    #             media,
    #             media_config,
    #             mm_info,
    #         )
    #         torch.cuda.synchronize()
    #         t2=time.time()
    #         time_list.append(t2-t1)
    #         print(t2-t1)
    #     print(f"embed time: {sum(time_list)/len(time_list)} s")
    #     return results

        

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
