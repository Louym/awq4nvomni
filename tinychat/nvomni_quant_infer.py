import os
from transformers import AutoProcessor, AutoModel, AutoConfig, GenerationConfig
import torch
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import sys
from awq.quantize.smooth import smooth_lm
import tinychat.utils.constants
tinychat.utils.constants.max_seq_len = 4*1024
from tinychat.models.modeling_quant_vila import NVOmniVideoInference
os.environ["HF_HUB_OFFLINE"] = "1"  # Use local cache for models
from tinychat.models import QuantVILAForCausalLM
from tinychat.models.nvomni.media import Video
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_to_sys_path_direct(model_path):
    """Add model path directly to sys.path"""
    if model_path not in sys.path:
        sys.path.insert(0, model_path)  # Insert at beginning for priority
        print(f"✓ Added to sys.path: {model_path}")
    else:
        print(f"Already in sys.path: {model_path}")

class QuantNVOmniVideoInference(NVOmniVideoInference):
    """A class to handle NVOmni video model inference with improved error handling and flexibility."""
    
    def __init__(self, model_path: str, quant_path=None, smooth_scale_path=None, alpha=0.5, torch_dtype="torch.float16", device_map="auto"):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.model = None
        self.processor = None
        self.config = None
        self.device = None
        
        
        self.quant_path=quant_path
        self.smooth_scale_path=smooth_scale_path
        self.alpha=alpha
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)    
        self.load_model()  
        
    
    def load_model(self) -> bool:
        """Load the model, processor, and config with error handling."""
        if not self.validate_paths(self.model_path):
            return False
            
        # try:
        if True:
            logger.info("Loading model configuration...")
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            logger.info("Loading model...")
            start_time = time.time()
            self.model=QuantVILAForCausalLM(
                self.config, 
                quant_path=self.quant_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True
            ).half()
            if self.smooth_scale_path is not None:
                act_scales = torch.load(self.smooth_scale_path)
                smooth_lm(self.model.vision_tower, act_scales, 0.3)
                smooth_lm(self.model.sound_tower, act_scales, 0.3)
                from tinychat.modules import QuantSiglipEncoder
                self.model.vision_tower.vision_tower.vision_model.encoder = QuantSiglipEncoder(
                    self.model.vision_tower.vision_tower.vision_model.encoder
                )   
                from tinychat.modules import QuantQwen2AudioEncoder
                self.model.sound_tower.audio_tower = QuantQwen2AudioEncoder(
                    self.model.sound_tower.audio_tower
                )
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

            # Set device for single-device setups
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = next(self.model.parameters()).device if self.model.parameters() else torch.device('cpu')
            
            logger.info(f"Model successfully loaded on device: {self.device}")
            self._print_model_info()
            return True
            
        # except Exception as e:
        #     logger.error(f"Failed to load model: {str(e)}")
        #     return False

    @torch.inference_mode()    
    def benchmark(
        self, 
        video_path: str, 
        text_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = None,
        num_video_frames: int = -1,
        load_audio_in_video: bool = True,
        audio_length: Union[int, str] = "fix_8",
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
            logger.error("Model or processor not loaded. Please initialize the model first.")
            return None
            
        if not self.validate_paths(self.model_path, video_path):
            return None
        
        # try:
        if True:
        
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Text prompt: {text_prompt}")
            
            # Create conversation
            conversation = self.create_conversation(video_path, text_prompt)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            logger.info(f"Chat template applied")

            # set model params
            self.model.config.load_audio_in_video = load_audio_in_video
            self.processor.config.load_audio_in_video = load_audio_in_video
            if num_video_frames > 0:
                self.model.config.num_video_frames = num_video_frames
                self.processor.config.num_video_frames = num_video_frames
            if audio_length != -1:
                self.model.config.audio_chunk_length = audio_length
                self.processor.config.audio_chunk_length = audio_length
            logger.info(f"Model config - load_audio_in_video: {self.model.config.load_audio_in_video}, num_video_frames: {self.model.config.num_video_frames}, audio_chunk_length: {self.model.config.audio_chunk_length}")
            
            # Process inputs
            start_time = time.time()
            inputs = self.processor([text])
            
            # Move inputs to the correct device if needed
            if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                inputs.input_ids = inputs.input_ids.to(self.device)
            
            processing_time = time.time() - start_time
            logger.info(f"Input processing completed in {processing_time:.2f} seconds")
            
            logger.info("Generating response...")
            torch.cuda.synchronize()
            start_time = time.time()


            with torch.no_grad():
                output_ids = self.model.benchmark(
                    input_ids=inputs.input_ids,
                    media=getattr(inputs, 'media', None),
                    media_config=getattr(inputs, 'media_config', None),
                    generation_config=None,
                )
            torch.cuda.synchronize()
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.5f} seconds")
            
            # Decode response
            response = self.processor.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0]

            # Extract only the new generated text (remove the input prompt)
            # if text in response:
            #     response = response[len(text):].strip()
            
            return response
    @torch.inference_mode()    
    def benchmark_content(
        self, 
        video_path: str, 
        text_prompt: str,
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
        video=Video(video_path)
        logger.info("Generating response...")
        start_time = time.time()


        with torch.no_grad():
            response = self.model.benchmark_content(
                prompt=[video, text_prompt],
                generation_config=None,
            )
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
        return response
   
    
    
def main():
    """Main function demonstrating usage of the NVOmni model."""
    
    # Configuration
    MODEL_PATH = "/home/yuming/workspace/nvomni/nvOmni-8B"
    VIDEO_PATH = "/home/yuming/workspace/nvomni/elon_musk2.mp4"
    # VIDEO_PATH = "/home/yuming/workspace/nvomni/elon_musk_trimmed_16s.mp4"
    # VIDEO_PATH = "/home/yuming/workspace/nvomni/draw.mp4"
    # TEXT_PROMPT = "Describe this video based on the audio."
    # TEXT_PROMPT = "Assess the video, followed by a detailed description of it's video and audio contents.  What is the person saying?"
    TEXT_PROMPT = "Describe what happens in this video in detail."
    quant_llm=True
    quant_tower=True
    # quant_llm=False
    # quant_tower=False
    video_length=2
    audio_length=video_length
    load_audio_in_video=True

    add_to_sys_path_direct(MODEL_PATH)
    
    # Initialize the inference class
    logger.info("Initializing NVOmni Video Inference...")
    if quant_llm:
        quant_path="/home/yuming/workspace/nvomni/awq4nvomni/quant_cache/nvomni-8B-w4-g128-v2.pt"
    else:
        quant_path=None
    if quant_tower:
        smooth_scale_path="/home/yuming/workspace/nvomni/awq4nvomni/awq_cache/nvomni-smooth-scale.pt"
    else:
        smooth_scale_path=None
    inferencer = QuantNVOmniVideoInference(
        MODEL_PATH, 
        torch_dtype="torch.float16", 
        quant_path=quant_path, 
        smooth_scale_path=smooth_scale_path, 
        alpha=0.3, 
        device_map="auto"
        )
    num_video_frames = int(2 * video_length)
    inferencer.model.config.num_video_frames = num_video_frames
    audio_chunk_length = audio_length
    inferencer.model.config.audio_chunk_length = f"fix_{audio_chunk_length}"
    if inferencer.model is None:
        logger.error("Failed to initialize model. Exiting.")
        return
    print(inferencer.model)
    inferencer.model = inferencer.model.cuda().eval()
    
    inferencer.model = inferencer.model.to("cuda:0")
    # Generate response
    logger.info("Starting inference...")
    # import cupyx
    # with cupyx.profiler.profile():
    for i in range(5):
        response = inferencer.benchmark_content(
            video_path=VIDEO_PATH,
            text_prompt=TEXT_PROMPT,
            
        )
    torch.cuda.synchronize()
    if response:
        print("\n" + "="*60)
        print("GENERATED RESPONSE")
        print("="*60)
        print(response)
        print("="*60)
    else:
        logger.error("Failed to generate response")
    

if __name__ == "__main__":
    main()