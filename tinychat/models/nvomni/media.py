import glob
import time
import random
import os
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import io
import cv2
import kaldiio
import librosa
import soundfile as sf
import torch
import numpy as np
import PIL
import PIL.Image
import requests
import tarfile
import whisper
import decord
from decord import AudioReader, cpu
import torchaudio
from transformers import PretrainedConfig

# from llava.constants import MEDIA_TOKENS
# from llava.media import Image, Video
# from llava.utils import make_list
# from llava.utils.logging import logger


MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
    "speech": "<speech>",
    "sound": "<sound>",
}

PROFILE_MODE = False
PROFILE_PROFILER_MODE = False

class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Video(File):
    pass

class Speech(File):
    def __init__(self, path, extension: str = None) -> None:
        self.path = path
        self.extension = extension

class Sound(File):
    def __init__(self, path, extension: str = None) -> None:
        self.path = path
        self.extension = extension


def make_list(obj: Any) -> List:
    return obj if isinstance(obj, list) else [obj]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image


def _load_video_bytesio(
    video_bytesio: BytesIO, *, num_frames: int, config: PretrainedConfig, load_aud: bool = False
) -> List[PIL.Image.Image]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(video_bytesio.read())
        temp_video_name = temp_video.name
        return _load_video(temp_video_name, num_frames=num_frames, load_aud=load_aud, config=config)

def get_overlap(inp1, inp2):
    """
    Calculates the overlapping time frame between a video clip and an audio segment.
    
    Args:
        inp1 (list): [start_sec, end_sec]
        inp2 (list): [start_sec, end_sec]

    Returns:
        tuple or None: (overlap_start, overlap_end) if overlap exists, else None.
    """
    # Calculate the maximum start time and minimum end time
    overlap_start = max(inp1[0], inp2[0])
    overlap_end = min(inp1[1], inp2[1])

    # Check if there is an actual overlap
    if overlap_start < overlap_end:
        return (overlap_start, overlap_end)
    else:
        return None


def _load_video(
    video_path: str, *, num_frames: int, config: PretrainedConfig, load_aud: bool = False
) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file

    if PROFILE_MODE:
        rank = torch.distributed.get_rank()
        time_0 = time.time()

    vidcap = cv2.VideoCapture(video_path)

    # load audio if available and needed
    audio_info = None
    if load_aud:
        # if True:
        try:
            aud_feature, audio_info = _load_speech(video_path, config)
            if PROFILE_MODE:
                time_1 = time.time()
                time_0 = time.time()
        except Exception as e:
            aud_feature = None
    else:
        aud_feature = None

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_duration = frame_count / fps

    # When load_audio_in_video and interleaved_vis_aud_in_video is True, we need to load frames for each video segment
    if config.load_audio_in_video and config.interleaved_vis_aud_in_video and aud_feature is not None:
        segment_duration = config.interleaved_video_segment_duration
        if segment_duration == -1:
            raise ValueError("video_segment_duration is not set")

        segment_vis_indices_list = []
        segment_aud_indices_list = []
        segment_counts = np.ceil(video_duration / segment_duration).astype(int) 

        # split the indices of indices into segments
        # segment_counts = 2
        # indices = [0, 1, 2, 3]
        # segment_indices_list = [[0, 1], [2, 3]]

        # audio feature length was padded to multiple of audio_chunk_length
        """
        aud_feature = {'input_features': tensor([[[-0.6806, -0.6806, -0.6806,  ...,  0.3303,  0.4267,  0.0335],
            [-0.6806, -0.6806, -0.6806,  ...,  0.4278,  0.5243,  0.1311],
            [-0.6806, -0.6806, -0.6806,  ...,  0.7774,  0.7901,  0.7618],
            ...,
            [-0.6806, -0.6806, -0.6806,  ..., -0.3868, -0.4061, -0.4405],
            [-0.6806, -0.6806, -0.6806,  ..., -0.4239, -0.4625, -0.5700],
            [-0.6806, -0.6806, -0.6806,  ..., -0.3374, -0.4024, -0.4679]]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], dtype=torch.int32)}
        """

        if type(aud_feature) == dict:
            aud_feas = aud_feature["input_features"] # torch.Size([1, 128, 3000])
        else:
            aud_feas = aud_feature
        audio_start_sec = audio_info['audio_start_sec']
        audio_end_sec = audio_info['audio_end_sample_sec']

        # audio_samples_per_second = aud_feas.shape[-1] / audio_info['new_audio_chunk_length']

        stft_frames_per_second = config.audio_sampling_rate // config.audio_hop_length

        _idx = 0
        aud_sample_start_idx = 0
        for i in range(segment_counts):
            end_frame = min((i+1) * segment_duration * fps, frame_count)

            _indices = []
            while _idx < len(indices) and indices[_idx] < end_frame and _idx < len(indices):
                _indices.append(indices[_idx])
                _idx += 1
            segment_vis_indices_list.append(_indices)
            clip_start_sec = i * segment_duration
            clip_end_sec = min(clip_start_sec + segment_duration, video_duration)

            # get the audio indices for the current clip
            overlap = get_overlap([clip_start_sec, clip_end_sec], [audio_start_sec, audio_end_sec])
            if overlap is not None:
                aud_sample_end_idx = round((overlap[1] - audio_start_sec) * stft_frames_per_second)
                segment_aud_indices_list.append([aud_sample_start_idx, aud_sample_end_idx])
                aud_sample_start_idx = aud_sample_end_idx
            else:
                segment_aud_indices_list.append([])
    frames = {}
    frame_times = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            print(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
        frame_times[index] = index / fps

    if PROFILE_MODE:
        time_2 = time.time()
        print(f"[Rank {rank}] video load frames time: {time_2 - time_0}. Video length: {video_duration}. video_path: {video_path}")
        time_0 = time.time()

    output_frames = [frames[index] for index in indices if index in frames]
    output_frame_times = [frame_times[index] for index in indices if index in frame_times]

    video_info = {}
    if config.load_audio_in_video and config.interleaved_vis_aud_in_video and aud_feature is not None:
        new_segment_vis_indices_list = []
        # vis_aud_segments = []
        processed_frame_index = 0
        for i, segment_indices in enumerate(segment_vis_indices_list):
            new_segment_vis_indices_list.append([])
            for index in segment_indices:
                if index in frames:
                    new_segment_vis_indices_list[-1].append(processed_frame_index)
                    processed_frame_index += 1
            # seg_vis_aud = {"frame_indices": new_segment_vis_indices_list[-1], "aud_indices": segment_aud_indices_list[i]}
        segment_vis_indices_list = new_segment_vis_indices_list
        # vis_aud_segments.append(seg_vis_aud)

        video_info["segment_vis_indices_list"] = segment_vis_indices_list
        video_info["segment_aud_indices_list"] = segment_aud_indices_list
        # video_info["vis_aud_segments"] = vis_aud_segments
        video_info['expected_frame_count'] = len(indices)
    video_info['video_path'] = video_path
    if audio_info is not None:
        audio_info['video_path'] = video_path
    video_info['has_audio'] = aud_feature is not None
    video_info['video_duration'] = video_duration
    video_info['audio_info'] = audio_info

    # calculate the time of each frame
    video_info['video_frame_times'] = output_frame_times

    return output_frames, aud_feature, video_info


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    aud_fea = None

    if getattr(config, "fps") != 0:
        print("Extracting frames from video with specified FPS is not supported yet. Ignored.")

    if isinstance(video.path, BytesIO):
        frames, aud_fea, video_info = _load_video_bytesio(
            video.path, num_frames=num_frames, config=config, load_aud=config.load_audio_in_video
        )
    else:
        frames, aud_fea, video_info = _load_video(
            video.path, num_frames=num_frames, config=config, load_aud=config.load_audio_in_video
        )

    if config.load_audio_in_video:
        return frames, aud_fea, video_info
    else:
        return frames


def soundFile_read_audio(audio_file, offset=None, duration=None, dtype='float32'):
        if dtype not in ['int32', 'float32']:
            print("audio dtype must be int32 or float32. Default to float32")
            dtype = 'float32'
        # return read audio and its sample rate
        if isinstance(audio_file, bytes):
            audio_file = io.BytesIO(audio_file)
        with sf.SoundFile(audio_file, 'r') as f:
            sample_rate = f.samplerate
            if offset is not None and offset > 0:
                f.seek(int(offset * sample_rate))
            if duration is not None and duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)
        return samples, sample_rate

def load_audio_from_tar(tar_file, audio_file):
    with tarfile.open(tar_file, 'r') as tar:
        audio_member = tar.getmember(audio_file)
        audio_file = tar.extractfile(audio_member)
        return librosa.load(audio_file)

def _load_audio_file(audio_path: str, config: PretrainedConfig):
    # Load video frames from a directory
    if audio_path is None:
        return None

    dirname = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)

    if dirname.endswith(".tar"):
        speech, sample_rate = load_audio_from_tar(dirname, filename)            
    else:
        sample_rate = config.audio_sampling_rate
        speech = whisper.load_audio(audio_path, sr=sample_rate)

    return speech, sample_rate


def _load_audio(audio: Union[str, dict], config: PretrainedConfig):
    if isinstance(audio, str):
        return _load_audio_file(audio, config)
    elif isinstance(audio, dict):
        audio_sample = audio['sample']
        if isinstance(audio_sample, (bytes, io.BytesIO)):
            offset = audio.get('offset', None)
            duration = audio.get('duration', None)
            dtype = audio.get('dtype', 'float32')
            return soundFile_read_audio(
                audio_sample, offset=offset, duration=duration, dtype=dtype
            )
        elif isinstance(audio_sample, np.ndarray):
            return audio_sample, audio.get('sample_rate')
        else:
            raise ValueError(f"Expect the loaded audio to be a processed numpy array or raw bytes. Got {type(audio_sample)}")
    else:
        raise ValueError(f"Expect input to be a path string or dict. Got {type(audio)}")

def _whisper_process(audio, sample_rate, audio_chunk_length, max_chunks_per_file):
    outputs = []
    num_audio_chunks = 0

    chunk_length = audio_chunk_length * sample_rate
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i : i + chunk_length]
        chunk = whisper.pad_or_trim(chunk)
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        mel = whisper.log_mel_spectrogram(chunk, n_mels=128)
        num_audio_chunks+=1
        outputs.append(mel)
        if num_audio_chunks == max_chunks_per_file:
            break

    frames = torch.stack(outputs, dim=0)
    return frames.numpy().tolist()
def _waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=0,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]

    fbank = fbank.unsqueeze(0)
    return fbank
def _load_speech(speech, config: PretrainedConfig):
    if PROFILE_MODE:
        time_0 = time.time()
        rank = torch.distributed.get_rank()

    if type(speech) == str:
        speech_path = speech
    else:
        speech_path = speech.path

    # Load video frames from a directory
    if speech_path is None:
        return None
    speech_outputs = []

    if config.audio_chunk_length and not (type(config.audio_chunk_length) == str and "max" in config.audio_chunk_length) and not (type(config.audio_chunk_length) == str and "fix" in config.audio_chunk_length):
        try:
            config.audio_chunk_length = int(config.audio_chunk_length)
            audio_n_samples_limit = config.audio_chunk_length * config.audio_sampling_rate
        except Exception as e:
            print(f"Error setting audio_chunk_length: {e}")
            raise e
    else:
        audio_n_samples_limit = None # not set here

    load_fix_audio = type(config.audio_chunk_length) == str and "fix" in config.audio_chunk_length

    if load_fix_audio:
        audio_chunk_length = int(config.audio_chunk_length.split("_")[1])
        audio_n_samples_limit = audio_chunk_length * config.audio_sampling_rate
    else:
        audio_chunk_length = config.audio_chunk_length

    def load_wav(speech_path):
        # speech = whisper.load_audio(speech_path, sr=config.audio_sampling_rate)
        speech, sr = librosa.load(speech_path, sr=config.audio_sampling_rate)
        cur_max_length = speech.shape[0]
        ori_audio_duration = cur_max_length / sr
        return speech, ori_audio_duration

    def get_audio(speech, audio_n_samples):

        if type(speech) == decord.audio_reader.AudioReader:
            ori_n_samples = speech.shape[1]
        else:
            ori_n_samples = speech.shape[0]

        # random audio smaple
        audio_start_sample_id = 0
        audio_end_sample_id = ori_n_samples


        load_max_audio = type(config.audio_chunk_length) == str and "max" in config.audio_chunk_length

        if hasattr(config, 'random_audio_sample') and not load_max_audio:
            if ori_n_samples > audio_n_samples:
                audio_start_sample_id = random.randint(0, ori_n_samples - audio_n_samples)
                audio_end_sample_id = audio_start_sample_id + audio_n_samples
        else:
            if load_max_audio:
                if "_" in config.audio_chunk_length:
                    max_audio_chunk_length = int(config.audio_chunk_length.split("_")[1])
                    max_audio_n_samples = max_audio_chunk_length * config.audio_sampling_rate
                    audio_n_samples = min(ori_n_samples, max_audio_n_samples)
                    audio_end_sample_id = audio_n_samples
                else:
                    audio_n_samples = ori_n_samples
                    audio_end_sample_id = audio_n_samples
            else:
                audio_end_sample_id = min(audio_n_samples, ori_n_samples)

        if type(speech) == decord.audio_reader.AudioReader:
            speech = speech[audio_start_sample_id:audio_end_sample_id].asnumpy()[0]
        else:
            speech = speech[audio_start_sample_id:audio_end_sample_id]


        return speech, audio_n_samples, audio_start_sample_id, audio_end_sample_id

    if isinstance(speech_path, dict):
        if "offset" in speech_path:
            # avlm dataset
            speech, ori_sample_rate = _load_audio(speech_path, config)

        else:
            speech = speech_path["sample"]
            ori_sample_rate = speech_path["sample_rate"]

        # resample the speech based on  current sample rate
        speech = librosa.resample(speech, orig_sr=ori_sample_rate, target_sr=config.audio_sampling_rate)
        # variable audio sequence lengths
        ori_audio_duration = speech.shape[0] / config.audio_sampling_rate
        speech, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(speech, audio_n_samples_limit)

    elif isinstance(speech_path, BytesIO):
        if speech.extension == ".wav":
            # speech, sr = librosa.load(speech_path, sr=config.audio_sampling_rate)
            # ori_audio_duration = speech.shape[0] / sr
            speech, ori_audio_duration = load_wav(speech_path)
            speech, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(speech, audio_n_samples_limit)
        else:
            raise ValueError(f"Unsupported audio extension: {speech.extension}")

    elif ".mat" in speech_path or ".ark" in speech_path:
        rate, speech = kaldiio.load_mat(speech_path)
        speech = librosa.resample(speech, orig_sr=rate, target_sr=config.audio_sampling_rate)
        speech, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(speech, audio_n_samples_limit)
        ori_audio_duration = speech.shape[0] / config.audio_sampling_rate
    elif ".mp4" in speech_path:
        # Load audio from video file
        ar = AudioReader(speech_path, ctx=cpu(0), sample_rate=config.audio_sampling_rate, mono=True)
        cur_max_length = ar.shape[1]
        ori_audio_duration = cur_max_length / config.audio_sampling_rate
        speech, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(ar, audio_n_samples_limit)
        # print(f"\033[31mFind speech in video: {speech_path}, speech shape: {speech.shape}\033[0m")
    else:
        assert os.path.exists(speech_path), f"File {speech_path} does not exist"
        speech, ori_audio_duration = load_wav(speech_path)
        speech, audio_n_samples, audio_start_sample_id, audio_end_sample_id = get_audio(speech, audio_n_samples_limit)

    # convert to float
    speech = speech.astype(np.float32)
    # speech = whisper.pad_or_trim(speech, length=cur_max_length) # this will leads to dis-sync issue in audio tower, where the audio tower futher split and process the feature sequence

    # Make the audio length a multiple of 30 seconds
    # if type(config.audio_chunk_length) == str and "max" in config.audio_chunk_length:
    #     audio_n_samples = int(np.ceil(speech.shape[0] / (config.audio_sampling_rate * 30)) * (config.audio_sampling_rate * 30))
    # else:
    #     audio_n_samples = speech.shape[0]

    if not load_fix_audio:
        audio_n_samples = int(np.ceil(speech.shape[0] / (config.audio_sampling_rate * 30)) * (config.audio_sampling_rate * 30))
    else:
        # debug
        audio_n_samples = audio_chunk_length * config.audio_sampling_rate

    speech = whisper.pad_or_trim(speech, length=audio_n_samples) # we will also pad based on the max length of all audio samples in the batch size later

    new_audio_chunk_length = int(audio_n_samples // config.audio_sampling_rate)
    # if config.audio_chunk_length == "max":
    #     config.audio_chunk_length = int(audio_n_samples // config.audio_sampling_rate)
    #     print(f"New audio chunk length: {config.audio_chunk_length}")

    audio_start_sec = audio_start_sample_id / config.audio_sampling_rate
    audio_end_sample_sec = audio_end_sample_id / config.audio_sampling_rate

    audio_info = {}
    audio_info['new_audio_chunk_length'] = new_audio_chunk_length 
    audio_info['new_audio_n_samples'] = audio_n_samples
    audio_info['ori_audio_duration'] = ori_audio_duration
    audio_info['audio_start_sec'] = audio_start_sec
    audio_info['audio_end_sample_sec'] = audio_end_sample_sec

    # calculate the time of each audio sample
    # audio_info['audio_sample_times'] = []
    # for i in range(audio_n_samples):
    #     audio_info['audio_sample_times'].append(i / config.audio_sampling_rate)

    if False:
        # speech = whisper.load_audio(speech_path)
        # print(f"new loader speech shape: {speech.shape}")
        speech = whisper.pad_or_trim(speech, length=4800000)
        mel = whisper.log_mel_spectrogram(speech, n_mels=128)
        # print(f"new loader mel shape: {speech.shape}")
        speech_outputs.append(mel)
        speech_frames = speech_outputs[0]  # torch.stack(speech_outputs, dim=0)
        return speech_frames.numpy().tolist()

    if False: # we do stft later after we align the length of all audio samples in the batch size
        whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", chunk_length=new_audio_chunk_length, sampling_rate=config.audio_sampling_rate
        )

    if False:
        whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/lustre/fsw/portfolios/adlr/users/sreyang/flamingo_v2/NV-Whisper",  chunk_length=config.audio_chunk_length, sampling_rate=config.audio_sampling_rate)

        # use WhisperFeatureExtractor in transformers to load
        speech_features = whisper_feature_extractor(
            speech,
            sampling_rate=config.audio_sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
            hop_length=config.audio_hop_length,
        )
        if "attention_mask" not in speech_features:
            return None

    if False:
        whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("/lustre/fsw/portfolios/adlr/users/sreyang/flamingo_v2/NV-Whisper",  chunk_length=config.audio_chunk_length, sampling_rate=config.audio_sampling_rate)

        # use WhisperFeatureExtractor in transformers to load
        speech_features = whisper_feature_extractor(
            speech,
            sampling_rate=config.audio_sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        if "attention_mask" not in speech_features:
            return None


    if PROFILE_MODE:
        time_3 = time.time()
        logger.warning(f"[Rank {rank}] audio load_speech time: {time_3 - time_0}. Audio length: {audio_info['ori_audio_duration']}. speech_path: {speech_path}")
        time_0 = time.time()
    return speech, audio_info


_load_sound = _load_speech

# def _load_sound(sound_path: str):
#     # Load video frames from a directory
#     if sound_path is None:
#         return None
#     sound_outputs = []
#     try:
#         sound = whisper.load_audio(sound_path)
#         sound = whisper.pad_or_trim(sound)
#         mel = whisper.log_mel_spectrogram(sound, n_mels=128)
#         sound_outputs.append(mel.unsqueeze(0))
#     except:
#         sound_outputs.append(torch.zeros(1,128,30000))
#     sound_frames = torch.stack(sound_outputs, dim=0)
#     return sound_frames.numpy().tolist()


def _extract_speech(speech: Speech, config: PretrainedConfig):
    frames, audio_info = _load_speech(speech, config)
    return frames, audio_info

_extract_sound = _extract_speech
def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)

    # if 'load_audio_in_video' not in config:
    if not hasattr(config, "load_audio_in_video"):
        print(f"Warning: load_audio_in_video not in config, set to False")
        config.load_audio_in_video = False

    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        print(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    # media["video"].append(_extract_video(part, config))
                    if config.load_audio_in_video:
                        output, aud_fea, video_info = _extract_video(part, config)
                        media["video"].append(output)
                        media["video_info"].append(video_info)
                        if aud_fea is not None:
                            media["sound"].append(aud_fea)
                            media["audio_info"].append(video_info['audio_info'])
                            text += MEDIA_TOKENS["sound"]
                    else:
                        output = _extract_video(part, config)
                        media["video"].append(output)
                text += MEDIA_TOKENS["video"]
            elif isinstance(part, Speech):
                if draft:
                    if config.unified_audio_encoder:
                        media["sound"].append(part)
                        text += MEDIA_TOKENS["sound"]
                    else:
                        media["speech"].append(part)
                        text += MEDIA_TOKENS["speech"]
                else:
                    output, audio_info = _extract_speech(part, config)
                    if output is not None:
                        if config.unified_audio_encoder:
                            media["sound"].append(output)
                            text += MEDIA_TOKENS["sound"]
                        else:
                            media["speech"].append(output)
                            text += MEDIA_TOKENS["speech"]
                        media["audio_info"].append(audio_info)
            elif isinstance(part, Sound):
                if draft:
                    media["sound"].append(part)
                    text += MEDIA_TOKENS["sound"]
                else:
                    # media["sound"].append(_extract_sound(part, config))
                    output, audio_info = _extract_sound(part, config)
                    if output is not None:
                        media["sound"].append(output)
                        media["audio_info"].append(audio_info)
                        text += MEDIA_TOKENS["sound"]
            else:
                print(f"part: {part}")
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
