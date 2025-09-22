import torch
from transformers import PretrainedConfig, Qwen2AudioEncoder, Qwen2AudioForConditionalGeneration

from .audio_encoder import AudioTower
from .modeling_qwen2_5_omni import Qwen2_5OmniModel

# class Qwen2AudioTower(SoundTower):
class Qwen2AudioTower(AudioTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.audio_tower = Qwen2AudioEncoder.from_pretrained(model_name_or_path, attn_implementation="flash_attention_2")
        self.is_loaded = True
        self.audio_chunk_unit_duration = 30
        self.audio_chunk_unit_length = 3000

    def forward(self, sounds):
        if type(sounds) is list:
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features") or (type(sound) is dict and "input_features" in sound):
                    sound = sound["input_features"]

                sound_feature = self.forward_audio_tower_batch(sound)
                # sound_feature = sound_feature.last_hidden_state
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            if len(sound_features) > 0:
                sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")

        return sound_features, audio_output_lengths

    def forward_audio_tower_batch(self, inp):
        """
        Process long audio input by splitting into fixed-size chunks (30s),
        pad if needed, batch them together, and run through the audio tower once.

        Args:
            inp: Tensor of shape (BS, n_mels, seq_len)

        Returns:
            Tensor of shape (BS, seq_chunks * chunk_seq_len, hidden_size)
        """
        try:
            batch_size, n_mels, seq_len = inp.shape
        except Exception as e:
            print(f"Error in audio tower forward: inp shape: {inp.shape if hasattr(inp, 'shape') else 'Invalid input'}")
            raise e

        chunk_length = self.audio_chunk_unit_length
        num_chunks = (seq_len + chunk_length - 1) // chunk_length  # Ceiling division

        padded_chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)

            # Extract the chunk
            chunk = inp[:, :, start_idx:end_idx]  # (BS, n_mels, chunk_len)

            # Pad if needed
            if chunk.shape[2] < chunk_length:
                pad_len = chunk_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len), mode='constant', value=0)

            padded_chunks.append(chunk)

        # Stack chunks along a new batch dimension: (BS * num_chunks, n_mels, chunk_len)
        all_chunks = torch.cat(padded_chunks, dim=0).reshape(batch_size * num_chunks, n_mels, chunk_length)

        # Forward pass through the audio tower once
        chunk_outputs = self.audio_tower(all_chunks)  # output shape: (BS * num_chunks, seq_len', hidden_size)
        hidden_states = chunk_outputs.last_hidden_state

        # Reshape back to (BS, num_chunks * seq_len', hidden_size)
        _, chunk_seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, num_chunks * chunk_seq_len, hidden_size)

        return hidden_states


    def forward_audio_tower(self, inp):
        """
        The audio_tower of Qwen2AudioEncoder can only process 30 seconds audio at a time.
        So we need to split or pad the audio into chunks of 30 seconds and process them one by one.
        mel-filter bank features (inp) shape: (BS, n_mels, seq_len)
        """
        # inp shape: (BS, n_mels, seq_len)
        try:
            batch_size, n_mels, seq_len = inp.shape
        except Exception as e:
            print(f"Error in audio tower forward: inp: {inp}")
            raise e
        
        # Calculate how many 30-second chunks we need
        # audio_chunk_unit_length = 3000 corresponds to 30 seconds of audio
        chunk_length = self.audio_chunk_unit_length
        num_chunks = (seq_len + chunk_length - 1) // chunk_length  # Ceiling division
        
        # Process each chunk
        chunk_outputs = []
        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)
            
            # Extract the chunk
            chunk = inp[:, :, start_idx:end_idx]
            
            # Pad the chunk if it's shorter than chunk_length
            if chunk.shape[2] < chunk_length:
                padding_length = chunk_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, padding_length), mode='constant', value=0)
            
            # Process the chunk through the audio tower
            chunk_output = self.audio_tower(chunk)
            
            chunk_outputs.append(chunk_output.last_hidden_state)
        
        # Concatenate all chunk outputs along the sequence dimension
        if len(chunk_outputs) == 1:
            # Single chunk, no need to concatenate
            final_output = chunk_outputs[0]
        else:
            # Multiple chunks, concatenate them
            final_output = torch.cat(chunk_outputs, dim=1)
        
        return final_output


class Qwen25omni_AudioTower(AudioTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        qwen = Qwen2_5OmniModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )  # need to specify dtype otherwise buggy in deepspeed
        self.audio_tower = qwen.thinker.audio_tower  # Qwen2AudioEncoder.from_pretrained(model_name_or_path)
        # self.sound_tower = Qwen2AudioEncoder.from_pretrained(model_name_or_path)
        self.is_loaded = True

    def qwen25o_process(self, audios_inputs):
        input_features = audios_inputs["input_features"]
        try:
            feature_attention_mask = audios_inputs.pop("attention_mask")
        except Exception as e:
            print(f"audios_inputs: {audios_inputs}")
            raise e

        # input_lengths = (feature_attention_mask.sum(-1).cpu().numpy() - 1) // 2 + 1  # becuause audio_tower does pooling
        """
        This line is bug:
        input_lengths = (feature_attention_mask.sum(-1).cpu().numpy() - 1) // 2 + 1 
        Numerical bug:
        (Pdb) feature_attention_mask.sum()
        tensor(27008., dtype=torch.float16)
        (Pdb) feature_attention_mask.shape
        torch.Size([1, 27000])
        (Pdb) feature_attention_mask.max()
        tensor(1., dtype=torch.float16)
        (Pdb) feature_attention_mask.min()
        tensor(1., dtype=torch.float16)
        (Pdb) torch.unique(feature_attention_mask)
        tensor([1.], dtype=torch.float16)
        (Pdb) torch.unique(input_lengths)
        tensor([27000], dtype=torch.int32)

        Reason: Data corruption or overflow
        Since this tensor is in float16, numerical issues can occur (e.g., due to limited precision).

        Some values might actually be slightly more than 1.0 due to rounding issues in float16, e.g., 1.00048828125.

        """
        input_lengths = ((feature_attention_mask==1).sum(-1).cpu().numpy() - 1) // 2 + 1  # becuause audio_tower does pooling

        audio_lengths = (input_lengths - 2) // 2 + 1

        # audio_feature_lengths = torch.sum(feature_attention_mask, dim=1).long() # buggy for the same reason as above
        audio_feature_lengths = torch.sum(feature_attention_mask==1, dim=1).long()
        input_features = (
            input_features.permute(0, 2, 1)[feature_attention_mask.bool()]
            .permute(1, 0)
            .to(device=self.device, dtype=self.dtype)
        )

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else (feature_attention_mask==1).sum(-1)
        ) # list of audio feature and embedding lengths of different samples in a batch
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else (feature_attention_mask==1).sum(-1)

        return input_features, feature_lens, audio_feat_lengths, audio_output_lengths

    def forward(self, sounds):
        if type(sounds) is list:
            sound_features = []
            # combine the batch
            batch_sounds = {k: torch.cat([sound[k] for sound in sounds], dim=0) for k in sounds[0].keys()}
            # batch_sounds: {"input_features": torch.tensor(BS, feature_size, sequence_length)}

            # print(f"audio batch_sounds shape: {batch_sounds['input_features'].shape}")
            # print(f"attention_mask.shape: {batch_sounds['attention_mask'].shape}")
            # print(f"attention_mask > 0: {batch_sounds['attention_mask'].bool().sum()}")

            input_features, feature_lens, audio_feat_lengths, audio_output_lengths = self.qwen25o_process(batch_sounds)
            # input_features: torch.tensor(feature_size, BS * sequence_length)
            # feature_lens: torch.tensor([sequence_length] * BS)

            # print(f"input_features shape: {input_features.shape}, feature_lens: {feature_lens}, audio_feat_lengths: {audio_feat_lengths}, audio_output_lengths: {audio_output_lengths}")

            audio_outputs = self.audio_tower(
                input_features,
                feature_lens=feature_lens,
                aftercnn_lens=audio_feat_lengths,
            )

            audio_features = audio_outputs.last_hidden_state
            # print(f"audio_features shape: {audio_features.shape}, audio_output_lengths: {audio_output_lengths}")

            if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                raise ValueError("length of audio_features should match audio_output_lengths")

            # split the batch
            if False:
                start = 0
                for length in audio_output_lengths:
                    sound_features.append(audio_features[start : start + length])
                    start += length

            # for sound in sounds:
            #     # sound: {'input_values': tensor([[-0.0001, -0.0001, -0.0001,  ..., -0.0001, -0.0001, -0.0001]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]])}
            #     input_features, feature_lens, audio_feat_lengths = self.qwen25o_process(sound)

            #     audio_outputs = self.audio_tower(
            #         input_features,
            #         feature_lens=feature_lens,
            #         aftercnn_lens=audio_feat_lengths,
            #     )

            #     sound_feature = sound_feature.to(sound.dtype)
            #     sound_feature = sound_feature.last_hidden_state
            #     sound_features.append(sound_feature)
        else:
            raise NotImplementedError
            sound_features = self.audio_tower(sounds)
            sound_features = sound_features.last_hidden_state
            sound_features = sound_features.to(sounds.dtype)

        return audio_features, audio_output_lengths
