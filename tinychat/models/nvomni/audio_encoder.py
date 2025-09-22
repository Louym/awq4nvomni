import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.cfg_only = None

    def forward(self, sounds):
        if type(sounds) is list:
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features"):
                    sound = sound["input_features"]
                sound_feature = self.audio_tower(sound)
                sound_feature = sound_feature.last_hidden_state
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")
            if type(sounds) is dict and "input_ids" in sounds:
                sounds = sounds["input_features"]
            sound_features = self.audio_tower(sounds)
            sound_features = sound_features.last_hidden_state
            sound_features = sound_features.to(sounds.dtype)

        return sound_features, audio_output_lengths

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size
