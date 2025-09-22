import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awq.quantize import W8A8OF16LinearDynamicInputScale

from tinychat.utils.input_metadata import ActivationBuffer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from flash_attn import flash_attn_func
import time

CLIP_RANGE = 5

save_act=False
import awq_inference_engine

# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoder with Whisper->Qwen2Audio
class QuantQwen2AudioEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2AudioEncoderLayer`].

    Args:
        config: Qwen2AudioEncoderConfig
    """

    def __init__(self, module, bsz=1, seqlen=1500):
        super().__init__()
        self.config=module.config
        self.layers = [QuantQwen2AudioEncoderLayer(layer) for layer in module.layers]
        self.buffer = ActivationBuffer(module)
        self.bsz = bsz
        self.seqlen = seqlen
        self.buffer.allocate_activation_buffer(self.bsz * self.seqlen)
        
        
        self.dropout = module.dropout
        self.layerdrop = module.layerdrop

        self.d_model = module.layers[0].self_attn.q_proj.in_features
        self.num_mel_bins = module.num_mel_bins
        self.padding_idx = module.padding_idx
        self.max_source_positions = module.max_source_positions
        self.embed_scale = module.embed_scale
        
        self.conv1 = module.conv1
        self.conv2 =  module.conv2

        self.embed_positions = module.embed_positions
        self.embed_positions.requires_grad_(False)
        self.layer_norm = module.layer_norm
        # Ignore copy
        self.avg_pooler = module.avg_pooler

        self.gradient_checkpointing = False

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Qwen2Audio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Qwen2Audio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        assert not output_attentions, "Quantized model does not return attentions. Please set output_attentions=False."
        # Quantized Model
        bsz, seqlen, _ = hidden_states.shape
        # print(f"Quantized Model - Batch Size: {bsz}, Sequence Length: {seqlen}")
        hidden_states=hidden_states.contiguous() # Important! Our kernel only supports contiguous input!
        if self.bsz != bsz or self.seqlen != seqlen:
            self.buffer.allocate_activation_buffer(bsz * seqlen)
            self.bsz = bsz
            self.seqlen = seqlen
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (
                    hidden_states.reshape(bsz, seqlen, -1),
                )
            hidden_states = encoder_layer(
                hidden_states, self.buffer, head_mask[idx] if head_mask is not None else None, bsz, seqlen
            )
        hidden_states = hidden_states.reshape(bsz, seqlen, -1)
        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths



class QuantQwen2AudioAttention(nn.Module):
    def __init__(
        self,
        module,
        init_only=False,
    ):
        super().__init__()
        self.config = module.config
        self.embed_dim = module.embed_dim
        self.num_heads = module.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = W8A8OF16LinearDynamicInputScale.from_qkv(
            module.q_proj, module.k_proj, module.v_proj, init_only=init_only
        )
        self.out_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.out_proj, init_only=init_only
        )
        self.invoke_quant = self.invoke_quant_wo

    def invoke_quant_wo(self, buffer, attn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            attn_output,
            buffer.quantized_scale_buffer,
        )

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self, buffer: ActivationBuffer, bsz=64, seqlen=1024
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # qkv
        self.qkv_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.qkv_proj_act_buffer,
        )
        q, k, v = buffer.qkv_proj_act_buffer.split(
            [self.embed_dim, self.embed_dim, self.embed_dim], dim=-1
        )
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        attn_output = flash_attn_func(q, k, v, softmax_scale=None, causal=False)
        attn_output = attn_output.reshape(bsz * seqlen, -1)
        # FP16 -> int8
        self.invoke_quant(buffer, attn_output)
        # INT8 in, FP16 out
        self.out_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )


class QuantQwen2AudioEncoderLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.embed_dim = module.embed_dim
        self.self_attn = QuantQwen2AudioAttention(module.self_attn)
        self.layer_norm1 = RMSNormGeneral(
            module.self_attn_layer_norm.weight.data,
            module.self_attn_layer_norm.bias.data,
            module.self_attn_layer_norm.eps,
            True,
        ).cuda()
        self.self_attn_layer_norm=module.self_attn_layer_norm
        #MLP
        self.fc1 = W8A8OF16LinearDynamicInputScale.from_linear(
            module.fc1, init_only=False
        )
        self.fc2 = W8A8OF16LinearDynamicInputScale.from_linear(
            module.fc2, init_only=False
        )
        self.layer_norm2 = RMSNormGeneral(
            module.final_layer_norm.weight.data,
            module.final_layer_norm.bias.data,
            module.final_layer_norm.eps,
            True,
        ).cuda()
        # Dummy
        self.invoke_quant = self.invoke_quant_wo

    def invoke_quant_wo(self, buffer, attn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            attn_output,
            buffer.quantized_scale_buffer,
        )
            

    def forward(
        self,
        hidden_states: torch.Tensor,
        buffer: ActivationBuffer,
        attention_mask,
        bsz,
        seqlen,
    ) -> Tuple[torch.FloatTensor]:
        # Attention block
        # FP16 in int8 out, layernorm & quantization
        residual = hidden_states
        self.layer_norm1(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
        )
        # # INT8 -> FP16
        self.self_attn(buffer, bsz, seqlen)
        if save_act:
            torch.save(hidden_states, "qafterATTN.pt")
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        # Fully Connected
        residual = hidden_states
        # FP16 in int8 out, layernorm & quantization
        self.layer_norm2(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
        )
        if save_act:
            torch.save(buffer.quantized_hidden_states_buffer*buffer.quantized_scale_buffer.reshape(-1,1), "qafterLN2.pt")
        # INT8 -> FP16
        self.fc1(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.fc1_buffer,
        )
        # Act & quantization
        awq_inference_engine.gelu_and_quant(
            buffer.quantized_mlp_act_buffer,
            buffer.fc1_buffer,
            buffer.quantized_scale_buffer,
            buffer.tmp,
            False,# Do not approximate gelu
        )
        if save_act:
            torch.save(buffer.quantized_mlp_act_buffer*buffer.quantized_scale_buffer.reshape(-1,1), "qafterACT.pt")
        # INT8 in, FP16 out
        self.fc2(
            buffer.quantized_mlp_act_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        if save_act:
            torch.save(hidden_states, "qafterRES.pt")
            exit()
        return hidden_states


class RMSNormGeneral(nn.Module):
    """Root mean square normalization (w/ per-token or per-tensor quant).

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        weight: torch.tensor,
        bias: torch.tensor,
        eps: float = 1e-6,
        use_per_token_quant: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.variance_epsilon = eps
        self.use_per_token_quant = use_per_token_quant

    def forward(
        self,
        x: torch.Tensor,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor = None,
    ) -> torch.Tensor:
        # quantized_sum_buffer is not used, only to keep the consistency of the interface
        awq_inference_engine.rms_norm_general(
            quantized_hidden_states_buffer,
            x,
            self.weight.data,
            self.bias.data,
            quantized_scale_buffer,
            self.variance_epsilon,
            self.use_per_token_quant,
        )
