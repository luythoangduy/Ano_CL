"""
UniADMemory with SEMA Integration
SEMA adapters added to Transformer Encoder/Decoder layers for Continual Learning
"""

import copy
import logging
from typing import Optional

import torch
from torch import Tensor, nn

# Import original components
from models.reconstructions.dumenet import (
    MemoryModule,
    TransformerEncoder,
    TransformerDecoder,
    _get_activation_fn,
    _get_clones,
    build_position_embedding,
)

# Import SEMA components
from models.sema_modules import SEMAModules


class SEMATransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with SEMA Adapters

    Architecture:
        input → Self-Attention → [+ SEMA Adapter] → FFN → [+ SEMA Adapter] → output

    SEMA adapters can be inserted:
        - After attention (parallel or sequential)
        - After FFN (parallel or sequential)
    """
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        sema_config=None,
        layer_id=0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_id = layer_id

        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # SEMA adapters
        self.use_sema = sema_config is not None and sema_config.get('use_sema', False)
        if self.use_sema:
            self.sema_position = sema_config.get('sema_position', 'ffn')  # 'attn', 'ffn', or 'both'
            self.sema_mode = sema_config.get('sema_mode', 'parallel')  # 'parallel' or 'sequential'

            if self.sema_position in ['attn', 'both']:
                self.sema_attn = SEMAModules(
                    layer_id=layer_id,
                    config=sema_config,
                    d_model=hidden_dim
                )
            else:
                self.sema_attn = None

            if self.sema_position in ['ffn', 'both']:
                self.sema_ffn = SEMAModules(
                    layer_id=layer_id,
                    config=sema_config,
                    d_model=hidden_dim
                )
            else:
                self.sema_ffn = None
        else:
            self.sema_attn = None
            self.sema_ffn = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Post-normalization forward (original UniAD style)"""
        sema_rd_loss = torch.tensor(0.0, device=src.device)
        sema_added = False

        # Self-attention
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # SEMA adapter after attention
        if self.sema_attn is not None:
            sema_out = self.sema_attn(src)
            if self.sema_mode == 'parallel':
                src = src + sema_out['output']
            else:  # sequential
                src = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # SEMA adapter after FFN
        if self.sema_ffn is not None:
            sema_out = self.sema_ffn(src)
            if self.sema_mode == 'parallel':
                src = src + sema_out['output']
            else:  # sequential
                src = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        return src, sema_rd_loss, sema_added

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Pre-normalization forward"""
        sema_rd_loss = torch.tensor(0.0, device=src.device)
        sema_added = False

        # Self-attention with pre-norm
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)

        # SEMA adapter after attention
        if self.sema_attn is not None:
            sema_out = self.sema_attn(src)
            if self.sema_mode == 'parallel':
                src = src + sema_out['output']
            else:
                src = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        # FFN with pre-norm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        # SEMA adapter after FFN
        if self.sema_ffn is not None:
            sema_out = self.sema_ffn(src)
            if self.sema_mode == 'parallel':
                src = src + sema_out['output']
            else:
                src = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        return src, sema_rd_loss, sema_added

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class SEMATransformerEncoder(nn.Module):
    """Transformer Encoder with SEMA layers"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        pos = torch.cat(
            [pos.unsqueeze(1)] * src.size(1), dim=1
        )  # (H X W) x B x C

        total_rd_loss = torch.tensor(0.0, device=src.device)
        any_added = False

        for layer in self.layers:
            output, rd_loss, added = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            total_rd_loss = total_rd_loss + rd_loss
            any_added = any_added or added

        if self.norm is not None:
            output = self.norm(output)

        return output, total_rd_loss, any_added


class SEMATransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer with SEMA Adapters"""
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        sema_config=None,
        layer_id=0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_id = layer_id

        # Standard transformer decoder components
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # SEMA adapters
        self.use_sema = sema_config is not None and sema_config.get('use_sema', False)
        if self.use_sema:
            self.sema_position = sema_config.get('sema_position', 'ffn')
            self.sema_mode = sema_config.get('sema_mode', 'parallel')

            if self.sema_position in ['attn', 'both']:
                self.sema_self_attn = SEMAModules(
                    layer_id=layer_id,
                    config=sema_config,
                    d_model=hidden_dim
                )
                self.sema_cross_attn = SEMAModules(
                    layer_id=layer_id,
                    config=sema_config,
                    d_model=hidden_dim
                )
            else:
                self.sema_self_attn = None
                self.sema_cross_attn = None

            if self.sema_position in ['ffn', 'both']:
                self.sema_ffn = SEMAModules(
                    layer_id=layer_id,
                    config=sema_config,
                    d_model=hidden_dim
                )
            else:
                self.sema_ffn = None
        else:
            self.sema_self_attn = None
            self.sema_cross_attn = None
            self.sema_ffn = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        sema_rd_loss = torch.tensor(0.0, device=tgt.device)
        sema_added = False

        # Self attention
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # SEMA after self-attention
        if self.sema_self_attn is not None:
            sema_out = self.sema_self_attn(tgt)
            if self.sema_mode == 'parallel':
                tgt = tgt + sema_out['output']
            else:
                tgt = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        # Cross attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # SEMA after cross-attention
        if self.sema_cross_attn is not None:
            sema_out = self.sema_cross_attn(tgt)
            if self.sema_mode == 'parallel':
                tgt = tgt + sema_out['output']
            else:
                tgt = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # SEMA after FFN
        if self.sema_ffn is not None:
            sema_out = self.sema_ffn(tgt)
            if self.sema_mode == 'parallel':
                tgt = tgt + sema_out['output']
            else:
                tgt = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        return tgt, sema_rd_loss, sema_added

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        sema_rd_loss = torch.tensor(0.0, device=tgt.device)
        sema_added = False

        # Self attention with pre-norm
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        # SEMA after self-attention
        if self.sema_self_attn is not None:
            sema_out = self.sema_self_attn(tgt)
            if self.sema_mode == 'parallel':
                tgt = tgt + sema_out['output']
            else:
                tgt = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        # Cross attention with pre-norm
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        # SEMA after cross-attention
        if self.sema_cross_attn is not None:
            sema_out = self.sema_cross_attn(tgt)
            if self.sema_mode == 'parallel':
                tgt = tgt + sema_out['output']
            else:
                tgt = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        # FFN with pre-norm
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # SEMA after FFN
        if self.sema_ffn is not None:
            sema_out = self.sema_ffn(tgt)
            if self.sema_mode == 'parallel':
                tgt = tgt + sema_out['output']
            else:
                tgt = sema_out['output']
            sema_rd_loss = sema_rd_loss + sema_out['rd_loss']
            sema_added = sema_added or sema_out['added']

        return tgt, sema_rd_loss, sema_added

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos
        )


class SEMATransformerDecoder(nn.Module):
    """Transformer Decoder with SEMA layers"""
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = tgt
        pos = torch.cat([pos.unsqueeze(1)] * tgt.size(1), dim=1)

        intermediate = []
        total_rd_loss = torch.tensor(0.0, device=tgt.device)
        any_added = False

        for layer in self.layers:
            output, rd_loss, added = layer(
                output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask, pos=pos
            )
            total_rd_loss = total_rd_loss + rd_loss
            any_added = any_added or added

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), total_rd_loss, any_added

        return output, total_rd_loss, any_added


# Note: UniADMemorySEMA will be in the next file due to length
