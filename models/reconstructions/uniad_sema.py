"""
UniADMemory with SEMA for Continual Learning
Complete reconstruction model with self-expanding adapters
"""

import os
import random
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import numpy as np

from models.initializer import initialize_from_cfg
from models.reconstructions.dumenet import MemoryModule, build_position_embedding
from models.reconstructions.dumenet_sema import (
    SEMATransformerEncoderLayer,
    SEMATransformerEncoder,
    SEMATransformerDecoderLayer,
    SEMATransformerDecoder,
)


class UniADMemorySEMA(nn.Module):
    """
    UniAD Memory Reconstruction Model with SEMA Continual Learning

    Architecture:
        Input Features → Projection → Encoder+SEMA → Memory → Decoder+SEMA → Output

    SEMA Features:
        - Adapters in Encoder/Decoder layers
        - Self-expansion when detecting distribution shift
        - Mixture of adapters with learned router

    Args:
        inplanes: Input channel list
        instrides: Input stride list
        feature_size: Spatial size [H, W]
        feature_jitter: Data augmentation config
        hidden_dim: Transformer dimension
        pos_embed_type: Type of positional embedding ('learned' or 'sine')
        save_recon: Config for saving reconstruction
        initializer: Weight initialization config
        sema_config: SEMA configuration dict
        **kwargs: Additional transformer args (nhead, num_layers, etc.)
    """

    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        sema_config=None,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1

        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.save_recon = save_recon
        self.hidden_dim = hidden_dim

        # Positional encoding
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )

        # Input projection
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)

        # Memory module (same as original UniAD)
        self.memory_size = kwargs.get('memory_size', 256)
        self.memory_module = MemoryModule(
            mem_dim=self.memory_size,
            feature_dim=hidden_dim,
            **kwargs
        )

        # SEMA configuration
        self.use_sema = sema_config is not None and sema_config.get('use_sema', False)
        self.sema_config = sema_config if self.use_sema else {}

        # Transformer Encoder with SEMA
        num_encoder_layers = kwargs.get('num_encoder_layers', 4)
        encoder_layer = SEMATransformerEncoderLayer(
            hidden_dim,
            kwargs.get('nhead', 8),
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False),
            sema_config=sema_config,
            layer_id=0,  # Will be overridden when cloning
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if kwargs.get('normalize_before', False) else None

        # Create encoder layers with unique IDs
        self.encoder = self._create_encoder(
            encoder_layer, num_encoder_layers, encoder_norm, sema_config
        )

        # Transformer Decoder with SEMA
        num_decoder_layers = kwargs.get('num_decoder_layers', 4)
        decoder_layer = SEMATransformerDecoderLayer(
            hidden_dim,
            kwargs.get('nhead', 8),
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False),
            sema_config=sema_config,
            layer_id=0,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)

        # Create decoder layers with unique IDs
        self.decoder = self._create_decoder(
            decoder_layer, num_decoder_layers, decoder_norm, sema_config, offset=num_encoder_layers
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])

        # Upsampling
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        # Initialize parameters
        initialize_from_cfg(self, initializer)

    def _create_encoder(self, encoder_layer, num_layers, norm, sema_config):
        """Create encoder with properly indexed SEMA layers"""
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SEMATransformerEncoderLayer(
                encoder_layer.hidden_dim,
                encoder_layer.self_attn.num_heads,
                encoder_layer.linear1.out_features,
                encoder_layer.dropout.p,
                'relu',
                encoder_layer.normalize_before,
                sema_config=sema_config,
                layer_id=i,
            )
            layers.append(layer)

        return SEMATransformerEncoder(layers[0], num_layers, norm) if num_layers == 1 else \
               self._manual_encoder(layers, norm)

    def _manual_encoder(self, layers, norm):
        """Create encoder from pre-made layers"""
        encoder = SEMATransformerEncoder(layers[0], len(layers), norm)
        encoder.layers = layers
        return encoder

    def _create_decoder(self, decoder_layer, num_layers, norm, sema_config, offset=0):
        """Create decoder with properly indexed SEMA layers"""
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SEMATransformerDecoderLayer(
                decoder_layer.hidden_dim,
                decoder_layer.self_attn.num_heads,
                decoder_layer.linear1.out_features,
                decoder_layer.dropout.p,
                'relu',
                decoder_layer.normalize_before,
                sema_config=sema_config,
                layer_id=offset + i,
            )
            layers.append(layer)

        return SEMATransformerDecoder(layers[0], num_layers, norm) if num_layers == 1 else \
               self._manual_decoder(layers, norm)

    def _manual_decoder(self, layers, norm):
        """Create decoder from pre-made layers"""
        decoder = SEMATransformerDecoder(layers[0], len(layers), norm, return_intermediate=False)
        decoder.layers = layers
        return decoder

    def add_jitter(self, feature_tokens, scale, prob):
        """Add random jitter to features during training"""
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        """
        Forward pass

        Args:
            input: Dict with 'feature_align' [B, C, H, W]

        Returns:
            Dict with:
                - feature_rec: Reconstructed features
                - feature_align: Normalized input features
                - pred: Anomaly map
                - sema_rd_loss: SEMA RD loss (for training)
                - attention: Memory attention weights
        """
        feature_align = input["feature_align"]  # B x C x H x W

        # Convert to tokens
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C

        # Add jitter during training
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )

        # Project input features
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        feature_tokens = F.layer_norm(feature_tokens, feature_tokens.shape[-1:])

        # Get positional embeddings
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C

        # Encode features with SEMA
        if self.use_sema:
            encoded_tokens, encoder_rd_loss, encoder_added = self.encoder(
                feature_tokens, pos=pos_embed
            )
        else:
            # Original encoder (no SEMA)
            from models.reconstructions.dumenet import TransformerEncoder
            if not hasattr(self, '_original_encoder'):
                # Fallback: treat as SEMA encoder with no loss
                encoded_tokens, encoder_rd_loss, encoder_added = self.encoder(
                    feature_tokens, pos=pos_embed
                )
            else:
                encoded_tokens = self._original_encoder(feature_tokens, pos=pos_embed)
                encoder_rd_loss = torch.tensor(0.0, device=feature_tokens.device)
                encoder_added = False

        # Memory retrieval
        memory_result = self.memory_module(encoded_tokens)
        memory_features = memory_result['output']  # (H x W) x B x C

        # Decode features with SEMA
        if self.use_sema:
            decoded_tokens, decoder_rd_loss, decoder_added = self.decoder(
                memory_features,
                encoded_tokens,
                pos=pos_embed
            )
        else:
            # Original decoder
            if not hasattr(self, '_original_decoder'):
                decoded_tokens, decoder_rd_loss, decoder_added = self.decoder(
                    memory_features, encoded_tokens, pos=pos_embed
                )
            else:
                decoded_tokens = self._original_decoder(
                    memory_features, encoded_tokens, pos=pos_embed
                )
                decoder_rd_loss = torch.tensor(0.0, device=feature_tokens.device)
                decoder_added = False

        # Project back to original dimension
        feature_rec_tokens = self.output_proj(decoded_tokens)  # (H x W) x B x C
        feature_rec_tokens = torch.sigmoid(feature_rec_tokens)

        # Reshape back to spatial
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C x H x W

        # Save reconstructed features if needed
        if not self.training and self.save_recon:
            self._save_reconstructions(input, feature_rec)

        # Compute prediction (reconstruction error)
        feature_align = torch.sigmoid(feature_align)
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W

        pred = self.upsample(pred)  # B x 1 x H x W

        # Prepare output
        output_dict = {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
            "attention": memory_result['att_weight'],
            "attention_scores": memory_result['attention_scores'],
        }

        # Add SEMA losses if using SEMA
        if self.use_sema:
            total_rd_loss = encoder_rd_loss + decoder_rd_loss
            output_dict["sema_rd_loss"] = total_rd_loss
            output_dict["sema_added"] = encoder_added or decoder_added

        return output_dict

    def _save_reconstructions(self, input, feature_rec):
        """Save reconstructed features to disk"""
        clsnames = input["clsname"]
        filenames = input["filename"]

        for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
            filedir, filename = os.path.split(filename)
            _, defename = os.path.split(filedir)
            filename_, _ = os.path.splitext(filename)

            save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
            os.makedirs(save_dir, exist_ok=True)

            feature_rec_np = feat_rec.detach().cpu().numpy()
            np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

    def get_sema_modules(self):
        """Get all SEMA modules for task-end operations"""
        sema_modules = []

        # Collect from encoder
        for layer in self.encoder.layers:
            if hasattr(layer, 'sema_attn') and layer.sema_attn is not None:
                sema_modules.append(layer.sema_attn)
            if hasattr(layer, 'sema_ffn') and layer.sema_ffn is not None:
                sema_modules.append(layer.sema_ffn)

        # Collect from decoder
        for layer in self.decoder.layers:
            if hasattr(layer, 'sema_self_attn') and layer.sema_self_attn is not None:
                sema_modules.append(layer.sema_self_attn)
            if hasattr(layer, 'sema_cross_attn') and layer.sema_cross_attn is not None:
                sema_modules.append(layer.sema_cross_attn)
            if hasattr(layer, 'sema_ffn') and layer.sema_ffn is not None:
                sema_modules.append(layer.sema_ffn)

        return sema_modules

    def end_task_training(self):
        """Called at end of each task - freeze adapters"""
        if not self.use_sema:
            return

        sema_modules = self.get_sema_modules()
        for sema_module in sema_modules:
            sema_module.end_task_training()

    def enable_outlier_detection(self):
        """Enable outlier detection for next task"""
        if not self.use_sema:
            return

        sema_modules = self.get_sema_modules()
        for sema_module in sema_modules:
            sema_module.enable_outlier_detection()

    def disable_outlier_detection(self):
        """Disable outlier detection"""
        if not self.use_sema:
            return

        sema_modules = self.get_sema_modules()
        for sema_module in sema_modules:
            sema_module.disable_outlier_detection()
