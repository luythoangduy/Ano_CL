"""
SEMA Components for Continual Learning
Adapted from SEMA-CL (CVPR 2025) for Anomaly Detection with Transformers
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


class Adapter(nn.Module):
    """
    Bottleneck Adapter module

    Architecture:
        input → down_proj → ReLU → up_proj → output

    Args:
        d_model: Input/output dimension
        bottleneck: Bottleneck dimension (default: d_model // 4)
        dropout: Dropout rate
        init_option: Initialization method ("lora" or "bert")
        adapter_scalar: Scaling factor for adapter output
    """
    def __init__(
        self,
        d_model=256,
        bottleneck=None,
        dropout=0.1,
        init_option="lora",
        adapter_scalar=1.0,
        adapter_layernorm_option="in"
    ):
        super().__init__()

        self.n_embd = d_model
        self.down_size = bottleneck if bottleneck is not None else d_model // 4
        self.adapter_layernorm_option = adapter_layernorm_option

        # Layer norm before adapter
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # Scalar for adapter output
        if isinstance(adapter_scalar, str) and adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # Bottleneck layers
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = nn.Dropout(dropout)

        # Initialization
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        else:
            raise NotImplementedError(f"Init option {init_option} not supported")

    def forward(self, x):
        """
        Args:
            x: Input tensor [..., d_model]
        Returns:
            Adapter output [..., d_model]
        """
        residual = x
        device = x.device
        if next(self.adapter_layer_norm_before.parameters()).device != device:
            self.adapter_layer_norm_before = self.adapter_layer_norm_before.to(device)
        if next(self.down_proj.parameters()).device != device:
            self.down_proj = self.down_proj.to(device)
        if next(self.up_proj.parameters()).device != device:
            self.up_proj = self.up_proj.to(device)
        
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # Bottleneck
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = self.dropout(down)

        output = self.up_proj(down)
        output = self.dropout(output)

        # Scale
        if isinstance(self.scale, nn.Parameter):
            output = output * self.scale
        else:
            output = output * self.scale

        return output


class RepresentationDescriptor(nn.Module):
    """
    Representation Descriptor (RD) - AutoEncoder for distribution shift detection

    Training: Learn to reconstruct normal representations
    Testing: High reconstruction error indicates distribution shift → trigger expansion

    Args:
        d_model: Input dimension
        rd_dim: Bottleneck dimension for RD (smaller = more sensitive)
    """
    def __init__(self, d_model=256, rd_dim=64):
        super().__init__()

        self.input_dim = d_model
        self.rd_dim = rd_dim

        # AutoEncoder
        self.encoder = nn.Linear(self.input_dim, rd_dim)
        self.decoder = nn.Linear(rd_dim, self.input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming initialization"""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.encoder.bias)
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor [..., d_model]
        Returns:
            Reconstruction of x
        """
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return reconstruction

    def compute_reconstruction_loss(self, x):
        """
        Compute per-sample reconstruction loss

        Args:
            x: Input tensor [num_tokens, batch, d_model] or [batch, num_tokens, d_model]
        Returns:
            Reconstruction loss per sample [batch]
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Average over token dimension
            if x.shape[0] < x.shape[1]:  # [num_tokens, batch, d_model]
                x = x.mean(dim=0)  # [batch, d_model]
            else:  # [batch, num_tokens, d_model]
                x = x.mean(dim=1)  # [batch, d_model]

        # Reconstruct
        reconstruction = self.forward(x)

        # Per-sample MSE loss
        batch_size = x.shape[0]
        reconstruction_losses = []
        for i in range(batch_size):
            loss = F.mse_loss(reconstruction[i], x[i], reduction='mean')
            reconstruction_losses.append(loss)

        reconstruction_losses = torch.stack(reconstruction_losses)
        return reconstruction_losses


class RDLossRecords:
    """
    Records of RD reconstruction losses for Z-score computation

    Tracks mean and std of RD losses to detect outliers
    Z-score = (loss - mean) / std
    High Z-score → distribution shift → trigger expansion
    """
    def __init__(self, max_len=500):
        self._max_len = max_len
        self._curr_len = 0
        self.record = torch.zeros(self._max_len)
        self._mean = 0.0
        self._var = 0.0
        self.updating = True
        self.device = 'cpu'  # Track device

    @property
    def length(self):
        return self._curr_len

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return math.sqrt(self._var) if self._var > 0 else 1e-8

    def add_record(self, losses):
        """
        Add new loss values to record

        Args:
            losses: Tensor of loss values [batch_size]
        """
        if not self.updating:
            return

        # Update device from input - keep tensors on same device
        if losses.device != torch.device('cpu'):
            self.device = losses.device
            self.record = self.record.to(self.device)

        losses = losses.detach()  # Remove .cpu() - keep on same device as record

        if self._curr_len < self._max_len:
            # Still have space
            place_left = self._max_len - self._curr_len
            if place_left >= len(losses):
                self.record[self._curr_len:self._curr_len + len(losses)] = losses
                self._curr_len += len(losses)
            else:
                # Fill remaining space
                self.record[self._curr_len:] = losses[:place_left]
                self._curr_len = self._max_len
        else:
            # Buffer full - rolling update
            self.record = torch.cat([self.record, losses])
            self.record = self.record[len(losses):]

        # Update statistics
        valid_records = self.record[:self._curr_len]
        self._mean = torch.mean(valid_records).item()
        self._var = torch.var(valid_records).item()

    def get_z_score(self, losses):
        """
        Compute Z-score for new losses

        Args:
            losses: Tensor [batch_size]
        Returns:
            Z-scores (absolute value) [batch_size]
        """
        if self._curr_len < 2:
            # Not enough data
            return torch.zeros_like(losses)

        # Compute z-score on same device as losses
        z_score = (losses - self._mean) / (self.stddev + 1e-8)
        z_score = torch.abs(z_score)
        return z_score  # Already on correct device

    def freeze(self):
        """Stop updating records (called after task ends)"""
        self.updating = False