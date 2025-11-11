"""
SEMA Modules Manager for Continual Learning
Manages multiple adapters with self-expansion capability
"""

import torch
from torch import nn
import copy
import logging
from models.sema_components import Adapter, RepresentationDescriptor, RDLossRecords


class AdapterModule(nn.Module):
    """
    Single Adapter Module with Representation Descriptor

    Components:
        - Functional adapter: Bottleneck adapter for feature transformation
        - Representation descriptor: AutoEncoder for distribution shift detection

    Args:
        adapter_id: Unique identifier (e.g., "layer_0.adapter_0")
        config: Configuration dict with SEMA settings
        d_model: Feature dimension
        enable_rd: Whether to use RD for this adapter
    """
    def __init__(self, adapter_id, config, d_model=256, enable_rd=True):
        super().__init__()

        self.adapter_id = adapter_id
        self.config = config
        self.d_model = d_model
        self.enable_rd = enable_rd

        # Functional adapter
        self.functional = Adapter(
            d_model=d_model,
            bottleneck=config.get('adapter_bottleneck', d_model // 4),
            dropout=config.get('adapter_dropout', 0.1),
            init_option="lora",
            adapter_scalar=config.get('adapter_scalar', 1.0),
            adapter_layernorm_option=config.get('adapter_layernorm', 'in')
        )

        # Representation descriptor (for distribution shift detection)
        if self.enable_rd:
            rd_dim = config.get('rd_dim', 64)
            self.rd = RepresentationDescriptor(d_model=d_model, rd_dim=rd_dim)
            buffer_size = config.get('rd_buffer_size', 500)
            self.rd_loss_record = RDLossRecords(max_len=buffer_size)
        else:
            self.rd = None
            self.rd_loss_record = None

        self.newly_added = True

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [num_tokens, batch, d_model] or [batch, seq_len, d_model]

        Returns:
            dict with:
                - func_out: Adapter output
                - rd_loss: RD reconstruction loss (if enabled)
                - z_score: Z-score for expansion decision (if enabled)
        """
        # Functional adapter
        func_out = self.functional(x)

        # Representation descriptor
        if not self.enable_rd or self.rd is None:
            return {
                'func_out': func_out,
                'rd_loss': torch.tensor(0.0, device=x.device),
                'z_score': torch.tensor(0.0, device=x.device)
            }

        # Compute RD loss
        rd_loss = self.rd.compute_reconstruction_loss(x)  # [batch]

        # Compute Z-score
        z_score = self.rd_loss_record.get_z_score(rd_loss)

        # Update RD loss records during training
        if self.training:
            self.rd_loss_record.add_record(rd_loss)

        return {
            'func_out': func_out,
            'rd_loss': rd_loss,
            'z_score': z_score
        }

    def freeze_functional(self):
        """Freeze functional adapter parameters"""
        for param in self.functional.parameters():
            param.requires_grad = False

    def freeze_rd(self):
        """Freeze RD and stop updating records"""
        if self.rd is not None:
            for param in self.rd.parameters():
                param.requires_grad = False
            self.rd_loss_record.freeze()

    def freeze(self):
        """Freeze both functional and RD"""
        self.freeze_functional()
        self.freeze_rd()
        self.newly_added = False


class SEMAModules(nn.Module):
    """
    SEMA Module Manager - Manages multiple adapters with self-expansion

    Features:
        - Multiple adapters per layer
        - Router network for mixing adapter outputs
        - Self-expansion based on distribution shift detection
        - Automatic adapter addition when Z-score exceeds threshold

    Args:
        layer_id: Layer identifier
        config: SEMA configuration dict
        d_model: Feature dimension
    """
    def __init__(self, layer_id, config, d_model=256):
        super().__init__()

        self.layer_id = layer_id
        self.config = config
        self.d_model = d_model

        # Adapter settings
        self.adapt_start_layer = config.get('sema_start_layer', 0)
        self.adapt_end_layer = config.get('sema_end_layer', 100)
        self.exp_threshold = config.get('expansion_threshold', 3.0)  # Z-score threshold

        # Check if this layer should use adapters
        self.is_adaptation_layer = (
            self.adapt_start_layer <= layer_id <= self.adapt_end_layer
        )

        # Adapter list
        self.adapters = nn.ModuleList()

        # Initialize with one adapter
        self._add_adapter(initialize=True)

        # Router for mixing adapter outputs
        self.router = nn.Linear(d_model, 1)
        self.new_router = None  # Temporary router for newly added adapter

        # Expansion control
        self.detecting_outlier = False  # Enable during inference to detect outliers
        self.added_for_task = True  # Track if adapter was added in current task

    @property
    def num_adapters(self):
        """Number of adapters in this layer"""
        return len(self.adapters)

    def _add_adapter(self, initialize=False):
        """
        Add a new adapter to this layer

        Args:
            initialize: If True, this is the first adapter (no router update)
        """
        adapter_id = f"layer_{self.layer_id}.adapter_{len(self.adapters)}"
        print(f"Adding adapter: {adapter_id}")
        # Create new adapter
        new_adapter = AdapterModule(
            adapter_id=adapter_id,
            config=self.config,
            d_model=self.d_model,
            enable_rd=self.is_adaptation_layer
        )

        self.adapters.append(new_adapter)
        self.added_for_task = True

        # Update router if not first adapter
        if not initialize:
            self.new_router = nn.Linear(self.d_model, 1)
            logging.info(f"✨ Adapter {adapter_id} added at layer {self.layer_id}")

    def _merge_routers(self):
        """Merge temporary router into main router"""
        if self.new_router is None:
            return

        # Create new router with increased output dimension
        new_dim = len(self.adapters)
        merged_router = nn.Linear(self.d_model, new_dim)

        # Copy weights from old router
        with torch.no_grad():
            old_weight = self.router.weight.data  # [old_dim, d_model]
            old_bias = self.router.bias.data  # [old_dim]

            new_weight = self.new_router.weight.data  # [1, d_model]
            new_bias = self.new_router.bias.data  # [1]

            # Concatenate
            merged_router.weight.data = torch.cat([old_weight, new_weight], dim=0)
            merged_router.bias.data = torch.cat([old_bias, new_bias], dim=0)

        self.router = merged_router
        self.new_router = None

    def forward(self, x):
        """
        Forward pass with self-expansion

        Args:
            x: Input tensor [num_tokens, batch, d_model] or [batch, seq_len, d_model]

        Returns:
            dict with:
                - output: Mixed adapter output
                - rd_loss: RD loss for training
                - added: Whether a new adapter was added
        """
        device = x.device

        # If not an adaptation layer, use only the first adapter
        if not self.is_adaptation_layer:
            out = self.adapters[0](x)
            return {
                'output': out['func_out'],
                'rd_loss': out['rd_loss'],
                'added': False
            }

        # Process all adapters
        func_outs = []
        rd_losses = []
        z_scores = []

        for adapter in self.adapters:
            out = adapter(x)
            func_outs.append(out['func_out'])
            rd_losses.append(out['rd_loss'])
            z_scores.append(out['z_score'])

        func_outs = torch.stack(func_outs)  # [num_adapters, ...]
        rd_losses = torch.stack(rd_losses)  # [num_adapters, batch]
        z_scores = torch.stack(z_scores)  # [num_adapters, batch]

        # Check expansion criteria
        should_expand = (
            self.detecting_outlier and
            not self.added_for_task and
            self.training and
            z_scores.mean(dim=1).min() > self.exp_threshold
        )
        print(f"Layer {self.layer_id} - Z-scores: {z_scores.mean(dim=1).tolist()} - Expand: {should_expand}")

        if should_expand:
            # Add new adapter
            self._add_adapter()

            # Return zero output (new adapter not trained yet)
            return {
                'output': torch.zeros_like(func_outs[0]),
                'rd_loss': torch.tensor(0.0, device=device),
                'added': True
            }

        # Mix adapter outputs using router
        # Get routing weights
        if x.dim() == 3:
            if x.shape[0] < x.shape[1]:  # [num_tokens, batch, d_model]
                router_input = x.mean(dim=0)  # [batch, d_model]
            else:  # [batch, seq_len, d_model]
                router_input = x.mean(dim=1)  # [batch, d_model]
        else:
            router_input = x

        logits = self.router(router_input)  # [batch, num_adapters-1]

        # Add new router logits if exists
        if self.new_router is not None:
            new_logits = self.new_router(router_input)  # [batch, 1]
            logits = torch.cat([logits, new_logits], dim=1)

        # Softmax to get weights
        weights = torch.softmax(logits, dim=1)  # [batch, num_adapters]

        # Mix outputs
        # func_outs: [num_adapters, ...] → need to broadcast weights
        if func_outs.dim() == 3:  # [num_adapters, num_tokens, batch, d_model]
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, batch, num_adapters, 1]
            weights = weights.permute(2, 0, 1, 3)  # [num_adapters, 1, batch, 1]
        elif func_outs.dim() == 4:  # [num_adapters, batch, seq_len, d_model]
            weights = weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, num_adapters, 1]
            weights = weights.permute(2, 0, 1, 3)  # [num_adapters, batch, 1, 1]

        mixed_output = (func_outs * weights).sum(dim=0)

        # RD loss (only for newly added adapter)
        if self.adapters[-1].newly_added:
            rd_loss = rd_losses[-1].mean()
        else:
            rd_loss = torch.tensor(0.0, device=device)

        return {
            'output': mixed_output,
            'rd_loss': rd_loss,
            'added': False
        }

    def end_task_training(self):
        """Called after each task - freeze adapters and merge routers"""
        # Freeze all adapters
        for adapter in self.adapters:
            adapter.freeze()

        # Merge routers if needed
        self._merge_routers()

        # Freeze router
        for param in self.router.parameters():
            param.requires_grad = False

        # Reset flags
        self.added_for_task = False

        logging.info(f"📌 Layer {self.layer_id}: Froze {len(self.adapters)} adapters")

    def enable_outlier_detection(self):
        """Enable outlier detection for next task"""
        self.detecting_outlier = True

    def disable_outlier_detection(self):
        """Disable outlier detection"""
        self.detecting_outlier = False
