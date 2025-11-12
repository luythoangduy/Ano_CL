"""
UniAD Learner with SEMA Support for Continual Learning
Extends UniADLearner to support SEMA self-expanding adapters
"""

import logging
import torch
from tqdm import tqdm

from models.uniad_learner import UniADLearner
from utils.criterion_helper import build_criterion
from utils.lr_helper import get_scheduler
from utils.optimizer_helper import get_optimizer


class UniADSEMALearner(UniADLearner):
    """
    SEMA-enhanced learner for UniAD anomaly detection

    Additional features:
        - SEMA RD loss training
        - Adapter self-expansion
        - End-of-task adapter freezing
        - Outlier detection mode
    """

    def __init__(self, args):
        super().__init__(args)

        # SEMA configuration
        self.use_sema = args.get('use_sema', False)
        if self.use_sema:
            self.sema_config = args.get('sema_config', {})
            self.rd_loss_weight = self.sema_config.get('rd_loss_weight', 0.1)
            logging.info(f"🔧 SEMA enabled with RD loss weight: {self.rd_loss_weight}")
        else:
            self.sema_config = None
            self.rd_loss_weight = 0.0

    def after_task(self):
        """
        Operations after each task
        Override to add SEMA-specific operations
        """
        # Call parent after_task
        super().after_task()

        # SEMA-specific: freeze adapters and enable outlier detection
        if self.use_sema and hasattr(self._network, 'reconstruction'):
            reconstruction = self._network.reconstruction

            # End task training (freeze adapters)
            if hasattr(reconstruction, 'end_task_training'):
                reconstruction.end_task_training()
                logging.info("🔒 SEMA adapters frozen after task")

            # Enable outlier detection for next task
            if hasattr(reconstruction, 'enable_outlier_detection'):
                reconstruction.enable_outlier_detection()
                logging.info("🔍 SEMA outlier detection enabled")

    def _train(self, train_loader, test_loader, config):
        """
        Train model on current task with SEMA support

        Args:
            train_loader: Training data loader
            test_loader: Test data loader for validation
            config: Training configuration
        """
        self._network.to(self._device)
        self._network.train()

        # Freeze specified layers
        for layer_name in self.frozen_layers:
            if hasattr(self._network, layer_name):
                layer = getattr(self._network, layer_name)
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

        # Setup optimizer for active layers only
        parameters = []
        for layer_name in self.active_layers:
            if hasattr(self._network, layer_name):
                layer = getattr(self._network, layer_name)
                # Filter for trainable parameters
                trainable_params = [p for p in layer.parameters() if p.requires_grad]
                if trainable_params:
                    parameters.append({"params": trainable_params})
                    logging.info(f"  Layer '{layer_name}': {len(trainable_params)} trainable params")
                else:
                    logging.warning(f"  Layer '{layer_name}': No trainable parameters!")

        # Fallback: If no parameters in active_layers, collect all trainable params
        if not parameters:
            logging.warning("⚠️ No trainable parameters in active_layers! Collecting all trainable params...")
            all_trainable = [p for p in self._network.parameters() if p.requires_grad]
            if all_trainable:
                parameters = [{"params": all_trainable}]
                logging.info(f"✓ Found {len(all_trainable)} trainable parameters across all layers")
            else:
                raise ValueError("No trainable parameters found in the entire network!")

        # Create optimizer and scheduler
        optimizer_config = config.trainer.optimizer
        optimizer = get_optimizer(parameters, optimizer_config)

        scheduler_config = config.trainer.lr_scheduler
        lr_scheduler = get_scheduler(optimizer, scheduler_config)

        # Build criterion
        criterion = build_criterion(config.criterion)

        # Training loop
        logging.info(f"Training for {self.epochs_per_task} epochs")

        total_params = sum(p.numel() for p in self._network.parameters())
        trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)

        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

        # SEMA: Check for adapters
        if self.use_sema:
            num_adapters = self._count_adapters()
            logging.info(f"🔧 SEMA adapters: {num_adapters}")

        prog_bar = tqdm(range(self.epochs_per_task))

        for epoch in prog_bar:
            self._network.train()

            # Freeze layers again
            for layer_name in self.frozen_layers:
                if hasattr(self._network, layer_name):
                    layer = getattr(self._network, layer_name)
                    layer.eval()

            epoch_loss = 0.0
            epoch_rd_loss = 0.0
            num_batches = 0
            num_expansions = 0

            for i, input in enumerate(train_loader):
                # Forward pass
                outputs = self._network(input)

                # Compute main loss
                loss = 0
                for name, criterion_loss in criterion.items():
                    weight = criterion_loss.weight
                    loss += weight * criterion_loss(outputs)

                # Add SEMA RD loss if available
                sema_rd_loss = 0
                if self.use_sema and 'sema_rd_loss' in outputs:
                    sema_rd_loss = outputs['sema_rd_loss']
                    loss = loss + self.rd_loss_weight * sema_rd_loss

                    epoch_rd_loss += sema_rd_loss.item()

                # Check if expansion occurred
                if self.use_sema and outputs.get('sema_added', False):
                    num_expansions += 1
                    logging.info(f"✨ SEMA expansion triggered at epoch {epoch+1}, batch {i}")

                    # Skip this batch (new adapter not trained yet)
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.clip_max_norm)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Step scheduler
            lr_scheduler.step()

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_rd_loss = epoch_rd_loss / num_batches if num_batches > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']

            # Update progress bar
            desc = (f"Task {self._cur_task}, Epoch {epoch+1}/{self.epochs_per_task} "
                   f"=> Loss {avg_loss:.5f}, LR {current_lr:.6f}")
            if self.use_sema:
                desc += f", RD {avg_rd_loss:.5f}"
            prog_bar.set_description(desc)

            # Periodic logging
            if (epoch + 1) % 10 == 0:
                log_msg = (f"Task {self._cur_task}, Epoch {epoch+1}/{self.epochs_per_task} "
                          f"=> Loss {avg_loss:.5f}, LR {current_lr:.6f}")
                if self.use_sema:
                    log_msg += f", RD Loss {avg_rd_loss:.5f}"
                    if num_expansions > 0:
                        log_msg += f", Expansions {num_expansions}"
                logging.info(log_msg)

        logging.info(f"Training on Task {self._cur_task} completed!\n")

        # SEMA: Report final adapter count
        if self.use_sema:
            final_num_adapters = self._count_adapters()
            logging.info(f"🔧 Final SEMA adapters: {final_num_adapters}")

    def _count_adapters(self):
        """Count total number of SEMA adapters in the network"""
        if not hasattr(self._network, 'reconstruction'):
            return 0

        reconstruction = self._network.reconstruction
        if not hasattr(reconstruction, 'get_sema_modules'):
            return 0

        sema_modules = reconstruction.get_sema_modules()
        total_adapters = sum(m.num_adapters for m in sema_modules)
        return total_adapters

    def get_adapter_statistics(self):
        """Get detailed statistics about SEMA adapters"""
        if not self.use_sema or not hasattr(self._network, 'reconstruction'):
            return {}

        reconstruction = self._network.reconstruction
        if not hasattr(reconstruction, 'get_sema_modules'):
            return {}

        sema_modules = reconstruction.get_sema_modules()
        stats = {
            'total_modules': len(sema_modules),
            'total_adapters': 0,
            'adapters_per_layer': [],
        }

        for module in sema_modules:
            stats['total_adapters'] += module.num_adapters
            stats['adapters_per_layer'].append({
                'layer_id': module.layer_id,
                'num_adapters': module.num_adapters,
            })

        return stats
