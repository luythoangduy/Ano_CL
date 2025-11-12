"""
UniAD Learner for Continual Learning on Anomaly Detection
Implements continual learning for UniAD (transformer-based anomaly detection model)
"""

import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.base_learner import BaseLearner
from models.model_helper import ModelHelper
from models.cl_strategies import build_cl_strategy
from utils.criterion_helper import build_criterion
from utils.lr_helper import get_scheduler
from utils.optimizer_helper import get_optimizer
import torch.nn.functional as F


class UniADLearner(BaseLearner):
    """
    Continual learner for UniAD anomaly detection model.
    Trains on tasks sequentially without replay buffer or anti-forgetting methods.
    """

    def __init__(self, args):
        super().__init__(args)

        # Build network
        self._network = ModelHelper(args["net"])
        self._network.to(self._device)

        # Training hyperparameters
        self.batch_size = args.get("batch_size", 8)
        self.init_lr = args.get("init_lr", 0.0001)
        self.weight_decay = args.get("weight_decay", 0.0001)
        self.min_lr = args.get("min_lr", 1e-8)
        self.epochs_per_task = args.get("epochs_per_task", 100)
        self.clip_max_norm = args.get("clip_max_norm", 0.1)

        # Layers to train/freeze
        self.frozen_layers = args.get("frozen_layers", ["backbone"])
        self.active_layers = self._get_active_layers(args["net"], self.frozen_layers)

        # Store full config for training
        self.config = args.get("config", None)

        # CL Strategy
        self.cl_strategy = build_cl_strategy(args.get("config", {}))
        logging.info(f"Using CL Strategy: {self.cl_strategy.strategy_name}")

        logging.info(f"Active layers for training: {self.active_layers}")
        logging.info(f"Frozen layers: {self.frozen_layers}")

    def _get_active_layers(self, net_config, frozen_layers):
        """Get list of layers that will be trained"""
        all_layers = [module["name"] for module in net_config]
        active_layers = list(set(all_layers) - set(frozen_layers))
        return active_layers

    def incremental_train(self, data_manager):
        """
        Train on a new task incrementally.

        Args:
            data_manager: Data manager providing train/test loaders for current task
        """
        self._cur_task += 1

        # Get new classes for this task
        new_classes = data_manager.get_task_classes(self._cur_task)
        self._total_classes.extend(new_classes)

        # Track task size
        self.task_sizes.append(len(new_classes))

        logging.info(f"\n{'='*80}")
        logging.info(f"Task {self._cur_task}: Training on classes {new_classes}")
        logging.info(f"Known classes so far: {self._total_classes}")
        logging.info(f"{'='*80}\n")

        # Build dataloaders for current task
        train_loader = data_manager.get_train_loader(new_classes)
        test_loader = data_manager.get_test_loader(new_classes)

        # Handle multi-GPU if needed
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # Train on current task
        self._train(train_loader, test_loader, self.config)

        # Remove DataParallel wrapper if used
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, config):
        """
        Train model on current task.

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
        logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        # Log trainable modules
        logging.info("Trainable modules:")
        for name, module in self._network.named_children():
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            module_total = sum(p.numel() for p in module.parameters())
            if module_total > 0:
                logging.info(f"  {name}: {module_trainable:,} / {module_total:,} ({100*module_trainable/module_total:.1f}%)")

        prog_bar = tqdm(range(self.epochs_per_task))

        for epoch in prog_bar:
            self._network.train()

            # Freeze layers again (in case they were unfrozen)
            for layer_name in self.frozen_layers:
                if hasattr(self._network, layer_name):
                    layer = getattr(self._network, layer_name)
                    layer.eval()

            epoch_loss = 0.0
            num_batches = 0

            for i, input in enumerate(train_loader):
                # Forward pass
                outputs = self._network(input)

                # Compute loss
                loss = 0
                for name, criterion_loss in criterion.items():
                    weight = criterion_loss.weight
                    loss += weight * criterion_loss(outputs)

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

            avg_loss = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']

            # Update progress bar
            prog_bar.set_description(
                f"Task {self._cur_task}, Epoch {epoch+1}/{self.epochs_per_task} "
                f"=> Loss {avg_loss:.5f}, LR {current_lr:.6f}"
            )

            # Periodic logging
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Task {self._cur_task}, Epoch {epoch+1}/{self.epochs_per_task} "
                    f"=> Loss {avg_loss:.5f}, LR {current_lr:.6f}"
                )

        logging.info(f"Training on Task {self._cur_task} completed!\n")

    def eval_task_simple(self, test_loader):
        """
        Simple evaluation returning average loss.
        Useful for quick validation during training.

        Args:
            test_loader: Test data loader

        Returns:
            avg_loss: Average loss on test set
        """
        self._network.eval()

        criterion = build_criterion(self.args.get("criterion", {}))
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input in test_loader:
                outputs = self._network(input)

                # Compute loss
                loss = 0
                for name, criterion_loss in criterion.items():
                    weight = criterion_loss.weight
                    loss += weight * criterion_loss(outputs)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def forward_network(self, input):
        """
        Forward pass through network.

        Args:
            input: Input data dict

        Returns:
            output: Network output dict
        """
        return self._network(input)

    def after_task(self):
        """
        Operations to perform after each task.
        Delegates to CL strategy.
        """
        super().after_task()

        # Apply CL strategy
        self.cl_strategy.after_task(self._network, self._cur_task)
