"""
Continual Learning Strategies for AnoCL

Implements different freeze and memory expansion strategies:
1. Baseline: No freeze (except backbone), retrain all
2. SEMA: Freeze old adapters
3. MemExpand: Freeze old modules, stack new memory
4. SEMA + MemExpand: Combine both strategies
"""

import logging
import torch
from torch import nn
from collections import OrderedDict


class CLStrategy:
    """Base class for CL strategies"""

    def __init__(self, config):
        self.config = config
        self.strategy_name = config.get('cl_strategy', 'baseline')

    def after_task(self, network, task_id):
        """Operations after each task"""
        raise NotImplementedError

    def before_task(self, network, task_id):
        """Operations before each task"""
        pass


class BaselineStrategy(CLStrategy):
    """
    Baseline: No special CL strategy
    - Freeze backbone only
    - Retrain neck, memory, transformer every task
    """

    def __init__(self, config):
        super().__init__(config)
        self.strategy_name = 'baseline'

    def after_task(self, network, task_id):
        """No special operations"""
        logging.info(f"📝 Baseline: No freeze after task {task_id}")
        pass


class FreezeAllStrategy(CLStrategy):
    """
    Freeze all reconstruction modules after each task
    - Freeze: neck, memory, transformer (all except backbone)
    - Result: No adaptation for new tasks (purely frozen)
    """

    def __init__(self, config):
        super().__init__(config)
        self.strategy_name = 'freeze_all'

    def after_task(self, network, task_id):
        """Freeze neck, memory, transformer"""
        logging.info(f"🔒 FreezeAll: Freezing all modules after task {task_id}")

        # Freeze neck
        if hasattr(network, 'neck'):
            for param in network.neck.parameters():
                param.requires_grad = False
            logging.info("  ❄️ Neck frozen")

        # Freeze reconstruction (memory + transformer)
        if hasattr(network, 'reconstruction'):
            for param in network.reconstruction.parameters():
                param.requires_grad = False
            logging.info("  ❄️ Reconstruction frozen")


class MemoryExpansionStrategy(CLStrategy):
    """
    Memory Expansion Strategy
    - After each task: Freeze current neck + memory + transformer
    - Before new task: Add new memory module (stacked)
    - Mix outputs from all memory modules
    """

    def __init__(self, config):
        super().__init__(config)
        self.strategy_name = 'mem_expand'
        self.task_modules = OrderedDict()  # Track modules per task

    def after_task(self, network, task_id):
        """
        Freeze current modules and save them
        """
        logging.info(f"🔒 MemExpand: Freezing modules for task {task_id}")

        # Clone and freeze current modules
        task_module = {
            'task_id': task_id,
            'neck': None,
            'reconstruction': None
        }

        # Freeze neck
        if hasattr(network, 'neck'):
            # Clone neck module
            neck_clone = self._clone_module(network.neck)
            neck_clone.eval()
            for param in neck_clone.parameters():
                param.requires_grad = False
            task_module['neck'] = neck_clone

            logging.info(f"  ❄️ Neck for task {task_id} frozen and saved")

        # Freeze reconstruction
        if hasattr(network, 'reconstruction'):
            # Clone reconstruction module
            recon_clone = self._clone_module(network.reconstruction)
            recon_clone.eval()
            for param in recon_clone.parameters():
                param.requires_grad = False
            task_module['reconstruction'] = recon_clone

            logging.info(f"  ❄️ Reconstruction for task {task_id} frozen and saved")

        # Store task modules
        self.task_modules[f'task_{task_id}'] = task_module
        logging.info(f"📦 Total task modules: {len(self.task_modules)}")

    def before_task(self, network, task_id):
        """
        Before new task: Reset/reinitialize modules for new learning
        """
        if task_id == 0:
            return  # First task, nothing to do

        logging.info(f"✨ MemExpand: Preparing new modules for task {task_id}")

        # Option 1: Keep current modules (they will be retrained)
        # Option 2: Reinitialize modules (fresh start)
        # For now, we keep current and will retrain them

        logging.info(f"  ✅ Current modules will be retrained for task {task_id}")

    def _clone_module(self, module):
        """Deep clone a module with its weights"""
        import copy
        return copy.deepcopy(module)

    def get_all_outputs(self, network, input_data):
        """
        Forward through all task modules and mix outputs

        Args:
            network: Current network
            input_data: Input dict

        Returns:
            Mixed output from all tasks
        """
        outputs = []

        # Forward through frozen modules from previous tasks
        for task_name, task_module in self.task_modules.items():
            neck = task_module['neck']
            reconstruction = task_module['reconstruction']

            if neck is not None and reconstruction is not None:
                # Forward through frozen modules
                with torch.no_grad():
                    neck_out = neck(input_data)
                    recon_input = {'feature_align': neck_out}
                    recon_out = reconstruction(recon_input)
                    outputs.append(recon_out)

        # Forward through current (trainable) modules
        if hasattr(network, 'neck') and hasattr(network, 'reconstruction'):
            neck_out = network.neck(input_data)
            recon_input = {'feature_align': neck_out}
            recon_out = network.reconstruction(recon_input)
            outputs.append(recon_out)

        # Mix outputs (simple average for now)
        if len(outputs) > 1:
            mixed_output = self._mix_outputs(outputs)
            return mixed_output
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return None

    def _mix_outputs(self, outputs):
        """
        Mix outputs from multiple modules

        Simple average for now, can be replaced with learned weighting
        """
        # Average predictions
        preds = [out['pred'] for out in outputs]
        mixed_pred = torch.stack(preds).mean(dim=0)

        # Use first output as template
        mixed_output = outputs[0].copy()
        mixed_output['pred'] = mixed_pred

        return mixed_output


class SEMAStrategy(CLStrategy):
    """
    SEMA Strategy
    - Freeze old adapters after each task
    - Enable outlier detection for new task
    """

    def __init__(self, config):
        super().__init__(config)
        self.strategy_name = 'sema'

    def after_task(self, network, task_id):
        """Freeze SEMA adapters"""
        if not hasattr(network, 'reconstruction'):
            logging.warning("⚠️ SEMA: No reconstruction module found")
            return

        reconstruction = network.reconstruction

        # Check if it's UniADMemorySEMA
        if hasattr(reconstruction, 'end_task_training'):
            logging.info(f"🔒 SEMA: Freezing adapters after task {task_id}")
            reconstruction.end_task_training()

            # Get adapter statistics
            if hasattr(reconstruction, 'get_sema_modules'):
                sema_modules = reconstruction.get_sema_modules()
                total_adapters = sum(m.num_adapters for m in sema_modules)
                logging.info(f"📊 Total adapters: {total_adapters}")

                # Log per-layer stats
                for module in sema_modules:
                    logging.info(f"   Layer {module.layer_id}: {module.num_adapters} adapters")

            # Enable outlier detection for next task
            if hasattr(reconstruction, 'enable_outlier_detection'):
                logging.info("🔍 SEMA: Enabling outlier detection for next task")
                reconstruction.enable_outlier_detection()


class SEMAMemExpandStrategy(CLStrategy):
    """
    Combined SEMA + Memory Expansion Strategy
    - Freeze old adapters (SEMA)
    - Freeze old neck/memory/transformer (MemExpand)
    - Stack new modules for new task
    """

    def __init__(self, config):
        super().__init__(config)
        self.strategy_name = 'sema_mem_expand'
        self.mem_expand = MemoryExpansionStrategy(config)
        self.sema = SEMAStrategy(config)

    def after_task(self, network, task_id):
        """Apply both strategies"""
        logging.info(f"🔒 SEMA+MemExpand: Applying combined strategy after task {task_id}")

        # 1. SEMA: Freeze adapters
        self.sema.after_task(network, task_id)

        # 2. MemExpand: Freeze and save modules
        self.mem_expand.after_task(network, task_id)

    def before_task(self, network, task_id):
        """Prepare for new task"""
        self.mem_expand.before_task(network, task_id)

    def get_all_outputs(self, network, input_data):
        """Forward through all modules"""
        return self.mem_expand.get_all_outputs(network, input_data)


def build_cl_strategy(config):
    """
    Factory function to build CL strategy

    Args:
        config: Configuration dict with 'cl_strategy' field

    Returns:
        CLStrategy instance
    """
    strategy_name = config.get('cl_strategy', 'baseline')

    if strategy_name == 'baseline':
        return BaselineStrategy(config)
    elif strategy_name == 'freeze_all':
        return FreezeAllStrategy(config)
    elif strategy_name == 'mem_expand':
        return MemoryExpansionStrategy(config)
    elif strategy_name == 'sema':
        return SEMAStrategy(config)
    elif strategy_name == 'sema_mem_expand':
        return SEMAMemExpandStrategy(config)
    else:
        logging.warning(f"Unknown strategy '{strategy_name}', using baseline")
        return BaselineStrategy(config)
