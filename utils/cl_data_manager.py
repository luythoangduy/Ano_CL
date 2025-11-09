"""
Continual Learning Data Manager for Anomaly Detection
Manages data loading for sequential task training
"""

import logging
import numpy as np
from datasets.data_builder import build_dataloader


class CLDataManager:
    """
    Data manager for continual learning on anomaly detection.
    Handles task-incremental learning where each task consists of one or more classes.
    """

    def __init__(self, config, class_order, classes_per_task):
        """
        Args:
            config: Dataset configuration (from config.dataset)
            class_order: List of classes in the order they should be learned
            classes_per_task: Number of classes per task (can be int or list)
        """
        self.config = config
        self.class_order = class_order
        self.num_classes = len(class_order)

        # Handle classes_per_task
        if isinstance(classes_per_task, int):
            # Same number of classes for each task
            self.num_tasks = self.num_classes // classes_per_task
            if self.num_classes % classes_per_task != 0:
                self.num_tasks += 1
            self.classes_per_task = [classes_per_task] * self.num_tasks
            # Adjust last task if needed
            if self.num_classes % classes_per_task != 0:
                self.classes_per_task[-1] = self.num_classes % classes_per_task
        else:
            # Custom number of classes for each task
            self.classes_per_task = classes_per_task
            self.num_tasks = len(classes_per_task)
            assert sum(classes_per_task) == self.num_classes, \
                f"Sum of classes_per_task ({sum(classes_per_task)}) must equal num_classes ({self.num_classes})"

        logging.info(f"\n{'='*60}")
        logging.info("Continual Learning Data Manager initialized")
        logging.info(f"Total classes: {self.num_classes}")
        logging.info(f"Class order: {self.class_order}")
        logging.info(f"Number of tasks: {self.num_tasks}")
        logging.info(f"Classes per task: {self.classes_per_task}")
        logging.info(f"{'='*60}\n")

    def get_task_classes(self, task_id):
        """
        Get the list of classes for a specific task.

        Args:
            task_id: Task ID (0-indexed)

        Returns:
            List of class names for this task
        """
        start_idx = sum(self.classes_per_task[:task_id])
        end_idx = start_idx + self.classes_per_task[task_id]
        return self.class_order[start_idx:end_idx]

    def get_train_loader(self, class_names, distributed=False):
        """
        Get training data loader for specified classes.

        Args:
            class_names: List of class names to include
            distributed: Whether to use distributed sampling

        Returns:
            DataLoader for training
        """
        # For single class, pass as string; for multiple, pass as comma-separated
        if len(class_names) == 1:
            class_filter = class_names[0]
        else:
            class_filter = ','.join(class_names)

        train_loader, _ = build_dataloader(
            self.config,
            distributed=distributed,
            class_name=class_filter
        )

        return train_loader

    def get_test_loader(self, class_names, distributed=False):
        """
        Get test data loader for specified classes.

        Args:
            class_names: List of class names to include
            distributed: Whether to use distributed sampling

        Returns:
            DataLoader for testing
        """
        # For single class, pass as string; for multiple, pass as comma-separated
        if len(class_names) == 1:
            class_filter = class_names[0]
        else:
            class_filter = ','.join(class_names)

        _, test_loader = build_dataloader(
            self.config,
            distributed=distributed,
            class_name=class_filter
        )

        return test_loader

    def get_task_size(self, task_id):
        """
        Get number of classes in a specific task.

        Args:
            task_id: Task ID (0-indexed)

        Returns:
            Number of classes in this task
        """
        return self.classes_per_task[task_id]

    def get_cumulative_classes(self, task_id):
        """
        Get all classes seen up to and including a specific task.

        Args:
            task_id: Task ID (0-indexed)

        Returns:
            List of all class names seen so far
        """
        all_classes = []
        for tid in range(task_id + 1):
            all_classes.extend(self.get_task_classes(tid))
        return all_classes
