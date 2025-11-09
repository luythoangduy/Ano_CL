"""
Base Learner for Continual Learning on Anomaly Detection
Adapted from SEMA-CL framework for anomaly detection tasks
"""

import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.eval_helper import dump, merge_together, performances
import shutil
import os

EPSILON = 1e-8


class BaseLearner(object):
    """
    Base class for continual learning of anomaly detection models.
    Manages task progression, model state, and evaluation without memory replay.
    """

    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = []  # List of classes seen so far
        self._total_classes = []  # List of all classes up to current task
        self._network = None
        self._device = args["device"]
        self._multiple_gpus = args.get("multiple_gpus", [])
        self.args = args

        # Task and class tracking
        self.task_sizes = []  # Number of classes per task
        self.all_classes = args.get("class_order", [])  # Order of classes

    @property
    def cur_task(self):
        return self._cur_task

    @property
    def known_classes(self):
        return self._known_classes

    @property
    def total_classes(self):
        return self._total_classes

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        self._network.cpu()
        save_dict = {
            "task": self._cur_task,
            "model_state_dict": self._network.state_dict(),
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
        }
        torch.save(save_dict, f"{filename}_task_{self._cur_task}.pkl")
        self._network.to(self._device)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self._device)
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._cur_task = checkpoint["task"]
        self._known_classes = checkpoint["known_classes"]
        self._total_classes = checkpoint["total_classes"]

    def after_task(self):
        """Operations to perform after each task"""
        self._known_classes = copy.deepcopy(self._total_classes)

    def incremental_train(self, data_manager):
        """
        Train on a new task incrementally.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def eval_task(self, test_loader, eval_classes, config):
        """
        Evaluate model on a specific set of classes.

        Args:
            test_loader: DataLoader for test data
            eval_classes: List of classes to evaluate on
            config: Configuration dict

        Returns:
            metrics: Dict of evaluation metrics
        """
        self._network.eval()

        eval_dir = config.evaluator.get('eval_dir', config.evaluator.save_dir)
        os.makedirs(eval_dir, exist_ok=True)

        with torch.no_grad():
            for i, input in enumerate(test_loader):
                # Forward pass
                outputs = self._network(input)
                dump(eval_dir, outputs)

        # Compute metrics
        fileinfos, preds, masks = merge_together(eval_dir)
        shutil.rmtree(eval_dir)

        metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)

        return metrics

    def eval_all_tasks(self, data_manager, config):
        """
        Evaluate model on all tasks seen so far.

        Returns:
            all_metrics: Dict mapping task_id -> metrics
        """
        all_metrics = {}

        for task_id, task_classes in enumerate(self.get_task_classes_list()):
            # Build test loader for this task
            test_loader = data_manager.get_test_loader(task_classes)

            # Evaluate
            metrics = self.eval_task(test_loader, task_classes, config)
            all_metrics[task_id] = metrics

            # Log results
            key_metric = config.evaluator.get("key_metric", "mean_pixel_auc")
            logging.info(f"Task {task_id} ({task_classes}): {key_metric}={metrics.get(key_metric, 0):.4f}")

        return all_metrics

    def get_task_classes_list(self):
        """
        Get list of classes for each task.

        Returns:
            List of lists, where each inner list contains classes for that task
        """
        task_classes_list = []
        start_idx = 0

        for task_size in self.task_sizes:
            end_idx = start_idx + task_size
            task_classes_list.append(self.all_classes[start_idx:end_idx])
            start_idx = end_idx

        return task_classes_list

    def calculate_cl_metrics(self, all_task_metrics, config):
        """
        Calculate continual learning metrics: average accuracy and forgetting.

        Args:
            all_task_metrics: Dict of {task_id: {eval_task_id: metrics}}
            config: Configuration

        Returns:
            cl_metrics: Dict of CL metrics
        """
        key_metric = config.evaluator.get("key_metric", "mean_pixel_auc")
        current_task = self._cur_task

        # Average performance on all seen tasks
        avg_performance = 0
        num_tasks = current_task + 1

        for eval_task_id in range(num_tasks):
            if eval_task_id in all_task_metrics[current_task]:
                metric_value = all_task_metrics[current_task][eval_task_id].get(key_metric, 0)
                avg_performance += metric_value

        avg_performance /= num_tasks

        # Calculate forgetting for previous tasks
        forgetting_per_task = {}
        avg_forgetting = 0

        if current_task > 0:
            total_forgetting = 0

            for eval_task_id in range(current_task):
                # Best performance (right after training this task)
                best_perf = all_task_metrics[eval_task_id][eval_task_id].get(key_metric, 0)

                # Current performance
                current_perf = all_task_metrics[current_task][eval_task_id].get(key_metric, 0)

                # Forgetting = best - current (positive means forgetting)
                forgetting = best_perf - current_perf
                forgetting_per_task[eval_task_id] = forgetting
                total_forgetting += forgetting

            avg_forgetting = total_forgetting / current_task

        cl_metrics = {
            'average_performance': avg_performance,
            'average_forgetting': avg_forgetting,
            'forgetting_per_task': forgetting_per_task,
        }

        # Log results
        logging.info(f"\n{'='*60}")
        logging.info(f"Continual Learning Metrics after Task {current_task}")
        logging.info(f"{'='*60}")
        logging.info(f"Average Performance: {avg_performance:.4f}")
        if current_task > 0:
            logging.info(f"Average Forgetting: {avg_forgetting:.4f}")
            for task_id, forgetting in forgetting_per_task.items():
                logging.info(f"  Task {task_id} forgetting: {forgetting:.4f}")
        logging.info(f"{'='*60}\n")

        return cl_metrics
