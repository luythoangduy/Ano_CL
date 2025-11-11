"""
SEMA-based Continual Learning Training Script for Anomaly Detection
Train UniAD model with self-expanding adapters (SEMA)
"""

import argparse
import logging
import os
import pprint
import sys
import json
from collections import defaultdict
from datetime import datetime

import torch
import yaml
from easydict import EasyDict

from models.uniad_sema_learner import UniADSEMALearner
from utils.cl_data_manager import CLDataManager
from utils.misc_helper import set_random_seed

# All MVTec-AD classes
MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

parser = argparse.ArgumentParser(description="SEMA Continual Learning for UniAD Anomaly Detection")
parser.add_argument("--config", default="./config_sema.yaml", help="Path to config file")
parser.add_argument("--seed", type=int, default=133, help="Random seed")
parser.add_argument("--device", type=int, nargs='+', default=[0], help="GPU device IDs")


def main():
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Override with command line args
    if args.seed is not None:
        config.random_seed = args.seed

    # Setup logging
    exp_path = os.path.dirname(args.config)
    log_path = os.path.join(exp_path, "logs_sema")
    os.makedirs(log_path, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_path, f"sema_train_{current_time}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("="*80)
    logging.info("🚀 SEMA Continual Learning for Anomaly Detection")
    logging.info("="*80)
    logging.info(f"Config file: {args.config}")
    logging.info(f"Config:\n{pprint.pformat(dict(config))}")

    # Set random seed
    set_random_seed(config.random_seed, config.get("reproduce", None))

    # Setup devices
    device_list = args.device
    if len(device_list) == 1:
        device = torch.device(f"cuda:{device_list[0]}" if torch.cuda.is_available() else "cpu")
        multiple_gpus = []
    else:
        device = torch.device(f"cuda:{device_list[0]}")
        multiple_gpus = device_list

    logging.info(f"Using device: {device}")
    if multiple_gpus:
        logging.info(f"Using multiple GPUs: {multiple_gpus}")

    # Get continual learning configuration
    cl_config = config.continual_learning

    # Task order
    class_order = cl_config.get("class_order", MVTEC_CLASSES)
    classes_per_task = cl_config.get("classes_per_task", 1)

    # Create data manager
    data_manager = CLDataManager(
        config=config.dataset,
        class_order=class_order,
        classes_per_task=classes_per_task
    )

    # SEMA configuration
    sema_config = config.get("sema", {})
    logging.info(f"\n🔧 SEMA Configuration:")
    logging.info(f"  - Use SEMA: {sema_config.get('use_sema', False)}")
    logging.info(f"  - Position: {sema_config.get('sema_position', 'ffn')}")
    logging.info(f"  - Mode: {sema_config.get('sema_mode', 'parallel')}")
    logging.info(f"  - Expansion threshold: {sema_config.get('expansion_threshold', 3.0)}")
    logging.info(f"  - RD loss weight: {sema_config.get('rd_loss_weight', 0.1)}")

    # Create learner arguments
    learner_args = {
        "net": config.net,
        "device": device,
        "multiple_gpus": multiple_gpus,
        "batch_size": config.dataset.batch_size,
        "init_lr": config.trainer.optimizer.kwargs.lr,
        "weight_decay": config.trainer.optimizer.kwargs.weight_decay,
        "min_lr": cl_config.get("min_lr", 1e-8),
        "epochs_per_task": cl_config.get("epochs_per_task", 100),
        "clip_max_norm": config.trainer.get("clip_max_norm", 0.1),
        "frozen_layers": config.get("frozen_layers", ["backbone"]),
        "class_order": class_order,
        "criterion": config.criterion,
        "config": config,
        # SEMA-specific
        "use_sema": sema_config.get("use_sema", False),
        "sema_config": sema_config,
    }

    # Create SEMA learner
    learner = UniADSEMALearner(learner_args)

    # Storage for metrics
    all_task_metrics = defaultdict(dict)
    cl_metrics_history = []
    adapter_stats_history = []

    # Main training loop
    for task_id in range(data_manager.num_tasks):
        logging.info("\n" + "="*80)
        logging.info(f"🎯 Starting Task {task_id}/{data_manager.num_tasks - 1}")
        logging.info("="*80)

        # Train on current task
        learner.incremental_train(data_manager)

        # Get adapter statistics
        if learner.use_sema:
            adapter_stats = learner.get_adapter_statistics()
            adapter_stats_history.append(adapter_stats)
            logging.info(f"\n📊 Adapter Statistics after Task {task_id}:")
            logging.info(f"  Total adapters: {adapter_stats.get('total_adapters', 0)}")
            for layer_stat in adapter_stats.get('adapters_per_layer', []):
                logging.info(f"    Layer {layer_stat['layer_id']}: {layer_stat['num_adapters']} adapters")

        # Evaluate on all seen tasks
        logging.info(f"\n📈 Evaluating after Task {task_id}...")
        for eval_task_id in range(task_id + 1):
            eval_classes = data_manager.get_task_classes(eval_task_id)
            test_loader = data_manager.get_test_loader(eval_classes)

            # Evaluate
            metrics = learner.eval_task(test_loader, eval_classes, config)
            all_task_metrics[task_id][eval_task_id] = metrics

            # Log
            key_metric = config.evaluator.get("key_metric", "mean_pixel_auc")
            metric_value = metrics.get(key_metric, 0)
            logging.info(f"  Task {eval_task_id} ({eval_classes}): {key_metric} = {metric_value:.4f}")

        # Calculate CL metrics
        cl_metrics = learner.calculate_cl_metrics(all_task_metrics, config)
        cl_metrics_history.append(cl_metrics)

        # After task operations (freeze adapters, enable outlier detection)
        learner.after_task()

        # Save checkpoint
        checkpoint_path = os.path.join(exp_path, "checkpoints_sema")
        os.makedirs(checkpoint_path, exist_ok=True)
        learner.save_checkpoint(os.path.join(checkpoint_path, "model"))

    # Final summary
    logging.info("\n" + "="*80)
    logging.info("✅ SEMA Continual Learning Training Complete!")
    logging.info("="*80)

    # Print final results
    print_final_results(all_task_metrics, cl_metrics_history, data_manager, config)

    # Print adapter expansion summary
    if learner.use_sema:
        print_adapter_summary(adapter_stats_history, data_manager)

    # Save results to JSON
    save_results(all_task_metrics, cl_metrics_history, adapter_stats_history,
                data_manager, config, exp_path)


def print_final_results(all_task_metrics, cl_metrics_history, data_manager, config):
    """Print final CL results in a formatted table"""
    key_metric = config.evaluator.get("key_metric", "mean_pixel_auc")
    num_tasks = data_manager.num_tasks

    logging.info("\n" + "="*80)
    logging.info("📊 Final Results Matrix")
    logging.info("="*80)

    # Print header
    header = "Task |"
    for i in range(num_tasks):
        header += f" T{i:2d} |"
    header += " Avg  |"
    logging.info(header)
    logging.info("-" * len(header))

    # Print each row
    for task_id in range(num_tasks):
        row = f"T{task_id:2d}  |"
        for eval_id in range(task_id + 1):
            if eval_id in all_task_metrics[task_id]:
                value = all_task_metrics[task_id][eval_id].get(key_metric, 0)
                row += f" {value:.2f}|"
            else:
                row += "  -  |"

        # Add padding for unseen tasks
        for _ in range(num_tasks - task_id - 1):
            row += "  -  |"

        # Add average
        avg = cl_metrics_history[task_id]['average_performance']
        row += f" {avg:.2f}|"

        logging.info(row)

    logging.info("="*80)

    # Print forgetting
    logging.info("\n📉 Forgetting per Task:")
    for task_id in range(1, num_tasks):
        avg_forgetting = cl_metrics_history[task_id]['average_forgetting']
        logging.info(f"After Task {task_id}: Average Forgetting = {avg_forgetting:.4f}")

    # Print overall statistics
    final_avg_perf = cl_metrics_history[-1]['average_performance']
    final_avg_forgetting = cl_metrics_history[-1]['average_forgetting'] if num_tasks > 1 else 0

    logging.info("\n" + "="*80)
    logging.info("📈 Overall Statistics:")
    logging.info(f"Final Average Performance: {final_avg_perf:.4f}")
    if num_tasks > 1:
        logging.info(f"Final Average Forgetting: {final_avg_forgetting:.4f}")
    logging.info("="*80 + "\n")


def print_adapter_summary(adapter_stats_history, data_manager):
    """Print summary of adapter expansion"""
    logging.info("\n" + "="*80)
    logging.info("🔧 SEMA Adapter Expansion Summary")
    logging.info("="*80)

    for task_id, stats in enumerate(adapter_stats_history):
        class_name = data_manager.get_task_classes(task_id)
        total_adapters = stats.get('total_adapters', 0)
        logging.info(f"Task {task_id} ({class_name}): {total_adapters} adapters")

    final_stats = adapter_stats_history[-1]
    logging.info(f"\nFinal total adapters: {final_stats.get('total_adapters', 0)}")
    logging.info(f"Average adapters per task: {final_stats.get('total_adapters', 0) / len(adapter_stats_history):.1f}")
    logging.info("="*80 + "\n")


def save_results(all_task_metrics, cl_metrics_history, adapter_stats_history,
                data_manager, config, exp_path):
    """Save results to JSON file"""
    results = {
        "config": {
            "class_order": data_manager.class_order,
            "num_tasks": data_manager.num_tasks,
            "classes_per_task": data_manager.classes_per_task,
            "key_metric": config.evaluator.get("key_metric", "mean_pixel_auc"),
            "sema_config": dict(config.get("sema", {})),
        },
        "task_performance": {},
        "cl_metrics": [],
        "adapter_stats": adapter_stats_history,
    }

    # Convert metrics to serializable format
    for task_id, eval_dict in all_task_metrics.items():
        results["task_performance"][f"task_{task_id}"] = {}
        for eval_id, metrics in eval_dict.items():
            results["task_performance"][f"task_{task_id}"][f"eval_{eval_id}"] = metrics

    # Add CL metrics history
    results["cl_metrics"] = cl_metrics_history

    # Save to file
    results_dir = os.path.join(exp_path, "results_sema")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"sema_results_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
