# Continual Learning for UniAD Anomaly Detection

Continual learning framework for anomaly detection inspired by SEMA-CL, adapted for the UniAD architecture and MVTec-AD dataset.

## Overview

This implementation provides a **baseline continual learning** framework for anomaly detection:
- **Task-incremental learning**: Learn new anomaly classes sequentially
- **No replay buffer**: Train on current task only (baseline to measure forgetting)
- **No anti-forgetting methods**: Pure sequential training to measure catastrophic forgetting
- **Learner-based architecture**: Following SEMA-CL's design pattern

## Architecture

### Key Components

1. **BaseLearner** (`models/base_learner.py`)
   - Base class for continual learning
   - Manages task progression and evaluation
   - Calculates CL metrics (forgetting, average performance)

2. **UniADLearner** (`models/uniad_learner.py`)
   - Learner class for UniAD transformer model
   - Handles incremental training on new tasks
   - Manages frozen/active layers

3. **CLDataManager** (`utils/cl_data_manager.py`)
   - Manages data loading for sequential tasks
   - Splits classes into tasks
   - Provides train/test loaders per task

4. **Training Script** (`tools/train_continual.py`)
   - Main CL training loop
   - Evaluates on all seen tasks after each new task
   - Saves results and checkpoints

## Installation

```bash
cd C:\Continual_learning_Anomaly_detection\Ano_CL

# Install dependencies (if not already done)
pip install -r requirements.txt
```

## Configuration

Edit `tools/config_continual.yaml`:

### Task Configuration

```yaml
continual_learning:
  # Order of classes to learn
  class_order:
    - bottle
    - cable
    - capsule
    # ... 15 classes total

  # Number of classes per task
  classes_per_task: 1  # 1 class = 15 tasks

  # Training epochs per task
  epochs_per_task: 100
```

### Options for `classes_per_task`:

- **Single class per task**: `classes_per_task: 1` → 15 tasks
- **Multiple classes per task**: `classes_per_task: 3` → 5 tasks
- **Custom split**: `classes_per_task: [5, 5, 5]` → 3 tasks with 5 classes each

## Usage

### Basic Training

```bash
cd tools
python train_continual.py --config config_continual.yaml
```

### With Custom Seed

```bash
python train_continual.py --config config_continual.yaml --seed 42
```

### With Specific GPU

```bash
python train_continual.py --config config_continual.yaml --device 0
```

### Multi-GPU (if available)

```bash
python train_continual.py --config config_continual.yaml --device 0 1
```

## Output

### During Training

The script logs:
- Current task being trained
- Training loss per epoch
- Evaluation on all seen tasks after each task
- CL metrics (average performance, forgetting)

### Results

Results are saved in:

1. **Checkpoints**: `checkpoints_cl/model_task_X.pkl`
2. **Logs**: `logs_cl/cl_train_TIMESTAMP.log`
3. **JSON Results**: `results_cl/cl_results_TIMESTAMP.json`

### Results Format

```json
{
  "config": {
    "class_order": ["bottle", "cable", ...],
    "num_tasks": 15,
    "classes_per_task": [1, 1, 1, ...],
    "key_metric": "mean_pixel_auc"
  },
  "task_performance": {
    "task_0": {
      "eval_0": {"mean_pixel_auc": 0.95, ...}
    },
    "task_1": {
      "eval_0": {"mean_pixel_auc": 0.88, ...},
      "eval_1": {"mean_pixel_auc": 0.96, ...}
    },
    ...
  },
  "cl_metrics": [
    {
      "average_performance": 0.95,
      "average_forgetting": 0.0,
      "forgetting_per_task": {}
    },
    ...
  ]
}
```

## Metrics

### Average Performance
- Mean of performance on all seen tasks after training on task T
- Formula: `AP_T = (1/T) * Σ R_{T,i}` where R_{T,i} is performance on task i after training on task T

### Average Forgetting
- Mean forgetting across all previous tasks
- Formula: `AF_T = (1/(T-1)) * Σ (R_{i,i} - R_{T,i})` where:
  - `R_{i,i}` = best performance on task i (right after training on it)
  - `R_{T,i}` = performance on task i after training on task T

## Example Results Table

```
Task | T 0 | T 1 | T 2 | T 3 | T 4 | Avg  |
-----|-----|-----|-----|-----|-----|------|
T 0  | 0.95|  -  |  -  |  -  |  -  | 0.95 |
T 1  | 0.88| 0.96|  -  |  -  |  -  | 0.92 |
T 2  | 0.82| 0.91| 0.94|  -  |  -  | 0.89 |
T 3  | 0.78| 0.87| 0.90| 0.93|  -  | 0.87 |
T 4  | 0.75| 0.84| 0.87| 0.89| 0.95| 0.86 |
```

Forgetting after T4:
- Task 0: 0.95 - 0.75 = 0.20
- Task 1: 0.96 - 0.84 = 0.12
- Task 2: 0.94 - 0.87 = 0.07
- Task 3: 0.93 - 0.89 = 0.04
- **Average Forgetting**: (0.20 + 0.12 + 0.07 + 0.04) / 4 = **0.11**

## Architecture Details

### Frozen vs Active Layers

By default:
- **Frozen**: `backbone` (EfficientNet-B4) - Pre-trained features not updated
- **Active**: `neck`, `reconstruction` - Updated during training

This can be configured in `config_continual.yaml`:

```yaml
frozen_layers:
  - backbone
```

### Model Structure

```
Input Image (224x224)
    ↓
Backbone (EfficientNet-B4) [FROZEN]
    ↓
Neck (MFCN)
    ↓
Reconstruction (UniADMemory - Transformer)
    ├─ Input Projection
    ├─ Memory Module (learnable memory bank)
    ├─ Transformer Encoder
    ├─ Transformer Decoder
    └─ Output Projection
    ↓
Anomaly Map
```

## Customization

### Add New Task Order

```yaml
continual_learning:
  class_order:
    - toothbrush  # Start with easier classes
    - bottle
    - capsule
    # ... your custom order
```

### Change Task Sizes

```yaml
continual_learning:
  # 3 tasks: first with 5 classes, then 5, then 5
  classes_per_task: [5, 5, 5]
```

### Adjust Training

```yaml
continual_learning:
  epochs_per_task: 200  # Train longer per task

trainer:
  optimizer:
    kwargs:
      lr: 0.0002  # Higher learning rate
```

## Differences from Standard Training

| Aspect | Standard Training | Continual Learning |
|--------|------------------|-------------------|
| Data | All classes at once | Sequential tasks |
| Evaluation | On test set | On all seen tasks |
| Metrics | Accuracy | Accuracy + Forgetting |
| Training | One long session | Multiple shorter sessions |

## Tips

1. **Start with fewer tasks**: Test with `classes_per_task: 5` (3 tasks) before full 15 tasks
2. **Monitor forgetting**: High forgetting (>0.2) indicates catastrophic forgetting
3. **Adjust epochs**: Shorter tasks (fewer classes) may need fewer epochs
4. **Save checkpoints**: Checkpoints saved after each task for analysis

## Future Extensions

This baseline can be extended with:
- [ ] Experience replay buffer
- [ ] Regularization methods (EWC, LwF)
- [ ] Parameter isolation (adapters, prompts)
- [ ] Dynamic architecture expansion
- [ ] Meta-learning approaches

## Citation

If you use this code, please cite:

```bibtex
@article{uniad2022,
  title={UniAD: Unified Anomaly Detection},
  author={...},
  journal={...},
  year={2022}
}

@article{sema2024,
  title={SEMA: Semantic Adaptation for Continual Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on the repository.
