# 🚀 SEMA-CL Quick Start Guide

## Chạy training với SEMA

### 1. Baseline CL (không có SEMA)
```bash
cd Ano_CL/tools
python train_continual.py --config config_continual.yaml --seed 133
```

### 2. SEMA-CL
```bash
cd Ano_CL/tools
python train_continual.py --config config_sema.yaml --seed 133
```

**Chỉ khác nhau ở config file!** ✨

---

## So sánh Config

### Baseline (`config_continual.yaml`)
```yaml
net:
  - name: reconstruction
    type: models.reconstructions.UniADMemory  # ← Baseline
    kwargs:
      # No SEMA settings
```

### SEMA (`config_sema.yaml`)
```yaml
# SEMA Configuration
sema:
  use_sema: true
  expansion_threshold: 3.0  # ⭐ Key parameter
  # ... other SEMA settings

# Loss with SEMA RD loss
criterion:
  - name: FeatureMSELoss
    type: FeatureMSELoss
    kwargs:
      weight: 1.0

net:
  - name: reconstruction
    type: models.reconstructions.UniADMemorySEMA  # ← SEMA
    kwargs:
      sema_config:
        use_sema: true
        expansion_threshold: 3.0
        # ... SEMA settings
```

---

## Expected Output

### Training Logs

**Baseline:**
```
Task 0: Training on classes ['bottle']
=> Loss 0.02345, LR 0.000100
Task 0 completed!
```

**SEMA:**
```
Task 0: Training on classes ['bottle']
=> Loss 0.02345, LR 0.000100
Task 0 completed!
🔒 SEMA: Freezing adapters after task...
📊 Total adapters: 8
   Layer 0: 1 adapters
   Layer 1: 1 adapters
   ...
🔍 SEMA: Enabling outlier detection for next task...

Task 1: Training on classes ['cable']
✨ Adapter layer_2.adapter_1 added at layer 2  ← NEW ADAPTER!
=> Loss 0.01987, LR 0.000100
...
```

---

## Kiểm tra kết quả

### Output files

```
Ano_CL/tools/
├── checkpoints_cl/        # Baseline checkpoints
├── checkpoints_sema/      # SEMA checkpoints
├── logs_cl/              # Baseline logs
├── logs_sema/            # SEMA logs  
├── results_cl/           # Baseline results JSON
└── results_sema/         # SEMA results JSON
```

### Metrics

**File:** `results_sema/sema_results_TIMESTAMP.json`

```json
{
  "cl_metrics": [
    {
      "average_performance": 0.897,  // After task 0
      "average_forgetting": 0.0
    },
    {
      "average_performance": 0.812,  // After task 1
      "average_forgetting": 0.085    // ← Lower = better
    },
    ...
  ]
}
```

---

## Troubleshooting

### ❌ Problem: Import error `UniADMemorySEMA`

**Solution:**
```bash
# Check if files exist
ls Ano_CL/models/reconstructions/uniad_sema.py
ls Ano_CL/models/sema_*.py

# Verify __init__.py updated
grep UniADMemorySEMA Ano_CL/models/reconstructions/__init__.py
```

### ❌ Problem: No adapters being added

**Solution:** Giảm `expansion_threshold` trong config:
```yaml
sema:
  expansion_threshold: 2.5  # Thay vì 3.0
```

### ❌ Problem: Too many adapters

**Solution:** Tăng `expansion_threshold`:
```yaml
sema:
  expansion_threshold: 4.0  # Thay vì 3.0
```

---

## Monitor Training

### Watch logs in real-time
```bash
tail -f logs_sema/sema_train_*.log
```

### Key things to watch:
- ✨ Adapter expansion events
- 📊 Total adapter count per layer
- 📉 RD loss values
- 🎯 Performance metrics

---

## Compare Results

### Quick comparison script
```python
import json

# Load results
with open('results_cl/cl_results_TIMESTAMP.json') as f:
    baseline = json.load(f)

with open('results_sema/sema_results_TIMESTAMP.json') as f:
    sema = json.load(f)

# Final metrics
baseline_final = baseline['cl_metrics'][-1]
sema_final = sema['cl_metrics'][-1]

print("Final Results:")
print(f"Baseline - Avg Perf: {baseline_final['average_performance']:.4f}, Forgetting: {baseline_final['average_forgetting']:.4f}")
print(f"SEMA     - Avg Perf: {sema_final['average_performance']:.4f}, Forgetting: {sema_final['average_forgetting']:.4f}")
```

---

For detailed documentation, see [README_SEMA.md](README_SEMA.md)
