# SEMA Integration for AnoCL

## 🎯 Tổng quan

Tích hợp **SEMA (Self-Expansion of pre-trained models with Mixture of Adapters)** từ CVPR 2025 vào **AnoCL** để cải thiện khả năng Continual Learning cho Anomaly Detection.

### Điểm khác biệt so với Baseline CL

| Feature | Baseline CL | SEMA-CL |
|---------|------------|---------|
| **Adapters** | ❌ Không có | ✅ Self-expanding adapters |
| **Anti-forgetting** | ❌ Không có | ✅ Adapter freezing + Router mixing |
| **Distribution shift detection** | ❌ Không có | ✅ Representation Descriptor (RD) |
| **Model expansion** | ❌ Fixed | ✅ Dynamic (auto-expand khi cần) |
| **Forgetting** | 🔴 Cao (~0.20) | 🟢 Thấp hơn (dự kiến) |

---

## 🏗️ Kiến trúc SEMA

### 1. **Components chính**

```
Input Features
    ↓
Transformer Encoder (4 layers)
    ├─ Self-Attention
    ├─ **SEMA Adapter** [NEW]
    ├─ Feedforward
    └─ **SEMA Adapter** [NEW]
    ↓
Memory Module
    ↓
Transformer Decoder (4 layers)
    ├─ Self-Attention
    ├─ **SEMA Adapter** [NEW]
    ├─ Cross-Attention
    ├─ **SEMA Adapter** [NEW]
    ├─ Feedforward
    └─ **SEMA Adapter** [NEW]
    ↓
Output
```

### 2. **SEMA Adapter Module**

Mỗi adapter gồm 2 phần:

#### **a) Functional Adapter** (Bottleneck)
```python
input [256] → down_proj [64] → ReLU → up_proj [256] → output
```

#### **b) Representation Descriptor (RD)** (AutoEncoder)
```python
input [256] → encoder [64] → decoder [256] → reconstruction
```

**Mục đích RD:**
- Train: Learn to reconstruct normal features
- Test: Detect distribution shift
  - High reconstruction error → Z-score cao → **Trigger expansion**

### 3. **Self-Expansion Mechanism**

```python
# Compute Z-score
z_score = (rd_loss - mean) / std

# Expansion criteria
if z_score.mean() > threshold (default: 3.0):
    ✨ Add new adapter
    📊 Update router (mix adapter outputs)
```

---

## 📁 Files được tạo

### **1. Core Components**

| File | Mô tả |
|------|-------|
| [`models/sema_components.py`](Ano_CL/models/sema_components.py) | Adapter, RD, RDLossRecords |
| [`models/sema_modules.py`](Ano_CL/models/sema_modules.py) | SEMAModules (manager cho multiple adapters) |

### **2. Transformer với SEMA**

| File | Mô tả |
|------|-------|
| [`models/reconstructions/dumenet_sema.py`](Ano_CL/models/reconstructions/dumenet_sema.py) | SEMA Encoder/Decoder Layers |
| [`models/reconstructions/uniad_sema.py`](Ano_CL/models/reconstructions/uniad_sema.py) | UniADMemorySEMA (full model) |

### **3. Training**

| File | Mô tả |
|------|-------|
| [`models/uniad_sema_learner.py`](Ano_CL/models/uniad_sema_learner.py) | SEMA Learner với RD loss |
| [`tools/train_sema.py`](Ano_CL/tools/train_sema.py) | Script huấn luyện SEMA |
| [`tools/config_sema.yaml`](Ano_CL/tools/config_sema.yaml) | Config cho SEMA |

---

## 🚀 Cách sử dụng

### **1. Cài đặt dependencies**

```bash
cd Ano_CL
pip install -r requirements.txt
```

### **2. Chuẩn bị data**

Đảm bảo MVTec-AD dataset đã được chuẩn bị (xem [`README_CL.md`](Ano_CL/README_CL.md))

### **3. Chỉnh config**

Edit [`tools/config_sema.yaml`](Ano_CL/tools/config_sema.yaml):

```yaml
# SEMA settings
sema:
  use_sema: true

  # Adapter position: 'ffn', 'attn', or 'both'
  sema_position: 'ffn'

  # Adapter mode: 'parallel' (residual) or 'sequential'
  sema_mode: 'parallel'

  # Expansion threshold (Z-score)
  expansion_threshold: 3.0

  # RD loss weight
  rd_loss_weight: 0.1
```

### **4. Chạy training**

```bash
cd tools
python train_sema.py --config config_sema.yaml
```

**Options:**
```bash
# Custom seed
python train_sema.py --config config_sema.yaml --seed 42

# Specific GPU
python train_sema.py --config config_sema.yaml --device 0

# Multi-GPU
python train_sema.py --config config_sema.yaml --device 0 1
```

---

## 📊 Output

### **1. Logs**
```
logs_sema/sema_train_TIMESTAMP.log
```

**Bao gồm:**
- Training loss per epoch
- **RD loss** (representation descriptor)
- **Adapter statistics** (số lượng adapters per layer)
- **Expansion events** (khi nào adapter được thêm)
- Evaluation metrics per task

### **2. Checkpoints**
```
checkpoints_sema/model_task_X.pkl
```

### **3. Results**
```
results_sema/sema_results_TIMESTAMP.json
```

**Format:**
```json
{
  "config": {
    "sema_config": {...},
    "num_tasks": 15,
    ...
  },
  "task_performance": {...},
  "cl_metrics": [...],
  "adapter_stats": [
    {
      "total_adapters": 8,
      "adapters_per_layer": [
        {"layer_id": 0, "num_adapters": 1},
        {"layer_id": 1, "num_adapters": 2},  ← Expanded
        ...
      ]
    },
    ...
  ]
}
```

---

## ⚙️ Hyperparameters chính

### **SEMA Adapters**

| Parameter | Default | Ý nghĩa |
|-----------|---------|---------|
| `adapter_bottleneck` | 64 | Dimension của adapter bottleneck |
| `adapter_dropout` | 0.1 | Dropout rate |
| `adapter_scalar` | 1.0 | Scaling factor cho adapter output |
| `sema_position` | 'ffn' | Vị trí thêm adapter ('ffn', 'attn', 'both') |
| `sema_mode` | 'parallel' | Mode ('parallel'=residual, 'sequential'=in-place) |

### **Representation Descriptor**

| Parameter | Default | Ý nghĩa |
|-----------|---------|---------|
| `rd_dim` | 64 | RD bottleneck dimension |
| `rd_buffer_size` | 500 | Số samples để tính mean/std |
| `expansion_threshold` | 3.0 | Z-score threshold để trigger expansion |
| `rd_loss_weight` | 0.1 | Weight của RD loss trong total loss |

### **Layer Range**

| Parameter | Default | Ý nghĩa |
|-----------|---------|---------|
| `sema_start_layer` | 0 | Layer đầu tiên có adapters |
| `sema_end_layer` | 7 | Layer cuối cùng có adapters (4 encoder + 4 decoder = 8 layers) |

---

## 🔬 So sánh với Baseline

### **Training Process**

| Step | Baseline CL | SEMA-CL |
|------|------------|---------|
| Task 0 | Train all params | Train adapters (init: 1 per layer) |
| After Task 0 | Continue | **Freeze adapters**, enable outlier detection |
| Task 1 | Train all params | Detect shift → **Expand if needed**, train new adapters |
| ... | Catastrophic forgetting | **Minimal forgetting** (frozen adapters) |

### **Expected Results**

| Metric | Baseline | SEMA (expected) |
|--------|----------|-----------------|
| Final Avg Performance | ~0.74 | ~0.78-0.82 |
| Final Avg Forgetting | ~0.20 | ~0.10-0.15 |
| Model size | Fixed | **Sub-linear expansion** |

---

## 🎓 Cơ chế hoạt động chi tiết

### **1. Task 0 - Khởi tạo**

```python
# Mỗi layer có 1 adapter
Layer 0: [Adapter_0]
Layer 1: [Adapter_0]
...
Layer 7: [Adapter_0]

# Train adapters + RD
Loss = MSE_loss + 0.1 * RD_loss
```

### **2. After Task 0**

```python
# Freeze tất cả adapters
for adapter in adapters:
    adapter.freeze_functional()  # Freeze adapter weights
    adapter.freeze_rd()           # Freeze RD weights
    adapter.rd_loss_record.freeze()  # Stop updating statistics

# Enable outlier detection
model.enable_outlier_detection()
```

### **3. Task 1 - Outlier Detection**

```python
# Forward pass
for sample in task_1_data:
    rd_loss = RD.compute_loss(sample)
    z_score = (rd_loss - mean) / std

    if z_score > 3.0:
        ✨ Add new adapter to this layer
        Layer_i: [Adapter_0, Adapter_1]  ← NEW
```

### **4. Task 1 - Mixing Adapters**

```python
# Router network
logits = Router(input)  # [batch, num_adapters]
weights = softmax(logits)

# Mix adapter outputs
output = Σ (weights[i] * Adapter_i(input))
```

### **5. After Task 1**

```python
# Freeze new adapters
# Enable outlier detection for Task 2
...
```

---

## 💡 Tips & Tricks

### **1. Điều chỉnh Expansion Threshold**

```yaml
# Conservative (expand ít hơn)
expansion_threshold: 4.0  # Chỉ expand khi shift rất lớn

# Aggressive (expand nhiều hơn)
expansion_threshold: 2.0  # Expand dễ hơn
```

**Trade-off:**
- High threshold → Ít adapters → Faster, nhưng có thể underfit new tasks
- Low threshold → Nhiều adapters → Slower, nhưng better adaptation

### **2. Điều chỉnh RD Loss Weight**

```yaml
# Tăng RD loss → RD học tốt hơn → Detection chính xác hơn
rd_loss_weight: 0.2

# Giảm RD loss → Focus vào main task
rd_loss_weight: 0.05
```

### **3. Chọn Adapter Position**

```yaml
# FFN only (faster, fewer adapters)
sema_position: 'ffn'

# Attention only (different feature space)
sema_position: 'attn'

# Both (maximum flexibility, more adapters)
sema_position: 'both'
```

---

## 🐛 Troubleshooting

### **Problem: Không có expansion**

**Nguyên nhân:**
- Threshold quá cao
- RD chưa học tốt (rd_loss_weight quá thấp)

**Giải pháp:**
```yaml
expansion_threshold: 2.5  # Giảm threshold
rd_loss_weight: 0.15      # Tăng RD loss weight
```

### **Problem: Expansion quá nhiều**

**Nguyên nhân:**
- Threshold quá thấp
- RD overfitting

**Giải pháp:**
```yaml
expansion_threshold: 3.5  # Tăng threshold
rd_buffer_size: 1000      # Tăng buffer size (stable statistics)
```

### **Problem: Training chậm**

**Giải pháp:**
```yaml
# Giảm số layers có adapters
sema_start_layer: 2
sema_end_layer: 5

# Hoặc chỉ dùng FFN
sema_position: 'ffn'
```

---

## 📚 References

1. **SEMA Paper** (CVPR 2025):
   - Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning
   - https://arxiv.org/abs/2403.18886

2. **UniAD** (Anomaly Detection):
   - Unified Anomaly Detection Framework

3. **AnoCL** (Baseline):
   - Continual Learning for Anomaly Detection on MVTec-AD

---

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Happy Training! 🚀**
