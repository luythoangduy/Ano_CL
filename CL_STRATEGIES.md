# 🎯 Continual Learning Strategies cho AnoCL

## 📊 Tổng quan 4 Strategies

| Strategy | SEMA Adapters | Memory | Freeze Strategy | Config File |
|----------|--------------|--------|-----------------|-------------|
| **1. Baseline** | ❌ No | Single (retrain) | Backbone only | `config_continual.yaml` |
| **2. SEMA** | ✅ Yes | Single (retrain) | Backbone + old adapters | `config_sema.yaml` |
| **3. MemExpand** | ❌ No | **Stack memories** | Backbone + old modules | `config_mem_expand.yaml` |
| **4. SEMA+Mem** | ✅ Yes | **Stack memories** | Backbone + all old | `config_sema_mem.yaml` |

---

## 🔍 Chi tiết từng Strategy

### **1️⃣ Baseline (No CL Strategy)**

```yaml
cl_strategy: baseline
```

**Cơ chế:**
- Freeze: **Chỉ backbone**
- Train: Neck + Memory + Transformer mỗi task
- Không có anti-forgetting mechanism

**Timeline:**
```
Task 0:
  Backbone:      ❄️ FROZEN
  Neck:          ✅ Train
  Memory:        ✅ Train
  Transformer:   ✅ Train

Task 1:
  Backbone:      ❄️ FROZEN
  Neck:          ✅ Retrain (overwrite T0 weights)
  Memory:        ✅ Retrain (overwrite T0 weights)
  Transformer:   ✅ Retrain (overwrite T0 weights)
  → CATASTROPHIC FORGETTING!
```

**Pros:**
- ✅ Simple, không cần extra memory
- ✅ Fast training

**Cons:**
- ❌ **High forgetting** (~0.20)
- ❌ Không có CL mechanism

---

### **2️⃣ SEMA (Adapters Only)**

```yaml
cl_strategy: sema

sema:
  use_sema: true
  expansion_threshold: 3.0
```

**Cơ chế:**
- Freeze: Backbone + **old SEMA adapters**
- Train: Neck + Memory + Transformer + **new adapters**
- Self-expansion khi detect distribution shift

**Timeline:**
```
Task 0:
  Backbone:      ❄️ FROZEN
  Neck:          ✅ Train
  Memory:        ✅ Train
  Transformer:   ✅ Train
  Adapters:      ✅ Train (8 adapters init, 1 per layer)

After Task 0:
  Adapters_T0:   🔒 FREEZE

Task 1:
  Backbone:      ❄️ FROZEN
  Neck:          ✅ Retrain
  Memory:        ✅ Retrain
  Transformer:   ✅ Retrain
  Adapters_T0:   🔒 FROZEN
  Adapters_T1:   ✅ Train (new adapters if Z-score > threshold)
```

**Pros:**
- ✅ **Lower forgetting** (~0.15) than baseline
- ✅ Self-expansion (automatic)
- ✅ Moderate memory increase

**Cons:**
- ❌ Vẫn có forgetting ở neck/memory/transformer
- ❌ Thêm ~2-4M params (adapters)

---

### **3️⃣ MemExpand (Memory Stacking)**

```yaml
cl_strategy: mem_expand
```

**Cơ chế:**
- Freeze: Backbone + **old neck + old memory + old transformer**
- Train: **New neck + new memory + new transformer** (fresh modules)
- Stack outputs từ tất cả modules

**Timeline:**
```
Task 0:
  Backbone:          ❄️ FROZEN
  Neck_T0:           ✅ Train
  Memory_T0:         ✅ Train
  Transformer_T0:    ✅ Train

After Task 0:
  Neck_T0:           🔒 FREEZE + Save
  Memory_T0:         🔒 FREEZE + Save
  Transformer_T0:    🔒 FREEZE + Save

Task 1:
  Backbone:          ❄️ FROZEN
  
  # Frozen modules from T0
  Neck_T0:           🔒 FROZEN
  Memory_T0:         🔒 FROZEN
  Transformer_T0:    🔒 FROZEN
  
  # New modules for T1
  Neck_T1:           ✅ Train (new instance)
  Memory_T1:         ✅ Train (new instance)
  Transformer_T1:    ✅ Train (new instance)

Inference:
  pred = average(T0_output, T1_output)
```

**Pros:**
- ✅ **Zero forgetting** (old modules frozen)
- ✅ Không cần SEMA complexity

**Cons:**
- ❌ **Linear memory growth** (~11M per task)
- ❌ Slower inference (forward through all modules)

---

### **4️⃣ SEMA + MemExpand (Combined)**

```yaml
cl_strategy: sema_mem_expand

sema:
  use_sema: true
  expansion_threshold: 3.0
```

**Cơ chế:**
- Freeze: Backbone + **old everything** (modules + adapters)
- Train: New modules + new adapters
- Best of both worlds

**Timeline:**
```
Task 0:
  Backbone:          ❄️ FROZEN
  Neck_T0:           ✅ Train
  Memory_T0:         ✅ Train
  Transformer_T0:    ✅ Train
  Adapters_T0:       ✅ Train (8 init)

After Task 0:
  Everything_T0:     🔒 FREEZE + Save

Task 1:
  Backbone:          ❄️ FROZEN
  Everything_T0:     🔒 FROZEN
  
  # New modules
  Neck_T1:           ✅ Train
  Memory_T1:         ✅ Train
  Transformer_T1:    ✅ Train
  Adapters_T1:       ✅ Train (+ expansion if needed)

Inference:
  pred = mix(T0_output, T1_output)
```

**Pros:**
- ✅ **Lowest forgetting** (everything frozen)
- ✅ Self-expansion adapters
- ✅ Flexible adaptation

**Cons:**
- ❌ **Highest memory** (~13M per task)
- ❌ Most complex

---

## 🚀 Cách chạy

### **Strategy 1: Baseline**
```bash
python train_continual.py --config config_continual.yaml
```

### **Strategy 2: SEMA**
```bash
python train_continual.py --config config_sema.yaml
```

### **Strategy 3: MemExpand**
```bash
python train_continual.py --config config_mem_expand.yaml
```

### **Strategy 4: SEMA + MemExpand**
```bash
python train_continual.py --config config_sema_mem.yaml
```

---

## 📊 So sánh Performance (Dự đoán)

| Strategy | Avg Perf (Final) | Avg Forgetting (Final) | Memory Growth | Speed |
|----------|-----------------|----------------------|---------------|-------|
| **Baseline** | 0.74 | **0.20** ❌ | 0 MB | ⚡⚡⚡ Fast |
| **SEMA** | 0.78 | **0.15** 🟡 | +2-4 MB | ⚡⚡ Medium |
| **MemExpand** | **0.82** | **0.05** ✅ | +165 MB (15 tasks) | ⚡ Slow |
| **SEMA+Mem** | **0.85** | **0.02** ✅✅ | +195 MB (15 tasks) | ⚡ Slow |

**Tính toán memory:**
- MemExpand: ~11M params/task × 15 tasks = 165M params ≈ 660 MB
- SEMA+Mem: ~13M params/task × 15 tasks = 195M params ≈ 780 MB

---

## 💡 Khi nào dùng Strategy nào?

### **Baseline** - Debug/Research
- Chỉ để baseline comparison
- Không recommend cho production

### **SEMA** - Balanced
- **Recommended** cho hầu hết use cases
- Balance giữa performance và memory
- Tốt khi có 5-15 tasks

### **MemExpand** - High Performance
- Khi cần **zero forgetting**
- Có đủ memory budget
- Số tasks ít (< 10)

### **SEMA + MemExpand** - Best Performance
- Khi cần **best possible results**
- Memory không phải vấn đề
- Production systems với high accuracy requirements

---

## 🔧 Implementation Details

### **File structure:**
```python
models/
├── cl_strategies.py          # ← All strategies
│   ├── BaselineStrategy
│   ├── SEMAStrategy
│   ├── MemoryExpansionStrategy
│   └── SEMAMemExpandStrategy
├── uniad_learner.py           # Uses strategies
└── ...
```

### **Core methods:**
```python
class CLStrategy:
    def after_task(network, task_id):
        # Freeze modules sau task
        pass
    
    def before_task(network, task_id):
        # Setup trước task mới
        pass
```

### **Memory Expansion forward:**
```python
# MemExpand inference
outputs = []

# Forward qua frozen modules
for task_module in frozen_modules:
    with torch.no_grad():
        out = task_module(input)
        outputs.append(out)

# Forward qua current module
out_current = current_module(input)
outputs.append(out_current)

# Mix outputs
final_pred = average(outputs)
```

---

## 🎓 References

1. **SEMA** (CVPR 2025): Self-Expansion with Mixture of Adapters
2. **MemExpand**: Inspired by Progressive Neural Networks
3. **AnoCL**: Continual Learning for Anomaly Detection

---

**Chọn strategy phù hợp với requirements của bạn! 🚀**
