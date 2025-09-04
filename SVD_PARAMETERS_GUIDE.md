# 📊 SVD Parameters Guide - Clear Technical Explanation

## 🎯 **What SVD Filtering Does**

SVD decomposes a matrix into: `Matrix = U × S × V^T`
- **U**: Orthogonal directions (what we keep)
- **S**: Importance weights (singular values, largest = most important)  
- **V^T**: Not used in our case

**Goal**: Keep only the **most important directions** from U, discard the rest.

---

## 🔧 **Parameter 1: Cumulative Energy Threshold**

### **What it means:**
```python
cumulative_energy_threshold = 0.85  # Keep directions explaining 85% of total variance
```

### **How it works:**
1. Calculate total energy: `total_energy = sum(S²)` 
2. For each direction i: `energy_i = S[i]²`
3. Calculate cumulative: `cumulative[i] = sum(energy_0 to energy_i) / total_energy`
4. Keep directions until cumulative ≥ threshold

### **Example:**
```
Singular values S = [10, 5, 3, 1, 0.5, 0.1]
Energy ratios     = [0.67, 0.17, 0.06, 0.007, 0.002, 0.0007] 
Cumulative       = [0.67, 0.84, 0.90, 0.907, 0.909, 0.910]

With threshold = 0.85:
- Direction 0: 67% ✅ (keep)
- Direction 1: 84% ✅ (keep) 
- Direction 2: 90% ✅ (keep, exceeds 85%)
- Directions 3-5: ❌ (discard)

Result: Keep first 3 directions (explain 90% of variance)
```

### **How to tune:**
- **Higher values (0.90-0.99)**: Keep more directions (more complete, less filtering)
- **Lower values (0.80-0.85)**: Keep fewer directions (more filtering, only essentials)
- **Use 0.95+ when you want to preserve most information**
- **Use 0.80-0.85 when you want maximum compression**

---

## 🔧 **Parameter 2: Spectral Gap Ratio**

### **What it means:**
```python
spectral_gap_ratio = 0.15  # Keep directions before a 15% drop in importance
```

### **How it works:**
1. Calculate ratios: `ratio[i] = S[i+1] / S[i]` (how much smaller the next singular value is)
2. Find first gap where `ratio[i] < spectral_gap_ratio`
3. Keep directions 0 to i (before the gap)

### **Example:**
```
Singular values S = [10,   5,   3,   0.4, 0.35, 0.1]  
Ratios             = [0.5, 0.6, 0.13, 0.87, 0.29]

With spectral_gap_ratio = 0.15:
- S[1]/S[0] = 0.5  ✅ (> 0.15, continue)
- S[2]/S[1] = 0.6  ✅ (> 0.15, continue) 
- S[3]/S[2] = 0.13 ❌ (< 0.15, STOP HERE!)

Result: Keep first 3 directions (before the gap)
```

### **How to tune:**
- **Higher values (0.2-0.5)**: More sensitive to gaps, keeps fewer directions
- **Lower values (0.05-0.1)**: Less sensitive to gaps, keeps more directions
- **Use 0.1-0.2 for most cases (detects natural breaks)**
- **Use 0.3+ when you want to be very selective**

---

## 🔧 **Parameter 3: Rank Limits**

### **What it means:**
```python
max_rank = self.subspace_dim // 2  # Never exceed half of allocated subspace
```

### **How it works:**
After energy and spectral analysis, enforce: `final_rank = min(energy_rank, spectral_rank, max_rank)`

### **Examples:**
```
subspace_dim = 20

// 2 → max_rank = 10  (allows up to half)
// 3 → max_rank = 6   (allows up to one-third)  
// 4 → max_rank = 5   (allows up to one-quarter)
```

### **How to tune:**
- **`// 2`**: Moderate limit (allows up to 50% of subspace)
- **`// 3`**: Stricter limit (allows up to 33% of subspace)
- **`// 4`**: Very strict limit (allows up to 25% of subspace)
- **Use `// 2` for balanced filtering**
- **Use `// 4` when you want maximum compression**

---

## 🎯 **Current Settings Explained**

### **Server Past Bases Stacking:**
```python
cumulative_energy_threshold = 0.95  # Keep 95% of variance (preserve most info)
spectral_gap_ratio = 0.1            # Detect small gaps (keep more)
max_rank = subspace_dim // 2         # Moderate limit
```
**Intent**: Preserve past knowledge well, moderate filtering

### **Knowledge Base Fusion:**
```python
cumulative_energy_threshold = 0.90  # Keep 90% of variance (more filtering)
spectral_gap_ratio = 0.15           # Detect larger gaps (keep less)  
max_rank = subspace_dim // 3         # Stricter limit
```
**Intent**: More selective fusion, avoid redundancy

### **Client Knowledge Distillation:**
```python
cumulative_energy_threshold = 0.85  # Keep 85% of variance (most filtering)
spectral_gap_ratio = 0.2            # Detect even larger gaps (keep least)
max_rank = subspace_dim // 4         # Strictest limit
```
**Intent**: Extract only the most essential knowledge

---

## 🛠 **How to Modify for Your Needs**

### **If you want LESS filtering (keep more components):**
- **Increase** cumulative_energy_threshold: `0.85 → 0.90 → 0.95`
- **Decrease** spectral_gap_ratio: `0.2 → 0.15 → 0.1`
- **Increase** rank limit: `// 4 → // 3 → // 2`

### **If you want MORE filtering (keep fewer components):**
- **Decrease** cumulative_energy_threshold: `0.95 → 0.90 → 0.85`
- **Increase** spectral_gap_ratio: `0.1 → 0.15 → 0.3`
- **Decrease** rank limit: `// 2 → // 3 → // 4`

### **Recommended starting points:**
```python
# For most cases (balanced)
cumulative_energy_threshold = 0.90
spectral_gap_ratio = 0.15
max_rank = subspace_dim // 2

# For maximum compression (if forgetting is still occurring)
cumulative_energy_threshold = 0.80  
spectral_gap_ratio = 0.25
max_rank = subspace_dim // 4

# For maximum preservation (if losing too much information)
cumulative_energy_threshold = 0.95
spectral_gap_ratio = 0.1
max_rank = subspace_dim // 2
```

---

## 📊 **Debugging Your Settings**

Monitor these logs to tune parameters:
```
[APOP] SVD analysis: energy_rank=8, spectral_rank=5
[APOP] Cumulative energy (90%): 0.9234
[APOP] Rank analysis: full_rank=25, effective_rank=5, using=5
```

**What to look for:**
- **energy_rank ≫ spectral_rank**: Spectral method is more restrictive (good)
- **effective_rank ≪ full_rank**: Filtering is working (e.g., 5 vs 25)
- **Cumulative energy > threshold**: Explained variance meets target (good)

**Red flags:**
- **effective_rank = max_rank**: Hitting the limit (may need higher limit)
- **energy_rank = 1**: Too restrictive energy threshold
- **spectral_rank = full_rank**: No gaps detected (may need higher gap ratio)
