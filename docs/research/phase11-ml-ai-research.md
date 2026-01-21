# Phase 11: ML/AI Integration Research Report

> **Date**: 2026-01-22
> **Status**: Research Analysis
> **Based on**: research-03.md Part 5 분석

---

## Executive Summary

Phase 11 explores ML/AI integration for U-Nesting. After analyzing recent research (2020-2025), we identify three promising approaches with varying feasibility levels:

| Approach | Feasibility | Expected Impact | Implementation Complexity |
|----------|-------------|-----------------|---------------------------|
| **11.1 GNN Efficiency Estimation** | High | Medium | Medium |
| **11.2 RL Policy Learning** | Medium | High | High |
| **11.3 ML-Guided Optimization** | High | High | Low-Medium |

**Recommendation**: Start with 11.3 (ML-Guided Optimization) for immediate practical value, then pursue 11.1 (GNN) for algorithm selection. Phase 11.2 (RL) should remain experimental due to high complexity and training requirements.

---

## 1. Current State-of-the-Art Analysis

### 1.1 GNN for Nesting Efficiency Estimation

**Key Paper**: Lallier et al. (J. Intelligent Manufacturing 2024)
- **Dataset**: 100,000 real nesting instances from MPEG company
- **Result**: MAE 1.65 for efficiency prediction
- **Architecture**: Message Passing Neural Network (MPNN)

**Graph Representation**:
```
Polygon → Graph:
- Nodes: vertices
- Edges: polygon edges
- Node features: (x, y, area_ratio, perimeter_ratio)
- Edge features: (length, angle)
```

**U-Nesting Integration Path**:
```rust
// Optional feature: ml-gnn
pub trait EfficiencyEstimator {
    fn estimate(&self, geometries: &[Geometry2D], boundary: &Boundary2D) -> f64;
}

pub struct GnnEstimator {
    model: TorchModel,  // ONNX runtime or tch-rs
}
```

**Pros**:
- Fast inference (~ms)
- No training data from user required
- Algorithm selection hint

**Cons**:
- Requires pre-trained model
- Model size (~10-50MB)
- Python/ONNX dependency

### 1.2 Reinforcement Learning for Placement

**Key Papers**:
- PCT (ICLR 2022): Graph-based packing state, ~75% utilization
- O4M-SP (2025): Single training for multiple bin sizes
- DMRL-BPP (2024): 7.8% improvement on small instances

**RL Formulation**:
```
State: Current placements + remaining items (graph or heightmap)
Action: (item_idx, position, rotation)
Reward: Utilization improvement or -waste penalty
```

**Challenges for U-Nesting**:
1. **Training Data**: 1M+ instances typically required
2. **Generalization**: Distribution shift between training/deployment
3. **Inference Speed**: Must be faster than heuristics to be useful
4. **Stability**: Often ignored in pure ML approaches

**Mitigation Strategies**:
- Symmetric Replay Training (Kim et al., ICML 2024): 2-10x sample efficiency
- Hierarchical decomposition for better generalization
- Hybrid approach: RL for sequence, heuristic for placement

### 1.3 ML-Guided Optimization (Most Practical)

**JD.com Production System**:
- 68.60% packing rate
- 0.16s per order
- Five-component hybrid: bin selection, item grouping, sequence, position, orientation

**Implementation Approaches**:

#### A. Warm Start for GA/BRKGA
```rust
pub trait WarmStartProvider {
    fn generate_initial_population(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
        population_size: usize,
    ) -> Vec<Vec<f64>>;  // Random keys for BRKGA
}
```

**Benefits**:
- 20-40% faster convergence observed in literature
- Requires simple feature extraction, no deep learning

#### B. Algorithm Selection
```rust
pub enum RecommendedStrategy {
    Blf,           // Simple shapes, time-critical
    NfpGuided,     // Complex polygons
    Brkga,         // High quality requirement
    Alns,          // Large instances
}

pub fn recommend_strategy(
    instance_features: &InstanceFeatures
) -> RecommendedStrategy;
```

**Feature Vector**:
```rust
pub struct InstanceFeatures {
    pub item_count: usize,
    pub avg_area_ratio: f64,      // avg(item_area) / boundary_area
    pub convexity_ratio: f64,     // fraction of convex items
    pub size_variance: f64,       // coefficient of variation
    pub aspect_ratio_max: f64,
    pub has_holes: bool,
}
```

#### C. ALNS Operator Selection
```rust
pub trait OperatorPredictor {
    fn predict_best_destroy(&self, state: &AlnsState) -> DestroyOperator;
    fn predict_best_repair(&self, state: &AlnsState) -> RepairOperator;
}
```

**Simple ML Model**: Random Forest or XGBoost on instance features
- Training data: Collect during ALNS runs (which operator succeeded)
- Online learning possible

---

## 2. Feasibility Analysis for U-Nesting

### 2.1 Rust ML Ecosystem

| Crate | Purpose | Maturity | Notes |
|-------|---------|----------|-------|
| `tch-rs` | PyTorch bindings | Stable | Requires libtorch |
| `burn` | Pure Rust ML | Growing | Limited pre-trained models |
| `ort` | ONNX Runtime | Stable | Best for inference |
| `linfa` | Classical ML | Stable | Good for RF/XGBoost |
| `smartcore` | Classical ML | Stable | Alternative to linfa |

**Recommendation**:
- Pre-trained models: ONNX Runtime (`ort`)
- Simple ML: `linfa` or `smartcore`
- Avoid training in Rust (use Python, export ONNX)

### 2.2 Integration Architecture

```
┌─────────────────────────────────────────┐
│              U-Nesting Core             │
├─────────────────────────────────────────┤
│  ml::estimator::GnnEstimator [ml-gnn]   │
│  ml::selector::StrategySelector [ml]    │
│  ml::warmstart::WarmStartProvider [ml]  │
├─────────────────────────────────────────┤
│         ONNX Runtime / linfa            │
└─────────────────────────────────────────┘
```

**Feature Flags**:
- `ml`: Basic ML integration (strategy selection, warm start)
- `ml-gnn`: GNN efficiency estimation (requires ONNX)
- `ml-rl`: RL policy (experimental, requires large models)

### 2.3 Training Data Strategy

**Option A: Public Benchmarks**
- ESICUP datasets (already integrated)
- BPPLIB for 3D
- Generate labels by running existing algorithms

**Option B: Synthetic Generation**
- Already have `benchmark/src/synthetic.rs`
- Generate diverse instances programmatically
- Label with best-known solutions from BRKGA

**Option C: User Data Collection (Privacy-Preserving)**
- Opt-in telemetry
- Only collect instance features, not geometry
- Federated learning approach

---

## 3. Implementation Roadmap

### Phase 11.1: GNN Efficiency Estimation (2주)

#### 11.1.1 Instance Graph Representation (3일)
```rust
// crates/core/src/ml/graph.rs
pub struct PolygonGraph {
    pub nodes: Vec<NodeFeatures>,
    pub edges: Vec<(usize, usize, EdgeFeatures)>,
}

pub struct NodeFeatures {
    pub x: f64,
    pub y: f64,
    pub area_contrib: f64,
    pub convexity_local: f64,
}

pub struct EdgeFeatures {
    pub length: f64,
    pub angle: f64,
}

impl From<&Geometry2D> for PolygonGraph { ... }
```

#### 11.1.2 ONNX Model Integration (3일)
```rust
// crates/core/src/ml/gnn.rs
#[cfg(feature = "ml-gnn")]
pub struct GnnEstimator {
    session: ort::Session,
}

impl GnnEstimator {
    pub fn from_model_path(path: &Path) -> Result<Self, MlError>;
    pub fn estimate(&self, graphs: &[PolygonGraph]) -> f64;
}
```

#### 11.1.3 Training Pipeline (외부)
- Python: PyTorch Geometric
- Export to ONNX
- Ship pre-trained model in `assets/models/`

#### 11.1.4 Integration API
```rust
pub fn estimate_efficiency(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
) -> Result<f64, MlError>;
```

### Phase 11.2: RL Policy Learning (2주) - 실험적

> ⚠️ **Warning**: High complexity, experimental only

#### 11.2.1 Environment Definition
```rust
pub struct NestingEnv {
    boundary: Boundary2D,
    geometries: Vec<Geometry2D>,
    placements: Vec<Option<Placement2D>>,
    remaining: Vec<usize>,
}

pub struct Action {
    item_idx: usize,
    position: (f64, f64),
    rotation: f64,
}

impl NestingEnv {
    pub fn step(&mut self, action: Action) -> (Observation, f64, bool);
    pub fn reset(&mut self);
}
```

#### 11.2.2 Policy Network
- Input: Graph representation or heightmap
- Output: Action probabilities
- Training: PPO (Proximal Policy Optimization)

#### 11.2.3 Training Infrastructure
- Requires Python for efficient training
- Export policy to ONNX for Rust inference
- 1M+ episodes for convergence

### Phase 11.3: ML-Guided Optimization (1.5주) - 권장

#### 11.3.1 Feature Extraction (1일)
```rust
// crates/core/src/ml/features.rs
pub struct InstanceFeatures {
    pub item_count: usize,
    pub total_area_ratio: f64,
    pub avg_item_area: f64,
    pub item_area_variance: f64,
    pub convex_ratio: f64,
    pub max_aspect_ratio: f64,
    pub has_holes: bool,
    pub has_concave: bool,
}

impl InstanceFeatures {
    pub fn from_instance(
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Self;

    pub fn to_vector(&self) -> Vec<f64>;
}
```

#### 11.3.2 Strategy Selector (2일)
```rust
// crates/core/src/ml/selector.rs
pub struct StrategySelector {
    model: linfa::RandomForest,  // or simple rules
}

impl StrategySelector {
    pub fn recommend(&self, features: &InstanceFeatures) -> Strategy {
        // Rule-based initial version:
        if features.item_count > 200 {
            return Strategy::Alns;
        }
        if features.convex_ratio > 0.8 && features.item_count < 50 {
            return Strategy::Blf;
        }
        if features.has_concave || features.has_holes {
            return Strategy::NfpGuided;
        }
        Strategy::Brkga
    }
}
```

#### 11.3.3 BRKGA Warm Start (2일)
```rust
// crates/core/src/ml/warmstart.rs
pub trait WarmStartProvider {
    fn generate_keys(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Vec<f64>;
}

// Simple heuristic-based warm start
pub struct AreaSortWarmStart;

impl WarmStartProvider for AreaSortWarmStart {
    fn generate_keys(&self, geometries: &[Geometry2D], _: &Boundary2D) -> Vec<f64> {
        // Sort by area descending, return keys reflecting this order
        let mut indexed: Vec<_> = geometries.iter()
            .enumerate()
            .map(|(i, g)| (i, g.area()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut keys = vec![0.0; geometries.len()];
        for (rank, (idx, _)) in indexed.into_iter().enumerate() {
            keys[idx] = rank as f64 / geometries.len() as f64;
        }
        keys
    }
}
```

#### 11.3.4 ALNS Operator Guidance (2일)
```rust
// Extend existing ALNS with ML guidance
pub struct MlGuidedAlns {
    inner: AlnsRunner,
    operator_model: Option<OperatorPredictor>,
}

impl MlGuidedAlns {
    pub fn select_destroy(&mut self) -> DestroyOperator {
        if let Some(model) = &self.operator_model {
            // Use ML prediction with some exploration
            if self.rng.gen::<f64>() < 0.8 {
                return model.predict_destroy(&self.state);
            }
        }
        // Fall back to weight-based selection
        self.inner.select_destroy_operator()
    }
}
```

---

## 4. Benchmarking Plan

### 4.1 Metrics

| Metric | Description |
|--------|-------------|
| Utilization | Placed area / boundary area |
| Time to Solution | Wall-clock time |
| Convergence Speed | Iterations to 95% of final quality |
| Prediction Accuracy | MAE for efficiency estimation |
| Recommendation Accuracy | % of correct strategy selections |

### 4.2 Test Datasets

1. **ESICUP**: Standard academic benchmarks
2. **Synthetic**: Controlled difficulty levels
3. **Large-scale**: 200-1000 items

### 4.3 Baselines

- Pure BRKGA (no ML)
- Pure ALNS (no ML)
- Random strategy selection
- Oracle (best strategy per instance)

---

## 5. Risk Assessment

### High Risk
- **RL Training**: May require significant compute resources and expertise
- **Model Maintenance**: Pre-trained models may degrade with distribution shift

### Medium Risk
- **ONNX Dependency**: Adds binary size and complexity
- **Cross-platform**: ONNX Runtime availability on all targets

### Low Risk
- **Rule-based Strategy Selection**: Simple, interpretable, maintainable
- **Heuristic Warm Start**: No ML dependency, guaranteed improvement

---

## 6. Recommendation

### Immediate (Phase 11.3)
1. Implement `InstanceFeatures` extraction
2. Rule-based strategy selector
3. Heuristic-based BRKGA warm start

### Short-term (Phase 11.1)
4. ONNX integration infrastructure
5. GNN efficiency estimator (with pre-trained model)

### Long-term (Phase 11.2)
6. RL policy exploration (experimental feature flag)
7. User data collection for model improvement

---

## 7. Code Structure Proposal

```
crates/core/src/ml/
├── mod.rs           # Feature-gated module
├── features.rs      # Instance feature extraction
├── selector.rs      # Strategy selection
├── warmstart.rs     # BRKGA warm start providers
├── gnn.rs          # GNN estimator [ml-gnn]
└── rl.rs           # RL policy [ml-rl]

assets/models/
├── gnn_efficiency_v1.onnx
└── strategy_selector_v1.json
```

---

## 8. References

1. Lallier et al. (2024) "Graph neural network comparison for 2D nesting efficiency estimation"
2. Kim et al. (2024) "Symmetric Replay Training: Enhancing Sample Efficiency in Deep Reinforcement Learning for Combinatorial Optimization"
3. PCT (2022) "Learning Efficient Online 3D Bin Packing on Packing Configuration Trees"
4. JD.com (2024) "A DRL-based hybrid approach for 3D bin packing"
5. Gasse et al. (2019) "Exact Combinatorial Optimization with Graph Convolutional Neural Networks"

---

## Appendix A: GNN Architecture Reference

```python
# PyTorch Geometric implementation reference
class NestingGNN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.node_encoder = Linear(4, hidden_dim)  # x, y, area, convexity
        self.edge_encoder = Linear(2, hidden_dim)  # length, angle

        self.convs = ModuleList([
            GATConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.readout = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index) + x  # Residual
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.readout(x).squeeze()
```

---

## Appendix B: Training Data Generation Script

```python
# scripts/generate_ml_training_data.py
import json
from u_nesting import solve_2d, Geometry2D, Boundary2D

def generate_instance(difficulty: str):
    # Use synthetic generator
    ...

def run_all_strategies(geometries, boundary):
    results = {}
    for strategy in ['blf', 'nfp', 'brkga', 'alns']:
        result = solve_2d(geometries, boundary, strategy=strategy, time_limit_ms=30000)
        results[strategy] = {
            'utilization': result.utilization,
            'time_ms': result.elapsed_ms,
        }
    return results

def main():
    dataset = []
    for _ in range(10000):
        geoms, boundary = generate_instance('random')
        features = extract_features(geoms, boundary)
        results = run_all_strategies(geoms, boundary)
        best = max(results.items(), key=lambda x: x[1]['utilization'])

        dataset.append({
            'features': features,
            'results': results,
            'best_strategy': best[0],
        })

    with open('training_data.json', 'w') as f:
        json.dump(dataset, f)
```
