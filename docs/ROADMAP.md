# U-Nesting 개발 로드맵

리서치 문서를 기반으로 상세한 다단계 로드맵을 구성했습니다.

> **마지막 업데이트**: 2026-01-21
> **현재 진행 단계**: Phase 1 완료, Phase 2 완료 (100%), Phase 3 완료 (100%), Phase 4 완료 (75%), Phase 5.2 완료, Phase 5.3 완료, Phase 6.1 완료, Phase 6.2 완료, Phase 6.3 완료, Phase 6.4 부분 완료

---

## 전체 타임라인 개요

| Phase | 기간 | 핵심 목표 | 상태 |
|-------|------|----------|------|
| **Phase 1** | 5-6주 | Geometry Core (2D/3D 기초) | ✅ 완료 |
| **Phase 2** | 4-5주 | NFP 엔진 및 배치 알고리즘 | ✅ 완료 |
| **Phase 3** | 5-6주 | 최적화 알고리즘 (GA/SA) | ✅ 완료 |
| **Phase 4** | 3-4주 | 성능 최적화 및 병렬화 | 🔄 진행 중 (75%) |
| **Phase 5** | 3-4주 | FFI 및 통합 API | 🔄 진행 중 (80%) |
| **Phase 6** | 2-3주 | 벤치마크 및 릴리스 준비 | 🔄 진행 중 (85%) |

**총 예상 기간: 22-28주**

---

## Phase 1: Geometry Core Foundation (5-6주) ✅ 완료

### 목표
2D/3D 기하학적 기초 구조 구축 및 기본 연산 구현

### 태스크

#### 1.1 프로젝트 구조 설정 (3일) ✅
- [x] Cargo workspace 구성 (`core`, `d2`, `d3`, `ffi` 크레이트)
- [x] 의존성 설정 (`geo`, `geo-types`, `parry2d`, `parry3d`, `nalgebra`)
- [x] CI/CD 파이프라인 구성 (GitHub Actions)
- [x] 코드 품질 도구 설정 (`clippy`, `rustfmt`, `cargo-deny`)

#### 1.2 Core Traits 정의 (1주) ✅
- [x] `Geometry` trait (2D/3D 공통 추상화) - `core/geometry.rs`
- [x] `Boundary` trait (컨테이너 추상화) - `core/geometry.rs`
- [x] `Placement` struct (위치 + 회전) - `core/placement.rs`
- [x] `SolveResult` struct (결과 표현) - `core/result.rs`
- [x] Error types 정의 (`thiserror` 기반) - `core/error.rs`

#### 1.3 2D Polygon 구현 (1.5주) ✅
- [x] `Geometry2D` 구조체 (외곽선 + 홀) - `d2/geometry.rs`
- [x] 기본 연산: 면적, 중심점, 바운딩 박스
- [x] Convex hull 계산 (`geo` crate 활용)
- [x] Convexity 판정
- [x] 둘레(perimeter) 계산
- [x] 헬퍼: `rectangle()`, `circle()`, `l_shape()`

#### 1.4 3D Geometry 구현 (1.5주) ✅
- [x] `Geometry3D` 구조체 (Box3D) - `d3/geometry.rs`
- [x] AABB (Axis-Aligned Bounding Box)
- [x] Volume 계산
- [x] `OrientationConstraint` (Any, Upright, Fixed)
- [x] 6가지 축 정렬 회전 지원

#### 1.5 Convex Decomposition (1주) ❌ 미구현
- [ ] Hertel-Mehlhorn 알고리즘 구현 (2D)
- [ ] V-HACD 통합 또는 구현 (3D)
- [ ] Decomposition 결과 캐싱

> **Note**: 현재 NFP 없이 BLF 알고리즘만 사용하므로 우선순위 낮음. Phase 2에서 NFP 구현 시 필요.

---

## Phase 2: NFP Engine & Placement Algorithms (4-5주) ✅ 완료

### 목표
No-Fit Polygon 계산 엔진 및 기본 배치 알고리즘 구현

### 태스크

#### 2.1 NFP 계산 - Convex Case (1주) ✅ 완료
- [x] Minkowski Sum for convex polygons (O(n+m))
- [x] Edge vector sorting and merging
- [x] Reference point tracking

#### 2.2 NFP 계산 - Non-Convex Case (2주) ✅ 완료
- [ ] Burke et al. Orbiting 알고리즘 구현 (대안 사용)
- [ ] Degenerate case 처리 (collinear, coincident) (향후 개선)
- [x] Decomposition + Union 방식 대안 구현
- [x] `i_overlay` 기반 Boolean 연산 통합 (정확한 non-convex NFP)
- [ ] Hole 처리 (내부 구멍이 있는 폴리곤) (향후 개선)

> **현재 상태**: Triangulation + Minkowski sum + i_overlay union 방식으로 non-convex NFP 구현 완료.

#### 2.3 Inner Fit Polygon (IFP) (0.5주) ✅ 완료
- [x] Container 경계에 대한 IFP 계산
- [x] Margin 적용 (`compute_ifp_with_margin()` 함수 추가)

#### 2.4 NFP 캐싱 시스템 (0.5주) ✅ 완료
- [x] `NfpCache` 구조체 정의
- [x] Thread-safe cache (`Arc<RwLock<HashMap>>`)
- [x] Cache key: `(geometry_id, geometry_id, rotation_angle)`
- [x] Simple eviction policy (half-cache clear when full)

#### 2.5 2D Placement Algorithms (1주) 🔄 부분 구현
- [x] **Bottom-Left Fill (BLF)**: 기본 구현 - `d2/nester.rs`
  - Row-based placement
  - Margin/spacing 지원
  - Cancellation 지원
- [x] **NFP-guided BLF**: NFP 경계 위 최적점 탐색 - `d2/nester.rs`
  - IFP 기반 유효 영역 계산
  - NFP 기반 충돌 회피
  - 다중 회전 각도 시도
  - Bottom-left 우선 배치
- [ ] **Deepest Bottom-Left Fill (DBLF)**: 개선된 BLF
- [ ] **Touching Perimeter**: 접촉 최대화

#### 2.6 3D Placement Algorithms (1주) ✅ 완료
- [x] **Layer Packing**: 기본 구현 - `d3/packer.rs`
  - Layer/row-based placement
  - Mass constraint 지원
  - Margin/spacing 지원
- [x] **Extreme Point Heuristic**: EP 생성 및 관리 - `d3/extreme_point.rs`
  - ExtremePointSet 데이터 구조
  - 배치된 박스로부터 새로운 EP 생성
  - Residual space 계산
  - Bottom-left-back 우선순위 기반 EP 선택
  - `Strategy::ExtremePoint` 지원
- [ ] **DBLF-3D**: 3D 확장 (선택적)
- [ ] GJK/EPA 기반 collision detection (`parry3d`) (선택적)

### Benchmark 추가
- [x] `d2/benches/nfp_bench.rs` - 벤치마크 파일 존재 (NFP 구현 후 활성화 필요)
- [x] `d3/benches/packer_bench.rs` - 벤치마크 파일 존재

---

## Phase 3: Optimization Algorithms (5-6주) ✅ 완료

### 목표
Genetic Algorithm 및 Simulated Annealing 최적화 엔진 구현

### 태스크

#### 3.1 GA Framework Core (1주) ✅ 완료
- [x] `Individual` trait 정의 - `core/ga.rs`
- [x] `GaProblem` trait 정의
- [x] `GaConfig` 설정 구조체
- [x] `GaRunner` evolution loop
  - [x] Tournament selection
  - [x] Elitism
  - [x] Time limit / target fitness 조기 종료
  - [x] Stagnation detection
  - [x] Cancellation support

#### 3.2 Permutation Chromosome (0.5주) ✅ 완료
- [x] `PermutationChromosome` 구조체
- [x] **Order Crossover (OX1)**: 순서 보존 교차
- [x] **Swap Mutation**: 위치 교환
- [x] **Inversion Mutation**: 구간 반전
- [x] Rotation gene 지원

#### 3.3 2D Nesting GA (2주) ✅ 완료
- [x] `NestingProblem` implementing `GaProblem` - `d2/ga_nesting.rs`
- [x] Decoder: chromosome → placement sequence (NFP-guided decoding)
- [x] Fitness function: placement ratio + utilization
- [x] Rotation gene integration with NFP
- [x] `Strategy::GeneticAlgorithm` 지원 - `d2/nester.rs`

> **구현 내용**:
> - `NestingChromosome`: 배치 순서(permutation) + 회전 유전자
> - Order Crossover (OX1) 및 Swap/Inversion/Rotation mutation
> - NFP-guided decoder로 collision-free placement 생성
> - Fitness = placement_ratio * 100 + utilization * 10

#### 3.4 BRKGA 구현 (1주) ✅ 완료
- [x] Random-key encoding - `core/brkga.rs`
- [x] Biased crossover (elite parent preference)
- [x] Decoder: random keys → placement sequence
- [x] 2D Nesting BRKGA - `d2/brkga_nesting.rs`
- [x] 3D Packing BRKGA - `d3/brkga_packing.rs`
- [x] `Strategy::Brkga` 지원

> **구현 내용**:
> - `RandomKeyChromosome`: [0,1) 범위의 random key 유전자
> - Biased crossover: elite parent 70% 확률로 선호
> - Population 구성: elite 20%, mutants 15%, crossover offspring 65%
> - Decoder: sorted indices로 permutation 변환, discrete decoding for rotations
> - Fitness = placement_ratio * 100 + utilization * 10

#### 3.5 3D Bin Packing GA (1주) ✅ 완료
- [x] Box orientation encoding (6가지 회전)
- [x] Layer-based decoder with orientation support
- [ ] Extreme Point 기반 decoder (향후 개선)
- [ ] Stability constraint 통합 (향후 개선)

> **구현 내용**:
> - `PackingChromosome`: 배치 순서(permutation) + orientation 유전자
> - Order Crossover (OX1) 및 Swap/Inversion/Orientation mutation
> - Layer-based decoder로 collision-free placement 생성
> - Mass constraint 지원
> - Fitness = placement_ratio * 100 + utilization * 10

#### 3.6 Simulated Annealing (1주) ✅ 완료
- [x] Cooling schedule: Geometric, Linear, Adaptive, LundyMees - `core/sa.rs`
- [x] Neighborhood operators: Swap, Relocate, Inversion, Rotation, Chain
- [x] Acceptance probability: exp(-ΔE/T)
- [x] Reheating 전략 (stagnation 감지 시)
- [x] 2D Nesting SA - `d2/sa_nesting.rs`
- [x] 3D Packing SA - `d3/sa_packing.rs`
- [x] `Strategy::SimulatedAnnealing` 지원

> **구현 내용**:
> - `SaConfig`: 온도, cooling rate, iterations 설정
> - `PermutationSolution`: sequence + rotation encoding
> - `SaRunner`: temperature-based acceptance, early stopping
> - Fitness = placement_ratio * 100 + utilization * 10

#### 3.7 Local Search / Hill Climbing (0.5주) ❌ 미구현
- [ ] First-improvement 전략
- [ ] Best-improvement 전략
- [ ] Variable Neighborhood Search (VNS) 기초

---

## Phase 4: Performance Optimization (3-4주) 🔄 진행 중

### 목표
병렬화 및 메모리 최적화를 통한 성능 향상

### 태스크

#### 4.1 NFP 병렬 계산 (1주) ✅ 완료
- [x] `rayon::par_iter()` 적용 - `d2/nfp.rs`
- [x] Pairwise Minkowski sum parallel computation
- [x] Work stealing 자동 최적화 (rayon 내장)

> **구현 내용**:
> - `compute_nfp_general()` 함수에서 triangulation 후 pairwise Minkowski sum을 병렬 계산
> - `par_iter().flat_map()` 패턴으로 모든 삼각형 쌍 병렬 처리

#### 4.2 GA Population 병렬 평가 (0.5주) ✅ 완료
- [x] Fitness 평가 병렬화 - `core/ga.rs`
- [x] `GaProblem::evaluate_parallel()` 기본 구현
- [x] Initial population 병렬 평가
- [x] Generation별 children 병렬 평가
- [ ] Island Model GA 구현 (선택적)

> **구현 내용**:
> - `GaProblem` trait에 `evaluate_parallel()` 메서드 추가 (기본값: rayon par_iter)
> - `GaRunner::run_with_rng()`에서 population 평가를 배치로 병렬 처리

#### 4.3 BRKGA Population 병렬 평가 (0.5주) ✅ 완료
- [x] Fitness 평가 병렬화 - `core/brkga.rs`
- [x] `BrkgaProblem::evaluate_parallel()` 기본 구현
- [x] Initial population, mutants, children 병렬 평가

#### 4.4 SA 병렬 재시작 (0.5주) ✅ 완료
- [x] `SaRunner::run_parallel()` 메서드 추가 - `core/sa.rs`
- [x] 여러 SA 인스턴스를 병렬로 실행하여 최적 결과 선택

> **구현 내용**:
> - `run_parallel(num_restarts)` 메서드: 지정된 수의 SA를 병렬 실행
> - 각 실행은 독립적인 RNG 사용
> - 가장 좋은 결과 반환

#### 4.5 Spatial Indexing (1주) ✅ 완료
- [x] `rstar` R*-tree 통합 (2D) - `d2/spatial_index.rs`
- [x] Custom AABB 기반 인덱스 (3D) - `d3/spatial_index.rs`
- [x] Broad-phase collision query API

> **구현 내용**:
> - `SpatialIndex2D`: R*-tree 기반 2D 공간 인덱스
> - `SpatialIndex3D`: AABB 리스트 기반 3D 공간 인덱스
> - 회전 지원 AABB 계산
> - Margin/spacing 지원 충돌 쿼리
> - 향후 solver 통합에서 활용 예정

#### 4.6 Memory Optimization (1주) ❌ 미구현
- [ ] Arena allocation (`bumpalo`) for temporary polygons
- [ ] Geometry instancing (shared vertex data)
- [ ] Zero-copy deserialization (`rkyv`) 평가

#### 4.7 SIMD Optimization (선택적, 0.5주) ❌ 미구현
- [ ] `simba` 기반 벡터 연산
- [ ] Batch point-in-polygon tests

---

## Phase 5: FFI & Integration API (3-4주) 🔄 진행 중

### 목표
C#/Python 소비자를 위한 안정적인 FFI 인터페이스

### 태스크

#### 5.1 C ABI 설계 (1주) ✅ 완료
- [x] `#[no_mangle] extern "C"` 함수 정의 - `ffi/api.rs`
  - [x] `unesting_solve()` - 자동 모드 감지
  - [x] `unesting_solve_2d()` - 2D 전용
  - [x] `unesting_solve_3d()` - 3D 전용
  - [x] `unesting_free_string()` - 메모리 해제
  - [x] `unesting_version()` - 버전 조회
- [x] Error codes 정의 (`UNESTING_OK`, `UNESTING_ERR_*`)
- [x] `cbindgen` 헤더 생성 설정 - `ffi/build.rs`

#### 5.2 JSON API 설계 (1주) ✅ 완료
- [x] Request/Response 구조체 - `ffi/types.rs`
  - [x] `Request2D`, `Request3D`
  - [x] `SolveResponse`
  - [x] `ConfigRequest`
- [x] Serde serialization 구현
- [x] JSON Schema 문서화 - `docs/json-schema/`
  - [x] `request-2d.schema.json` - 2D 요청 스키마
  - [x] `request-3d.schema.json` - 3D 요청 스키마
  - [x] `response.schema.json` - 응답 스키마
- [ ] Version 필드 추가

#### 5.3 Progress Callback (0.5주) ✅ 완료
- [x] `ProgressCallback` type 정의 - `core/solver.rs`
- [x] `ProgressInfo` 구조체 (builder pattern 메서드 포함)
- [x] `solve_with_progress()` 메서드 시그니처
- [x] `GaProgress` 구조체 - `core/ga.rs`
- [x] `GaRunner::run_with_progress()` 메서드
- [x] `BrkgaProgress` 구조체 - `core/brkga.rs`
- [x] `BrkgaRunner::run_with_progress()` 메서드
- [x] `run_ga_nesting_with_progress()` 함수 - `d2/ga_nesting.rs`
- [x] `Nester2D::solve_with_progress()` GA 전략 지원 - `d2/nester.rs`
- [ ] FFI callback function pointer 지원 (향후 개선)

#### 5.4 Python Bindings (1주) ❌ 미구현
- [ ] `PyO3` 기반 바인딩
- [ ] `maturin` 빌드 설정
- [ ] Type stubs (`.pyi`) 생성
- [ ] PyPI 배포 준비

#### 5.5 C# Integration Example (0.5주) 🔄 부분 구현
- [x] P/Invoke 사용 예제 - README.md
- [ ] NuGet 패키지 구조
- [ ] 완전한 사용 예제 프로젝트

---

## Phase 6: Benchmark & Release (2-3주) 🔄 진행 중

### 목표
표준 벤치마크 검증 및 릴리스 준비

### 태스크

#### 6.1 ESICUP Benchmark Suite (1주) ✅ 완료
- [x] 데이터셋 파서 구현 - `benchmark/src/parser.rs`
- [x] Benchmark runner 구축 - `benchmark/src/runner.rs`
- [x] 결과 기록 시스템 - `benchmark/src/result.rs`
- [x] CLI 도구 구현 - `benchmark/src/main.rs` (bench-runner)

**데이터셋** ([ESICUP](https://oscar-oliveira.github.io/2D-Cutting-and-Packing/pages/datset.htm)):
- ALBANO, BLAZ1-3, DIGHE1-2
- FU, JAKOBS1-2, MARQUES
- POLY1-5, SHAPES, SHIRTS, SWIM, TROUSERS

#### 6.2 3D Benchmark (0.5주) ✅ 완료
- [x] Martello-Pisinger-Vigo (MPV) 인스턴스 생성기 - `benchmark/src/dataset3d.rs`
- [x] 9개 인스턴스 클래스 (MPV1-5, BW6-8, Custom)
- [x] 3D Benchmark runner - `benchmark/src/runner3d.rs`
- [x] BenchmarkConfig3D, BenchmarkRunner3D, BenchmarkSummary3D 구현
- [ ] BPPLIB 1D 인스턴스 (검증용) - 1D only이므로 우선순위 낮음

#### 6.3 결과 분석 및 리포트 (0.5주) ✅ 완료
- [x] `Analyzer` 클래스 - `benchmark/src/analyzer.rs`
  - [x] 전체 통계 (OverallStats)
  - [x] 전략별 분석 (StrategyAnalysis)
  - [x] 데이터셋별 분석 (DatasetAnalysis)
  - [x] 전략 비교 매트릭스 (win matrix, improvement matrix)
  - [x] 성능 랭킹 (utilization, speed, consistency, wins)
- [x] `ReportGenerator` - Markdown/JSON 리포트 생성
- [ ] 기존 솔버(SVGnest, libnest2d) 대비 비교 (실제 벤치마크 실행 필요)
- [ ] 성능 그래프 생성 (외부 도구 활용)

#### 6.4 문서화 (0.5주) 🔄 부분 구현
- [x] README.md 기본 문서
- [x] CLAUDE.md (AI 어시스턴트 가이드)
- [x] API 문서 (`cargo doc`) - 모든 크레이트에 모듈 문서 및 사용 예제 추가
- [x] 코드 예제 문서 테스트 통과
- [ ] 사용자 가이드 확장
- [ ] 알고리즘 해설 문서

#### 6.5 릴리스 준비 (0.5주) ❌ 미구현
- [ ] CHANGELOG 작성
- [ ] 버전 태깅 (SemVer)
- [ ] crates.io 배포
- [ ] GitHub Release

---

## 현재 구현 요약

### 완료된 기능 ✅
| 기능 | 위치 | 설명 |
|------|------|------|
| Workspace 구조 | `Cargo.toml` | core, d2, d3, ffi 크레이트 |
| CI/CD | `.github/workflows/` | 테스트, lint, 보안 감사 |
| Geometry2D | `d2/geometry.rs` | 폴리곤, 홀, 면적, convex hull |
| Geometry3D | `d3/geometry.rs` | Box, 6방향 회전, mass |
| Boundary2D | `d2/boundary.rs` | 직사각형, 폴리곤 경계 |
| Boundary3D | `d3/boundary.rs` | Box 컨테이너, mass 제한 |
| Nester2D (BLF) | `d2/nester.rs` | Row-based BLF 배치 |
| Nester2D (NFP-guided) | `d2/nester.rs` | NFP 기반 최적 배치 |
| Nester2D (GA) | `d2/nester.rs`, `d2/ga_nesting.rs` | GA 기반 최적화 |
| Packer3D (Layer) | `d3/packer.rs` | Layer-based 배치 |
| Packer3D (GA) | `d3/packer.rs`, `d3/ga_packing.rs` | GA 기반 최적화 |
| GA Framework | `core/ga.rs` | Individual, GaProblem, GaRunner |
| BRKGA Framework | `core/brkga.rs` | RandomKeyChromosome, BrkgaProblem, BrkgaRunner |
| Nester2D (BRKGA) | `d2/brkga_nesting.rs` | BRKGA 기반 2D nesting |
| Packer3D (BRKGA) | `d3/brkga_packing.rs` | BRKGA 기반 3D packing |
| SA Framework | `core/sa.rs` | SaConfig, SaProblem, SaRunner |
| Nester2D (SA) | `d2/sa_nesting.rs` | SA 기반 2D nesting |
| Packer3D (SA) | `d3/sa_packing.rs` | SA 기반 3D packing |
| Packer3D (EP) | `d3/extreme_point.rs` | Extreme Point heuristic 3D packing |
| FFI JSON API | `ffi/api.rs` | C ABI, JSON 요청/응답 |
| NFP Convex | `d2/nfp.rs` | Minkowski sum 기반 NFP 계산 |
| NFP Non-convex | `d2/nfp.rs` | Triangulation + i_overlay union 방식 |
| NFP Cache | `d2/nfp.rs` | Thread-safe 캐싱 시스템 |
| IFP | `d2/nfp.rs` | Inner-Fit Polygon 계산 |
| IFP with Margin | `d2/nfp.rs` | Margin 적용 가능한 IFP 계산 |
| ESICUP Parser | `benchmark/src/parser.rs` | ESICUP JSON 데이터셋 파서 |
| Benchmark Runner | `benchmark/src/runner.rs` | 다중 전략 벤치마크 실행 |
| Result Recording | `benchmark/src/result.rs` | JSON/CSV 결과 기록 |
| Benchmark CLI | `benchmark/src/main.rs` | bench-runner CLI 도구 |
| NFP 병렬 계산 | `d2/nfp.rs` | rayon 기반 pairwise Minkowski sum 병렬화 |
| GA 병렬 평가 | `core/ga.rs` | Population fitness 병렬 평가 |
| BRKGA 병렬 평가 | `core/brkga.rs` | Population fitness 병렬 평가 |
| SA 병렬 재시작 | `core/sa.rs` | 다중 SA 인스턴스 병렬 실행 |
| Spatial Index 2D | `d2/spatial_index.rs` | R*-tree 기반 2D 공간 인덱스 |
| Spatial Index 3D | `d3/spatial_index.rs` | AABB 기반 3D 공간 인덱스 |
| GA Progress Callback | `core/ga.rs` | GaProgress 구조체, run_with_progress() 메서드 |
| BRKGA Progress Callback | `core/brkga.rs` | BrkgaProgress 구조체, run_with_progress() 메서드 |
| ProgressInfo Builder | `core/solver.rs` | Builder pattern 메서드로 확장된 ProgressInfo |
| MPV Instance Generator | `benchmark/src/dataset3d.rs` | 3D 벤치마크 인스턴스 생성기 (MPV1-5, BW6-8) |
| 3D Benchmark Runner | `benchmark/src/runner3d.rs` | 3D 벤치마크 실행기 |
| 3D Dataset Types | `benchmark/src/dataset3d.rs` | Dataset3D, Item3D, InstanceClass 타입 |
| API Documentation | `*/src/lib.rs` | 모든 크레이트에 모듈 문서 및 사용 예제 추가 |
| Benchmark Analyzer | `benchmark/src/analyzer.rs` | 벤치마크 결과 분석 및 리포트 생성 |
| Analysis Report | `benchmark/src/analyzer.rs` | 전략별/데이터셋별 분석, 랭킹, 비교 매트릭스 |
| JSON Schema | `docs/json-schema/` | 2D/3D 요청 및 응답 스키마 |

### 미구현 핵심 기능 ❌
| 기능 | 우선순위 | 설명 |
|------|----------|------|
| ~~NFP 계산 (non-convex 정밀)~~ | ~~**중간**~~ | ~~i_overlay 통합~~ ✅ 완료 |
| ~~NFP-guided BLF~~ | ~~**높음**~~ | ~~NFP 기반 최적 배치점 탐색~~ ✅ 완료 |
| ~~GA-based Nesting~~ | ~~**중간**~~ | ~~GA + BLF/NFP decoder~~ ✅ 완료 |
| ~~Extreme Point (3D)~~ | ~~**중간**~~ | ~~EP heuristic for bin packing~~ ✅ 완료 |
| ~~병렬 처리~~ | ~~**중간**~~ | ~~rayon 기반 NFP/GA 병렬화~~ ✅ 완료 |
| ~~Spatial Indexing~~ | ~~**중간**~~ | ~~R*-tree/AABB 통합~~ ✅ 완료 |
| Python Bindings | **낮음** | PyO3/maturin |

---

## 우선순위 권장사항

### 다음 단계 (권장 순서)

1. ~~**Non-convex NFP 정밀 구현** (Phase 2.2)~~ ✅ 완료
   - Triangulation + i_overlay union 방식으로 구현 완료

2. ~~**IFP Margin 적용** (Phase 2.3)~~ ✅ 완료
   - `compute_ifp_with_margin()` 함수 추가 완료

3. ~~**벤치마크 설정** (Phase 6.1)~~ ✅ 완료
   - ESICUP 데이터셋 파서 구현
   - Benchmark runner 및 CLI 도구 구현
   - JSON/CSV 결과 기록 시스템 구현

4. ~~**병렬 처리** (Phase 4)~~ ✅ 완료
   - rayon 기반 NFP/GA/BRKGA/SA 병렬화 완료

5. ~~**Spatial Indexing** (Phase 4.5)~~ ✅ 완료
   - R*-tree 기반 2D 공간 인덱스 구현
   - AABB 기반 3D 공간 인덱스 구현
   - 향후 solver에 통합하여 broad-phase collision culling 적용 예정

6. ~~**3D 벤치마크** (Phase 6.2)~~ ✅ 완료
   - MPV 인스턴스 생성기 구현
   - 3D 벤치마크 러너 구현

7. **Memory Optimization** (Phase 4.6)
   - Arena allocation
   - Geometry instancing

---

## 리스크 및 완화 전략

| 리스크 | 영향 | 확률 | 완화 전략 |
|--------|------|------|-----------|
| NFP 수치 불안정 | High | Medium | `robust` crate 사용, 정수 좌표 스케일링 |
| GA 수렴 부족 | Medium | Medium | Adaptive parameter tuning, Island model |
| 3D 성능 병목 | Medium | High | BVH 최적화, LOD 적용 |
| FFI 메모리 누수 | High | Low | Valgrind/Miri 테스트, RAII 패턴 |

---

## 참조 링크 종합

### 핵심 논문
1. [Burke et al. (2007) - Complete NFP Generation](https://www.graham-kendall.com/papers/bhkw2007.pdf)
2. [Bennell & Oliveira (2008) - Nesting Tutorial](https://eprints.soton.ac.uk/154797/)
3. [Gonçalves & Resende (2013) - BRKGA](https://www.semanticscholar.org/paper/A-biased-random-key-genetic-algorithm-for-2D-and-Goncalves-Resende)

### Rust 생태계
4. [geo crate](https://docs.rs/geo)
5. [i_overlay](https://github.com/iShape-Rust/iOverlay)
6. [parry](https://parry.rs/docs/)
7. [rstar](https://docs.rs/rstar)

### 벤치마크
8. [ESICUP Datasets](https://oscar-oliveira.github.io/2D-Cutting-and-Packing/pages/datset.htm)
9. [BPPLIB](https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library)

### 기존 구현
10. [SVGnest](https://github.com/Jack000/SVGnest)
11. [libnest2d](https://github.com/tamasmeszaros/libnest2d)
12. [OR-Tools](https://developers.google.com/optimization)

---

이 로드맵은 리서치 문서의 권장사항을 기반으로 구성되었으며, 각 Phase는 이전 단계의 완료에 의존합니다. 필요에 따라 Phase 간 병렬 진행이 가능한 태스크도 있습니다.
