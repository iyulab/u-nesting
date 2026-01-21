# U-Nesting 개발 로드맵

리서치 문서를 기반으로 상세한 다단계 로드맵을 구성했습니다.

> **마지막 업데이트**: 2026-01-21
> **현재 진행 단계**: Phase 0.4 완료 (시나리오 실행) - Phase 0.5 대기

---

## 전체 타임라인 개요

| Phase | 기간 | 핵심 목표 | 상태 |
|-------|------|----------|------|
| **Phase 0** | 2-3주 | 품질 검증 및 결함 도출 | 🔥 **최우선** |
| **Phase 1** | 5-6주 | Geometry Core (2D/3D 기초) | ✅ 완료 |
| **Phase 2** | 4-5주 | NFP 엔진 및 배치 알고리즘 | ✅ 완료 |
| **Phase 3** | 5-6주 | 최적화 알고리즘 (GA/SA) | ✅ 완료 |
| **Phase 4** | 3-4주 | 성능 최적화 및 병렬화 | ✅ 완료 |
| **Phase 5** | 3-4주 | FFI 및 통합 API | ✅ 완료 (98%) |
| **Phase 6** | 2-3주 | 벤치마크 및 릴리스 준비 | 🔄 릴리스 대기 (95%) |
| **Phase 7** | 5-6주 | 배포 확장 및 문서화 | ⬜ 대기 |

**총 예상 기간: 29-37주**

---

## Phase 0: 품질 검증 및 결함 도출 (2-3주) 🔥 최우선

### 목표
실제 벤치마크 데이터셋을 활용한 포괄적인 품질 검증 및 결함/개선사항 도출

### 배경
- v0.1.0 릴리스 전 실제 데이터셋 기반 성능 검증 필수
- Dogfooding 원칙에 따른 자체 결함 발견 및 개선
- 학계 표준 벤치마크 대비 품질 수준 파악

---

### Phase 0.1: 벤치마크 데이터셋 확보 (3일) ✅ 완료

#### 2D Nesting 데이터셋

##### 0.1.1 ESICUP 표준 데이터셋 ✅ 완료
**소스**: [ESICUP/datasets](https://github.com/ESICUP/datasets)

| 데이터셋 | 아이템 수 | 특성 | 우선순위 |
|----------|----------|------|----------|
| ALBANO | 24 | 볼록/비볼록 혼합 | **높음** |
| BLAZ1-3 | 7-28 | 단순 폴리곤 | **높음** |
| DAGLI | 30 | 산업용 패턴 | 중간 |
| FU | 12 | 복잡한 형상 | **높음** |
| JAKOBS1-2 | 25 | 클래식 벤치마크 | **높음** |
| MAO | 20 | 다양한 크기 | 중간 |
| MARQUES | 24 | 홀 포함 | **높음** |
| SHAPES | 43 | 다양한 기하학 | **높음** |
| SHIRTS | 99 | 대규모 산업용 | **높음** |
| SWIM | 48 | 곡선 근사 | 중간 |
| TROUSERS | 64 | 산업용 의류 패턴 | **높음** |

**태스크**:
- [x] ESICUP GitHub 데이터셋 일괄 다운로드 스크립트 작성 (`download.rs`, `DatasetManager`)
- [x] `datasets/2d/esicup/` 디렉토리 구성
- [x] 각 데이터셋의 best-known solution 값 수집 (`ESICUP_DATASETS` 상수)

**다운로드 결과**: 13/16 성공 (ALBANO, BLAZ1, DAGLI, FU, JAKOBS1-2, MAO, MARQUES, SHAPES0-1, SHIRTS, SWIM, TROUSERS)

##### 0.1.2 seanys 전처리 데이터셋 ⬜
**소스**: [seanys/2D-Irregular-Packing-Algorithm](https://github.com/seanys/2D-Irregular-Packing-Algorithm)

**태스크**:
- [ ] CSV 형식 데이터셋 다운로드
- [ ] CSV → JSON 변환기 구현 (`benchmark/src/csv_parser.rs`)
- [ ] `datasets/2d/seanys/` 디렉토리 구성

##### 0.1.3 Jigsaw Puzzle 인스턴스 ⬜
**소스**: López-Camacho et al. (2013)

| 세트 | 인스턴스 수 | 특성 |
|------|------------|------|
| JP1 | 540 | 볼록 도형, 100% 최적해 |
| JP2 | 480 | 볼록/비볼록 혼합 |

**태스크**:
- [ ] JP1/JP2 데이터셋 확보 (웹 검색 또는 논문 부록)
- [ ] 파서 구현 (`benchmark/src/jigsaw_parser.rs`)
- [ ] `datasets/2d/jigsaw/` 디렉토리 구성

##### 0.1.4 합성 테스트 데이터셋 생성 ✅ 완료
**목적**: Edge case 및 스트레스 테스트용

| 카테고리 | 설명 |
|----------|------|
| `convex_only` | 순수 볼록 폴리곤 (삼각형, 사각형, 오각형 등) |
| `concave_complex` | 복잡한 비볼록 폴리곤 |
| `with_holes` | 구멍이 있는 폴리곤 |
| `extreme_aspect` | 극단적 종횡비 (매우 길거나 좁은) |
| `tiny_items` | 매우 작은 아이템 (정밀도 테스트) |
| `large_count` | 1000+ 아이템 (스케일 테스트) |
| `near_collinear` | 거의 직선인 엣지 (수치 안정성) |
| `self_touching` | 자기 접촉 폴리곤 |

**태스크**:
- [x] 합성 데이터 생성기 구현 (`benchmark/src/synthetic.rs`)
- [x] 각 카테고리별 10-50개 인스턴스 생성
- [x] `datasets/2d/synthetic/` 디렉토리 구성

**생성 결과**: 8개 합성 데이터셋 (convex, concave, with_holes, extreme_aspect, tiny, large, near_collinear, jigsaw)

#### 3D Bin Packing 데이터셋

##### 0.1.5 MPV 인스턴스 (기존 구현 활용) ✅
**상태**: `benchmark/src/dataset3d.rs`에 생성기 구현 완료

- [ ] MPV1-5, BW6-8 인스턴스 생성 및 저장
- [ ] `datasets/3d/mpv/` 디렉토리 구성

##### 0.1.6 BPPLIB 데이터셋 ⬜
**소스**: [BPPLIB](https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library)

**태스크**:
- [ ] BPPLIB 3D 인스턴스 다운로드
- [ ] 파서 구현 또는 확장
- [ ] `datasets/3d/bpplib/` 디렉토리 구성

##### 0.1.7 BED-BPP 실제 주문 데이터 ⬜
**소스**: 학술 논문 (Alibaba Cloud 기반)

**태스크**:
- [ ] 공개 데이터셋 확보 가능 여부 조사
- [ ] 가능시 파서 구현

##### 0.1.8 합성 3D 테스트 데이터셋 ⬜

| 카테고리 | 설명 |
|----------|------|
| `uniform_cubes` | 동일 크기 정육면체 |
| `varied_boxes` | 다양한 크기 직육면체 |
| `extreme_ratios` | 극단적 종횡비 (판재, 막대) |
| `heavy_items` | 무게 제약 테스트 |
| `orientation_restricted` | 회전 제한 아이템 |

**태스크**:
- [ ] 3D 합성 데이터 생성기 확장
- [ ] `datasets/3d/synthetic/` 디렉토리 구성

---

### Phase 0.2: 테스트 시나리오 정의 (2일) ✅ 완료

**구현 완료**: `benchmark/src/scenario.rs`, `benchmark/src/scenario_runner.rs`

#### 2D Nesting 시나리오

| ID | 시나리오 | 목적 | 데이터셋 |
|----|----------|------|----------|
| 2D-S01 | **기본 기능 검증** | 모든 전략이 올바르게 동작하는지 확인 | SHAPES, BLAZ |
| 2D-S02 | **볼록 폴리곤 최적화** | NFP 정확성 및 배치 품질 | convex_only |
| 2D-S03 | **비볼록 폴리곤 처리** | Triangulation + NFP union 정확성 | FU, concave_complex |
| 2D-S04 | **홀 처리** | 구멍 있는 폴리곤 처리 | MARQUES, with_holes |
| 2D-S05 | **대규모 인스턴스** | 스케일 성능 및 메모리 | SHIRTS, large_count |
| 2D-S06 | **회전 최적화** | 다중 회전 각도 효과 | JAKOBS, TROUSERS |
| 2D-S07 | **수치 안정성** | Edge case 처리 | near_collinear, tiny_items |
| 2D-S08 | **전략 비교** | BLF vs NFP vs GA vs BRKGA vs SA | 전체 ESICUP |
| 2D-S09 | **100% 최적해 검증** | 알려진 최적해 달성 가능 여부 | JP1 (jigsaw) |
| 2D-S10 | **시간 제약 성능** | 제한 시간 내 최대 품질 | ALBANO, DAGLI |

#### 3D Bin Packing 시나리오

| ID | 시나리오 | 목적 | 데이터셋 |
|----|----------|------|----------|
| 3D-S01 | **기본 기능 검증** | Layer/EP 전략 동작 확인 | MPV1-3 |
| 3D-S02 | **다양한 크기 처리** | 크기 변동이 큰 아이템 | MPV4-5, varied_boxes |
| 3D-S03 | **회전 최적화** | 6방향 회전 효과 | BW6-8 |
| 3D-S04 | **무게 제약** | mass constraint 처리 | heavy_items |
| 3D-S05 | **Extreme Point** | EP 전략 품질 | 전체 MPV |
| 3D-S06 | **GA/BRKGA/SA 비교** | 최적화 전략 비교 | MPV1-5 |
| 3D-S07 | **대규모 인스턴스** | 100+ 아이템 처리 | large_count_3d |

#### 공통 시나리오

| ID | 시나리오 | 목적 |
|----|----------|------|
| C-S01 | **FFI 통합 테스트** | JSON API 경계 조건 |
| C-S02 | **취소 기능** | Cancellation token 동작 |
| C-S03 | **진행 콜백** | Progress callback 정확성 |
| C-S04 | **메모리 사용량** | 대규모 인스턴스 메모리 프로파일링 |
| C-S05 | **병렬 성능** | 멀티스레드 스케일링 |

---

### Phase 0.3: 테스트 인프라 구축 (3일) ✅ 완료

#### 0.3.1 자동화 테스트 러너 확장 ✅
**태스크**:
- [x] 시나리오 기반 테스트 러너 (`benchmark/src/scenario_runner.rs`)
- [x] TOML 기반 시나리오 정의 (`benchmark/src/scenario.rs`)
- [ ] 병렬 테스트 실행 지원 (rayon) - 향후 개선
- [ ] 중간 결과 저장 및 재개 기능 - 향후 개선

**CLI 명령어 추가**:
- `bench-runner list-scenarios` - 시나리오 목록 조회
- `bench-runner run-scenarios --category 2d` - 카테고리별 실행
- `bench-runner run-scenarios --scenario-id 2D-S01` - 개별 실행

#### 0.3.2 결과 수집 및 분석 확장 ⬜
**태스크**:
- [ ] 상세 메트릭 수집
  - Utilization (%)
  - Strip length / Volume used
  - Computation time (ms)
  - Memory peak (MB)
  - NFP cache hit rate
  - GA/SA convergence history
- [ ] Best-known 대비 gap 계산
- [ ] 통계 분석 (평균, 표준편차, 최소/최대)

#### 0.3.3 리포트 생성 확장 ⬜
**태스크**:
- [ ] Markdown 리포트 템플릿
- [ ] 전략별/데이터셋별 히트맵
- [ ] 수렴 그래프 (GA/SA)
- [ ] 이슈 자동 생성 템플릿

#### 0.3.4 시각화 도구 ⬜
**태스크**:
- [ ] SVG 결과 출력 (`benchmark/src/visualizer.rs`)
- [ ] 배치 결과 이미지 생성
- [ ] 3D 결과 OBJ/STL 출력

---

### Phase 0.4: 테스트 실행 및 결함 도출 (1주) 🔄 진행 중

#### 0.4.1 전체 시나리오 실행 🔄
**태스크**:
- [x] 2D 시나리오 일부 실행 (2D-S02, 2D-S07)
- [ ] 2D 시나리오 전체 (2D-S01 ~ 2D-S10) 실행
- [ ] 3D 시나리오 (3D-S01 ~ 3D-S07) 실행 - MPV 데이터셋 미지원
- [x] 공통 시나리오 일부 실행 (C-S01)
- [x] 결과 JSON/CSV 저장 (`benchmark/results/`)

**실행 결과 요약**:
- 2D-S02 (Convex Optimization): PASSED, 1 defect (BLF > NFP quality)
- 2D-S07 (Numerical Stability): PASSED, 2 defects (BLF > NFP quality)
- C-S01 (FFI Integration): PASSED, 0 defects

#### 0.4.2 결과 분석 ✅
**분석 항목**:
- [x] 전략별 Utilization 평균/편차 - ScenarioRunner에서 자동 수집
- [x] Best-known 대비 gap 분석 - RunResult.gap_percent로 계산
- [x] 실패 케이스 분류 - Defect 구조체로 분류
- [ ] 성능 병목 식별 - 향후 프로파일링 필요
- [ ] 메모리 누수 검사 - 향후 valgrind/miri 필요

#### 0.4.3 결함/개선사항 도출 ✅
**발견된 결함**:

| ID | 심각도 | 카테고리 | 제목 |
|----|--------|----------|------|
| ISSUE-20260121-nfp-quality-anomaly | **P1** | Quality | NFP-guided가 BLF보다 긴 strip length 생성 |

**도출 카테고리**:

| 카테고리 | 설명 | 이슈 라벨 |
|----------|------|----------|
| **버그** | 잘못된 결과, 크래시, 무한 루프 | `bug` |
| **정확성** | NFP 계산 오류, 충돌 감지 실패 | `accuracy` |
| **성능** | 느린 속도, 높은 메모리 사용 | `performance` |
| **품질** | 낮은 utilization, 최적해 미달 | `quality` |
| **API** | 불편한 인터페이스, 누락된 기능 | `api` |
| **문서** | 부족한 설명, 예제 필요 | `docs` |

**태스크**:
- [x] 각 결함/개선사항별 GitHub 이슈 초안 작성
- [x] `claudedocs/issues/` 디렉토리에 저장
- [x] 우선순위 및 심각도 분류

---

### Phase 0.5: 개선 계획 수립 (2일)

#### 0.5.1 결함 우선순위 결정 ⬜
**기준**:
- **P0 (Critical)**: 크래시, 데이터 손상, 보안 문제
- **P1 (High)**: 잘못된 결과, 심각한 성능 저하
- **P2 (Medium)**: 품질 저하, 사용성 문제
- **P3 (Low)**: 마이너 개선, 코드 정리

#### 0.5.2 개선 로드맵 업데이트 ⬜
**태스크**:
- [ ] P0/P1 결함을 Phase 6.5에 추가 (릴리스 전 수정)
- [ ] P2 결함을 Phase 7에 추가
- [ ] P3 결함을 Backlog에 추가

#### 0.5.3 회귀 테스트 추가 ⬜
**태스크**:
- [ ] 발견된 버그에 대한 단위 테스트 추가
- [ ] CI에 벤치마크 회귀 테스트 통합

---

### Phase 0 산출물

| 산출물 | 위치 | 상태 |
|--------|------|------|
| 데이터셋 | `datasets/2d/`, `datasets/3d/` | ✅ |
| 시나리오 정의 | `benchmark/src/scenario.rs` | ✅ |
| 테스트 결과 | `benchmark/results/` | ✅ |
| 분석 리포트 | `benchmark/results/scenario_report.md` | ✅ |
| 이슈 초안 | `claudedocs/issues/` | ✅ |
| 시각화 결과 | `benchmark/visualizations/` | ⬜ 향후 |

### Phase 0 완료 기준

- [ ] 최소 15개 ESICUP 데이터셋 테스트 완료
- [ ] 모든 전략(BLF, NFP, GA, BRKGA, SA)에 대한 벤치마크
- [ ] Best-known 대비 gap < 15% (GA/BRKGA/SA 전략)
- [ ] 모든 발견된 P0/P1 결함 이슈 생성
- [ ] 품질 검증 리포트 작성 완료

---

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

## Phase 4: Performance Optimization (3-4주) ✅ 완료

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

#### 4.6 Memory Optimization (1주) ✅ 완료
- [x] `ObjectPool<T>` - 재사용 가능한 객체 풀 - `core/memory.rs`
- [x] `ClearingPool<T>` - 자동 초기화 객체 풀
- [x] `SharedGeometry<V>` - Geometry instancing (shared vertex data)
- [x] `GeometryCache<V>` - 지오메트리 캐시/중복 제거
- [x] `ScratchBuffer<T>` - Thread-local 임시 버퍼
- [x] `MemoryStats` - 메모리 사용량 모니터링

> **구현 내용**:
> - `Clearable` trait으로 재사용 가능한 객체 정의
> - `ObjectPool` 및 `ClearingPool`으로 반복 할당 최소화
> - `SharedGeometry`와 `GeometryCache`로 공유 정점 데이터 관리
> - `ScratchBuffer`로 thread-local 임시 저장소 제공

#### 4.7 SIMD Optimization (선택적, 0.5주) ❌ 미구현
- [ ] `simba` 기반 벡터 연산
- [ ] Batch point-in-polygon tests

> **Note**: SIMD 최적화는 성능 프로파일링 후 필요시 구현 예정

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
- [x] Version 필드 추가 - `SolveResponse`에 API 버전 포함

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

#### 5.4 Python Bindings (1주) ✅ 완료
- [x] `PyO3` 기반 바인딩 - `python/src/lib.rs`
- [x] `maturin` 빌드 설정 - `python/pyproject.toml`
- [x] Type stubs (`.pyi`) 생성 - `python/python/u_nesting/__init__.pyi`
- [x] Python 패키지 구조 - `python/python/u_nesting/__init__.py`
- [ ] PyPI 배포 준비 (향후)

> **구현 내용**:
> - `solve_2d()`, `solve_3d()` 함수로 Python에서 직접 호출 가능
> - `version()`, `available_strategies()` 유틸리티 함수
> - TypedDict 기반 type stubs로 IDE 자동완성 지원
> - JSON 기반 데이터 변환으로 Python 딕셔너리 직접 사용 가능

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

#### 6.5 릴리스 준비 (0.5주) 🔄 진행 중
- [x] CHANGELOG 작성 - `CHANGELOG.md`
- [x] 버전 설정 (SemVer) - workspace version 0.1.0, 내부 크레이트 버전 의존성 추가
- [x] crates.io 배포 준비 - `cargo publish --dry-run` 검증 완료 (core)
- [ ] crates.io 실제 배포 (순서: core → d2 → d3 → ffi)
- [ ] GitHub Release 태그 생성 (v0.1.0)

---

## Phase 7: 배포 확장 및 문서화 (5-6주) ⬜ 대기

### 목표
다양한 언어 생태계 배포 및 종합 문서 제공

---

### Phase 7.1: FFI Callback Function Pointer 지원 (1주)

#### 목표
C/C# 소비자가 실시간 진행 상태를 받을 수 있는 콜백 메커니즘 제공

#### 태스크

##### 7.1.1 C ABI 콜백 타입 정의 (1일)
- [ ] `typedef void (*UnestingProgressCallback)(const char* progress_json)`
- [ ] `ProgressCallbackContext` opaque 핸들 정의

##### 7.1.2 FFI API 확장 (2일) - 의존: 7.1.1
- [ ] `unesting_solve_2d_with_progress(request, callback, context, result)`
- [ ] `unesting_solve_3d_with_progress(request, callback, context, result)`
- [ ] 콜백 호출 주기 설정 파라미터 추가

##### 7.1.3 Thread-safe 콜백 래퍼 구현 (1일) - 의존: 7.1.2
- [ ] unsafe extern "C" 콜백을 Rust closure로 변환
- [ ] Panic guard 적용 (FFI boundary)

##### 7.1.4 cbindgen 헤더 업데이트 (0.5일) - 의존: 7.1.3
- [ ] `unesting.h`에 콜백 타입 및 함수 추가

##### 7.1.5 C 사용 예제 작성 (0.5일) - 의존: 7.1.4
- [ ] `examples/c/progress_callback.c`

#### 산출물
- `ffi/api.rs`: `_with_progress` 함수 추가
- `ffi/callback.rs`: 콜백 타입 및 래퍼 (신규)
- `include/unesting.h`: 콜백 타입 포함 헤더
- `examples/c/`: C 예제 코드

---

### Phase 7.2: PyPI 배포 (1주)

#### 목표
`pip install u-nesting`으로 설치 가능한 Python 패키지 배포

#### 태스크

##### 7.2.1 maturin 빌드 검증 (0.5일)
- [ ] Linux/macOS/Windows 크로스 컴파일 테스트
- [ ] `maturin build --release` 검증

##### 7.2.2 CI/CD 워크플로우 구성 (1일) - 의존: 7.2.1
- [ ] `.github/workflows/python-publish.yml` 생성
- [ ] maturin-action 설정 (manylinux, musllinux, macOS, Windows)
- [ ] 태그 기반 자동 배포 트리거

##### 7.2.3 PyPI 계정 및 토큰 설정 (0.5일)
- [ ] PyPI API 토큰 발급
- [ ] GitHub Secrets에 `PYPI_API_TOKEN` 등록

##### 7.2.4 TestPyPI 배포 테스트 (1일) - 의존: 7.2.2, 7.2.3
- [ ] TestPyPI에 먼저 배포
- [ ] `pip install --index-url https://test.pypi.org/simple/ u-nesting` 검증

##### 7.2.5 Python README 작성 (0.5일)
- [ ] `crates/python/README.md` (PyPI 페이지용)
- [ ] 설치 가이드, 빠른 시작, 예제 코드

##### 7.2.6 PyPI 정식 배포 (0.5일) - 의존: 7.2.4, 7.2.5
- [ ] 태그 생성 → 자동 배포
- [ ] PyPI 페이지 확인

#### 산출물
- `.github/workflows/python-publish.yml`
- `crates/python/README.md`
- PyPI 패키지: `u-nesting`

---

### Phase 7.3: C# NuGet 패키지 (1.5주)

#### 목표
.NET 개발자를 위한 NuGet 패키지 배포

#### 태스크

##### 7.3.1 C# 프로젝트 구조 생성 (0.5일)
- [ ] `bindings/csharp/UNesting/UNesting.csproj`
- [ ] `bindings/csharp/UNesting.Tests/`

##### 7.3.2 P/Invoke 래퍼 클래스 구현 (2일) - 의존: 7.3.1
- [ ] `NativeLibrary.cs`: DLL import 선언
- [ ] `Nester2D.cs`: 2D nesting API
- [ ] `Packer3D.cs`: 3D packing API
- [ ] `ProgressCallback.cs`: 콜백 델리게이트 (7.1 완료 후)

##### 7.3.3 JSON 직렬화 모델 (1일) - 의존: 7.3.2
- [ ] `Models/Request2D.cs`, `Response.cs` 등
- [ ] `System.Text.Json` 또는 `Newtonsoft.Json` 사용

##### 7.3.4 네이티브 라이브러리 번들링 (1일) - 의존: 7.3.2
- [ ] `runtimes/win-x64/native/unesting.dll`
- [ ] `runtimes/linux-x64/native/libunesting.so`
- [ ] `runtimes/osx-x64/native/libunesting.dylib`
- [ ] `.nuspec` 또는 `.csproj` 번들 설정

##### 7.3.5 단위 테스트 (0.5일) - 의존: 7.3.3
- [ ] xUnit 기반 테스트
- [ ] 2D/3D 기본 시나리오 검증

##### 7.3.6 NuGet 패키지 구성 (0.5일) - 의존: 7.3.4, 7.3.5
- [ ] `UNesting.nuspec` 메타데이터
- [ ] `dotnet pack` 검증

##### 7.3.7 CI/CD 워크플로우 (0.5일) - 의존: 7.3.6
- [ ] `.github/workflows/nuget-publish.yml`
- [ ] 태그 기반 NuGet.org 배포

##### 7.3.8 NuGet.org 배포 (0.5일) - 의존: 7.3.7
- [ ] API 키 설정
- [ ] 정식 배포

#### 산출물
- `bindings/csharp/UNesting/` C# 프로젝트
- `bindings/csharp/UNesting.Tests/` 테스트 프로젝트
- `.github/workflows/nuget-publish.yml`
- NuGet 패키지: `UNesting`

---

### Phase 7.4: 사용자 가이드 및 알고리즘 해설 문서 (1.5주)

#### 목표
개발자와 연구자를 위한 종합 문서 제공

#### 태스크

##### 7.4.1 문서 사이트 구조 설계 (0.5일)
- [ ] mdBook 또는 Docusaurus 선택
- [ ] `docs/book/` 디렉토리 구조

##### 7.4.2 시작 가이드 (1일) - 의존: 7.4.1
- [ ] 설치 방법 (Rust/Python/C#/C)
- [ ] 빠른 시작 예제
- [ ] 기본 개념 설명

##### 7.4.3 API 사용 가이드 (1일) - 의존: 7.4.2
- [ ] 2D Nesting 가이드 (입력 형식, 옵션, 출력 해석)
- [ ] 3D Packing 가이드
- [ ] 전략 선택 가이드 (BLF vs NFP vs GA vs BRKGA vs SA)
- [ ] 성능 튜닝 팁

##### 7.4.4 알고리즘 해설 (2일)
- [ ] NFP (No-Fit Polygon) 개념 및 계산 방법
- [ ] Bottom-Left Fill 알고리즘
- [ ] Genetic Algorithm 구조 및 파라미터
- [ ] BRKGA 특징 및 장점
- [ ] Simulated Annealing 쿨링 스케줄
- [ ] Extreme Point Heuristic (3D)

##### 7.4.5 아키텍처 문서 (0.5일)
- [ ] 크레이트 구조 다이어그램
- [ ] 핵심 trait/struct 관계
- [ ] 데이터 흐름

##### 7.4.6 기여 가이드 (0.5일)
- [ ] `CONTRIBUTING.md`
- [ ] 코드 스타일 가이드
- [ ] PR 프로세스

##### 7.4.7 문서 사이트 배포 (0.5일) - 의존: 7.4.1~7.4.6
- [ ] GitHub Pages 설정
- [ ] 자동 빌드 워크플로우

#### 산출물
- `docs/book/`: mdBook 소스
- `docs/algorithms/`: 알고리즘 해설 (그림 포함)
- `CONTRIBUTING.md`
- GitHub Pages 문서 사이트

---

### Phase 7 요약

| Sub-Phase | 기간 | 핵심 산출물 |
|-----------|------|-------------|
| 7.1 FFI Callback | 1주 | `_with_progress` API, C 예제 |
| 7.2 PyPI 배포 | 1주 | PyPI 패키지, CI/CD |
| 7.3 C# NuGet | 1.5주 | NuGet 패키지, P/Invoke 래퍼 |
| 7.4 문서 확장 | 1.5주 | 문서 사이트, 알고리즘 해설 |

**총 예상 기간: 5-6주**

### 의존성 그래프

```
Phase 7.1 (FFI Callback)
    ↓
Phase 7.3 (C# NuGet) ← 콜백 델리게이트 지원 시 의존

Phase 7.2 (PyPI) ← 독립적, 바로 시작 가능

Phase 7.4 (문서) ← 독립적, 병렬 진행 가능
```

### 권장 실행 순서

1. **Phase 7.2 (PyPI)** - 이미 Python 바인딩 완료, 즉시 배포 가능
2. **Phase 7.1 (FFI Callback)** - C# 통합 전 선행 필요
3. **Phase 7.3 (C# NuGet)** - FFI Callback 완료 후
4. **Phase 7.4 (문서)** - 전 기간 병렬 진행 가능

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
| Memory Optimization | `core/memory.rs` | ObjectPool, GeometryCache, ScratchBuffer |
| Python Bindings | `python/src/lib.rs` | PyO3 기반 Python 바인딩 |
| Python Type Stubs | `python/python/u_nesting/__init__.pyi` | TypedDict 기반 타입 힌트 |

### 미구현 핵심 기능 ❌
| 기능 | 우선순위 | 설명 |
|------|----------|------|
| ~~NFP 계산 (non-convex 정밀)~~ | ~~**중간**~~ | ~~i_overlay 통합~~ ✅ 완료 |
| ~~NFP-guided BLF~~ | ~~**높음**~~ | ~~NFP 기반 최적 배치점 탐색~~ ✅ 완료 |
| ~~GA-based Nesting~~ | ~~**중간**~~ | ~~GA + BLF/NFP decoder~~ ✅ 완료 |
| ~~Extreme Point (3D)~~ | ~~**중간**~~ | ~~EP heuristic for bin packing~~ ✅ 완료 |
| ~~병렬 처리~~ | ~~**중간**~~ | ~~rayon 기반 NFP/GA 병렬화~~ ✅ 완료 |
| ~~Spatial Indexing~~ | ~~**중간**~~ | ~~R*-tree/AABB 통합~~ ✅ 완료 |
| ~~Python Bindings~~ | ~~**낮음**~~ | ~~PyO3/maturin~~ ✅ 완료 |

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

7. ~~**Memory Optimization** (Phase 4.6)~~ ✅ 완료
   - ObjectPool, ClearingPool for reusable allocations
   - SharedGeometry, GeometryCache for geometry instancing
   - ScratchBuffer for thread-local temporary storage

8. ~~**Python Bindings** (Phase 5.4)~~ ✅ 완료
   - PyO3 기반 Python 바인딩 구현
   - maturin 빌드 설정
   - Type stubs 생성

9. **릴리스 준비** (Phase 6.5)
   - CHANGELOG 작성
   - 버전 태깅
   - crates.io / PyPI 배포

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
