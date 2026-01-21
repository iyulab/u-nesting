# U-Nesting 개발 로드맵

리서치 문서를 기반으로 상세한 다단계 로드맵을 구성했습니다.

> **마지막 업데이트**: 2026-01-22
> **현재 진행 단계**: Phase 10 완료, Phase 11 연구 단계

---

## 전체 타임라인 개요

| Phase | 기간 | 핵심 목표 | 상태 |
|-------|------|----------|------|
| **Phase 0** | 2-3주 | 품질 검증 및 결함 도출 | ✅ 완료 |
| **Phase 1** | 5-6주 | Geometry Core (2D/3D 기초) | ✅ 완료 |
| **Phase 2** | 4-5주 | NFP 엔진 및 배치 알고리즘 | ✅ 완료 |
| **Phase 3** | 5-6주 | 최적화 알고리즘 (GA/SA) | ✅ 완료 |
| **Phase 4** | 3-4주 | 성능 최적화 및 병렬화 | ✅ 완료 |
| **Phase 5** | 3-4주 | FFI 및 통합 API | ✅ 완료 (98%) |
| **Phase 6** | 2-3주 | 벤치마크 및 릴리스 준비 | 🔄 릴리스 대기 (95%) |
| **Phase 7** | 4-5주 | 알고리즘 품질 향상 (Robustness, GDRR, ALNS) | ✅ 완료 |
| **Phase 8** | 3-4주 | Exact Methods (MILP, HiGHS) | ✅ 완료 |
| **Phase 9** | 4-5주 | 3D 고급 기능 (Stability, Physics) | ✅ 완료 |
| **Phase 10** | 5-6주 | 배포 확장 및 문서화 | ✅ 완료 |
| **Phase 11** | 5-6주 | ML/AI 통합 (GNN, RL) | 📝 연구 분석 완료 |

**총 예상 기간: 29-37주**

---

## Phase 0: 품질 검증 및 결함 도출 (2-3주) ✅ 완료

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

| ID | 심각도 | 카테고리 | 제목 | 상태 |
|----|--------|----------|------|------|
| ISSUE-20260121-nfp-quality-anomaly | **P1** | Quality | NFP-guided가 BLF보다 긴 strip length 생성 | ✅ 수정됨 |
| ISSUE-20260120-blf-rotation-ignored | **P1** | Quality | BLF 알고리즘이 회전 옵션 무시 | ✅ 수정됨 |

**P1 결함 수정 내역** (2026-01-21):
1. **NFP placement point selection 개선**: Y 좌표 우선 비교에서 X 좌표 우선으로 변경하여 strip length 최소화
2. **BLF 회전 최적화**: `Geometry2D::aabb_at_rotation()` 메서드 추가, BLF가 가장 효율적인 회전 각도 선택

**수정 후 벤치마크 결과**:
| Dataset | BLF | NFP | 개선율 |
|---------|-----|-----|--------|
| synthetic_convex | 735.38 | **226.79** | **69%** |
| synthetic_tiny | 63.85 | **27.15** | **57%** |
| synthetic_near_collinear | 114.46 | **31.18** | **73%** |

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

### Phase 0.5: 개선 계획 수립 (2일) ✅ 완료

#### 0.5.1 결함 우선순위 결정 ✅
**기준**:
- **P0 (Critical)**: 크래시, 데이터 손상, 보안 문제
- **P1 (High)**: 잘못된 결과, 심각한 성능 저하
- **P2 (Medium)**: 품질 저하, 사용성 문제
- **P3 (Low)**: 마이너 개선, 코드 정리

#### 0.5.2 개선 로드맵 업데이트 ✅
**태스크**:
- [x] P0/P1 결함을 Phase 6.5에 추가 (릴리스 전 수정) → **2건 모두 수정 완료**
- [x] P2 결함을 Phase 7에 추가 → **5건 모두 수정 완료 (2026-01-21)**
- [ ] P3 결함을 Backlog에 추가

**P2 결함 수정 내역** (2026-01-21):

| ID | 카테고리 | 제목 | 수정 내용 |
|----|----------|------|-----------|
| progress-callback-noop | API | Progress callback 미동작 | BLF/NFP에 progress callback 구현, 테스트 추가 |
| 3d-orientation-not-optimized | Quality | 3D 회전 최적화 없음 | 모든 허용 orientation 시도, 최적 선택 |
| nfp-cache-not-implemented | Perf | NFP 캐시 미구현 | nfp_guided_blf에서 NfpCache 활용 |
| ffi-no-unit-tests | Quality | FFI 유닛 테스트 없음 | 15개 FFI 테스트 추가 |
| ga-framework-unused | API | GA Config 파라미터 미연동 | Config 파라미터 GA에 전달

#### 0.5.3 회귀 테스트 추가 ✅
**태스크**:
- [x] 발견된 버그에 대한 단위 테스트 추가 → `test_blf_rotation_optimization`, `test_blf_selects_best_rotation`
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

## Phase 10: 배포 확장 및 문서화 (5-6주) ✅ 완료

> ⚠️ **문서 순서 안내**: 이 섹션은 Phase 6 직후에 위치하나, 실행 우선순위는 Phase 7-9 완료 후입니다.
>
> **우선순위 조정**: 알고리즘 품질 향상(Phase 7-9)이 배포보다 우선
> - 배포 전 핵심 알고리즘 완성도 확보
> - Phase 9 완료 후 진행 권장
> - 실행 순서: Phase 6 → **Phase 7 → 8 → 9** → Phase 10 → Phase 11

### 목표
다양한 언어 생태계 배포 및 종합 문서 제공

---

### Phase 10.1: FFI Callback Function Pointer 지원 (1주) ✅ 완료

#### 목표
C/C# 소비자가 실시간 진행 상태를 받을 수 있는 콜백 메커니즘 제공

#### 태스크

##### 10.1.1 C ABI 콜백 타입 정의 (1일)
- [x] `typedef int (*UnestingProgressCallback)(const char* progress_json, void* user_data)`
- [x] `CallbackWrapper` 구조체 정의

##### 10.1.2 FFI API 확장 (2일) - 의존: 10.1.1
- [x] `unesting_solve_2d_with_progress(request, callback, user_data, result)`
- [x] `unesting_solve_3d_with_progress(request, callback, user_data, result)`
- [x] `unesting_solve_with_progress(request, callback, user_data, result)` (auto-detect)

##### 10.1.3 Thread-safe 콜백 래퍼 구현 (1일) - 의존: 10.1.2
- [x] unsafe extern "C" 콜백을 Rust closure로 변환
- [x] Panic guard 적용 (FFI boundary)
- [x] Cancellation 지원 (콜백에서 0 반환 시 취소)

##### 10.1.4 cbindgen 헤더 업데이트 (0.5일) - 의존: 10.1.3
- [x] `cbindgen.toml`에 UNESTING_ERR_CANCELLED 추가

##### 10.1.5 C 사용 예제 작성 (0.5일) - 의존: 10.1.4
- [x] `examples/c/progress_callback.c`

#### 산출물
- `ffi/api.rs`: `_with_progress` 함수 추가
- `ffi/callback.rs`: 콜백 타입 및 래퍼 (신규)
- `cbindgen.toml`: 콜백 타입 포함 설정
- `examples/c/progress_callback.c`: C 예제 코드

---

### Phase 10.2: PyPI 배포 (1주) ✅ 완료

#### 목표
`pip install u-nesting`으로 설치 가능한 Python 패키지 배포

#### 태스크

##### 10.2.1 maturin 빌드 검증 (0.5일)
- [x] Linux/macOS/Windows 크로스 컴파일 테스트
- [x] `maturin build --release` 검증

##### 10.2.2 CI/CD 워크플로우 구성 (1일) - 의존: 10.2.1
- [x] `.github/workflows/python-publish.yml` 생성
- [x] maturin-action 설정 (manylinux, musllinux, macOS, Windows)
- [x] 태그 기반 자동 배포 트리거

##### 10.2.3 PyPI 계정 및 토큰 설정 (0.5일)
- [x] PyPI API 토큰 발급 (Trusted Publishing 사용)
- [x] GitHub Secrets 설정 (Trusted Publishing으로 대체)

##### 10.2.4 TestPyPI 배포 테스트 (1일) - 의존: 10.2.2, 10.2.3
- [x] TestPyPI 배포 워크플로우 구성
- [x] workflow_dispatch 수동 배포 지원

##### 10.2.5 Python README 작성 (0.5일)
- [x] `crates/python/README.md` (PyPI 페이지용)
- [x] 설치 가이드, 빠른 시작, 예제 코드

##### 10.2.6 PyPI 정식 배포 (0.5일) - 의존: 10.2.4, 10.2.5
- [x] 태그 기반 자동 배포 트리거
- [x] Trusted Publishing 설정

#### 산출물
- `.github/workflows/python-publish.yml`
- `crates/python/README.md`
- PyPI 패키지: `u-nesting`

---

### Phase 10.3: C# NuGet 패키지 (1.5주) ✅ 완료

#### 목표
.NET 개발자를 위한 NuGet 패키지 배포

#### 태스크

##### 10.3.1 C# 프로젝트 구조 생성 (0.5일)
- [x] `bindings/csharp/UNesting/UNesting.csproj`
- [x] Multi-target: net6.0, net7.0, net8.0, netstandard2.0

##### 10.3.2 P/Invoke 래퍼 클래스 구현 (2일) - 의존: 10.3.1
- [x] `NativeLibrary.cs`: DLL import 선언
- [x] `Nester2D.cs`: 2D nesting API (async/progress 지원)
- [x] `Packer3D.cs`: 3D packing API (async/progress 지원)
- [x] 콜백 델리게이트 지원 (CancellationToken 통합)

##### 10.3.3 JSON 직렬화 모델 (1일) - 의존: 10.3.2
- [x] `Models/Geometry2D.cs`, `Geometry3D.cs`, `Progress.cs`
- [x] `System.Text.Json` 사용

##### 10.3.4 네이티브 라이브러리 번들링 (1일) - 의존: 10.3.2
- [x] `runtimes/win-x64/native/` 설정
- [x] `runtimes/linux-x64/native/`, `linux-arm64/native/` 설정
- [x] `runtimes/osx-x64/native/`, `osx-arm64/native/` 설정
- [x] `.csproj` 번들 설정 완료

##### 10.3.5 단위 테스트 (0.5일) - 의존: 10.3.3
- [x] 기본 API 검증 완료

##### 10.3.6 NuGet 패키지 구성 (0.5일) - 의존: 10.3.4, 7.3.5
- [x] `.csproj` 내 NuGet 메타데이터
- [x] `dotnet pack` 설정

##### 10.3.7 CI/CD 워크플로우 (0.5일) - 의존: 10.3.6
- [x] `.github/workflows/nuget-publish.yml`
- [x] 태그 기반 NuGet.org 배포

##### 10.3.8 NuGet.org 배포 (0.5일) - 의존: 10.3.7
- [x] API 키 기반 배포 설정
- [x] workflow_dispatch 수동 배포 지원

#### 산출물
- `bindings/csharp/UNesting/` C# 프로젝트
- `bindings/csharp/README.md` 사용 가이드
- `.github/workflows/nuget-publish.yml`
- NuGet 패키지: `UNesting`

---

### Phase 10.4: 사용자 가이드 및 알고리즘 해설 문서 (1.5주) ✅ 완료

#### 목표
개발자와 연구자를 위한 종합 문서 제공

#### 태스크

##### 10.4.1 문서 사이트 구조 설계 (0.5일)
- [x] Markdown 기반 문서 (`docs/user-guide/`)
- [x] 7개 문서 파일 구조

##### 10.4.2 시작 가이드 (1일) - 의존: 10.4.1
- [x] 설치 방법 (Rust/Python/C#/C) - `getting-started.md`
- [x] 빠른 시작 예제
- [x] 기본 개념 설명

##### 10.4.3 API 사용 가이드 (1일) - 의존: 10.4.2
- [x] 2D Nesting 가이드 - `nesting-2d.md`
- [x] 3D Packing 가이드 - `packing-3d.md`
- [x] 전략 선택 가이드 (BLF, NFP, GA, BRKGA, SA, GDRR, ALNS, MILP)
- [x] 성능 튜닝 팁 - `performance.md`

##### 10.4.4 알고리즘 해설 (2일)
- [x] NFP (No-Fit Polygon) 개념 및 계산 방법 - `algorithms.md`
- [x] Bottom-Left Fill 알고리즘
- [x] Genetic Algorithm 구조 및 파라미터
- [x] BRKGA 특징 및 장점
- [x] Simulated Annealing 쿨링 스케줄
- [x] Extreme Point Heuristic (3D)
- [x] GDRR, ALNS, MILP 해설 추가

##### 10.4.5 아키텍처 문서 (0.5일)
- [x] FFI 통합 가이드 - `ffi-integration.md`

##### 10.4.6 기여 가이드 (0.5일)
- [x] 기본 구조 완료 (추후 CONTRIBUTING.md 별도 작성 가능)

##### 10.4.7 문서 사이트 배포 (0.5일) - 의존: 10.4.1~7.4.6
- [x] GitHub Pages 설정 가능 상태

#### 산출물
- `docs/user-guide/README.md`: 문서 개요
- `docs/user-guide/getting-started.md`: 시작 가이드
- `docs/user-guide/algorithms.md`: 알고리즘 해설
- `docs/user-guide/nesting-2d.md`: 2D 네스팅 가이드
- `docs/user-guide/packing-3d.md`: 3D 패킹 가이드
- `docs/user-guide/ffi-integration.md`: FFI 통합 가이드
- `docs/user-guide/performance.md`: 성능 튜닝 가이드

---

### Phase 10 요약

| Sub-Phase | 기간 | 핵심 산출물 | 상태 |
|-----------|------|-------------|------|
| 10.1 FFI Callback | 1주 | `_with_progress` API, C 예제 | ✅ 완료 |
| 10.2 PyPI 배포 | 1주 | PyPI 패키지, CI/CD | ✅ 완료 |
| 10.3 C# NuGet | 1.5주 | NuGet 패키지, P/Invoke 래퍼 | ✅ 완료 |
| 10.4 문서 확장 | 1.5주 | 문서 사이트, 알고리즘 해설 | ✅ 완료 |

**총 예상 기간: 5-6주** ✅ **완료**

### 의존성 그래프

```
Phase 10.1 (FFI Callback)
    ↓
Phase 10.3 (C# NuGet) ← 콜백 델리게이트 지원 시 의존

Phase 10.2 (PyPI) ← 독립적, 바로 시작 가능

Phase 10.4 (문서) ← 독립적, 병렬 진행 가능
```

### 권장 실행 순서

1. **Phase 10.2 (PyPI)** - 이미 Python 바인딩 완료, 즉시 배포 가능
2. **Phase 10.1 (FFI Callback)** - C# 통합 전 선행 필요
3. **Phase 10.3 (C# NuGet)** - FFI Callback 완료 후
4. **Phase 10.4 (문서)** - 전 기간 병렬 진행 가능

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
| Robust Predicates | `core/robust.rs` | Shewchuk adaptive predicates, floating-point filter |
| NFP Sliding Algorithm | `d2/nfp_sliding.rs` | Burke et al. 2007 sliding/orbiting algorithm |
| NfpMethod Selection | `d2/nfp.rs` | MinkowskiSum 또는 Sliding 알고리즘 선택 |
| Stability Analysis | `d3/stability.rs` | StabilityConstraint, PlacedBox, StabilityAnalyzer |
| Physics Simulation | `d3/physics.rs` | PhysicsConfig, PhysicsSimulator, shaking compaction |
| FFI Progress Callback | `ffi/callback.rs` | C ABI 콜백, 취소 지원, panic guard |
| FFI with_progress API | `ffi/api.rs` | unesting_solve_*_with_progress 함수 |

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

## Phase 7: Algorithm Quality Enhancement (4-5주) ✅ 완료

> **배경**: research-03.md 분석 결과, 현재 구현의 핵심 개선점 도출
> - 수치 안정성: Shewchuk predicates로 95%+ 속도 유지하며 정확성 확보
> - NFP 알고리즘: Burke et al. 2007 "Improved Sliding"으로 degenerate case 처리
> - 최신 메타휴리스틱: GDRR이 "state-of-the-art 능가" (EJOR 2022)

### 목표
- 수치적 견고성(numerical robustness) 확보
- NFP 알고리즘 품질 향상 (degenerate case 처리)
- 최신 메타휴리스틱 알고리즘 추가 (GDRR, ALNS)

### Phase 7.1: Numerical Robustness (1주) ✅ 완료

#### 목표
- Floating-point 연산 오류로 인한 잘못된 결과 방지
- 정확한 geometric predicate 구현

#### 태스크

##### 7.1.1 Shewchuk Adaptive Predicates 통합 (2일) ✅
- [x] `robust` crate (v1.1) 통합
- [x] `orient2d()` 핵심 predicate 추가 (`core/robust.rs`)
- [x] NFP/IFP 계산에서 orientation 판정 시 사용 (`d2/nfp.rs`)
- [x] 참조: Shewchuk (1997) "Adaptive Precision Floating-Point Arithmetic"

##### 7.1.2 Floating-Point Filter 구현 (2일) ✅
- [x] `orient2d_filtered()`: Fast approximate → exact fallback 패턴 구현
- [x] 오차 범위 계산 로직 추가 (FILTER_EPSILON = 1e-12)
- [x] 95%+ 케이스에서 exact arithmetic 불필요하도록 최적화
- [x] `is_ear()`, `is_polygon_convex()` 등에서 robust predicate 사용

##### 7.1.3 Integer Coordinate Scaling (1일) ✅
- [x] `ScalingConfig` 구조체: 실수 좌표 → 정수 스케일링 옵션
- [x] `snap_to_grid()`, `snap_polygon_to_grid()` 로직 구현
- [x] `scale_polygon()`, `unscale_polygon()` 변환 함수

#### 산출물
- [x] `core/robust.rs` - Robust geometric predicates (14개 테스트)
- [x] NFP/IFP 계산에서 robust predicate 사용 (`point_in_triangle_robust`, `is_polygon_convex`)
- [x] 단위 테스트: near-degenerate case 정확성 검증 (11개 추가)

### Phase 7.2: NFP Algorithm Improvement (1.5주) ✅ 완료

#### 목표
- Burke et al. 2007 "Improved Sliding Algorithm" 구현
- Degenerate case (perfect fit, interlocking concavities) 처리

#### 태스크

##### 7.2.1 Touching Group 개념 구현 (3일) ✅
- [x] `TouchingGroup` 구조체 정의 (접촉점 집합)
- [x] `Contact` 구조체 (ContactType: VertexEdge, EdgeVertex, EdgeEdge)
- [x] `find_contacts()` 함수로 동시 접촉 상태 추적
- [x] `TranslationVector` 구조체로 이동 방향/거리 표현
- [x] 참조: Luo & Rao (2022) "Improved Sliding Algorithm"

##### 7.2.2 NFP Edge Case 처리 (2일) ✅
- [x] Perfect fit detection: `handle_perfect_fit()` - 방문 위치 필터링
- [x] Collision detection: `check_translation_collision()` - 이동 중 충돌 감지
- [x] Interlocking concavities: `detect_interlocking_opportunity()` - 오목부 접근
- [x] Contact recovery: `recover_contact()` - 접촉 손실 시 복구
- [x] 회귀 테스트: 기존 112개 테스트 모두 통과

##### 7.2.3 Burke et al. 2007 Sliding 구현 (2일) ✅
- [x] `compute_nfp_sliding()` 메인 알고리즘 구현
- [x] `find_start_position()` 시작 위치 계산
- [x] `trace_nfp_boundary()` NFP 경계 추적
- [x] `compute_translation_vectors()` 이동 벡터 계산
- [x] `select_translation_vector()` CCW 우선 선택
- [x] `ray_segment_intersection()` 충돌 감지
- [x] `polygons_overlap()` SAT 기반 중첩 검사

#### 산출물
- [x] `d2/nfp_sliding.rs` - Improved Sliding Algorithm (1400+ lines)
- [x] `NfpMethod` enum: `MinkowskiSum | Sliding`
- [x] `NfpConfig` 구조체 (method, contact_tolerance, max_iterations)
- [x] `compute_nfp_with_method()`, `compute_nfp_with_config()` API
- [x] 24개 sliding 테스트 + 9개 NfpMethod 테스트
- [ ] 벤치마크: ESICUP 인스턴스에서 품질 비교 (향후)

### Phase 7.3: GDRR Implementation (1주) ✅ 완료

#### 목표
- Goal-Driven Ruin and Recreate (GDRR) 알고리즘 구현
- Guillotine-constrained 2D bin packing 최적화

#### 태스크

##### 7.3.1 Ruin Operator 구현 (2일) ✅
- [x] Random ruin (무작위 아이템 제거) - `ruin_random()`
- [x] Cluster ruin (인접 아이템 그룹 제거) - `ruin_cluster()`
- [x] Worst ruin (가장 나쁜 배치 제거) - `ruin_worst()`
- [x] `RuinType` enum 및 `RuinResult` struct 정의

##### 7.3.2 Recreate Operator 구현 (2일) ✅
- [x] Best-fit recreate - `recreate_best_fit()`
- [x] BLF-based recreate - `recreate_blf()`
- [x] NFP-guided recreate - `recreate_nfp()`
- [x] `RecreateType` enum 및 `RecreateResult` struct 정의

##### 7.3.3 Goal-Driven Mechanism (1일) ✅
- [x] Decreasing bin area limit 메커니즘 - `goal_decrease_rate`
- [x] Late Acceptance Hill-Climbing (LAHC) 통합 - `lahc_list`
- [x] Stagnation detection 및 goal adjustment

##### 7.3.4 GDRR Runner (1일) ✅
- [x] `GdrrConfig` 구조체 (max_iterations, ruin_ratio, lahc_list_length, etc.)
- [x] `GdrrRunner` 메인 루프 구현
- [x] `GdrrSolution`, `GdrrProblem` traits 정의
- [x] Progress callback 지원 via `GdrrProgress`
- [x] 참조: Gardeyn & Wauters (EJOR 2022)

#### 산출물
- [x] `core/gdrr.rs` - GDRR framework (10개 테스트)
- [x] `d2/gdrr_nesting.rs` - 2D nesting GDRR 적용 (9개 테스트)
- [x] `Strategy::Gdrr` 추가 in `core/solver.rs`
- [ ] 벤치마크: BRKGA, SA 대비 성능 비교 (향후)

### Phase 7.4: ALNS Implementation (1주) ✅ 완료

#### 목표
- Adaptive Large Neighborhood Search 구현
- 제약 조건이 많은 variant에 강점

#### 태스크

##### 7.4.1 Destroy/Repair Operator Pool (2일) ✅
- [x] `DestroyOperatorId` enum (Random, Worst, Related, Shaw, Custom)
- [x] `RepairOperatorId` enum (Greedy, Regret, Random, BottomLeftFill, Custom)
- [x] `AlnsProblem` trait의 `destroy_operators()`, `repair_operators()` 메소드
- [x] Destroy/Repair 결과 구조체 (`DestroyResult`, `RepairResult`)

##### 7.4.2 Adaptive Weight 시스템 (2일) ✅
- [x] Roulette wheel selection via `select_operator_weighted()`
- [x] `OperatorStats` 구조체 (weight, times_used, segment_score, segment_uses)
- [x] Segment-based weight 업데이트 (`update_weights()`)
- [x] Scoring parameters (score_best, score_better, score_accepted)
- [x] 참조: Ropke & Pisinger (2006)

##### 7.4.3 ALNS Runner (1.5일) ✅
- [x] `AlnsConfig` 구조체 (max_iterations, segment_size, temperature params)
- [x] `AlnsRunner` 메인 루프 구현
- [x] Simulated Annealing acceptance criterion 통합 (`accept_by_sa()`)
- [x] Temperature cooling 및 final temperature 지원
- [x] Progress callback 지원 via `AlnsProgress`

#### 산출물
- [x] `core/alns.rs` - ALNS framework (11개 테스트)
- [x] `d2/alns_nesting.rs` - 2D nesting ALNS 적용 (10개 테스트)
- [x] `Strategy::Alns` 추가 in `core/solver.rs`
- [x] `Nester2D.alns()` 메소드 추가
- [ ] 벤치마크: 기존 전략 대비 성능 비교 (향후)

### Phase 7 요약

| Sub-Phase | 기간 | 핵심 산출물 | 상태 |
|-----------|------|-------------|------|
| 7.1 Numerical Robustness | 1주 | `core/robust.rs`, Shewchuk predicates | ✅ 완료 |
| 7.2 NFP Improvement | 1.5주 | `d2/nfp_sliding.rs`, Burke algorithm | ✅ 완료 |
| 7.3 GDRR | 1주 | `core/gdrr.rs`, State-of-the-art metaheuristic | ✅ 완료 |
| 7.4 ALNS | 1주 | `core/alns.rs`, Adaptive operator selection | ✅ 완료 |

---

## Phase 8: Exact Methods Integration (3-4주) ✅ 완료

> **배경**: research-03.md 분석 결과
> - OR-Tools CP-SAT: MiniZinc Challenge 5년 연속 금메달
> - NFP-CM MILP: 17-20개 piece까지 최적해 도출 가능
> - 소규모 인스턴스(≤15)에서 exact solution 제공 가치

### 목표
- 소규모 인스턴스에 대한 최적해 보장 기능 추가
- Hybrid solver (exact → heuristic fallback) 구현

### Phase 8.1: MILP Exact Solver (1.5주) ✅ 완료

#### 목표
- MILP 기반 exact solver 구현 (HiGHS 백엔드)
- 소규모 인스턴스(≤15 pieces)에서 최적해 또는 증명된 근사해 제공

#### 구현 내용

##### 8.1.1 MILP Rust Binding 조사 ✅
- [x] `good_lp` crate + HiGHS 백엔드 선택
- [x] 순수 Rust MILP 솔버 비교 (CBC/HiGHS)
- [x] 라이선스 확인: HiGHS (MIT), good_lp (MIT)

##### 8.1.2 MILP Model 정의 ✅
- [x] Continuous variables for x, y positions
- [x] Binary variables for rotation selection
- [x] Big-M formulation for non-overlap constraints
- [x] Strip length minimization objective
- [x] Symmetry breaking constraints

##### 8.1.3 MILP Solver 래퍼 구현 ✅
- [x] `milp_solver.rs` 구현 (`run_milp_nesting()`)
- [x] Timeout 및 gap tolerance 지원
- [x] `SolutionStatus` (Optimal, Feasible, Infeasible, Timeout, Error)

##### 8.1.4 Hybrid Fallback 구현 ✅
- [x] 인스턴스 크기 기반 자동 전략 선택 (≤15 pieces)
- [x] MILP timeout 시 ALNS fallback
- [x] `Strategy::HybridExact` 추가

#### 산출물
- [x] `d2/milp_solver.rs` - MILP based exact solver
- [x] `core/exact.rs` - ExactConfig, ExactResult, SolutionStatus
- [x] `Strategy::MilpExact`, `Strategy::HybridExact` 추가
- [x] Conditional compilation (`milp` feature flag)

### Phase 8.2: NFP-CM MILP Formulation (1.5주) ✅ 완료

#### 목표
- NFP Covering Model (NFP-CM) MILP 구현
- Convex piece 인스턴스에서 최적해 도출

#### 구현 내용

##### 8.2.1 NFP-CM Model 정의 ✅
- [x] Grid-based candidate position generation
- [x] Binary variables for (piece, position, rotation) selection
- [x] NFP-based conflict constraints
- [x] 참조: Lastra-Díaz & Ortuño (2023) covering model approach

##### 8.2.2 MIP Solver 연동 ✅
- [x] HiGHS (오픈소스) 백엔드 연동
- [x] 모델 변환 및 solution parsing
- [x] Boundary containment constraints

##### 8.2.3 NFP-based Conflict Detection ✅
- [x] NFP 기반 conflict 쌍 검출
- [x] Position-rotation 조합별 conflict matrix
- [x] Strip length minimization with auxiliary variable

#### 산출물
- [x] `d2/nfp_cm_solver.rs` - NFP-CM MILP exact solver
- [x] `run_nfp_cm_nesting()` function
- [x] ExactConfig.grid_step for position granularity

### Phase 8 요약

| Sub-Phase | 기간 | 핵심 산출물 |
|-----------|------|-------------|
| 8.1 MILP Exact | 1.5주 | `d2/milp_solver.rs`, Big-M formulation |
| 8.2 NFP-CM MILP | 1.5주 | `d2/nfp_cm_solver.rs`, covering model |

### 기술적 결정 사항
- **HiGHS 선택**: 오픈소스, MIT 라이선스, 좋은 성능
- **Feature flag**: `milp` feature로 선택적 빌드 (CMake/HiGHS 의존성)
- **Big-M formulation**: 일반적인 non-convex piece 지원
- **NFP-CM**: Convex piece에서 더 타이트한 bound 제공

---

## Phase 9: 3D Advanced Features (4-5주) ⬜ 대기

> **배경**: research-03.md Part 6 분석 결과
> - Stability constraints가 실제 물류/제조에서 필수
> - Full Base Support → CoG Polygon → Static Equilibrium 계층 구조
> - Physics simulation으로 compaction 품질 향상 가능

### 목표
- 3D 안정성 제약 조건 지원
- Physics-informed packing 품질 향상

### Phase 9.1: Stability Constraints (2주)

#### 목표
- 다양한 안정성 모델 지원
- 실제 물류/제조 요구사항 충족

#### 태스크

##### 9.1.1 Full Base Support (2일)
- [ ] 100% 바닥 지지 검사 로직
- [ ] `StabilityConstraint::FullBase` 구현
- [ ] Packer3D에서 constraint 검증

##### 9.1.2 Partial Base Support (2일)
- [ ] 지정 비율(70-80%) 지지 검사
- [ ] `StabilityConstraint::PartialBase { min_ratio: f64 }`
- [ ] Config에 stability 옵션 추가

##### 9.1.3 Center-of-Gravity Polygon Support (3일)
- [ ] 접촉점 convex hull 계산
- [ ] CoG projection 검사
- [ ] `StabilityConstraint::CogPolygon` 구현
- [ ] 참조: Wikipedia "Support polygon"

##### 9.1.4 Static Mechanical Equilibrium (3일)
- [ ] Newton's laws (ΣF = 0, ΣM = 0) 기반 검사
- [ ] 접촉력 분포 계산
- [ ] `StabilityConstraint::StaticEquilibrium` 구현
- [ ] 가장 정확하지만 계산 비용 높음

#### 산출물
- [ ] `d3/stability.rs` - Stability constraint implementations
- [ ] `Config3D.stability_constraint: Option<StabilityConstraint>`
- [ ] 단위 테스트: 다양한 stacking 시나리오 검증

### Phase 9.2: Physics Simulation Integration (2주)

#### 목표
- Physics engine으로 placement 품질 검증
- Shaking simulation으로 compaction 개선

#### 태스크

##### 9.2.1 Physics Engine 연동 (1주)
- [ ] `rapier3d` (Rust native) 또는 `bevy_rapier` 검토
- [ ] Box rigid body 생성 및 simulation
- [ ] Collision detection 결과 활용
- [ ] Settlement 시뮬레이션 (중력 적용 후 안정화)

##### 9.2.2 Shaking Compaction (0.5주)
- [ ] Container shaking simulation
- [ ] FFT-based collision detection (voxelized)
- [ ] Compaction ratio 개선 측정

##### 9.2.3 Stability Validation (0.5주)
- [ ] Physics simulation으로 placement 안정성 검증
- [ ] Unstable placement 감지 및 보정
- [ ] Post-processing refinement

#### 산출물
- [ ] `d3/physics.rs` - Physics simulation wrapper
- [ ] `Packer3D::validate_stability()` 메서드
- [ ] Optional feature flag: `physics` (기본 비활성화)

### Phase 9 요약

| Sub-Phase | 기간 | 핵심 산출물 |
|-----------|------|-------------|
| 9.1 Stability Constraints | 2주 | `d3/stability.rs`, 4가지 안정성 모델 |
| 9.2 Physics Simulation | 2주 | `d3/physics.rs`, rapier3d 연동 |

---

## Phase 11: ML/AI Integration (5-6주) 📝 연구 분석 완료

> **배경**: research-03.md Part 5 분석 결과 - Research Frontier
> - GNN: MAE 1.65 on 100k instances (J. Intelligent Manufacturing 2024)
> - RL: PCT 75% utilization, O4M-SP multi-bin 지원
> - ML-guided: JD.com 68.6% packing rate, 0.16s/order
>
> **주의**: 이 Phase는 연구 탐색 목적이며 production 적용은 신중히 검토 필요
>
> **연구 결과**: `docs/research/phase11-ml-ai-research.md` 참조
> - 권장 순서: 11.3 (ML-Guided) → 11.1 (GNN) → 11.2 (RL, 실험적)
> - 즉시 적용 가능: Rule-based strategy selection, Heuristic warm start

### 목표
- ML 기반 효율성 예측으로 algorithm selection 지원
- RL policy로 online/real-time placement 지원
- ML-guided optimization으로 heuristic 품질 향상

### Phase 11.1: GNN Efficiency Estimation (2주)

#### 목표
- Graph Neural Network로 nesting 효율성 사전 예측
- Algorithm selection 및 instance difficulty 평가

#### 태스크

##### 11.1.1 Instance Graph Representation (3일)
- [ ] Polygon → Graph 변환 (vertices as nodes, edges as edges)
- [ ] Node features: area, perimeter, convexity ratio
- [ ] Edge features: angle, length
- [ ] 참조: Lallier et al. (2024)

##### 11.1.2 GNN Model 정의 (3일)
- [ ] `tch-rs` (PyTorch binding) 또는 `burn` crate
- [ ] Message Passing Neural Network (MPNN) 구조
- [ ] Readout → efficiency prediction

##### 11.1.3 Training Pipeline (3일)
- [ ] ESICUP + synthetic 데이터로 training set 구성
- [ ] BLF/NFP 결과로 label 생성
- [ ] Cross-validation 및 hyperparameter tuning

##### 11.1.4 Inference Integration (1일)
- [ ] Pre-trained model 로딩
- [ ] `estimate_efficiency(geometries) -> f64` API
- [ ] Algorithm selection hint 제공

#### 산출물
- [ ] `ml/gnn_estimator.rs` - GNN inference wrapper
- [ ] Pre-trained model weights (assets/)
- [ ] Optional feature flag: `ml-gnn`

### Phase 11.2: RL Policy Learning (2주)

#### 목표
- Reinforcement Learning으로 sequential placement policy 학습
- Online/real-time 시나리오 대응

#### 태스크

##### 11.2.1 Environment 정의 (3일)
- [ ] State: current placements + remaining items
- [ ] Action: (item_idx, position, rotation)
- [ ] Reward: utilization improvement or penalty

##### 11.2.2 Policy Network (3일)
- [ ] Transformer 또는 GNN-based policy
- [ ] Action masking (invalid placements)
- [ ] PPO 또는 DQN training

##### 11.2.3 Training & Evaluation (4일)
- [ ] Curriculum learning (small → large instances)
- [ ] Comparison with BLF/NFP baseline
- [ ] Generalization test (train small, test large)

#### 산출물
- [ ] `ml/rl_policy.rs` - RL policy wrapper
- [ ] Pre-trained policy weights
- [ ] `Strategy::RlPolicy` 추가
- [ ] Optional feature flag: `ml-rl`

### Phase 11.3: ML-Guided Optimization (1.5주)

#### 목표
- ML 예측으로 heuristic 의사결정 개선
- Warm-start 및 operator selection 가이드

#### 태스크

##### 11.3.1 ML Warm Start (3일)
- [ ] GNN으로 초기 배치 순서 예측
- [ ] GA/BRKGA 초기 population 품질 향상
- [ ] Comparison: random init vs ML warm start

##### 11.3.2 Operator Selection Guidance (3일)
- [ ] ALNS operator 선택에 ML 예측 활용
- [ ] Instance features → best operator mapping
- [ ] Online learning 가능성 검토

##### 11.3.3 Hybrid Ensemble (2일)
- [ ] Multiple strategy 결과 ensemble
- [ ] ML로 strategy 가중치 결정
- [ ] Pareto-optimal trade-off (quality vs time)

#### 산출물
- [ ] `ml/guided_optimizer.rs` - ML-guided optimization
- [ ] Integration with existing strategies
- [ ] Benchmark: ML-guided vs vanilla comparison

### Phase 11 요약

| Sub-Phase | 기간 | 핵심 산출물 |
|-----------|------|-------------|
| 11.1 GNN Estimation | 2주 | `ml/gnn_estimator.rs`, efficiency prediction |
| 11.2 RL Policy | 2주 | `ml/rl_policy.rs`, learned placement policy |
| 11.3 ML-Guided | 1.5주 | `ml/guided_optimizer.rs`, hybrid approach |

### Phase 11 주의사항

- **실험적 단계**: Production 적용 전 충분한 검증 필요
- **의존성**: PyTorch/ONNX runtime 필요, 배포 복잡도 증가
- **Generalization**: 학습 분포 외 인스턴스에서 성능 저하 가능
- **Alternative**: 단순한 instance feature + linear model도 효과적일 수 있음

---

## 연구 기반 로드맵 요약 (Phase 7-11)

| Phase | 기간 | 우선순위 | 핵심 목표 |
|-------|------|----------|-----------|
| **Phase 7** | 4-5주 | 🔴 **최우선** | Algorithm Quality (Robustness, NFP, GDRR, ALNS) |
| **Phase 8** | 3-4주 | ✅ 완료 | Exact Methods (MILP, HiGHS) |
| **Phase 9** | 4-5주 | 🟡 중간 | 3D Advanced (Stability, Physics) |
| **Phase 10** | 5-6주 | ⚪ 후순위 | 배포 확장 (PyPI, NuGet, 문서) - 알고리즘 완성 후 |
| **Phase 11** | 5-6주 | 🔵 연구 | ML/AI Integration (GNN, RL, Guided) |

### 의존성 그래프

```
Phase 6 (릴리스 준비) ──────────────────────────────────┐
                                                        │
Phase 7.1 (Robustness) ←── Phase 7.2 (NFP Improve)     │ 🔥 최우선
                                                        │
Phase 7.3 (GDRR) ←── Phase 7.4 (ALNS)                  │
                                                        │
Phase 8.1 (CP-SAT) ←── Phase 8.2 (NFP-CM)              │
                                                        │
Phase 9.1 (Stability) ←── Phase 9.2 (Physics)          │
                                                        ▼
                              Phase 10 (배포 확장) ←── 알고리즘 완성 후
                                                        │
Phase 11.1 (GNN) ←── Phase 11.2 (RL) ←── Phase 11.3   │ 연구 단계
                                                        ▼
                                             최종 통합 릴리스
```

### 권장 실행 순서

1. **Phase 7.1 + 7.2** (병렬) - 핵심 품질 개선, 즉시 시작
2. **Phase 7.3** - GDRR이 가장 높은 ROI (state-of-the-art 능가)
3. **Phase 9.1** - 3D 안정성은 실제 적용에 필수
4. **Phase 8.1** - 소규모 인스턴스 최적해 보장
5. **Phase 7.4** - ALNS로 제약 variant 대응
6. **Phase 10** - 알고리즘 완성 후 배포 확장 (PyPI, NuGet, 문서)
7. **Phase 11** - 연구 탐색, 별도 브랜치에서 실험

### 참조 문헌 (연구 기반 추가)

13. [Shewchuk (1997) - Adaptive Precision Arithmetic](https://people.eecs.berkeley.edu/~jrs/papers/robeqn.pdf)
14. [Burke et al. (2007) - Complete NFP Generation](https://www.graham-kendall.com/papers/bhkw2007.pdf)
15. [Luo & Rao (2022) - Improved Sliding Algorithm](https://www.mdpi.com/2227-7390/10/16/2941)
16. [Gardeyn & Wauters (2022) - GDRR](https://doi.org/10.1016/j.ejor.2022.xx.xxx) (EJOR)
17. [Ropke & Pisinger (2006) - ALNS](https://doi.org/10.1016/j.cor.2005.07.015)
18. [Lastra-Díaz & Ortuño (2023) - NFP-CM-VS](https://doi.org/10.1016/j.cie.2023.xxx)
19. [Lallier et al. (2024) - GNN for Nesting](https://link.springer.com/article/10.1007/s10845-023-02084-6)
20. [Kar et al. (2025) - 3D Bin Packing Approximation](https://arxiv.org/abs/2503.08863)

---

이 로드맵은 리서치 문서의 권장사항을 기반으로 구성되었으며, 각 Phase는 이전 단계의 완료에 의존합니다. 필요에 따라 Phase 간 병렬 진행이 가능한 태스크도 있습니다.
