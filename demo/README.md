# SigmaNest XML → U-Nesting 데모

이 폴더는 **소비 앱 시뮬레이션**을 포함합니다. SigmaNest XML 파일을 파싱하여 U-Nesting으로 네스팅하는 전체 워크플로우를 보여줍니다.

## 구조

```
dev/
├── 우등산업.XML            # SigmaNest 내보내기 파일 (3.1MB)
├── 우등산업_note.md        # XML 구조 분석 노트
├── sigmanest_to_json.py    # XML → U-Nesting JSON 변환기
├── sigmanest_parts.json    # 변환된 JSON (벤치마크 도구용)
├── nesting_ui.py           # 대화형 데스크톱 UI (tkinter)
├── nesting_viewer.html     # 웹 기반 시각화 도구
└── README.md               # 이 파일
```

## 대화형 UI

### 옵션 1: 데스크톱 UI (Python/tkinter)
```bash
cd dev
python nesting_ui.py
```
기능:
- SigmaNest XML / JSON 파일 로드
- 파라미터 설정 (strip height, strategy, time limit)
- U-Nesting 벤치마크 도구 호출
- 결과 시각화

### 옵션 2: 웹 뷰어 (HTML/JavaScript)
```bash
# 브라우저에서 직접 열기
start nesting_viewer.html   # Windows
open nesting_viewer.html    # macOS
```
기능:
- JSON 데이터셋 로드
- 샘플 데이터 제공
- 실시간 BLF 시뮬레이션
- 드래그/줌 지원
- JSON 내보내기

## 사용법

### 1. XML → JSON 변환
```bash
cd dev
python sigmanest_to_json.py [input.xml] [output.json]
```

### 2. 네스팅 실행
```bash
cd ..
cargo run --release -p u-nesting-benchmark --bin bench-runner -- \
    run-file dev/sigmanest_parts.json -s blf -s nfp -t 60
```

### 3. 결과 예시
```
Dataset         Strategy         Length     Placed    Time
sigmanest_import BottomLeftFill  199.62mm   2/2       0ms
sigmanest_import NfpGuided       ---        2/2       20560ms
```

## 핵심 포인트

### 소비 앱 책임 (이 폴더에서 시뮬레이션)
- SigmaNest XML 파싱
- TSNLine, TSNArc 좌표 추출
- Arc → 폴리라인 변환 (16 세그먼트)
- 폐곡선 재구성 (세그먼트 체이닝)
- U-Nesting JSON 포맷으로 변환

### U-Nesting 라이브러리 책임
- Geometry2D 수신 (JSON으로)
- BLF/NFP/GA 알고리즘 실행
- 배치 결과 반환

## XML 구조 (참고)

```xml
TSNPart
├── PartName: "J-해성 26 01 02 한일농장"
├── AbsAngles: "0,90,180,270"
├── DList
│   ├── TSNContour (OutSideC=True)  → 외곽선
│   │   └── DList: 71 lines + 16 arcs
│   └── TSNContour (OutSideC=False) → 홀
│       └── DList: 1 line + 1 arc
└── WOList
    └── TSNWorkOrder
        └── QtyOrdered: 1
```

## 변환 결과

| 항목 | 값 |
|------|-----|
| 파트 수 | 2 |
| 정점 수 | 328개/파트 |
| 홀 | 1개/파트 |
| 크기 | 199.6 × 199.6 mm |
| 허용 회전 | 0°, 90°, 180°, 270° |
