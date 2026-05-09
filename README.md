# 📊 서울시 상권 쇠퇴 예측 모델 (SNU12 통계 개인 프로젝트)

서울시 상권분석서비스 공공데이터(2020~2024)를 통합해 **상권별 폐업률 상위 40%** 를 `declining`(쇠퇴)으로 라벨링하고, **XGBoost 이진분류 + threshold tuning** 으로 어떤 상권이 쇠퇴 위험인지 예측한 단독 통계 프로젝트.

## 문제 정의

자영업·소상공인 입장에서 어떤 상권이 위험한지(폐업률이 높은지) 사전에 알기 어려움. 점포·매출·유동인구가 분리된 공공데이터를 직접 조합해야 함.

## 데이터 (서울시 상권분석서비스, 5년치 통합)

- 점포-상권 (2020~2024) — 점포 수 / 폐업 점포 수 / 폐업률
- 추정매출-상권 (2020~2024) — 당월 매출 건수
- 길단위인구-상권 — 총 유동인구 수

> 인코딩 `cp949`, 파일 용량 30MB+ × 10개 → `.gitignore`. 직접 [서울 열린데이터광장](https://data.seoul.go.kr/) 에서 받아 `data/` 에 배치.

## 파생변수 5종 (`code/data_preprocess.py`)

| 변수 | 정의 |
|------|------|
| `avg_sales` | 월 매출건수 ÷ 총 점포수 (점포당 매출) |
| `log_pop` | log(총 유동인구 + 1) (스케일 보정) |
| `density` | 총 점포수 ÷ 총 유동인구 (상권 밀도) |
| `sales_growth` | 상권별 분기 매출 변화율 (`pct_change`) |
| `fail_rate` | 평균 폐업률 (라벨 산출용) |

라벨: `fail_rate` 60 분위수(상위 40%) 이상이면 `declining=1`, 그 외 `growing=0`

## XGBoost 모델 (`code/model_xgb.py`)

```python
XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.5, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss"
)
```

- **클래스 불균형 보정**: `compute_sample_weight(class_weight="balanced")` 로 declining 가중치 상향
- **stratify split** 8:2 — 라벨 비율 유지

## Threshold 튜닝 (recall 우선)

- 0.20 ~ 0.70 (0.05 step) 임계값 비교 (`precision / recall / F1 / balanced acc / overall acc`)
- **선택 기준**: `declining_recall ≥ 0.65` 조건 하에 declining F1 최대 (쇠퇴 누락이 더 큰 비용이라는 도메인 판단)
- 후보 없으면 recall 최대로 fallback

## 결과 출력

- Threshold 비교 표 (전체 임계값 5개 지표)
- 최종 `classification_report`
- Confusion Matrix (`growing` vs `declining`)
- Feature Importance (Gain 기준)

## 실행

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
# 데이터를 data/ 에 배치
python code/data_preprocess.py   # 5년치 CSV 통합 → input.csv
python code/model_xgb.py         # 학습 → threshold tuning → 평가 → 시각화
```

`code/서울시_상권분석_코드.py` 는 위 둘을 합친 단일 파이프라인 버전.

## 배운 점

- 모델 정확도(accuracy)가 아니라 **'무엇을 놓치면 안 되는가'** 를 먼저 정의하고 거기에 맞춰 임계값을 튜닝해야 함 — 쇠퇴 탐지에서는 recall 우선
- 공공데이터는 `cp949` 인코딩·연도별 칼럼명 차이 같은 잡일이 많아 전처리 단계 분리가 필수
- 점포 절대 수보다 `avg_sales`(점포당 매출), `density`(점포/유동인구) 같은 비율형 파생변수가 모델에서 더 큰 정보를 제공
- threshold tuning 시 recall 하한 + F1 최대라는 다단계 선택 규칙으로 단일 지표에 끌려가지 않음
