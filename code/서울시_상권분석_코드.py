import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, plot_importance



# 점포 데이터 연도별 병합
store_df_2020 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(점포-상권)_2020년.csv', encoding= 'cp949')
store_df_2021 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(점포-상권)_2021년.csv', encoding= 'cp949')
store_df_2022 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(점포-상권)_2022년.csv', encoding= 'cp949')
store_df_2023 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(점포-상권)_2023년.csv', encoding= 'cp949')
store_df_2024 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(점포-상권)_2024년.csv', encoding= 'cp949')

stores = pd.concat([store_df_2020, store_df_2021, store_df_2022, store_df_2023, store_df_2024])

# 매출 데이터 연도별 병합
sales_df_2020 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(추정매출-상권)_2020년.csv', encoding= 'cp949')
sales_df_2021 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(추정매출-상권)_2021년.csv', encoding= 'cp949')
sales_df_2022 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(추정매출-상권)_2022년.csv', encoding= 'cp949')
sales_df_2023 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(추정매출-상권)_2023년.csv', encoding= 'cp949')
sales_df_2024 = pd.read_csv('Personal_Statistic/data/서울시_상권분석서비스(추정매출-상권)_2024년.csv', encoding= 'cp949')

sales = pd.concat([sales_df_2020, sales_df_2021, sales_df_2022, sales_df_2023, sales_df_2024])

# join을 위한 칼럼 설정
index_col = ['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명']
sales_col = ['당월_매출_건수']
stores_col = ['점포_수', '폐업_점포_수', '폐업_률']

# 점포 칼럼 정리 후 저장
storecol = index_col + stores_col
stores = stores[storecol]
stores.to_csv('Personal_Statistic/stores.csv', index=False, encoding='utf-8-sig')

# 매출 칼럼 정리 후 저장
salescol = index_col + sales_col
sales = sales[salescol]
sales.to_csv('Personal_Statistic/sales.csv', index=False, encoding='utf-8-sig')

# 유동인구 칼럼 정리 후 저장
population= pd.read_csv('Personal_Statistic/data/서울시 상권분석서비스(길단위인구-상권).csv', encoding= 'cp949')
floating_col = ['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명', '총_유동인구_수']
population = population[floating_col]
population.to_csv('Personal_Statistic/population.csv', index=False, encoding='utf-8-sig')


# 정리한 데이터 불러오기
population = pd.read_csv('Personal_Statistic/population.csv')
sales = pd.read_csv('Personal_Statistic/sales.csv')
stores = pd.read_csv('Personal_Statistic/stores.csv')

index_col = ['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명']

# 점포, 매출 데이터 병합
df_store_sales = pd.merge(stores, sales, on=index_col, how='left')
df_store_sales = df_store_sales.fillna(0)

# 상권/분기별 각 칼럼 합계
df_commercial_stats = df_store_sales.groupby(index_col).agg(
    총_점포수=('점포_수', 'sum'),         
    총_폐업점포수=('폐업_점포_수', 'sum'),
    평균_폐업률=('폐업_률', 'mean'),    
    월_매출건수=('당월_매출_건수', 'sum')       
).reset_index()

# 유동인구 데이터와 병합
df = pd.merge(population, df_commercial_stats, on=index_col, how='left')

# 전처리된 데이터 저장
df.to_csv('Personal_Statistic/df.csv', index=False, encoding='utf-8-sig')

# 전처리된 데이터 불러오기
df1 = pd.read_csv('Personal_Statistic/df.csv')
# 결측치 제거
df = df1.copy().dropna()

# 변수 설정
# 1. 점포당 매출 
df['avg_sales'] = df['월_매출건수'] / df['총_점포수']
# 2. 유동인구 
df["log_pop"] = np.log(df["총_유동인구_수"] + 1)
# 3. 점포 밀도 
df['density'] = df['총_점포수'] / df['총_유동인구_수']
# 4. 폐업률 
df['fail_rate'] = df['평균_폐업률']
# 5. 매출 변화율 
df = df.sort_values(['상권_코드', '기준_년분기_코드'])
df['sales_growth'] = df.groupby('상권_코드')['월_매출건수'].pct_change()
# 결측치 처리 
df['sales_growth'] = df['sales_growth'].fillna(0)



# 모델에 사용할 데이터 칼럼 정리 및 저장
input = df[['기준_년분기_코드', '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명',
           'avg_sales', 'log_pop', 'density', 'fail_rate', 'sales_growth']]
input.to_csv('Personal_Statistic/input.csv', index=False, encoding='utf-8-sig')




###########################################################################################
"=========================================================================================="
###########################################################################################



# 전처리된 입력 데이터 불러오기
df = pd.read_csv("Personal_Statistic/input.csv")

# inf/-inf를 결측치로 치환 (sales_growth 계산 시 발생 가능)
df = df.replace([np.inf, -np.inf], np.nan)

# 모델에 필요한 컬럼만 선택
required_cols = ["avg_sales", "log_pop", "density", "sales_growth", "fail_rate"]
df = df[required_cols].copy()

# 결측치 제거
df = df.dropna().copy()


# 라벨 생성: fail_rate 0.6 분위수 기준으로 declining / growing 이진분류
# 상위 40% 폐업률 = declining(쇠퇴), 나머지 = growing(비쇠퇴)
fail_thr = df["fail_rate"].quantile(0.6)
df["label"] = np.where(df["fail_rate"] >= fail_thr, "declining", "growing")

# 문자열 라벨을 숫자로 변환 (growing=0, declining=1)
label_map = {"growing": 0, "declining": 1}
inverse_label_map = {0: "growing", 1: "declining"}
df["target"] = df["label"].map(label_map)


# 입력 변수(X)와 타겟(y) 설정
feature_cols = ["avg_sales", "log_pop", "density", "sales_growth"]
X = df[feature_cols].copy()
y = df["target"].copy()

# 학습/테스트 데이터 분할 (8:2, stratify로 라벨 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 클래스 불균형 보정: declining 클래스에 더 높은 가중치 부여
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)


# XGBoost 모델 정의
model = XGBClassifier(
    n_estimators=300,       # 트리 300개 순차 학습
    max_depth=3,            # 트리 깊이 3: 과적합 방지
    learning_rate=0.03,     # 낮은 학습률로 점진적 수렴
    subsample=0.8,          # 80% 샘플만 사용
    colsample_bytree=0.8,   # 80% 변수만 사용
    min_child_weight=3,     # 리프 노드 최소 가중치
    reg_alpha=0.5,          # L1 정규화
    reg_lambda=1.0,         # L2 정규화
    objective="binary:logistic",
    random_state=42,
    eval_metric="logloss"
)

# 모델 학습
model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)


# 예측 확률 추출: [:, 1]은 declining으로 분류될 확률
y_prob = model.predict_proba(X_test)[:, 1]


# Threshold 비교: 쇠퇴 탐지 목적이므로 declining recall 우선
# threshold를 낮추면 recall 증가, precision 감소
thresholds = np.arange(0.20, 0.71, 0.05)
results = []

for th in thresholds:
    y_pred_th = (y_prob >= th).astype(int)
    results.append({
        "threshold": round(th, 2),
        "declining_precision": precision_score(y_test, y_pred_th, zero_division=0),
        "declining_recall": recall_score(y_test, y_pred_th, zero_division=0),
        "declining_f1": f1_score(y_test, y_pred_th, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_th),
        "overall_accuracy": (y_test == y_pred_th).mean()
    })

result_df = pd.DataFrame(results)
print("=== Threshold Comparison ===")
print(result_df.round(4))


# 최적 threshold 선택: recall >= 0.65 조건 하에 declining F1이 가장 큰 값
candidate_df = result_df[result_df["declining_recall"] >= 0.65].copy()

# 후보가 없으면 recall 최대 기준으로 fallback
if len(candidate_df) == 0:
    candidate_df = result_df.copy()

best_row = candidate_df.sort_values(
    by=["declining_f1", "balanced_accuracy", "declining_precision"],
    ascending=[False, False, False]
).iloc[0]

best_threshold = float(best_row["threshold"])


# 최적 threshold로 최종 예측
y_pred = (y_prob >= best_threshold).astype(int)

# 최종 성능 출력
print("=== Final Classification Report ===")
print(classification_report(
    y_test,
    y_pred,
    target_names=["growing", "declining"],
    zero_division=0
))


# Confusion Matrix 시각화
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["growing", "declining"]
)
disp.plot(cmap="Blues")
plt.title(f"Growing vs Declining (threshold={best_threshold})")
plt.tight_layout()
plt.show()

# Feature Importance 시각화 (Gain 기준)
plt.figure(figsize=(8, 5))
plot_importance(model, importance_type="gain", show_values=False)
plt.title("Feature Importance (Gain)")
plt.tight_layout()
plt.show()