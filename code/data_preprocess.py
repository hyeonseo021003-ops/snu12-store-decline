import pandas as pd
import numpy as np


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