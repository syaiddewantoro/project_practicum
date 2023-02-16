# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

# ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor

import streamlit as st
import io


def kode(codes):
    st.code(codes, language='python')

def buffer(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    detail = buffer.getvalue()
    return st.text(detail)


st.title('Find the Best Place for a New Well')


st.markdown('''# Table of Contents

* [OilyGiant](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=VqwilnTpNjII)
* [1. Initialization](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=f216a9d6)
* [2. Data Preparation](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=df6691a7)
    * [2.1 Memuat Semua Data dan Menampilkan Sample Data](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=RQMUi6I0ZLIp)
    * [2.2 Print the Information of Datasets](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=bfe21e52-6526-41a6-8e80-7ad55c356d8f&line=1&uniqifier=1)
    * [2.3 Menampilkan Statistik Deskriptif dari Data](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=Hez-TZEKaFIb)
* [3. EDA and Data Visualization](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=ck0zHdn6lNO2&line=5&uniqifier=1)
    * [3.1 Region 0](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=bmQM2sXZZFDN&line=1&uniqifier=1)
    * [3.2 Region 1](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=Til7usIYZI2r&line=1&uniqifier=1)
    * [3.3 Region 2](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=LpyIObMkZNJ-&line=1&uniqifier=1)
* [4. Split the Data](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=JNogScXo4HeA&line=2&uniqifier=1)
* [5. Create Model](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=xGCOQVD_hz5-)
    * [5.1 Linear Regression](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=levZlEbkh4l0)
* [6. Calculate the Profit](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=dxrQ4R3Wrrif)
    
* [Consclusions](https://colab.research.google.com/drive/1cITvek3ekCpfzXM8U975nBoTeSi95krp#scrollTo=ODjbHKLCbc7F&line=23&uniqifier=1)

# OilyGiant

We work for the OilyGiant mining company. Our task is to find the best place for a new well.

Steps to choose the location:

- Collecting the oil well parameters in the selected region: oil quality and volume of reserves;
- Build a model for predicting the volume of reserves in the new wells;
- Pick the oil wells with the highest estimated values;
- Pick the region with the highest total profit for the selected oil wells.

We have data on oil samples from three regions. Parameters of each oil well in the region are already known. We are going to build a model that will help to pick the region with the highest profit margin. Analyze potential profit and risks using the *Bootstrapping* technique.

**Goals:**
- Collect the oil well parameters each regions: oil quality and volume of reserves.
- Build a model for predicting the volume of reserves in the new wells.
- Pick the oil wells with the highest estimated values.
- Pick the region with the highest total profit for the selected oil wells.

## 1. Initialization
''')


code1 = (''' 
# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

# ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
''')
kode(code1)


st.markdown('''## 2. Data Preparation

### 2.1. Memuat Semua Data dan Menampilkan Sample Data
''')


code2 = (''' 
# load datasets
try:
    df0 = pd.read_csv('/datasets/geo_data_0.csv')
    df1 = pd.read_csv('/datasets/geo_data_1.csv')
    df2 = pd.read_csv('/datasets/geo_data_2.csv')
    
except:
    try:
        df0 = pd.read_csv('/home/syaid/Downloads/Sprint9/geo_data_0.csv')
        df1 = pd.read_csv('/home/syaid/Downloads/Sprint9/geo_data_1.csv')
        df2 = pd.read_csv('/home/syaid/Downloads/Sprint9/geo_data_2.csv')

    except:
        df0 = pd.read_csv('/content/geo_data_0.csv')
        df1 = pd.read_csv('/content/geo_data_1.csv')
        df2 = pd.read_csv('/content/geo_data_2.csv')
''')
kode(code2)

df0 = pd.read_csv('/home/syaid/Downloads/Sprint9/geo_data_0.csv')
df1 = pd.read_csv('/home/syaid/Downloads/Sprint9/geo_data_1.csv')
df2 = pd.read_csv('/home/syaid/Downloads/Sprint9/geo_data_2.csv')
    

code3 = (''' 
# menampilkan sample data df0
df0
''')
kode(code3)

df0


code4 = (''' 
# menampilkan sample data df1
df1
''')
kode(code4)

df1


code5 = (''' 
# menampilkan sample data df2
df2
''')
kode(code5)

df2


st.markdown('''### 2.2. Print the Information of Datasets
''')


code6 = (''' 
# menampilkan informasi df0
df0.info()
''')
kode(code6)

buffer(df0)


code7 = (''' 
# menampilkan informasi df1
df1.info()
''')
kode(code7)

buffer(df1)


code8 = (''' 
# menampilkan informasi df2
df2.info()
''')
kode(code8)

buffer(df2)


st.markdown('''- Ketiga dataset diatas memiliki jumlah kolom dan baris yang sama yaitu **5** kolom dan **100.000** baris.
- Tidak ada *missing value* ditemukan pada ketiga dataset tersebut.
- Tipe data data pada kolom-kolom sudah terdefinisi dengan benar, kolom `id` dedefinisikan sebagai **object** sedangkan kolom `f0`, `f1`, `f2`, dan `product` didefinisikan sebagai **float**.

### 2.3. Menampilkan Statistik Deskriptif dari Data
''')


code9 = (''' 
# menampilkan statistik deskriptif df0
df0.describe()
''')
kode(code9)

st.write(df0.describe()
)


code10 = (''' 
# menampilkan statistik deskriptif df1
df1.describe()
''')
kode(code10)

st.write(df1.describe()
)


code11 = (''' 
# menampilkan statistik deskriptif df2
df2.describe()
''')
kode(code11)

st.write(df2.describe()
)


st.markdown('''## 3. EDA and Data Visualization

### 3.1. Region 0
''')


code12 = (''' 
# menampilkan visualisasi df0
columns = ['f0', 'f1', 'f2', 'product']

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Histograms Distribution of the df0', fontsize=18, y=0.95)

# loop through the length of tickers and keep track of index
for n, column in enumerate(columns):
    # add a new subplot iteratively
    ax = plt.subplot(2, 2, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.histplot(x=df0[column], kde=True, ax=ax)

    # chart formatting
    ax.set_title(column.upper())
    ax.set_xlabel('')
''')
kode(code12)

sns.set()

columns = ['f0', 'f1', 'f2', 'product']

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Histograms Distribution of the df0', fontsize=18, y=0.95)

# loop through the length of tickers and keep track of index
for n, column in enumerate(columns):
    # add a new subplot iteratively
    ax = plt.subplot(2, 2, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.histplot(x=df0[column], kde=True, ax=ax)

    # chart formatting
    ax.set_title(column.upper())
    ax.set_xlabel('')
st.pyplot(fig)


st.markdown('''**Findings:**
- Kolom `f0`, `f1`, dan `product` memiliki bentuk multimodal, artinya kolom-kolom tersebut memiliki 2 atau lebih *mode*.
- Kolom `f2` memiliki distribusi normal.

### 3.2. Region 1
''')


code13 = (''' 
# menampilkan visualisasi df1
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Histograms Distribution of the df1', fontsize=18, y=0.95)

# loop through the length of tickers and keep track of index
for n, column in enumerate(columns):
    # add a new subplot iteratively
    ax = plt.subplot(2, 2, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.histplot(x=df1[column], kde=True, ax=ax)

    # chart formatting
    ax.set_title(column.upper())
    ax.set_xlabel('')
''')
kode(code13)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Histograms Distribution of the df1', fontsize=18, y=0.95)

# loop through the length of tickers and keep track of index
for n, column in enumerate(columns):
    # add a new subplot iteratively
    ax = plt.subplot(2, 2, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.histplot(x=df1[column], kde=True, ax=ax)

    # chart formatting
    ax.set_title(column.upper())
    ax.set_xlabel('')
st.pyplot(fig)


st.markdown('''**Findings:**
- Kolom `f0` pada dataset df1 memiliki distribusi bimodal, atau memiliki dua *mode*.
- Kolom `f1` pada dataset df1 memiliki distibusi normal.
- Kolom `f2` dan `product` pada dataset df1 memiiliki distribusi abnormal.


### 3.3. Region 2
''')


code14 = (''' 
# menampilkan visualisasi df1
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Histograms Distribution of the df2', fontsize=18, y=0.95)

# loop through the length of tickers and keep track of index
for n, column in enumerate(columns):
    # add a new subplot iteratively
    ax = plt.subplot(2, 2, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.histplot(x=df2[column], kde=True, ax=ax)

    # chart formatting
    ax.set_title(column.upper())
    ax.set_xlabel('')
''')
kode(code14)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('Histograms Distribution of the df2', fontsize=18, y=0.95)

# loop through the length of tickers and keep track of index
for n, column in enumerate(columns):
    # add a new subplot iteratively
    ax = plt.subplot(2, 2, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.histplot(x=df2[column], kde=True, ax=ax)

    # chart formatting
    ax.set_title(column.upper())
    ax.set_xlabel('')
st.pyplot(fig)


st.markdown('''**Findings:**
- Kolom `f0`, `f1`, dan `f2`, memiliki distribusi multimodal.
- Kolom `product` memiliki distribusi normal.

## 4. Split the Data

Kita akan membagi data menjadi dua set yaitu *train set* dan *validation set* dengan rasio **75 : 25**. 

Kita akan melatih dengan model Linear Regression untuk menghitung nilai **RMSE** dan menemukan cadangan sumur minyak baru.
''')


code15 = (''' 
# split the datasets
datasets = [df0, df1, df2]

X_train = []
X_valid = []
y_train = []
y_valid = []
''')
kode(code15)

datasets = [df0, df1, df2]

X_train = []
X_valid = []
y_train = []
y_valid = []


code16 = (''' 
# create for loop function to split data
for data in datasets:
        X = data[data.columns.drop(['id', 'product'])]
        y = data['product']
        features_train, features_valid, target_train, target_valid = train_test_split(X, y, test_size = 0.25, 
                                                              random_state=42)
        X_train.append(features_train)
        X_valid.append(features_valid)
        
        y_train.append(target_train)
        y_valid.append(target_valid)
''')
kode(code16)

for data in datasets:
        X = data[data.columns.drop(['id', 'product'])]
        y = data['product']
        features_train, features_valid, target_train, target_valid = train_test_split(X, y, test_size = 0.25, 
                                                              random_state=42)
        X_train.append(features_train)
        X_valid.append(features_valid)
        
        y_train.append(target_train)
        y_valid.append(target_valid)


code17 = (''' 
# display the X_train sample
for region in range(len(datasets)):
    st.write(X_train[region])
''')
kode(code17)

for region in range(len(datasets)):
    st.write(X_train[region])


code18 = (''' 
# display the y_valid sample
for region in range(len(datasets)):
    st.write(y_valid[region])
''')
kode(code18)

for region in range(len(datasets)):
    st.write(y_valid[region])


st.markdown('''## 5. Create Model

### 5.1. Linear Regression
''')


code19 = (''' 
# create RMSE function
def rmse_score(x, y):
    square_root = sqrt(mean_squared_error(x, y))
    return square_root
''')
kode(code19)

def rmse_score(x, y):
    square_root = sqrt(mean_squared_error(x, y))
    return square_root


code20 = (''' 
# create linear regression model function
lr_results = defaultdict(list)

predictions_valid = [] # for profit calculation

for region in range(len(datasets)):
    model = LinearRegression()
    model.fit(X_train[region], y_train[region])

    y_pred_train = model.predict(X_train[region])
    y_pred_valid = model.predict(X_valid[region])
    
    predictions_valid.append(pd.Series(y_pred_valid))

    lr_results['region'].append(region)
    lr_results['rmse_train'].append(rmse_score(y_train[region], y_pred_train))
    lr_results['rmse_valid'].append(rmse_score(y_valid[region], y_pred_valid))
    lr_results['avg_train'].append(y_train[region].mean())
    lr_results['avg_valid'].append(y_valid[region].mean()) 
''')
kode(code20)

lr_results = defaultdict(list)

predictions_valid = [] # for profit calculation

for region in range(len(datasets)):
    model = LinearRegression()
    model.fit(X_train[region], y_train[region])

    y_pred_train = model.predict(X_train[region])
    y_pred_valid = model.predict(X_valid[region])
    
    predictions_valid.append(pd.Series(y_pred_valid))

    lr_results['region'].append(region)
    lr_results['rmse_train'].append(rmse_score(y_train[region], y_pred_train))
    lr_results['rmse_valid'].append(rmse_score(y_valid[region], y_pred_valid))
    lr_results['avg_train'].append(y_train[region].mean())
    lr_results['avg_valid'].append(y_valid[region].mean()) 


code21 = (''' 
# print the result of linear regression model
lr_results = pd.DataFrame(lr_results)
lr_results
''')
kode(code21)

lr_results = pd.DataFrame(lr_results)
lr_results


code22 = (''' 
# create pie chart to show the results
palette_color = sns.color_palette('pastel')[0:5]
label = ['Region 0', 'Region 1', 'Region 2']

fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2);
plt.subplots_adjust(wspace=0.5)

ax[0].pie(lr_results['rmse_valid'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(lr_results["rmse_valid"]):,.3f})')
ax[0].set_title('rmse_valid')
ax[1].pie(lr_results['avg_valid'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(lr_results["avg_valid"]):,.3f})')
ax[1].set_title('avg_valid')

plt.legend(loc='best', bbox_to_anchor=(1,0), title='Region')
plt.suptitle('Linear Regression Results')
st.pyplot(fig)
''')
kode(code22)

palette_color = sns.color_palette('pastel')[0:5]
label = ['Region 0', 'Region 1', 'Region 2']

fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2);
plt.subplots_adjust(wspace=0.5)

ax[0].pie(lr_results['rmse_valid'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(lr_results["rmse_valid"]):,.3f})')
ax[0].set_title('rmse_valid')
ax[1].pie(lr_results['avg_valid'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(lr_results["avg_valid"]):,.3f})')
ax[1].set_title('avg_valid')

plt.legend(loc='best', bbox_to_anchor=(1,0), title='Region')
plt.suptitle('Linear Regression Results')
st.pyplot(fig)


st.markdown('''**Temuan:**
- Kita bisa melihat bahwa nilai **RMSE** dari *train set* dan *validation set* memiliki *score* yang mirip, artinya tidak terjadi *underfiting* maupun *overfitting* pada kedua data tersebut.
- Kita dapat mempertimbangkan **region 2** untuk melakukan pengembangan karena memiliki nilai rata-rata cadangan tertinggi sebesar **95.150999**, tetapi perlu diingat bahwa wilayah tersebut juga memiliki **error** yang paling tinggi sebesar **40.145872**.

Kita perlu melakukan analisa lebih lanjut untuk dapat menentukan probabilitas profit maupun kerugian yang dapat terjadi. 

## 6. Caclulate the Profit
''')


code23 = (''' 
# counting the number of products to gain 0 profit

budget = 100000000 # anggaran
wells_to_dig = 200 # jumlah sumur untuk digali

cost_per_well = budget / wells_to_dig # biaya menggali satu sumur
points_per_budget = budget // cost_per_well # point per budget

product_price = 4500 # harga per produk = 1000 barrels

cost_per_point = budget / points_per_budget # biaya per point
zero_profit_product = cost_per_point / product_price # produk dibutuhkan untuk menghasilkan 0 profit

st.write('Jumlah produk yang dibutuhkan untuk menghasilkan profit $0 =', zero_profit_product)
''')
kode(code23)

budget = 100000000 # anggaran
wells_to_dig = 200 # jumlah sumur untuk digali

cost_per_well = budget / wells_to_dig # biaya menggali satu sumur
points_per_budget = budget // cost_per_well # point per budget

product_price = 4500 # harga per produk = 1000 barrels

cost_per_point = budget / points_per_budget # biaya per point
zero_profit_product = cost_per_point / product_price # produk dibutuhkan untuk menghasilkan 0 profit

st.write('Jumlah produk yang dibutuhkan untuk menghasilkan profit $0 =', zero_profit_product)


st.markdown('''**Temuan:**
- Produk yang dibutuhkan untuk untuk mendapatkan *Return of Investment* adalah sebanyak **111** produk atau **111.000** *barrels*.
- Pada analisa sebelumnya kita mendapatkan bahwa **Region 2** memiliki rata-rata cadangan minyak tertinggi dengan **95** produk yang bisa dihasilkan, tetapi jumlah tersebut masih jauh untuk mencukupi kebutuhan dalam menghasilkan profit **0**.
''')


code24 = (''' 
# create fuction to calculate the profit7

def profit(target, predictions):
    predictions_sorted = predictions.reset_index(drop=True).sort_values(ascending=False) # hasil prediksi jumlah produk validation set
    selected_points = target.reset_index(drop=True).iloc[predictions_sorted.index][:200] # jumlah produk validation set
    product = selected_points.sum() # jumlah produk dalam validation set
    revenue = product * product_price # pendapatan = produk × harga produk
    cost = budget
    
    return revenue - cost # pendapatan - biaya
''')
kode(code24)

def profit(target, predictions):
    predictions_sorted = predictions.reset_index(drop=True).sort_values(ascending=False) # hasil prediksi jumlah produk validation set
    selected_points = target.reset_index(drop=True).iloc[predictions_sorted.index][:200] # jumlah produk validation set
    product = selected_points.sum() # jumlah produk dalam validation set
    revenue = product * product_price # pendapatan = produk × harga produk
    cost = budget
    
    return revenue - cost # pendapatan - biaya


code25 = (''' 
# calculate the profit without bootstrapping
profit_outcome = defaultdict(list)

for region in range(len(datasets)):
    profit_outcome['region'].append(region)
    profit_outcome['profit'].append(profit(y_valid[region], predictions_valid[region]))

pd.set_option('display.float_format',  '{:,f}'.format)
pd.DataFrame(profit_outcome)
''')
kode(code25)

profit_outcome = defaultdict(list)

for region in range(len(datasets)):
    profit_outcome['region'].append(region)
    profit_outcome['profit'].append(profit(y_valid[region], predictions_valid[region]))

pd.set_option('display.float_format',  '{:,f}'.format)
st.write(pd.DataFrame(profit_outcome)
)


st.markdown('''**Temuan:**
- Tanpa **bootstrapping** laba tertinggi diperoleh pada **Region 0**.
''')


code26 = (''' 
# create boostrap function to calculet the profit

state = np.random.RandomState(42)
profit_boost = defaultdict(list)

for region in range(3):
    target = y_valid[region]
    predictions = predictions_valid[region]
    
    profit_values = []
    for i in range(1000):
        target_subsample = target.reset_index(drop=True).sample(n=500, replace=True, random_state=state)
        predicted_valid_subsample = pd.Series(predictions).iloc[target_subsample.index]
        profit_values.append(profit(target_subsample, predicted_valid_subsample))
    profit_values = pd.Series(profit_values)
        
    mean_profit = profit_values.mean()
    lower_quantile = profit_values.quantile(0.025)
    upper_quantile = profit_values.quantile(0.975)
    negative_profit_chance = (profit_values < 0).mean() * 100 
  
    profit_boost['region'].append(region)
    profit_boost['avg_profit'].append(mean_profit)
    profit_boost['lower_quantile'].append(lower_quantile)
    profit_boost['upper_quantile'].append(upper_quantile)
    profit_boost['risk_of_losses'].append(negative_profit_chance)
''')
kode(code26)

state = np.random.RandomState(42)
profit_boost = defaultdict(list)

for region in range(3):
    target = y_valid[region]
    predictions = predictions_valid[region]
    
    profit_values = []
    for i in range(1000):
        target_subsample = target.reset_index(drop=True).sample(n=500, replace=True, random_state=state)
        predicted_valid_subsample = pd.Series(predictions).iloc[target_subsample.index]
        profit_values.append(profit(target_subsample, predicted_valid_subsample))
    profit_values = pd.Series(profit_values)
        
    mean_profit = profit_values.mean()
    lower_quantile = profit_values.quantile(0.025)
    upper_quantile = profit_values.quantile(0.975)
    negative_profit_chance = (profit_values < 0).mean() * 100 
  
    profit_boost['region'].append(region)
    profit_boost['avg_profit'].append(mean_profit)
    profit_boost['lower_quantile'].append(lower_quantile)
    profit_boost['upper_quantile'].append(upper_quantile)
    profit_boost['risk_of_losses'].append(negative_profit_chance)


code27 = (''' 
# show the profit and risk of losses results
pd.set_option('display.float_format',  '{:,.2f}'.format)
profit_boost = pd.DataFrame(profit_boost)
profit_boost
''')
kode(code27)

pd.set_option('display.float_format',  '{:,.2f}'.format)
profit_boost = pd.DataFrame(profit_boost)
profit_boost


code28 = (''' 
# create pie chart to show the profit results
palette_color = sns.color_palette('pastel')[0:5]
label = ['Region 0', 'Region 1', 'Region 2']

fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2);
plt.subplots_adjust(wspace=0.5)

ax[0].pie(profit_boost['avg_profit'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(profit_boost["avg_profit"]):,.3f})')
ax[0].set_title('avg_profit')
ax[1].pie(profit_boost['risk_of_losses'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(profit_boost["risk_of_losses"]):,.3f})')
ax[1].set_title('risk_of_losses')

plt.legend(loc='best', bbox_to_anchor=(1,0), title='Region')
plt.suptitle('Profit and Risk of Loss Results')
plt.show()
''')
kode(code28)

# create pie chart to show the profit results
palette_color = sns.color_palette('pastel')[0:5]
label = ['Region 0', 'Region 1', 'Region 2']

fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2);
plt.subplots_adjust(wspace=0.5)

ax[0].pie(profit_boost['avg_profit'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(profit_boost["avg_profit"]):,.3f})')
ax[0].set_title('avg_profit')
ax[1].pie(profit_boost['risk_of_losses'], labels=label, 
            autopct=lambda x: f'{x:.1f}%\n({(x/100)*sum(profit_boost["risk_of_losses"]):,.3f})')
ax[1].set_title('risk_of_losses')

plt.legend(loc='best', bbox_to_anchor=(1,0), title='Region')
plt.suptitle('Profit and Risk of Loss Results')
st.pyplot(fig)


st.markdown('''**Temuan:**
- Hasil dari teknik *boostrapping* yang kita lakukan untuk kita menetapkan nilai untuk subsampel sebesar **1.000** dan nilai *confidence interval* sebesar **0.90%**  untuk menghitung rata-rata profit, *confidence interval* dan *risk of losses*, didapatkan wilayah terbaik untuk lokasi pembangunan sumur baru pada **Region 2**. 
- Pada wilayah tersebut didapatkan rata-rata profit tertinggi sebesar **4.525.765**, resiko kerugian terendah dengan **0.90%** dan dibawah nilai ambang batas yang ditentukan sebesar **2.5%**.
- Wilayah tersebut juga memiliki rentang *confidence interval* paling rendah dengan tidak ada nilai minus pada *lower quantile*-nya antara **523.094** sampai **8.301.463** untuk *upper quantile*.

# Consclusions

**1. Data Preparation**
- Kita memulai dengan memuat 3 dataset yang masing-masing terdiri dari **5** kolom dan **100.000** baris.
- Tidak ada *missing value* ditemukan pada ketiga dataset tersebut, tipe data data pada kolom-kolom sudah terdefinisi dengan benar, kolom `id` dedefinisikan sebagai **object** sedangkan kolom `f0`, `f1`, `f2`, dan `product` didefinisikan sebagai **float**.
- Pada **Region 2** kolom `f0`, `f1`, dan `f2`, memiliki distribusi multimodal, kolom `product` memiliki distribusi normal.

**2. EDA and Data Visualization**
- Pada **Region 0** kolom `f0`, `f1`, dan `product` memiliki bentuk multimodal, artinya kolom-kolom tersebut memiliki 2 atau lebih *mode*, kolom `f2` memiliki distribusi normal..
- Pada **Region 1** kolom `f0` pada dataset df1 memiliki distribusi bimodal, kolom `f1` pada dataset df1 memiliki distibusi normal, kolom `f2` dan `product` pada dataset df1 memiiliki distibusi abnormal.

**3. Split the Data**
- Kita akan membagi data menjadi dua set yaitu *train set* dan *validation set* dengan rasio **75 : 25**. 
- Kita menggunakan kolom `f0`, `f1`, dan `f2` sebagai *features set* dan kolom `product` sebagai *target set*, kita juga melakukan drop pada kolom `id` yang tidak diperlukan untuk membuat model.

**4. Model**
- Kita bisa mendapatkan hasil nilai **RMSE** dari *train set* dan *validation set* memiliki *score* yang mirip, artinya tidak terjadi *underfiting* maupun *overfitting* pada kedua data tersebut.
- Kita akan mempertimbangkan **region 2** untuk melakukan pengembangan karena memiliki nilai rata-rata cadangan tertinggi, tetapi perlu diingat bahwa wilayah tersebut juga memiliki **error** yang paling tinggi.

**5. Calculate the Profit**
- Hasil dari teknik *boostrapping* yang kita lakukan untuk menghitung rata-rata profit, *confidence interval* dan *risk of losses*, didapatkan wilayah terbaik untuk lokasi pembangunan sumur baru pada **Region 2**. 
- Pada wilayah tersebut didapatkan rata-rata profit tertinggi sebesar **4.525.765**, resiko kerugian terendah dengan **0.90%** dan dibawah nilai ambang batas yang ditentukan.
- Wilayah tersebut juga memiliki rentang *confidence interval* paling rendah dengan tidak ada nilai minus pada *lower quantile*-nya.


**Main Consclusion**

Kita menemukan wilyah terbaik untuk melakukan pengembangan sumur baru pada **Region 1**. Pada wilayah ini kita mendapatkan hasil rata-rata profit tertinggi dan resiko kerugian paling kecil.
''')