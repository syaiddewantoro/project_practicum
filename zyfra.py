import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import sidetable as stb

# model libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.datasets import make_classification
from math import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, make_scorer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import streamlit as st
import io


def kode(codes):
    st.code(codes, language='python')

def buffer(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    detail = buffer.getvalue()
    return st.text(detail)


st.title('Predict the Amount of Gold Recovered')


st.markdown('''# Table of Contents

* [Table of Contents](#scrollTo=yWeNjdXc54LP)

* [Zyfra](#scrollTo=BELuEq6BnW9J)

    * [Initialization](#scrollTo=D3ltqRpjDqZc)

    * [Data Preparation](#scrollTo=YmNzfIkCDwtn)

        * [Learn Data](#scrollTo=fySzvJFFoLob)

            * [Load all Datasets and Print the Sample of Data](#scrollTo=3DjG_CRlCmW4)

            * [Show the Information of Datasets](#scrollTo=0uBEhRARCv3t)

            * [Checking Missing Values](#scrollTo=63pfX9niDVvo)

        * [Recovery Calculation](#scrollTo=UX_xRylIcVgC)

        * [Analyze the Feature not Available in the Test Set](#scrollTo=pn464iFQfmQu)

        * [Perform Data Preprocessing](#scrollTo=dqn1d_eUoWkT)

            * [Fill Missing Values](#scrollTo=dEdwAmRBYLfg)

            * [Fix date Column](#scrollTo=lcfQlN6iYSH6)

    * [Data Analytics and Data Visualization](#scrollTo=-HRrO7Vy_bvJ)

        * [The Changes in Metal Concentration](#scrollTo=HLG-KuAWwiW4)

            * [Gold (Au)](#scrollTo=xehYDp9-q_OP)

            * [Silver (Ag)](#scrollTo=SVW2F-iHrbX1)

            * [Lead/Timbal (Pb)](#scrollTo=65p-PZHluBYB)

        * [Compare the Feed Particle Size](#scrollTo=HG_GWjz_RxRj)

        * [Distribution the total Concentrations](#scrollTo=8CFzzvPGbLCq)

            * [Raw Feed](#scrollTo=M_2uXVnh0wOL)

            * [Rougher Output Concentrate](#scrollTo=GB_yeYwF00GM)

            * [Final Output Concentrate](#scrollTo=eorjDT1c2Gh5)

            * [Remove Anomali Values](#scrollTo=IE0CxlibJe78)

    * [Build the Model](#scrollTo=jbI6c1zNPBYk)

        * [Calculate the Final sMAPE](#scrollTo=vZ13a12cPF5b)

        * [Train the Models](#scrollTo=j2hLmfwkqr0e)

            * [Split the Data into Features and Target](#scrollTo=gO47_auCqvyo)

            * [Define Function to Return Model Score](#scrollTo=XpCcR_aT8NA0)

            * [Linear Regression](#scrollTo=XCJmZeta8Q5c)

            * [Decision Tree Model](#scrollTo=ten3H6Qk5CWn)

            * [Random Forest Model](#scrollTo=wuCKBC5ixPGZ)

            * [Lasso Regressor](#scrollTo=5OOPZb8h64Cz)

            * [Ridge Regressor](#scrollTo=MrB-a-kJKVWC)

            * [Neural Network](#scrollTo=iedwMYnKMxIr)

        * [Perform Best Model using Test Set](#scrollTo=Y3maZKFXPJP-)

* [Consclusions](#scrollTo=upaWagdBPaIX)



# Zyfra

Prepare a prototype of a machine learning model for Zyfra. The company develops efficiency solutions for heavy industry.
The model should predict the amount of gold recovered from gold ore. You have the data on extraction and purification.
The model will help to optimize the production and eliminate unprofitable parameters. You need to:
- Prepare the data;
- Perform data analysis;
- Develop and train a model.

To complete the project, you may want to use documentation from pandas, matplotlib, and sklearn.

## Initialization
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
import sidetable as stb

# model libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.datasets import make_classification
from math import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, make_scorer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
''')
kode(code1)


st.markdown('''## 1. Data Preparation

### 1.1. Learn Data

#### 1.1.1. Load all Datasets and Print the Sample of Data
''')


code2 = ('''
# load datasets
try:
    df_full = pd.read_csv('/datasets/gold_recovery_full.csv')
    df_train = pd.read_csv('/datasets/gold_recovery_train.csv')
    df_test = pd.read_csv('/datasets/gold_recovery_test.csv')
    
except:
    try:
        df_full = pd.read_csv('gold_recovery_full_new.csv')
        df_train = pd.read_csv('gold_recovery_train.csv')
        df_test = pd.read_csv('gold_recovery_test.csv')

    except:
        df_full = pd.read_csv('/content/gold_recovery_full_new.csv')
        df_train = pd.read_csv('/content/gold_recovery_train.csv')
        df_test = pd.read_csv('/content/gold_recovery_test.csv')
''')
kode(code2)

df_full = pd.read_csv('/home/syaid/Downloads/Sprint10/gold_recovery_full_new.csv')
df_train = pd.read_csv('/home/syaid/Downloads/Sprint10/gold_recovery_train.csv')
df_test = pd.read_csv('/home/syaid/Downloads/Sprint10/gold_recovery_test.csv')
    

code3 = ('''
# all data
datasets = [df_full, df_train, df_test]

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
''')
kode(code3)

datasets = [df_full, df_train, df_test]

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


st.markdown('''**Data Description**

**Technological process**
- `Rougher feed` — raw material
- `Rougher additions` (or reagent additions) — flotation reagents: Xanthate, Sulphate, Depressant
    - `Xanthate` — promoter or flotation activator;
    - `Sulphate` — sodium sulphide for this particular process;
    - `Depressant` — sodium silicate.
- `Rougher process` — flotation
- `Rougher tails` — product residues
- `Float banks` — flotation unit
- `Cleaner process` — purification
- `Rougher Au` — rougher gold concentrate
- `Final Au` — final gold concentrate

<p align="center">
    <img src="https://pictures.s3.yandex.net/resources/ore_1591699963.jpg" alt="drawing" width="600" height="390">
  </a>
</p>

**Parameters of stages**
- `air amount` — volume of air
- `fluid levels`
- `feed size` — feed particle size
- `feed rate`

''')


code4 = ('''
# print the sample of df_full
df_full.head()
''')
kode(code4)

st.write(df_full.head()
)


code5 = ('''
# print the sample of df_train
df_train.head()
''')
kode(code5)

st.write(df_train.head()
)


code6 = ('''
# print the sample of df_test
df_test.head()
''')
kode(code6)

st.write(df_test.head()
)


st.markdown('''#### 1.1.2. Show the Information of Datasets
''')

code7 = ('''
# show the information of df_full
df_full.info()
''')
kode(code7)

buffer(df_full)


code8 = ('''
# print the infortmation of df_train
df_train.info()
''')
kode(code8)

buffer(df_train)


code9 = ('''
# print the information of df_test
df_test.info()
''')
kode(code9)

buffer(df_test)


code10 = ('''
# show the descriptive statistic of datasets
for i in datasets:
    st.write(i.describe()), st.write()
''')
kode(code10)

for i in datasets:
    st.write(i.describe()), st.write()


code11 = ('''
# len the shape of datasets
for i in datasets:
    st.write(i.shape), st.write()
''')
kode(code11)

for i in datasets:
    st.write(i.shape), st.write()


st.markdown('''**Findings:**
- Kita memiliki **3** dataset yang terdapat perbedaan antara jumlah kolom pada `train set` yang berjumlah **87** dan `test set` yang berjumlah **53** kolom.

#### 1.1.3. Checking Missing Values
''')


code12 = ('''
# checking sd_full missing values
df_full.stb.missing()
''')
kode(code12)

st.write(df_full.stb.missing()
)


code13 = ('''
# checkking df_train missing values
df_train.stb.missing()
''')
kode(code13)

st.write(df_train.stb.missing()
)


code14 = ('''
# checking df_test missing values
df_test.stb.missing()
''')
kode(code14)

st.write(df_test.stb.missing()
)


code15 = ('''
df_train.shape[0] * df_train.shape[1]
''')
kode(code15)

st.write(df_train.shape[0] * df_train.shape[1]
)


st.markdown('''**Findings:**
- *Missing value* terbanyak ada pada datasets `df_train` pada kolom `rougher.output.recovery` sebanyak **13%** dari data.

### 1.2. Recovery Calculation

We need to simulate the process of recovering gold from gold ore.

Using the following formula to simulate the recovery process:

<p align="center">
    <img src="https://pictures.s3.yandex.net/resources/Recovery_1576238822_1589899219.jpg" alt="drawing" width="400" height="70">
  </a>
</p>

where:
- **C** — share of gold in the concentrate right after flotation (for finding the rougher concentrate recovery)/after purification (for finding the final concentrate recovery)
- **F** — share of gold in the feed before flotation (for finding the rougher concentrate recovery)/in the concentrate right after flotation (for finding the final concentrate recovery)
- **T** — share of gold in the rougher tails right after flotation (for finding the rougher concentrate recovery)/after purification (for finding the final concentrate recovery)
''')


code16 = ('''
# calculate the recovery
C = df_train['rougher.output.concentrate_au']
F = df_train['rougher.input.feed_au']
T = df_train['rougher.output.tail_au']

# formula
recovery = (C * (F - T)) / (F * (C - T)) * 100

rougher_output_recovery = df_train['rougher.output.recovery'].dropna()
recovery = recovery.iloc[rougher_output_recovery.index]

st.write('MAE:', mean_absolute_error(rougher_output_recovery, recovery))
''')
kode(code16)

C = df_train['rougher.output.concentrate_au']
F = df_train['rougher.input.feed_au']
T = df_train['rougher.output.tail_au']

# formula
recovery = (C * (F - T)) / (F * (C - T)) * 100

rougher_output_recovery = df_train['rougher.output.recovery'].dropna()
recovery = recovery.iloc[rougher_output_recovery.index]

st.write('MAE:', mean_absolute_error(rougher_output_recovery, recovery))


st.markdown('''### 1.3. Analyze the Feature not Available in the Test Set
''')


code17 = ('''
# find which columns are not in df_test
cols_not_in = df_train.columns.difference(df_test.columns)
list(cols_not_in)
''')
kode(code17)

cols_not_in = df_train.columns.difference(df_test.columns)
st.write(list(cols_not_in)
)


code18 = ('''
# count the  columns not shown in the test_set
st.write(len(cols_not_in)), st.write()
st.write(len(cols_not_in) / len(df_train.columns) * 100, '%')
''')
kode(code18)

st.write(len(cols_not_in)), st.write()
st.write(len(cols_not_in) / len(df_train.columns) * 100, '%')


st.markdown('''### 1.4. Perform Data Preprocessing

#### 1.4.1. Fill Missing Values

**Check the Missing Value in DataFrame**
''')


code19 = ('''
# data full isna
df_full[df_full.isna().any(axis=1)].head(10)
''')
kode(code19)

st.write(df_full[df_full.isna().any(axis=1)].head(10)
)


st.markdown('''Karena telah disebutkan dalam deskripsi data

    Data is indexed with the date and time of acquisition (date feature). 
    Parameters that are next to each other in terms of time are often similar.

Kita akan mangisi *mising value* dengan metode `KNN Imputer`.
''')


code20 = ('''
# create function to fill missing value
def fill_missing (dataset):
    df_date = pd.DataFrame(dataset.copy().loc[:, 'date']) # menyimpan kolom tanggal, karena masih diutuhkan
    drop_date = dataset.copy().drop(['date'], axis=1)

    knn_imputer = KNNImputer(n_neighbors=3) # model to fill missing values
    knn_imputer.fit(drop_date)
    data_trans = knn_imputer.transform(drop_date)

    data_new = pd.DataFrame(data_trans, columns=dataset.columns.drop(['date'])) # replace columns name
    data_new.insert(0, 'date', df_date.loc[:,'date']) # add the 'date' column

    return data_new
''')
kode(code20)

def fill_missing (dataset):
    df_date = pd.DataFrame(dataset.copy().loc[:, 'date']) # menyimpan kolom tanggal, karena masih diutuhkan
    drop_date = dataset.copy().drop(['date'], axis=1)

    knn_imputer = KNNImputer(n_neighbors=3) # model to fill missing values
    knn_imputer.fit(drop_date)
    data_trans = knn_imputer.transform(drop_date)

    data_new = pd.DataFrame(data_trans, columns=dataset.columns.drop(['date'])) # replace columns name
    data_new.insert(0, 'date', df_date.loc[:,'date']) # add the 'date' column

    return data_new


code21 = ('''
# perform the function and create new data_full
data_full = fill_missing(df_full)
data_full.head()
''')
kode(code21)

data_full = fill_missing(df_full)
st.write(data_full.head()
)


code22 = ('''
# # perform the function and create new data_train
data_train = fill_missing(df_train)
data_train.head()
''')
kode(code22)

data_train = fill_missing(df_train)
st.write(data_train.head()
)


code23 = ('''
# # perform the function and create new data_test
data_test = fill_missing(df_test)
data_test.head()
''')
kode(code23)

data_test = fill_missing(df_test)
st.write(data_test.head()
)


code24 = ('''
# checking missing values
datasets = [data_full, data_train, data_test]

for i in datasets:
    st.write(i.isna().sum().sum()), st.write()
''')
kode(code24)

datasets = [data_full, data_train, data_test]

for i in datasets:
    st.write(i.isna().sum().sum()), st.write()


code25 = ('''
# len the shape of datassets
for i in datasets:
    st.write(i.shape), st.write()
''')
kode(code25)

for i in datasets:
    st.write(i.shape), st.write()


st.markdown('''Jumlah datasets sebelum dan setelah *missing value* diatasi tetap sama, artinya tidak ada kesalahan saat proses pengisian nilai yang hilang.

#### 1.4.2. Fix `date` Column
''')


code26 = ('''
# change date column format
for df in datasets:
    df['date'] = pd.to_datetime(df.loc[:, 'date'], format='%Y-%m-%dT%H:%M:%S')
''')
kode(code26)

for df in datasets:
    df['date'] = pd.to_datetime(df.loc[:, 'date'], format='%Y-%m-%dT%H:%M:%S')


code27 = ('''
for df in datasets:
    st.write(df.loc[:, 'date'].dtype)
''')
kode(code27)

for df in datasets:
    st.write(df.loc[:, 'date'].dtype)


st.markdown('''We have already fixed the column format.

## 2. Data Analytics and Data Visualization

### 2.1. The Changes in Metal Concentration

Kita akan melihat prosesnya sekali lagi:

<p align="center">
    <img src="https://pictures.s3.yandex.net/resources/ore_1591699963.jpg" alt="drawing" width="500" height="340">
  </a>
</p>

Kita hanya akan memilih kolom yang berisi *metals concentrate* seperti pada proses **1**, **3**, **5**, dan **8** di atas

#### 2.1.1. Gold (Au)
''')


code28 = ('''
# we just select columns that contains roger.input.feed, and concetrate_au
data_au = data_full.loc[:, data_full.columns.str.contains('rougher.input.feed_au|concentrate_au')]
st.write(data_au.head(2)), print
st.write('Data Au Shape', data_au.shape), st.write()
list(data_au.columns)
''')
kode(code28)

data_au = data_full.loc[:, data_full.columns.str.contains('rougher.input.feed_au|concentrate_au')]
st.write(data_au.head(2)), print
st.write('Data Au Shape', data_au.shape), st.write()
st.write(list(data_au.columns)
)


code29 = ('''
# create data_au plot
sns.set()

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_au, bins=100)
plt.xlabel('Ore Concentrate')
plt.ylabel('Amount')
plt.title('Changes in Gold (Au) Concentration')
st.pyplot(fig)
''')
kode(code29)

sns.set()

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_au, bins=100)
plt.xlabel('Ore Concentrate')
plt.ylabel('Amount')
plt.title('Changes in Gold (Au) Concentration')
st.pyplot(fig)


st.markdown('''**Findings**
- Pada proses ekstraksi emas dari tahap awal hingga tahap akhir terlihat jelas bahwa konsentrasi emas meningkat secara signifikan pada tiap proses ekstraksi.
- Pada tiap proses ekstraksi juga terlihat dengan jelas hampir tidak ada data pada tiap proses yang beririsan, sehingga kita dapat membedakan karakteristik jumlah konsentrasi pada tiap proses.

#### 2.1.2. Silver (Ag)
''')


code30 = ('''
# we just select columns that contains roger.input.feed, and concetrate_ag
data_ag = data_full.loc[:, data_full.columns.str.contains('rougher.input.feed_ag|concentrate_ag')]
st.write(data_ag.head(2)), print
st.write('Data Ag Shape', data_ag.shape), st.write()
list(data_ag.columns)
''')
kode(code30)

data_ag = data_full.loc[:, data_full.columns.str.contains('rougher.input.feed_ag|concentrate_ag')]
st.write(data_ag.head(2)), print
st.write('Data Ag Shape', data_ag.shape), st.write()
st.write(list(data_ag.columns)
)


code31 = ('''
# create data_ag plot
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_ag, bins=100)
plt.xlabel('Ore Concentrate')
plt.ylabel('Amount')
plt.title('Changes in Silver (Ag) Concentration')
st.pyplot(fig)
''')
kode(code31)

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_ag, bins=100)
plt.xlabel('Ore Concentrate')
plt.ylabel('Amount')
plt.title('Changes in Silver (Ag) Concentration')
st.pyplot(fig)


st.markdown('''**Findings**
- Pada tahap ekstraksi perak kita bisa melihat konsentrasi bijih perak meningkat dari bahan mentah pada proses awal, kemudian menurun pada proses selanjutnya hingga *final concentrate*.
- Kita juga melihat distribusi pada konsentrasi perak dimulai dari bahan mentah sama sama persis dengan distribusi *primary cleaner*, dan juga kita dapat melihat hampir hampir setengah dari tiap distribusi data beririsan.

#### 2.1.3. Lead/Timbal (Pb)
''')


code32 = ('''
# we just select columns that contains roger.input.feed, and concetrate_pb
data_pb = data_full.loc[:, data_full.columns.str.contains('rougher.input.feed_pb|concentrate_pb')]
st.write(data_pb.head(2)), print
st.write('Data Pb Shape', data_pb.shape), st.write()
list(data_pb.columns)
''')
kode(code32)

data_pb = data_full.loc[:, data_full.columns.str.contains('rougher.input.feed_pb|concentrate_pb')]
st.write(data_pb.head(2)), print
st.write('Data Pb Shape', data_pb.shape), st.write()
st.write(list(data_pb.columns)
)


code33 = ('''
# create data_pb plot
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_pb, bins=100)
plt.xlabel('Ore Concentrate')
plt.ylabel('Amount')
plt.title('Changes in Lead (Pb) Concentration')
st.pyplot(fig)
''')
kode(code33)

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_pb, bins=100)
plt.xlabel('Ore Concentrate')
plt.ylabel('Amount')
plt.title('Changes in Lead (Pb) Concentration')
st.pyplot(fig)


st.markdown('''**Finding**
- Pada proses ekstraksi timbal, kita melihat peningkatan distribusi data pada tiap proses tetapi memiliki distibusi yang sama memasuki tahap primer(*primary cleaner*) dan *final concentrate*.


**Kesimpulan Umum pada Perubahan Konsentrasi Metal**
- Terdapat perbedaan distribusi konsentrasi pada tiap proses ekstraksi ketiga logam tersebut:
    - Pada emas konsentrasi meningkat signifikan pada tiap proses ekstraksi.
    - Pada perak penigkatan terjadi pada tahap awal tetapi menurun pada proses selanjutnya hingga *final output*.
    - Pada timbal tidak ada penurunan konsentrasi hanya saja pada proses akhir distribusi konsentrasinya sama dengan *final output*.

### 2.2. Compare the Feed Particle Size
''')


code34 = ('''
# filter the data
train_feed = data_train.loc[:, 'rougher.input.feed_size']
test_feed = data_test.loc[:, 'rougher.input.feed_size']
''')
kode(code34)

train_feed = data_train.loc[:, 'rougher.input.feed_size']
test_feed = data_test.loc[:, 'rougher.input.feed_size']


code35 = ('''
# create the chart of feed size
fig, ax = plt.subplots(figsize=(9, 6))
plt.xlim(0, 120)
plt.title('Distribution of the Feed Particle Size of Train Set and Test Set')
sns.distplot(train_feed, bins=300, color='blue', label='train_set_feed_size')
sns.distplot(test_feed, bins=300, color='red', label='test_set_feed_size')
plt.legend()
st.pyplot(fig)
''')
kode(code35)

fig, ax = plt.subplots(figsize=(9, 6))
plt.xlim(0, 120)
plt.title('Distribution of the Feed Particle Size of Train Set and Test Set')
sns.distplot(train_feed, bins=300, color='blue', label='train_set_feed_size')
sns.distplot(test_feed, bins=300, color='red', label='test_set_feed_size')
plt.legend()
st.pyplot(fig)


st.markdown('''Secara umum distribusi `rougher.input` pada `train_set` dan `test_set` memiliki distribusi yang serupa. Kita bisa menggunakan model pada kedua datasets.

### 2.3. Distribution the total Concentrations 

Consider the total concentrations of all substances at different stages: raw feed, rougher concentrate, and final concentrate.

#### 2.3.1. Raw Feed
''')


code36 = ('''
# distribution total rougher.input.concentrate
rougher_input = data_full.loc[:, ((~data_full.columns.str.contains('_size|_rate')) & (data_full.columns.str.contains('rougher.input.feed_')))]
rougher_input['total'] = rougher_input.copy().sum(axis=1)
st.write(rougher_input.head(2)), print

st.write(rougher_input.shape), st.write()
list(rougher_input.columns)
''')
kode(code36)

rougher_input = data_full.loc[:, ((~data_full.columns.str.contains('_size|_rate')) & (data_full.columns.str.contains('rougher.input.feed_')))]
rougher_input['total'] = rougher_input.copy().sum(axis=1)
st.write(rougher_input.head(2)), print

st.write(rougher_input.shape), st.write()
st.write(list(rougher_input.columns)
)


st.markdown('''- Ada **4** kolom yang termasuk ke dalam *raw material* dibedakan berdasarkan zat-nya `ag`, `au`, `pb`, dan `sol`.
- Kita menambahkan kolom baru untuk mendapatkan *total concentration* dari semua zat tersebut. 
''')


code37 = ('''
# create chart of the distribution rougher.input.concentrate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=rougher_input, x=rougher_input['total'], kde=True, ax=axes[0])
sns.boxplot(data=rougher_input, x=rougher_input['total'], ax=axes[1])

fig.suptitle('Distribution Total rougher.input.concentrate')
st.pyplot(fig)
''')
kode(code37)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=rougher_input, x=rougher_input['total'], kde=True, ax=axes[0])
sns.boxplot(data=rougher_input, x=rougher_input['total'], ax=axes[1])

fig.suptitle('Distribution Total rougher.input.concentrate')
st.pyplot(fig)


st.markdown('''#### 2.3.2. Rougher Output Concentrate
''')


code38 = ('''
# distribution total rougher.output.concentrate
rougher_output = data_full.loc[:, ((~data_full.columns.str.contains('_size|_rate')) & (data_full.columns.str.contains('rougher.output.concentrate_')))]
rougher_output['total'] = rougher_output.copy().sum(axis=1)
st.write(rougher_output.head(2)), print

st.write(rougher_output.shape), st.write()
list(rougher_output.columns)
''')
kode(code38)

rougher_output = data_full.loc[:, ((~data_full.columns.str.contains('_size|_rate')) & (data_full.columns.str.contains('rougher.output.concentrate_')))]
rougher_output['total'] = rougher_output.copy().sum(axis=1)
st.write(rougher_output.head(2)), print

st.write(rougher_output.shape), st.write()
st.write(list(rougher_output.columns)
)


st.markdown('''- Ada **4** kolom yang termasuk ke dalam *rougher.output* dibedakan berdasarkan zat-nya `ag`, `au`, `pb`, dan `sol`.
- Kita menambahkan kolom baru untuk mendapatkan *total concentration* dari semua zat tersebut. 
''')


code39 = ('''
# create chart of the distribution rougher.output.concentrate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=rougher_output, x=rougher_output['total'], kde=True, ax=axes[0])
sns.boxplot(data=rougher_output, x=rougher_output['total'], ax=axes[1])

fig.suptitle('Distribution Total rougher.output.concentrate')
st.pyplot(fig)
''')
kode(code39)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=rougher_output, x=rougher_output['total'], kde=True, ax=axes[0])
sns.boxplot(data=rougher_output, x=rougher_output['total'], ax=axes[1])

fig.suptitle('Distribution Total rougher.output.concentrate')
st.pyplot(fig)


st.markdown('''#### 2.3.3. Final Output Concentrate
''')


code40 = ('''
# distribution total final.output.concentrate
final_output = data_full.loc[:, ((~data_full.columns.str.contains('_size|_rate')) & (data_full.columns.str.contains('final.output.concentrate_')))]
final_output['total'] = final_output.copy().sum(axis=1)
st.write(final_output.head(2)), print

st.write(final_output.shape), st.write()
list(final_output.columns)
''')
kode(code40)

final_output = data_full.loc[:, ((~data_full.columns.str.contains('_size|_rate')) & (data_full.columns.str.contains('final.output.concentrate_')))]
final_output['total'] = final_output.copy().sum(axis=1)
st.write(final_output.head(2)), print

st.write(final_output.shape), st.write()
st.write(list(final_output.columns)
)


st.markdown('''- Ada **4** kolom yang termasuk ke dalam *final.output* dibedakan berdasarkan zat-nya `ag`, `au`, `pb`, dan `sol`.
- Kita menambahkan kolom baru untuk mendapatkan *total concentration* dari semua zat tersebut. 
''')


code41 = ('''
# create chart of the distribution final.output.concentrate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=final_output, x=final_output['total'], kde=True, ax=axes[0])
sns.boxplot(data=final_output, x=final_output['total'], ax=axes[1])

fig.suptitle('Distribution Total final.output.concentrate')
st.pyplot(fig)
''')
kode(code41)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=final_output, x=final_output['total'], kde=True, ax=axes[0])
sns.boxplot(data=final_output, x=final_output['total'], ax=axes[1])

fig.suptitle('Distribution Total final.output.concentrate')
st.pyplot(fig)


st.markdown('''**Consclusion**
- Terdapat cukup banyak *concentrate* yang memiliki nilai **0**, hal ini mungkin terjadi karena *error* yang terjadi saat bahan mentah dimasukkan ke dalam proses ekstraksi pertama kali sama dengan **0**, jika bahan mentah yang dimasukkan ke dalam proses pertama sama dengan **0** maka proses selanjutnya juga tidak akan memiliki nilai *output* karena sama saja tidak ada bahan baku (*concentrate*) untuk dihasilkan.
- Kita akan menghapus *values* dari kolom-kolom tersebut dari `train_set` dan `test_set`, model akan cenderung *malas* untuk mempelajari *features* yang memiliki nilai **0**

#### 2.3.4. Remove Anomali Values
''')


code42 = ('''
# len dataset
st.write(data_full.shape)
st.write(data_train.shape)
data_test.shape
''')
kode(code42)

st.write(data_full.shape)
st.write(data_train.shape)
st.write(data_test.shape
)


code43 = ('''
# filter columns for remove abnormal values
raw_feed = list(rougher_input.drop(['total'], axis=1))
output_concentrate = list(rougher_output.drop(['total'], axis=1))
final_concentrate = list(final_output.drop(['total'], axis=1))

all_concentrate = []

for out in [raw_feed, output_concentrate, final_concentrate]:
    all_concentrate.extend(out)

all_concentrate
''')
kode(code43)

raw_feed = list(rougher_input.drop(['total'], axis=1))
output_concentrate = list(rougher_output.drop(['total'], axis=1))
final_concentrate = list(final_output.drop(['total'], axis=1))

all_concentrate = []

for out in [raw_feed, output_concentrate, final_concentrate]:
    all_concentrate.extend(out)

all_concentrate


code44 = ('''
# remove abnormal value
data_train = data_train.loc[(data_train[all_concentrate] != 0).all(axis=1)].reset_index(drop=True)

data_full_test = data_full.loc[data_test.index]
data_test = data_test.loc[(data_full_test[all_concentrate] != 0).all(axis=1)].reset_index(drop=True)

data_full = data_full.loc[(data_full[all_concentrate] != 0).all(axis=1)].reset_index(drop=True)
''')
kode(code44)

data_train = data_train.loc[(data_train[all_concentrate] != 0).all(axis=1)].reset_index(drop=True)

data_full_test = data_full.loc[data_test.index]
data_test = data_test.loc[(data_full_test[all_concentrate] != 0).all(axis=1)].reset_index(drop=True)

data_full = data_full.loc[(data_full[all_concentrate] != 0).all(axis=1)].reset_index(drop=True)


code45 = ('''
# len dataset
st.write(data_full.shape)
st.write(data_train.shape)
data_test.shape
''')
kode(code45)

st.write(data_full.shape)
st.write(data_train.shape)
st.write(data_test.shape
)


st.markdown('''## 3. Build the Model

### 3.1. Calculate the Final sMAPE

The sMAPE formula: 
<p align="center">
    <img src="https://pictures.s3.yandex.net/resources/smape_1576239058_1589899769.jpg" alt="drawing" width="500" height="70">
  </a>
</p>


Final sMAPE formula:
<p align="center">
    <img src="https://pictures.s3.yandex.net/resources/_smape_1_1589900649.jpg" alt="drawing" width="500" height="70">
  </a>
</p>
''')


code46 = ('''
# create smape function

def smape_final(actual, forecast):
    result = (np.mean(abs(actual - forecast) / ((abs(actual) + abs(forecast)) / 2))) * 100
    final_smape = 0.25 * result[0] + 0.75 * result[1]
    return final_smape

smape_score = make_scorer(smape_final, greater_is_better=False)
''')
kode(code46)

def smape_final(actual, forecast):
    result = (np.mean(abs(actual - forecast) / ((abs(actual) + abs(forecast)) / 2))) * 100
    final_smape = 0.25 * result[0] + 0.75 * result[1]
    return final_smape

smape_score = make_scorer(smape_final, greater_is_better=False)


st.markdown('''### 3.2. Train the Models

#### 3.2.1. Split the Data into Features and Target
''')


code47 = ('''
# split data to features and targets
X = list(data_test.columns.drop('date'))
y = ['rougher.output.recovery', 'final.output.recovery']
''')
kode(code47)

X = list(data_test.columns.drop('date'))
y = ['rougher.output.recovery', 'final.output.recovery']


code48 = ('''
# apply the filtered list to split data
X_train = data_train[X].reset_index(drop=True)
y_train = data_train[y].reset_index(drop=True)

y_train.columns = [0, 1]
''')
kode(code48)

X_train = data_train[X].reset_index(drop=True)
y_train = data_train[y].reset_index(drop=True)

y_train.columns = [0, 1]


code49 = ('''
# print the y_train samples
y_train.head()
''')
kode(code49)

st.write(y_train.head()
)


code50 = ('''
# split data_test into features and target
target_extract = data_full[['date', 'rougher.output.recovery', 'final.output.recovery']].reset_index(drop=True)
data_test = data_test.merge(target_extract, on='date')

X_test = data_test[X].reset_index(drop=True)
y_test = data_test[y].reset_index(drop=True)

y_test.columns = [0, 1]
''')
kode(code50)

target_extract = data_full[['date', 'rougher.output.recovery', 'final.output.recovery']].reset_index(drop=True)
data_test = data_test.merge(target_extract, on='date')

X_test = data_test[X].reset_index(drop=True)
y_test = data_test[y].reset_index(drop=True)

y_test.columns = [0, 1]


code51 = ('''
# print the X_test samples
X_test.head()
''')
kode(code51)

st.write(X_test.head()
)


code52 = ('''
# check deta X_train, y_train, X_test, y_test shapes
for i in [X_train, y_train, X_test, y_test]:
    st.write(i.shape)
''')
kode(code52)

for i in [X_train, y_train, X_test, y_test]:
    st.write(i.shape)


st.markdown('''#### 3.2.2. Define Function to Return Model Score
''')


code53 = ('''
# create kfold model
def kfold_model(model):
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, scoring=smape_score, cv=cv, n_jobs=-1)
    scores = np.abs(scores)
    return scores.mean()
''')
kode(code53)

def kfold_model(model):
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, scoring=smape_score, cv=cv, n_jobs=-1)
    scores = np.abs(scores)
    return scores.mean()


st.markdown('''#### 3.2.3. Linear Regression
''')


code54 = ('''
# perform linear regression
model = LinearRegression()
kfold_model(model)
''')
kode(code54)

model = LinearRegression()
kfold_model(model)


st.markdown('''#### 3.2.4. Decision Tree Model
''')


code55 = ('''
# search best params for decision tree model
dt_results = defaultdict(list)

for depth in [1, 2, 5, 8, 10, 15, 20, 50]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_results['max_depth'].append(depth)
    dt_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(dt_results))
''')
kode(code55)

dt_results = defaultdict(list)

for depth in [1, 2, 5, 8, 10, 15, 20, 50]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_results['max_depth'].append(depth)
    dt_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(dt_results))


st.markdown('''#### 3.2.5. Random Forest Model
''')


code56 = ('''
# search best params for random forest model
rf_results = defaultdict(list)

for depth in range(1, 10):
    model = RandomForestRegressor(n_estimators=10, max_depth=depth, random_state=42)
    rf_results['max_depth'].append(depth)
    rf_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(rf_results))
''')
kode(code56)

rf_results = defaultdict(list)

for depth in range(1, 10):
    model = RandomForestRegressor(n_estimators=10, max_depth=depth, random_state=42)
    rf_results['max_depth'].append(depth)
    rf_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(rf_results))


st.markdown('''#### 3.2.6. Lasso Regressor
''')


code57 = ('''
# search best params for lasso regressor model
lasso_results = defaultdict(list)

for alpha in [0.00, 1.0, 0.01, 2, 3, 5, 10, 20, 50]:
    model = Lasso(alpha=alpha, random_state=42)
    lasso_results['alpha'].append(alpha)
    lasso_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(lasso_results))
''')
kode(code57)

lasso_results = defaultdict(list)

for alpha in [0.00, 1.0, 0.01, 2, 3, 5, 10, 20, 50]:
    model = Lasso(alpha=alpha, random_state=42)
    lasso_results['alpha'].append(alpha)
    lasso_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(lasso_results))


st.markdown('''#### 3.2.7. Ridge Regressor
''')


code58 = ('''
# search best params for ridge regressor model
ridge_results = defaultdict(list)

for alpha in np.linspace(0, 0.2, 11):
    model = Ridge(alpha=alpha, random_state=42)
    ridge_results['alpha'].append(alpha)
    ridge_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(ridge_results))
''')
kode(code58)

ridge_results = defaultdict(list)

for alpha in np.linspace(0, 0.2, 11):
    model = Ridge(alpha=alpha, random_state=42)
    ridge_results['alpha'].append(alpha)
    ridge_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(ridge_results))


st.markdown('''#### 3.2.8. Neural Network
''')


code59 = ('''
# search best params for neural network regressor model
nn_results = defaultdict(list)

for activation in ['identity', 'logistic', 'tanh', 'relu']:
    model = MLPRegressor(activation=activation, random_state=42)
    nn_results['activation'].append(activation)
    nn_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(nn_results))
''')
kode(code59)

nn_results = defaultdict(list)

for activation in ['identity', 'logistic', 'tanh', 'relu']:
    model = MLPRegressor(activation=activation, random_state=42)
    nn_results['activation'].append(activation)
    nn_results['mean_smape'].append(kfold_model(model))

st.write(pd.DataFrame(nn_results))


st.markdown('''**Consclusion**
- Model terbaik dengan nilai **sMAPE** paling rendah adalah `Random Forest` dengan `max_depth` sama dengan **9**.

### 3.3. Perform Best Model using Test Set
''')


code60 = ('''
# create best random forest model

rf_model = RandomForestRegressor(n_estimators=10, max_depth=9, random_state=42)
rf_model.fit(X_train, y_train)

y_train_rf_pred = rf_model.predict(X_train)
y_test_rf_pred = rf_model.predict(X_test)

st.write('Train:', smape_final(y_train, y_train_rf_pred))
st.write('Test:', smape_final(y_test, y_test_rf_pred))
''')
kode(code60)

rf_model = RandomForestRegressor(n_estimators=10, max_depth=9, random_state=42)
rf_model.fit(X_train, y_train)

y_train_rf_pred = rf_model.predict(X_train)
y_test_rf_pred = rf_model.predict(X_test)

st.write('Train:', smape_final(y_train, y_train_rf_pred))
st.write('Test:', smape_final(y_test, y_test_rf_pred))


st.markdown('''# Consclusions

**1. Data Preparation**
- Kita memulai dengan memuat 3 dataset yang memiliki perbedaan pada jumlah *features* dan *rows*.
- Cukup banyak *missing value* ditemukan pada tiap datasets.
- Kita mengisi *missing value* menggunakan metode `KNN Imputer` karena data memiliki kesamaan berdasarkan waktu yang berdekatan.

**2. EDA and Data Visualization**
- Terdapat perbedaan distribusi konsentrasi pada tiap proses ekstraksi ketiga logam tersebut:
    - Pada emas konsentrasi meningkat signifikan pada tiap proses ekstraksi.
    - Pada perak penigkatan terjadi pada tahap awal tetapi menurun pada proses selanjutnya hingga *final output*.
    - Pada timbal tidak ada penurunan konsentrasi hanya saja pada proses akhir distribusi konsentrasinya sama dengan *final output*.
- Secara umum distribusi `rougher.input` pada `train_set` dan `test_set` memiliki distribusi yang serupa.
- Terdapat cukup banyak *concentrate* yang memiliki nilai **0**, hal ini mungkin terjadi karena *error* yang terjadi saat bahan mentah dimasukkan ke dalam proses ekstraksi pertama kali sama dengan **0**.

**3. Model**
- Kita menggunakan **6** model untuk mendapatkan nilai **sMAPE** paling rendah.
- Model terbaik dengan nilai **sMAPE** paling rendah adalah `Random Forest` dengan `max_depth` sama dengan **10**.

**Main Consclusion**
- Kita mendapatkan hasil **sMAPE** pada `test set` dengan model terbaik sebesar **7.623%**, hasil ini tentu sangat baik karena memiliki nilai **error** dibawah 10%.
''')
