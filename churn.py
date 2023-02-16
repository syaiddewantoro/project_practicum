import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sidetable as stb
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

# ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_confusion_matrix

# mengimpor SMOTE
from imblearn.over_sampling import SMOTE

import streamlit as st
import io


def kode(codes):
    st.code(codes, language='python')

def buffer(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    detail = buffer.getvalue()
    return st.text(detail)


st.markdown('''# Table of Contents

* [Beta Bank](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Beta-Bank)
* [1. Initialization](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Iniatialization)
* [2. Data Preparation](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Data-Preparation)
* [3. EDA and Data Visualization](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#EDA-and-Data-Visulization)
    * [3.1 Visualisasi Data Kolom Continues](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Visualisasi-Data-Kolom-Continues)
    * [3.2 Visualisasi Data Kolom Kategori](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Visualisasi-Data-Kolom-Kategori)
* [4. Split the Data](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#EDA-and-Data-Visulization)
    * [4.1 Memeriksa Keseimbangan Kelas](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Memeriksa-Keseimbangan-Kelas)
    * [4.2 Split the Data to Train, Validation and Test](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Split-the-Data-to-Train,-Validation,-and-Test)
* [5. Create Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Decision-Tree-Model)
    * [5.1 Decision Tree Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Decision-Tree-Model)
    * [5.2 Logistic Regression Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Logistic-Regression-Model)
    * [5.3 Random Forest Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Random-Forest-Model)
    * [5.4 K-Nearest Neighbors Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#K-Nearest-Neighbors-Model)
* [6. Meningkatkan Kualitas Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Meningkatkan-Kualitas-Model)
    * [6.1 Upsampling](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Upsampling)
    * [6.2 Standard Scaller](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Standard-Scaller)
* [7. Membuat Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Membuat-Model)
    * [7.1 Decision Tree Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Decision-Tree)
    * [7.2 Logistic Regression Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Logistic-Regression)
    * [7.3 Random Forest Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Random-Forest)
    * [7.4 K-Nearest Neighbors Model](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#K-Nearest-Neighbors)  
    
* [Consclusions](https://jupyterhub.practicum-services.com/user/user-3-511a834a-cb53-4a25-8768-72c2f21b226f/notebooks/882d3d75-03c1-4793-a93f-cca500235605.ipynb#Consclusions)

# Beta Bank

Nasabah Bank Beta pergi meninggalkan perusahaan: sedikit demi sedikit, jumlah mereka berkurang setiap bulannya. Para pegawai bank menyadari bahwa lebih murah untuk mempertahankan nasabah lama mereka yang setia daripada menarik nasabah baru.

Pada kasus ini, tugas kami adalah untuk memprediksi apakah seorang nasabah akan segera meninggalkan bank atau tidak. Kami memiliki data terkait perilaku para klien di masa lalu dan riwayat pemutusan kontrak mereka dengan bank.

Kami akan membuat sebuah model dengan skor F1 semaksimal mungkin. Untuk bisa dinyatakan lulus dari peninjauan, Kami memerlukan skor F1 minimal 0,59 untuk test dataset. Periksa nilai F1 untuk test set.

Selain itu, kita akan mengukur metrik AUC-ROC dan membandingkan metrik tersebut dengan skor F1.

**Tujuan:**
- Memprediksi apakah seorang nasabah akan segera meninggalkan bank atau tidak.
- Melatih model tanpa mempertimbangkan ketidakseimbangan kelas.
- Meningkatkan kualitas data untuk menjalankan model.
- Menemukan model terbaik untuk memprediksi nasabah Bank.

## 1. Initialization

**Loading Libraries**
''')


code1 = ('''
# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install -U sidetable
import sidetable as stb
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

# ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_confusion_matrix
''')
kode(code1)



code2 = ('''
# mengimpor SMOTE
!pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE
''')
kode(code2)


st.markdown('''## 2. Data Preparation

### 2.1. Memuat Data, Menampilkan Sample dan Informasi

**Deskripsi Data:**

**Fitur-fitur**
- `RowNumber` — indeks string data
- `CustomerId` — ID pelanggan
- `Surname` — nama belakang
- `CreditScore` — skor kredit
- `Geography` — negara domisili
- `Gender` — gender
- `Age` — umur
- `Tenure` — jangka waktu jatuh tempo untuk deposito tetap nasabah (tahun)
- `Balance` — saldo rekening
- `NumOfProducts` — jumlah produk bank yang digunakan oleh nasabah
- `HasCrCard` — apakah nasabah memiliki kartu kredit
- `IsActiveMember` — tingkat keaktifan nasabah
- `EstimatedSalary` — estimasi gaji

**Target**
- `Exited` — apakah nasabah telah berhenti
''')


code3 = ('''
# load the dataset
try:
    df = pd.read_csv('/datasets/Churn.csv')
except:
    df = pd.read_csv('Churn.csv')
''')
kode(code3)

df = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/Churn.csv')


code4 = ('''
# display the sample of dataset
df
''')
kode(code4)

df


st.markdown('''Sebelum kita melakukan analisa lebih jauh, untuk mempermudah dalam melakukan analisis kira akan mengubah register semua kolom menjadi **lower**.
''')


code5 = ('''
# mengubah register kolom
df.columns = df.columns.str.lower()
df.columns.values
''')
kode(code5)

df.columns = df.columns.str.lower()
st.write(df.columns.values
)


code6 = ('''
# show the information of data
df.info()
''')
kode(code6)

buffer(df)


st.markdown('''### 2.2. Memeriksa *Missing Value*
''')


code7 = ('''
# menampilkan missing value
df.stb.missing().reset_index()
''')
kode(code7)

st.write(df.stb.missing().reset_index()
)


st.markdown('''Data yang hilang hanya terjadi di kolom `tenure` sekitar **9%** dari data. Kolom `tenure` berisi jangka waktu jatuh tempo untuk deposito tetap nasabah (tahun), kita akan memeriksa apakah *missing value* tersebut sama dengan nilai **0** sebelum melakukan analisa lebih lanjut.
''')


code8 = ('''
# memeriksa value 0 kolom tenure
df['tenure'].value_counts(dropna=False).sort_index()
''')
kode(code8)

st.write(df['tenure'].value_counts(dropna=False).sort_index()
)


st.markdown('''Pada kolom `tenure` juga terdapat value **0** berisi nasabah yang tidak memiliki jangka waktu pembayaran deposito, artinya *missing value* bukan merupakan nilai **0**.
''')


code9 = ('''
# memeriksa distribusi
df.loc[df['tenure'].isna()]
''')
kode(code9)

st.write(df.loc[df['tenure'].isna()]
)


st.markdown('''### 2.3. Handle Missing Value
''')


code10 = ('''
# Mengisi missing value kolom tenure berdasarkan usia
df['tenure'] = np.round(df['tenure'].fillna(df.groupby(['age'])['tenure'].transform('mean')))
df['tenure'].isna().sum()
''')
kode(code10)

# Mengisi missing value kolom tenure berdasarkan usia
df['tenure'] = np.round(df['tenure'].fillna(df.groupby(['age'])['tenure'].transform('mean')))
st.write(df['tenure'].isna().sum()
)


code11 = ('''
# memeriksa value 0 kolom tenure
df['tenure'].value_counts(dropna=False).sort_index()
''')
kode(code11)

st.write(df['tenure'].value_counts(dropna=False).sort_index()
)


st.markdown('''*Missing value* sudah terisi.

**Drop Unnecessary Columns**
''')


code12 = ('''
# menghapus kolom yang tidak diperlukan
df.drop(['rownumber', 'customerid', 'surname'], axis=1, inplace=True)
df.head()
''')
kode(code12)

df.drop(['rownumber', 'customerid', 'surname'], axis=1, inplace=True)
st.write(df.head()
)


st.markdown('''## 3. EDA and Data Visulization

### 3.1. Visualisasi Data Kolom Continues
''')


code13 = ('''
for column in ['creditscore', 'balance', 'estimatedsalary', 'age']:
    fig, ax = plt.subplots(figsize=(9,6))
    sns.histplot(x=df[column], hue=df['exited'], bins=70, kde=True)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)
''')
kode(code13)

for column in ['creditscore', 'balance', 'estimatedsalary', 'age']:
    fig, ax = plt.subplots(figsize=(9,6))
    sns.histplot(x=df[column], hue=df['exited'], bins=70, kde=True)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)


st.markdown('''**Temuan :**
- Dari kedua kelompok nasabah yang *exited* dan yang tidak, mayoritas dari mereka memiliki rata-rata credit score antara **550 - 750**.
- Terdapat banyak nasabah yang tidak memiliki sisa saldo sama sekali dalam *account* mereka ada sekitar **500** orang dari kelompok **exited** dan lebih dari **3000** nasabah dari kelompok yang *stay*, mayoritas lain dari mereka juga memiliki rata-rata *balance* antara **100.000 - 150.000**
- *Estimated Salary* memiliki penyebaran yang merata mulai dari **0 - 200.000**.
- Mayoritas dari kelompok yang tetap setia memiliki rentang usia di bawah **50** tahun, sedangkan dari kelompok **exited** memiliki rentang usia **40 - 60** tahun.

### 3.2. Visualisasi Data Kolom Kategori
''')


code14 = ('''
for label in ['geography', 'gender', 'tenure', 'numofproducts', 'hascrcard']:
    fig, ax = plt.subplots(figsize=(9, 6))
    splot = sns.countplot(data=df, x=df[label], hue=df['exited'])
    plt.title(label)
    plt.ylabel('Amount')
    plt.xlabel(label)
    for p in splot.patches:
        splot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
    st.pyplot(fig)
''')
kode(code14)

for label in ['geography', 'gender', 'tenure', 'numofproducts', 'hascrcard']:
    fig, ax = plt.subplots(figsize=(9, 6))
    splot = sns.countplot(data=df, x=df[label], hue=df['exited'])
    plt.title(label)
    plt.ylabel('Amount')
    plt.xlabel(label)
    for p in splot.patches:
        splot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
    st.pyplot(fig)


st.markdown('''**Temuan :**
- Nasabah Bank Beta mayoritas berasal dari Prancis
- Mayoritas dari nasabah Bank Beta menggunakan setidaknya **1-2** produk yang dikeluarkan oleh bank.
- Mayoritas dari nasabah Bank Beta juga memiliki kartu.

## 4. Split the Data

### 4.1. Memeriksa Keseimbangan Kelas
''')


code15 = ('''
# Menampilkan jumlah kolom is_ultra
df.groupby('exited').agg(count=('creditscore', 'count')).reset_index()
''')
kode(code15)

st.write(df.groupby('exited').agg(count=('creditscore', 'count')).reset_index()
)


code16 = ('''
# menghitung rasio
df['exited'].value_counts() / df.shape[0] * 100
''')
kode(code16)

st.write(df['exited'].value_counts() / df.shape[0] * 100
)


code17 = ('''
# Membuat diagram
fig, ax = plt.subplots(figsize=(4,4))
splot = sns.countplot(x='exited', data=df, order=df['exited'].value_counts().index)
plt.xlabel('Is Exited')
plt.ylabel('Amount')
plt.title('Count of Exited')
for p in splot.patches:
        splot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
st.pyplot(fig)
''')
kode(code17)

fig, ax = plt.subplots(figsize=(4,4))
splot = sns.countplot(x='exited', data=df, order=df['exited'].value_counts().index)
plt.xlabel('Is Exited')
plt.ylabel('Amount')
plt.title('Count of Exited')
for p in splot.patches:
        splot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
st.pyplot(fig)


st.markdown('''Perbandingan data pada kolom `exited` sebesar **80** dengan **20**.

### 4.2. Split the Data to Train, Validation, and Test
''')


code18 = ('''
# mengkonversi variabel katgorikal ke variable indikator
df = pd.get_dummies(df, drop_first=True)
df.head()
''')
kode(code18)

df = pd.get_dummies(df, drop_first=True)
st.write(df.head()
)


st.markdown('''Kolom dengan *value* kategorikal sudah terkonversi dengan benar.
''')


code19 = ('''
# membagi dataset menjadi features dan target
X = df.drop(['exited'], axis=1)
y = df['exited']
''')
kode(code19)

X = df.drop(['exited'], axis=1)
y = df['exited']


code20 = ('''
# memeriksa ukuran data
st.write(X.shape)
st.write(y.shape)
''')
kode(code20)

st.write(X.shape)
st.write(y.shape)


code21 = ('''
# Split the data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.15, random_state = 8)

# Use the same function above for the validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
    test_size=0.18, random_state= 8) 
''')
kode(code21)

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.15, random_state = 8)

# Use the same function above for the validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
    test_size=0.18, random_state= 8) 


code22 = ('''
# menampilkan dimensi dataset
st.write('X_train shape', X_train.shape)
st.write('X_valid shape', X_valid.shape)
st.write('X_test shape', X_test.shape)
''')
kode(code22)

st.write('X_train shape', X_train.shape)
st.write('X_valid shape', X_valid.shape)
st.write('X_test shape', X_test.shape)


code23 = ('''
# menampilkan sample data X-train
X_train
''')
kode(code23)

X_train


st.markdown('''## 4. Create Model

### 4.1. Decision Tree Model
''')


code24 = ('''
# membuat model decision tree
st.write('Decision Tree Model')
st.write()
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('max_depth =', depth)
    st.write('Train f1  =', f1_score(y_train, y_train_predict))
    st.write('Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')
''')
kode(code24)

st.write('Decision Tree Model')
st.write()
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('max_depth =', depth)
    st.write('Train f1  =', f1_score(y_train, y_train_predict))
    st.write('Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')


st.markdown('''Dari beberapa parameter yang kita gunakan untuk melatih model pada **Decision Tree Classifier**, dan hasil-hasil yang didapat tidak dapat mencapai batas untuk *fi score* pada `train set` dan `validation set`. Parameter yang paling baik sejauh ini untuk model ini adalah menetapkan nilai **max_dept = 7**.

### 4.2. Logistic Regression Model
''')


code25 = ('''
# membuat model logistic regression
st.write('Logistic Regression Model')
st.write()
for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    model = LogisticRegression(solver=solv)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('solver =', solv)
    st.write('Train f1  =', f1_score(y_train, y_train_predict))
    st.write('Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')
''')
kode(code25)

st.write('Logistic Regression Model')
st.write()
for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    model = LogisticRegression(solver=solv)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('solver =', solv)
    st.write('Train f1  =', f1_score(y_train, y_train_predict))
    st.write('Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')


st.markdown('''Model **Logistik Regressiion** setelah kita menetapkan beberapa parameter hasil yang didapat lebih tinggi daripada **Decision Tree**, dengan nilai pada *train* dan *validation* set sebesar **84%** dan **82%** dan telah melewati nilai *treshold*. Pada model ini model terbaik didapatkan dengan menetapkan parameter **solver = lbfgs**.

### 4.3. Random Forest Model
''')


code26 = ('''
# membuat model random forest
st.write('Random Forest Model')
st.write()
for n in range(1,21):
    model = RandomForestClassifier(random_state=42, max_depth=n)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('max_depth =', n)
    st.write('Train f1  =', f1_score(y_train, y_train_predict), '|| Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')
''')
kode(code26)

st.write('Random Forest Model')
st.write()
for n in range(1,21):
    model = RandomForestClassifier(random_state=42, max_depth=n)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('max_depth =', n)
    st.write('Train f1  =', f1_score(y_train, y_train_predict), '|| Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')


st.markdown('''Pada model **Random Forest** model yang paling baik didapatkan dengan menetapkan **max_depth = 15** meskipun memiliki selisih yang sangat jauh antara *training set* dan *validation set*-nya.

### 4.4. K-Nearest Neighbors Model
''')


code27 = ('''
# membuat model knn
st.write('K-Nearest Neighbors Model')
st.write()
for n in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('n-neighbors =', n)
    st.write('Train f1  =', f1_score(y_train, y_train_predict))
    st.write('Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')
''')
kode(code27)

st.write('K-Nearest Neighbors Model')
st.write()
for n in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_valid_predict = model.predict(X_valid)
    st.write('n-neighbors =', n)
    st.write('Train f1  =', f1_score(y_train, y_train_predict))
    st.write('Valid f1  =', f1_score(y_valid, y_valid_predict))
    st.write('')


st.markdown('''Pada model **K-Nearest Neighbors** kita menetapkan **n-neighbors** dalam rentang **1 - 20** dan model terbaik didapatkan pada nilai **n-neigbors = 13** meskipun **f1**-nya tidak ada yang bisa melebihi nilai ambang batas yang ditentukan.

## 5. Meningkatkan Kualitas Model

### 5.1. Upsampling
''')


code28 = ('''
# meningkatkan jumlah sample dengan SMOTE
X_upsampled, y_upsampled = SMOTE(random_state=42).fit_resample(X_train, y_train)
''')
kode(code28)

X_upsampled, y_upsampled = SMOTE(random_state=42).fit_resample(X_train, y_train)


code29 = ('''
# menampilkan dimensi data sebelum dan sesudah dilakukan upsampling
st.write(y_train.value_counts()), st.write()
st.write(y_upsampled.value_counts())
st.write()

# Membuat diagram
fig, ax = plt.subplots(figsize=(4,4))
splot = sns.countplot(x=y_upsampled, order=df['exited'].value_counts().index)
plt.xlabel('Is Exited')
plt.ylabel('Amount')
plt.title('Count of Exited')
for p in splot.patches:
        splot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
st.pyplot(fig)
''')
kode(code29)

st.write(y_train.value_counts()), st.write()
st.write(y_upsampled.value_counts())
st.write()

# Membuat diagram
fig, ax = plt.subplots(figsize=(4,4))
splot = sns.countplot(x=y_upsampled, order=df['exited'].value_counts().index)
plt.xlabel('Is Exited')
plt.ylabel('Amount')
plt.title('Count of Exited')
for p in splot.patches:
        splot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
st.pyplot(fig)


st.markdown('''Kita telah meningkatkan jumlah data *customer* yang *exited* dari sebelumnya **1423** menjadi **4269**.
''')


code30 = ('''
# print the datashape
st.write(X_upsampled.shape)
st.write(y_upsampled.shape)
''')
kode(code30)

st.write(X_upsampled.shape)
st.write(y_upsampled.shape)


code31 = ('''
# menampilkan dimensi dataset
st.write('X_upsampled shape', X_upsampled.shape)
st.write('X_valid shape', X_valid.shape)
st.write('X_test shape', X_test.shape)
''')
kode(code31)

st.write('X_upsampled shape', X_upsampled.shape)
st.write('X_valid shape', X_valid.shape)
st.write('X_test shape', X_test.shape)


code32 = ('''
# show the sample of upsampled data
X_upsampled
''')
kode(code32)

X_upsampled


st.markdown('''### 5.2. Standard Scaller

Pertama-kita akan menstandarkan data dengan mengubah *training set* dan *validation set* menggunakan *transform* ke **StandardScaller**. 
''')


code33 = ('''
# membuat sebuah instance dari kelas dan melakukan penyetelan terhadap data menggunakan training dataset
scaler = StandardScaler()
scaler.fit(X_upsampled) 

# mengubah training set dan validation set
X_upsampled_scaled = scaler.transform(X_upsampled)
X_valid_scaled = scaler.transform(X_valid) 
X_test_scaled = scaler.transform(X_test)
''')
kode(code33)

scaler = StandardScaler()
scaler.fit(X_upsampled) 

# mengubah training set dan validation set
X_upsampled_scaled = scaler.transform(X_upsampled)
X_valid_scaled = scaler.transform(X_valid) 
X_test_scaled = scaler.transform(X_test)


code34 = ('''
# mengubah tabel array ke dalam format dataframe
X_upsampled = pd.DataFrame(X_upsampled_scaled, columns=X_upsampled.columns)
X_valid = pd.DataFrame(X_valid_scaled, columns=X_valid.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
''')
kode(code34)

X_upsampled = pd.DataFrame(X_upsampled_scaled, columns=X_upsampled.columns)
X_valid = pd.DataFrame(X_valid_scaled, columns=X_valid.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)


code35 = ('''
# menampilkan sample data
X_upsampled
''')
kode(code35)

X_upsampled


code36 = ('''
# menampilkan sample data
X_valid.head()
''')
kode(code36)

X_valid.head()


code37 = ('''
# menampilkan dimensi data scalled
st.write('X_train scaled shape', X_upsampled.shape)
st.write('X_valid scaled shape', X_valid.shape)
st.write('X_test scaled shape', X_test.shape)
''')
kode(code37)

st.write('X_train scaled shape', X_upsampled.shape)
st.write('X_valid scaled shape', X_valid.shape)
st.write('X_test scaled shape', X_test.shape)


st.markdown('''## 6. Membuat Model

### 6.1. Decision Tree
''')


code38 = ('''
# membuat model decision tree
st.write('Decision Tree Model')
dt_result = defaultdict(list)

for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    y_valid_predict = model.predict(X_valid)
    dt_result['max_depth'].append(depth)
    dt_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    dt_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(dt_result))
''')
kode(code38)

st.write('Decision Tree Model')
dt_result = defaultdict(list)

for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    y_valid_predict = model.predict(X_valid)
    dt_result['max_depth'].append(depth)
    dt_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    dt_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(dt_result))


st.markdown('''- Pada model **Decision Tree** yang kita latih sebelum dilakukan peningkatan kualitas data nilai **max_depth = 7** merupakan parameter paling baik dengan tingkat *f1 score* sebesar **61%** dan **58%** untuk *train* dan *validation* set-nya. 

- Sedangkan pada model setelah kita meningkatkan kualitas data nilai **max_depth = 5** merupakan parameter terbaik dengan peningkatan *f1 score* pada *train* sekitar **19%** tetapi terjadi penurunan pada *validation* set, menjadi **80%** dan **55%** untuk data *train* dan *validation* set.

- Setelah dilakukan peningkatan kualitas data pada model ini telah memberikan peningkatan nilai *f1 score* meskipun tidak teralu signifikan.
''')


code39 = ('''
# create decision tree model
dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
dt_model.fit(X_upsampled, y_upsampled)
''')
kode(code39)

dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
dt_model.fit(X_upsampled, y_upsampled)


code40 = ('''
# accuracy score dt model compare
st.write('Decision Tree Model'), st.write('')
y_train_dt_pred = dt_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_dt_pred))
y_valid_dt_pred = dt_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_dt_pred))
y_test_dt_pred = dt_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_dt_pred))
''')
kode(code40)

st.write('Decision Tree Model'), st.write('')
y_train_dt_pred = dt_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_dt_pred))
y_valid_dt_pred = dt_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_dt_pred))
y_test_dt_pred = dt_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_dt_pred))


st.markdown('''Nilai *accuracy* pada *training* lebih rendah daripada *validation* dan *test*, dan memiliki tingkat *accuracy* cukup tinggi sebesar **79%**.
''')


code41 = ('''
# membuat kurva ROC
probabilities_valid = dt_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('Decision Tree'), st.write()
st.write('f1 Score =', f1_score(y_test, y_test_dt_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)
''')
kode(code41)

probabilities_valid = dt_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('Decision Tree'), st.write()
st.write('f1 Score =', f1_score(y_test, y_test_dt_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)


st.markdown('''**AUC-ROC Score** menunjukan nilai **84%** dengan kurva cukup tinggi, hal ini menunjukan model kita sudah cukup baik dari model acak.
''')


code42 = ('''
# classification report the dt model of data test
st.write('Decision Tree Classification Report'), st.write()
st.write(classification_report(y_test, y_test_dt_pred)), st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_dt_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)''')
kode(code42)

st.write('Decision Tree Classification Report'), st.write()
st.write(classification_report(y_test, y_test_dt_pred)), st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_dt_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)

st.markdown('''- **Precision** pada pada nasabah yang *exited* dan *stay* memiliki nilai persentase yang cukup tinggi hanya pada *train* yaitu **92%** daripada *validation* ynag hanya **50%**, model ini hanya mampu memprediksi satu kelas saja. 


- Nilai **recall** pada nasabah yang setia dan  tidak sebesar **81%** dan **74%** yang artinya dari semua data nasabah yang benar-benar *stay* dan *exited* model ini hanya bisa memprediksi pada nasabah yang *exited* dengan cukup baik pada kedua kelompok nasabah Bank Beta.

### 6.2. Logistic Regression
''')


code43 = ('''
# membuat model logistic regression
st.write('Logistic Regression Model')
lr_result = defaultdict(list)

for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    model = LogisticRegression(solver=solv, class_weight='balanced')
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    y_valid_predict = model.predict(X_valid)
    lr_result['solver'].append(solv)
    lr_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    lr_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(lr_result)) 
''')
kode(code43)

st.write('Logistic Regression Model')
lr_result = defaultdict(list)

for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    model = LogisticRegression(solver=solv, class_weight='balanced')
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    y_valid_predict = model.predict(X_valid)
    lr_result['solver'].append(solv)
    lr_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    lr_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(lr_result)) 


st.markdown('''- Pada model **Logistic Regression** yang kita latih sebelum dilakukan peningkatan kualitas data parameter **solver = lbfgs** merupakan parameter paling baik dengan tingkat *f1 score* sebesar **84%** dan **82%** untuk *train* dan *validation* set-nya. 

- Sedangkan pada model ini setelah kita meningkatkan kualitas data semua parameter memiliki hasil yang serupa tetapi mengalami penurunan *f1 score* pada dataset yang cukup signifikan atau menjadi **78%** untuk data *train*, bahkan *validation* set tidak mencapai minimal ambang batas yang ditentukan hanya mendapatkan angka **46%**.

- Setelah dilakukan peningkatan kualitas data pada model **Logistic Regression** justru mengalami penurunan nilai *f1 score* dari model sebelumnya dengan paramater **solver = lbfgs**, tetapi mengalami peningkatan pada parameter lainnya.
''')


code44 = ('''
# create the best logisticregression model
lr_model = LogisticRegression(solver='newton-cg', class_weight='balanced')
lr_model.fit(X_upsampled, y_upsampled)
''')
kode(code44)

lr_model = LogisticRegression(solver='newton-cg', class_weight='balanced')
lr_model.fit(X_upsampled, y_upsampled)


code45 = ('''
# accuracy score lg model compare
st.write('Logistic Regression Model'), st.write('')
y_train_lr_pred = lr_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_lr_pred))
y_valid_lr_pred = lr_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_lr_pred))
y_test_lr_pred = lr_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_lr_pred))
''')
kode(code45)

st.write('Logistic Regression Model'), st.write('')
y_train_lr_pred = lr_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_lr_pred))
y_valid_lr_pred = lr_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_lr_pred))
y_test_lr_pred = lr_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_lr_pred))


st.markdown('''Pada model ini juga nilai *accuracy* pada data *training* lebih rendah daripada *validation* dan *test* memiliki persentase antara **69% - 72%**.
''')


code46 = ('''
# membuat kurva ROC
probabilities_valid = lr_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('Logistic Regression'), st.write()
st.write('f1 Score =', f1_score(y_test, y_test_lr_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)
''')
kode(code46)

probabilities_valid = lr_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('Logistic Regression'), st.write()
st.write('f1 Score =', f1_score(y_test, y_test_lr_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)


st.markdown('''**AUC-ROC Score** menunjukan nilai **83%** dengan kurva cukup tinggi, hal ini menunjukan model kita sudah cukup baik dari model acak.
''')


code47 = ('''
# classification report the lr model of data test
st.write('Logistic Regression Classification Report'), st.write()
st.write(classification_report(y_test, y_test_lr_pred)), st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_lr_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)''')
kode(code47)

st.write('Logistic Regression Classification Report'), st.write()
st.write(classification_report(y_test, y_test_lr_pred)), st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_lr_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)

st.markdown('''- **Precision** pada pada nasabah yang *exited* dan *stay* memiliki nilai persentase **91%** dan **39%**, model ini hanya baik saat memprediksi nasabah yang setia daripada yang *exited*. 

- Nilai **recall** pada nasabah yang setia dan tidak sebesar **70%** dan **75%** yang artinya dari semua data nasabah yang benar-benar *stay* dan *exited* model ini cukup baik untuk memprediksi kedua kelompok tersebut.

### 6.3. Random Forest
''')


code48 = ('''
# membuat model random forest
st.write('Random Forest Model')
rf_result = defaultdict(list)

for n in range(1,21):
    model = RandomForestClassifier(random_state=42, max_depth=n, class_weight='balanced')
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    y_valid_predict = model.predict(X_valid)
    rf_result['max_depth'].append(n)
    rf_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    rf_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(rf_result))
''')
kode(code48)

st.write('Random Forest Model')
rf_result = defaultdict(list)

for n in range(1,21):
    model = RandomForestClassifier(random_state=42, max_depth=n, class_weight='balanced')
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    y_valid_predict = model.predict(X_valid)
    rf_result['max_depth'].append(n)
    rf_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    rf_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(rf_result))


st.markdown('''- Model **Random Forest** yang kita latih sebelum dilakukan peningkatan kualitas data, nilai **max_depth = 55** merupakan parameter paling baik dengan tingkat *f1 score* sebesar **95%** dan **59%** untuk *train* dan *validation* set-nya dan memiliki selisih yang sangat jauh untuk kedua data tersebut. 

- Sedangkan pada model setelah kita meningkatkan kualitas data, nilai **max_depth = 13** merupakan parameter terbaik dang mengalami kenaikan *f1 score* pada *validation set* sehingga antara kedua data *train* dan *validation* memiliki selisih cukup signifikan atau menjadi **96%** dan **59%** untuk data *train* dan *validation* set.

- Setelah dilakukan peningkatan kualitas data pada model **Random Forest** mengalami tidak terlalu mengalami perubahan  yang signifikan pada nilai *f1 score*.
''')


code49 = ('''
# create RandomForestClassifier best model
rf_model = RandomForestClassifier(random_state=42, max_depth=13, class_weight='balanced') 
rf_model.fit(X_upsampled, y_upsampled)
''')
kode(code49)

rf_model = RandomForestClassifier(random_state=42, max_depth=13, class_weight='balanced') 
rf_model.fit(X_upsampled, y_upsampled)


code50 = ('''
# accuracy score rf model compare
st.write('Random Forest Model')
st.write('')
y_train_rf_pred = rf_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_rf_pred))
y_valid_rf_pred = rf_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_rf_pred))
y_test_rf_pred = rf_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_rf_pred))
''')
kode(code50)

st.write('Random Forest Model')
st.write('')
y_train_rf_pred = rf_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_rf_pred))
y_valid_rf_pred = rf_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_rf_pred))
y_test_rf_pred = rf_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_rf_pred))


st.markdown('''Pada model ini tingkat *accuracy* pada *training* set sangat tinggi hampir mendekati nilai sempurna sebesar **99%** sedangkan *validation* dan *test* memiliki persentase cukup jauh meskipun masih tergolong cukup tinggi sebesar **86%** dan **84%**.
''')


code51 = ('''
# membuat kurva ROC
probabilities_valid = rf_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('Random Forest'), st.write()
st.write('f1 Score =', f1_score(y_test, y_test_rf_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)
''')
kode(code51)

probabilities_valid = rf_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('Random Forest'), st.write()
st.write('f1 Score =', f1_score(y_test, y_test_rf_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)


st.markdown('''**AUC-ROC Score** menunjukan nilai **83%** dengan kurva cukup tinggi, hal ini menunjukan model kita sudah cukup baik dari model acak.
''')


code52 = ('''
# classification report the rf model of data test
st.write('Random Forest Classification Report'), st.write()
st.write(classification_report(y_test, y_test_rf_pred))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_rf_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)''')
kode(code52)

st.write('Random Forest Classification Report'), st.write()
st.write(classification_report(y_test, y_test_rf_pred))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_rf_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)

st.markdown('''- **Precision** pada pada nasabah yang *exited* dan *stay* memiliki nilai persentase **89%** dan **63%**, artinya ini model cukup baik dalam memprediksi nasabah yang tetap *stay* dan *exited* meskipun memiliki selisih yang cukup tinggi.


- Nilai **recall** pada nasabah yang setia dan  tidak sebesar **91%** dan **58%** yang artinya dari semua data nasabah yang benar-benar *stay* dan *exited* model ini hanya mampu memprediksi dengan benar pada nasabah yang setia dengan sangat baik daripada nasabah yang *exited*.

### 6.4. K-Nearest Neighbors
''')


code53 = ('''
# membuat model knn
st.write('K-Nearest Neighbors Model'), st.write()

st.write('K-Nearest Neighbors Model')
knn_result = defaultdict(list)

for n in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    knn_result['n_neighbors'].append(n)
    knn_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    knn_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(knn_result))
''')
kode(code53)

st.write('K-Nearest Neighbors Model'), st.write()

st.write('K-Nearest Neighbors Model')
knn_result = defaultdict(list)

for n in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_upsampled, y_upsampled)
    y_train_predict = model.predict(X_upsampled)
    knn_result['n_neighbors'].append(n)
    knn_result['train_f1_score'].append(f1_score(y_upsampled, y_train_predict))
    knn_result['valid_f1_score'].append(f1_score(y_valid, y_valid_predict))

st.write(pd.DataFrame(knn_result))


st.markdown('''- Model **K-Nearest Neighbors** yang kita latih sebelum dilakukan peningkatan kualitas data, nilai **n-neighbors = 13** merupakan parameter paling baik dengan tingkat *f1 score* sebesar **94%** dan **42%** untuk *train* dan *validation* set-nya dan memiliki selisih yang sangat jauh untuk kedua data tersebut dan tidak ada parameter yang bisa mencapai *treshold* yang ditentukan.

- Sedangkan pada model ini setelah kita meningkatkan kualitas data, nilai **n-neighbors = 18** merupakan parameter terbaik dan mengalami kenaikan *f1 score* pada *validation set* atau menjadi **82%** dan **50%** untuk data *train* dan *validation* set meskipun belum dapat mencapi nilai ambang batas yang ditentukan dengan beberapa *tuning* parameter.

- Setelah dilakukan peningkatan kualitas data pada model **K-Nearest Neighbors** mengalami peningkatan pada nilai *f1 score* artinya peningkatan kualitas data cukup memengaruhi pada pembelajaran model ini meskipun tidak dapat mencapai nilai *treshold* yang ditentukan.
''')


code54 = ('''
# create the best KNeighborsClassifier model
knn_model = KNeighborsClassifier(n_neighbors=18)
knn_model.fit(X_upsampled, y_upsampled)
''')
kode(code54)

knn_model = KNeighborsClassifier(n_neighbors=18)
knn_model.fit(X_upsampled, y_upsampled)


code55 = ('''
# accuracy score knn model compare
st.write('K-Nearest Neighbors Model'), st.write('')
y_train_knn_pred = knn_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_knn_pred))
y_valid_knn_pred = knn_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_knn_pred))
y_test_knn_pred = knn_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_knn_pred))
''')
kode(code55)

st.write('K-Nearest Neighbors Model'), st.write('')
y_train_knn_pred = knn_model.predict(X_upsampled)
st.write('Training set accuracy =', accuracy_score(y_upsampled, y_train_knn_pred))
y_valid_knn_pred = knn_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_knn_pred))
y_test_knn_pred = knn_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_knn_pred))


st.markdown('''Pada model ini juga memiliki kemiripan dengan dua model sebelumnya yang memiliki nilai *accuracy* pada *training* lebih rendah dibandingkan *validation* dan *test* set, dengan persentese antara **76% - 82%**.
''')


code56 = ('''
# membuat kurva ROC
probabilities_valid = knn_model.predict_proba(X_test)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('K-Nearest Neighbors'), st.write('')
st.write('f1 Score =', f1_score(y_test, y_test_knn_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)
''')
kode(code56)

probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
auc_roc = roc_auc_score(y_test, probabilities_one_valid)

st.write('K-Nearest Neighbors'), st.write('')
st.write('f1 Score =', f1_score(y_test, y_test_knn_pred))
st.write('AUC-ROC Score =', auc_roc), st.write('')

fig, ax = plt.subplots()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC')
st.pyplot(fig)


st.markdown('''**AUC-ROC Score** menunjukan nilai **80%** dengan kurva cukup tinggi, hal ini menunjukan model kita sudah cukup baik dari model acak.
''')


code57 = ('''
# classification report the knn model of data test
st.write('K-Nearest Neighbors Classisfication Report'), st.write('')
st.write(classification_report(y_test, y_test_knn_pred))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_knn_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)''')
kode(code57)

st.write('K-Nearest Neighbors Classisfication Report'), st.write('')
st.write(classification_report(y_test, y_test_knn_pred))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_knn_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)

st.markdown('''- **Precision** pada pada nasabah yang tidak *exited* memiliki nilai persentase **90%**, artinya prediksi pada nasabah yang tetap menggunakan layanan bank memiliki nilai sangat tinggi dengan selisih yang sangan jauh daripada yang tidak setia yang memiliki persentase ketepatan sebesar **46%**.


- Nilai **recall** pada nasabah yang setia sebesar **80%** yang artinya dari semua data nasabah yang benar-benar *stay* cukup banyak terprediksi dengan benar, sedangkan nilai **recall** untuk nasabah yang *exited* mencapai **66%**.

# Consclusions

**1. Data Preparation**
- Kita memulai dengan memauat dataset yang terdiri dari **14** kolom dan **10.000** baris, tipe-tipe kolom sudah didefinisikan dengan benar.
- Terdapat *missing value* sebanyak **909** baris atau **9%** dari data pada kolom `tenure` dan telah kita isi dengan rata-rata dari usia.
- Kita juga mengubah *register* nama-nama kolom menjadi huruf kecil untuk memudahkan dalam analisis.

**2. EDA and Data Visualization**
- Dari kedua kelompok nasabah yang *exited* dan yang tidak, mayoritas dari mereka memiliki rata-rata credit score antara **550 - 750**.
- Terdapat banyak nasabah yang tidak memiliki sisa saldo sama sekali dalam *account* mereka ada sekitar **500** orang dari kelompok **exited** dan lebih dari **3000** nasabah dari kelompok yang *stay*, mayoritas lain dari mereka juga memiliki rata-rata *balance* antara **100.000 - 150.000**
- *Estimated Salary* memiliki penyebaran yang merata mulai dari **0 - 200.000**.
- Mayoritas dari kelompok yang tetap setia memiliki rentang usia di bawah **50** tahun, sedangkan dari kelompok **exited** memiliki rentang usia **40 - 60** tahun.
- Nasabah Bank Beta mayoritas berasal dari Prancis
- Mayoritas dari nasabah Bank Beta menggunakan setidaknya **1-2** produk yang dikeluarkan oleh bank.
- Mayoritas dari nasabah Bank Beta juga memiliki kartu.

**3. Split the Data**
- Perbandingan data pada kolom `exited` sebesar **80** dengan **20** kita membagi data menjadi **3** jenis dengan proporsi **70%** *training set*, **15%** *validation set*, dan **15%** untuk *test set*.

**4. Model tanpa Penyesuaian Kelas**
1. Pada model **Decision Tree Classifier**, hasil-hasil yang didapat tidak dapat mencapai batas untuk *fi score* pada `train set` dan `validation set`. Parameter yang paling baik adalah **max_dept = 7**.
2. Model **Logistik Regressiion** hasil yang didapat lebih tinggi daripada **Decision Tree**, dengan nilai pada *train* dan *validation* set sebesar **84%** dan **82%** dan telah melewati nilai *treshold*. Model terbaik didapatkan dengan menetapkan parameter **solver = lbfgs**.
3. Pada model **Random Forest** model yang paling baik didapatkan dengan menetapkan **n-estimators = 5**, memiliki selisih yang sangat jauh antara *training set* dan *validation set*-nya.
4. Pada model **K-Nearest Neighbors** kita menetapkan **n-neighbors** dalam rentang **1 - 20** dan model terbaik didapatkan pada nilai **n-neigbors = 13** meskipun **f1**-nya tidak ada yang bisa melebihi nilai ambang batas yang ditentukan.

**5. Meningkatkan Kualitas Model**
- Kita meningkatkan kualitas model dengan metode **upsampling** dengan mengacak sebanyak **3** kali untuk menyesuaikan keseimbangan kelas dan menggunakan metode **StandardScaller** untuk menyesuaikan skala.
- Kita juga menerapkan paramater `class_weight = balanced` pada beberapa model.

**6. Models Setelah Kualitas Ditingkatkan**
1. Pada model **Decision Tree Classifier** nilai **max_depth = 5** merupakan parameter terbaik dengan peningkatan *f1 score* pada masing-masing dataset sekitar **19%** atau menjadi **80%** dan **55%** untuk data *train* dan *validation* set.
2. Model **Logistik Regression** mengalami penurunan *f1 score* pada dataset sekitar **15%** atau menjadi **78%** untuk data *train*, bahkan *validation* set tidak mencapai minimal ambang batas yang ditentukan hanya mendapatkan angka **46%**, model ini lebih baik sebelum data *balanced*.
3. Model **Random Forest** nilai **n-estimators = 9** merupakan parameter terbaik dang mengalami kenaikan *f1 score* pada *validation set* sehingga antara kedua data *train* dan *validation* memiliki seilisih cukup signifikan atau menjadi **82%** dan **50%** untuk data *train* dan *validation* set dan tidak mencapai nilai ambang batas.
4. Model **K-Nearest Neighbors** nilai **n-neighbors = 18** merupakan parameter terbaik dan mengalami kenaikan *f1 score* pada *validation set* atau menjadi **69%** dan **57%** untuk data *train* dan *validation* set meskipun belum dapat mencapi nilai ambang batas yang ditentukan dengan beberapa *tuning* parameter.

Nilai **AUC-ROC Score** pada semua model berada diatas **70 - 80%** hal ini menunjukan model kita lebih baik daripada model acak.

**Main Consclusion**

Kita menemukan model terbaik setelah *balancing data* dengan nilai *f1 score* paling tinggi pada **Random Forest** dengan parameter `max_depth` di-set ke angka **13**, memperoleh *f1 score* **60%** pada *test set*.
''')
