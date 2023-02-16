# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ml libraries
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_confusion_matrix

import streamlit as st
import io

sns.set()

def kode(codes):
    st.code(codes, language='python')

def buffer(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    detail = buffer.getvalue()
    return st.text(detail)


st.markdown('''# Megaline (Introduce to Machine Learning)

Operator seluler Megaline merasa tidak puas karena banyak pelanggan mereka yang masih menggunakan paket lama. Perusahaan tersebut ingin mengembangkan sebuah model yang dapat menganalisis perilaku konsumen dan merekomendasikan salah satu dari kedua paket terbaru Megaline: Smart atau Ultra.

Kami memiliki akses terhadap data perilaku para pelanggan yang sudah beralih ke paket terbaru (dari proyek kursus Analisis Data Statistik). Dalam tugas klasifikasi ini, Kami perlu mengembangkan sebuah model yang mampu memilih paket dengan tepat. Mengingat Kami telah menyelesaikan langkah pra-pemrosesan data, Kami bisa langsung menuju ke tahap pembuatan model.

Kami akan mengembangkan sebuah model yang memiliki accuracy setinggi mungkin. Pada proyek ini, ambang batas untuk tingkat accuracy-nya adalah 0,75. Kami akan memeriksa metrik accuracy model dengan menggunakan test dataset.

**Goals :**
1. Memisahkan data sumber menjadi training set, validation set, dan test set.
2. Memeriksa kualitas model yang berbeda dengan mengubah hyperparameter-nya. Menjelaskan secara singkat temuan-temuan yang  didapatkan dari penelitian ini.
3. Memeriksa kualitas model dengan menggunakan test set.
4. Melakukan sanity check terhadap model. 

## 1. Initialization
''')


code1 = ('''
# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ml libraries
from sklearn.model_selection import train_test_split
!pip install -U fast-ml
from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
''')
kode(code1)


st.markdown('''## 2. Load the Data and Preparation

### 2.1. Menampilkan *sample* data and mempelajarinya
''')


code2 = ('''
# Load the data
df = pd.read_csv('/datasets/users_behavior.csv')
''')
kode(code2)

df = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/users_behavior.csv')


code3 = ('''
# menampilkan sample dataset
df
''')
kode(code3)

df


code4 = ('''
# menampilkan informasi dataset
df.info()
''')
kode(code4)

buffer(df)


st.markdown('''Dataset kita terdiri dari **4** kolom dan **3214** baris, ada 4 kolom dideskripsikan sebagai **float** yaitu kolom `calls`, `minutes`, `messages`, dan `mb_used`. Kolom `minutes` dan `mb_used` seharusnya sudah memiliki type data yang sesuai karena memuat jumlah menit panggilan dan penggunaan data dalam *megabytes*, sedangkan kolom `calls` dan `messages` seharusnya memiliki type data **int** karena memuat jumlah panggilan dan jumlah pesan teks. Kolom `is_ultra` sudah memiliki type data yang sesuai.

### 2.2. Memperbaiki kualitas data
''')


code5 = ('''
# memperbaiki type data
for col in ['calls', 'messages']:
    df[col] = df[col].apply(np.int64)
    
df.info()
''')
kode(code5)

for col in ['calls', 'messages']:
    df[col] = df[col].apply(np.int64)
    
buffer(df)


st.markdown('''Kita telah memperbaiki type data kolom.
''')


code6 = ('''
# menampilkan statistik deskriptif dari dataset
df.describe()
''')
kode(code6)

st.write(df.describe()
)


st.markdown('''Dari tabel diatas kita mengetahui rata-rata pelanggan menghabiskan **63** jumlah panggilan, **438** menit waktu bicara, **38** pesan teks, dan **17** gb data dalam periode tersebut. Sementara untuk jumlah minimun pemakaian oleh pelanngan adalah tidak menggunakan semua kuota yang terjadi pada setiap layanan yang kita sediakan. Pemakaian tertinggi yaitu sebanyak **244** panggilan, **1632** menit, **244** teks, dan **49** gb data dalam periode tersebut.
''')


code7 = ('''
# memeriksa missing value
df.isna().sum()
''')
kode(code7)

st.write(df.isna().sum()
)


st.markdown('''Tidak ada *missing value* ditemukan.

## 3. EDA and Data Visualization
''')


code8 = ('''
# Menampilkan visualisasi data
for label in df.columns[:-1]:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plt.hist(df[df['is_ultra']==1][label], color='blue', label='Ultimate', alpha=0.7, density=True)
    plt.hist(df[df['is_ultra']==0][label], color='red', label='Surf', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('Amount')
    plt.xlabel(label)
    plt.legend()
    st.pyplot(fig)
''')
kode(code8)

for label in df.columns[:-1]:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plt.hist(df[df['is_ultra']==1][label], color='blue', label='Ultimate', alpha=0.7, density=True)
    plt.hist(df[df['is_ultra']==0][label], color='red', label='Surf', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('Amount')
    plt.xlabel(label)
    plt.legend()
    st.pyplot(fig)


st.markdown('''**Temuan :**
- Dari grafik banyaknya jumlah panggilan yang dilakukan, rata-rata pengguna melakukan panggilan sebanyak **30-80** dalam satu bulan.
- Rata-rata pengguna menghabiskan **250-500** menit panggilan dalam satu bulan.
- Para pengguna dari kedua paket rata-rata tidak menggunakan kuota pesan teks sama sekali, hal ini mungkin terjadi karena mereka sudah beralih pada aplikasi pesan *instant* atau *chatting* menggunakan data internet.
- Data yang digunakan oleh pengguna dalam kedua paket berada di sekitar angka **10.000-20.000 mb** dalam satu bulan.

## 4. Split the Data
''')


code9 = ('''
# Menampilkan jumlah kolom is_ultra
df.groupby('is_ultra').agg(count=('calls', 'count')).reset_index()
''')
kode(code9)

st.write(df.groupby('is_ultra').agg(count=('calls', 'count')).reset_index()
)


code10 = ('''
# menghitung rasio
df['is_ultra'].value_counts() / df.shape[0]
''')
kode(code10)

st.write(df['is_ultra'].value_counts() / df.shape[0]
)


code11 = ('''
# Membuat diagram
fig, ax = plt.subplots(figsize=(4,4))
sns.countplot(x='is_ultra', data=df, order=df['is_ultra'].value_counts().index)
plt.xlabel('Service')
plt.ylabel('Amount')
plt.title('Count of Package')
st.pyplot(fig)
''')
kode(code11)

fig, ax = plt.subplots(figsize=(4,4))
sns.countplot(x='is_ultra', data=df, order=df['is_ultra'].value_counts().index)
plt.xlabel('Service')
plt.ylabel('Amount')
plt.title('Count of Package')
st.pyplot(fig)


st.markdown('''- Kita melihat jumlah yang tidak seimbang antara jumlah pengguna paket pada dataset, hal ini tentu akan berdampak pada model yang akan kita buat. 

- Rasio kolom `is_ultra` memiliki perbandingan **70:30 %**, hal ini dapat mengakibatkan model tidak dapat mempelajari data dengan baik.
''')


code12 = ('''
# split the data in to training, validation, and test
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'is_ultra', train_size=0.70,
                                                                            valid_size=0.15, test_size=0.15, random_state=12)
''')
kode(code12)

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'is_ultra', train_size=0.70,
                                                                            valid_size=0.15, test_size=0.15, random_state=12)



st.markdown('''Kita akan menggunakan proporsi data sebanyak **70%** untuk *data training*, **15%** untuk *data validation*, dan **15%** untuk *data test*, kita menetapkan nilai `random_state` untuk menghindari perbedaan hasil jika kita menjalankan ulang kode pada *notebook*.
''')


code13 = ('''
# memeriksa dimensi data X_train
X_train.shape
''')
kode(code13)

st.write(X_train.shape
)


code14 = ('''
# memeriksa dimensi data X_valid
X_valid.shape
''')
kode(code14)

st.write(X_valid.shape
)


code15 = ('''
# memeriksa dimensi data X_test
X_test.shape
''')
kode(code15)

st.write(X_test.shape
)


code16 = ('''
# menampilkan sample data training
X_train.head()
''')
kode(code16)

st.write(X_train.head()
)


st.markdown('''## 5. Models

### 5.1. Logistic Regression
''')


code17 = ('''
# accuracy score lr model training set and validation set
st.write('Logistic Regression Model')
lr_result = defaultdict(list)

for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:    
    lr_model = LogisticRegression(solver=solv)
    lr_model.fit(X_train, y_train)
    lr_model_train_pred = lr_model.predict(X_train)
    lr_model_valid_pred = lr_model.predict(X_valid)
    lr_result['solver'].append(solv)
    lr_result['train_accuracy'].append(accuracy_score(y_train, lr_model_train_pred))
    lr_result['valid_accuracy'].append(accuracy_score(y_valid, lr_model_valid_pred))
    
st.write(pd.DataFrame(lr_result))
''')
kode(code17)

# accuracy score lr model training set and validation set
st.write('Logistic Regression Model')
lr_result = defaultdict(list)

for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:    
    lr_model = LogisticRegression(solver=solv)
    lr_model.fit(X_train, y_train)
    lr_model_train_pred = lr_model.predict(X_train)
    lr_model_valid_pred = lr_model.predict(X_valid)
    lr_result['solver'].append(solv)
    lr_result['train_accuracy'].append(accuracy_score(y_train, lr_model_train_pred))
    lr_result['valid_accuracy'].append(accuracy_score(y_valid, lr_model_valid_pred))
    
st.write(pd.DataFrame(lr_result))


st.markdown('''Hasil dari **Logistic Regression Model** menghasilkan tingkat *accuracy* sebesar **75%** untuk *training test* dan **74%** pada *validation test* dengan menetapkan **newton-cg** pada parameter `solver`, artinya *validation set* tidak mampu mencapai batas ketentuan tingkat *accuracy* sebesar 75%. Selanjutnya kita akan mencoba menerapkan model ini pada *test set*.
''')


code19 = ('''
# create the best logisticregression model
lr_model = LogisticRegression(solver='newton-cg')
lr_model.fit(X_train, y_train)
''')
kode(code19)

lr_model = LogisticRegression(solver='newton-cg')
lr_model.fit(X_train, y_train)


code20 = ('''
# accuracy score lg model compare
st.write('Logistic Regression Model')
st.write('-------------------------')
y_train_lr_pred = lr_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_lr_pred))
y_valid_lr_pred = lr_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_lr_pred))
y_test_lr_pred = lr_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_lr_pred))
''')
kode(code20)

st.write('Logistic Regression Model')
st.write('-------------------------')
y_train_lr_pred = lr_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_lr_pred))
y_valid_lr_pred = lr_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_lr_pred))
y_test_lr_pred = lr_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_lr_pred))


st.markdown('''Hasil penerapan model pada data *test set* tidak mampu menaikkkan tingkat *accuracy* dan semakin terjadi penurunan pada *accuracy* menjadi **70.8%** yang menjadikan model ini tidak memenuh syarat untuk dijadikan sebagai model *machine learning*.
''')


code21 = ('''
# classification report the lr model of data test
st.write(classification_report(y_test, y_test_lr_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_lr_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)
''')
kode(code21)

st.write(classification_report(y_test, y_test_lr_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_lr_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)


st.markdown('''- **Precision** pada paket ultimate lebih tinggi daripada paket surf, artinya prediksi pada paket ultimate lebih banyak yang benar daripada paket surf.

- Nilai **recall** pada paket ultimate sangat rendah yang artinya dari semua data yang benar-benar ultimate hanya sedikit yang terprediksi dengan benar.

Model ini tidak mampu memprediksi paket ultimate dengan baik, mungkin karena data yang kita punya tidak seimbang.

### 5.2. Decision Tree 
''')


code22 = ('''
# Data train and validation set test
st.write('Decision Tree Model')
dt_result = defaultdict(list)

for depth in range(1, 11):
    dt_model = DecisionTreeClassifier(max_depth=depth)
    dt_model.fit(X_train, y_train)
    dt_model_train_pred = dt_model.predict(X_train)
    dt_model_valid_pred = dt_model.predict(X_valid)
    dt_result['max_depth'].append(depth)
    dt_result['train_accuracy'].append(accuracy_score(y_train, dt_model_train_pred))
    dt_result['valid_accuracy'].append(accuracy_score(y_valid, dt_model_valid_pred))
    
st.write(pd.DataFrame(dt_result))
''')
kode(code22)

st.write('Decision Tree Model')
dt_result = defaultdict(list)

for depth in range(1, 11):
    dt_model = DecisionTreeClassifier(max_depth=depth)
    dt_model.fit(X_train, y_train)
    dt_model_train_pred = dt_model.predict(X_train)
    dt_model_valid_pred = dt_model.predict(X_valid)
    dt_result['max_depth'].append(depth)
    dt_result['train_accuracy'].append(accuracy_score(y_train, dt_model_train_pred))
    dt_result['valid_accuracy'].append(accuracy_score(y_valid, dt_model_valid_pred))
    
st.write(pd.DataFrame(dt_result))


st.markdown('''Dari beberapa proses *looping* yang kita gunakan dalam **Decision Tree Model** kita menetapkan parameter **max_dept** dengan *range* **1-11** dan mendapatkan beberapa hasil tingkat *accuracy* maka kita menetapkan menggunakan nilai **max_depth = 3** karena nilai tersebut telah melewati nilai ambang batas yang ditentukan dan tingkat *accuracy* antara *data train testing*, *data validation testing*, dan *data test testing* menghasilkan tingkat *accuracy* yang paling konsisten atau tidak tejadi *overfitting*.
''')


code24 = ('''
# create decision tree model
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)
''')
kode(code24)

dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)


code25 = ('''
# accuracy score dt model compare
st.write('Decision Tree Model')
st.write('-------------------------')
y_train_dt_pred = dt_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_dt_pred))
y_valid_dt_pred = dt_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_dt_pred))
y_test_dt_pred = dt_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_dt_pred))
''')
kode(code25)

st.write('Decision Tree Model')
st.write('-------------------------')
y_train_dt_pred = dt_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_dt_pred))
y_valid_dt_pred = dt_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_dt_pred))
y_test_dt_pred = dt_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_dt_pred))


st.markdown('''Pada model ini tingkat *accuracy* pada *training set* menghasilkan angka **80%**, pada *validation set* menghasilkan angka **79.2%**, dan **76.6%** pada *test set* yang menjadikan model ini telah melewati *threshold* tingkat *accuracy* yang ditentukan.
''')


code26 = ('''
# classification report the dt model of data test
st.write(classification_report(y_test, y_test_dt_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_dt_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)
''')
kode(code26)

st.write(classification_report(y_test, y_test_dt_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_dt_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)


st.markdown('''- **Precision** pada paket ultimate hampir sama daripada paket surf, artinya prediksi yang benar pada paket ultimate dan paket surf ultimate hampir setara.

- Pada model ini juga nilai **recall** pada paket ultimate sangat rendah yang artinya dari semua data yang benar-benar ultimate hanya sedikit yang terprediksi dengan benar.

Model ini juga tidak mampu memprediksi paket ultimate dengan baik, hal ini mungkin terjadi karena data kita punya tidak seimbang.

### 5.3. Random Forest
''')


code27 = ('''
# accuracy score rf model training and validation set test
st.write('Random Forest Model')
rf_result = defaultdict(list)

for n in range(1,6):
    rf_model = RandomForestClassifier(random_state=42, n_estimators=n)
    rf_model.fit(X_train, y_train) 
    rf_model_train_pred = rf_model.predict(X_train)
    rf_model_valid_pred = rf_model.predict(X_valid)
    rf_result['n_estimators'].append(n)
    rf_result['train_accuracy'].append(accuracy_score(y_train, rf_model_train_pred))
    rf_result['valid_accuracy'].append(accuracy_score(y_valid, rf_model_valid_pred))
    
st.write(pd.DataFrame(rf_result))
''')
kode(code27)

st.write('Random Forest Model')
rf_result = defaultdict(list)

for n in range(1,6):
    rf_model = RandomForestClassifier(random_state=42, n_estimators=n)
    rf_model.fit(X_train, y_train) 
    rf_model_train_pred = rf_model.predict(X_train)
    rf_model_valid_pred = rf_model.predict(X_valid)
    rf_result['n_estimators'].append(n)
    rf_result['train_accuracy'].append(accuracy_score(y_train, rf_model_train_pred))
    rf_result['valid_accuracy'].append(accuracy_score(y_valid, rf_model_valid_pred))
    
st.write(pd.DataFrame(rf_result))


st.markdown('''Hasil dari **Random Forest Model** menunjukan terjadinya *overfitting* pada *training set* dengan angka **95%** terhadap *validation set* menghasilkan angka **77%**, dengan parameter `n-estimator` ditetapkan pada nilai **3**.
''')


code29 = ('''
# create RandomForestClassifier best model
rf_model = RandomForestClassifier(random_state=42, n_estimators=3) 
rf_model.fit(X_train, y_train)
''')
kode(code29)

rf_model = RandomForestClassifier(random_state=42, n_estimators=3) 
rf_model.fit(X_train, y_train)


code30 = ('''
# accuracy score rf model compare
st.write('Random Forest Model')
st.write('-------------------------')
y_train_rf_pred = rf_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_rf_pred))
y_valid_rf_pred = rf_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_rf_pred))
y_test_rf_pred = rf_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_rf_pred))
''')
kode(code30)

st.write('Random Forest Model')
st.write('-------------------------')
y_train_rf_pred = rf_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_rf_pred))
y_valid_rf_pred = rf_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_rf_pred))
y_test_rf_pred = rf_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_rf_pred))


st.markdown('''Pada *test set* juga tidak menghasilkan perbedaan yang signifikan dengan menghasilkan tingkat *accuracy* sebesar **75%**. Dengan demikian model ini memang melewati nilai *threshold* yang ditentukan tetapi terjadinya *overfitting* menjadikan model ini tidak bisa kita gunakan.
''')


code31 = ('''
# classification report the rf model of data test
st.write(classification_report(y_test, y_test_rf_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_rf_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)
''')
kode(code31)

st.write(classification_report(y_test, y_test_rf_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_rf_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)


st.markdown('''- **Precision** pada paket surf lebih tinggi daripada paket ultimate, artinya prediksi pada paket surf lebih banyak yang benar daripada paket ultimate, hal ini berbeda dengan dua prediksi sebelumnya.

- Nilai **recall** pada paket ultimate lebih rendah dari paket surf tetapi lebih tinggi dari dua model sebelumnya yang artinya dari semua data yang benar-benar ultimate hanya sedikit yang terprediksi dengan benar.

Model ini juga tidak mampu memprediksi paket ultimate dengan baik, ketindakseimbangan data mungkin berpengaruh pada hasil tersebut.

### 5.4. KNN
''')


code33 = ('''
# accuracy score knn model training setand validation set test
st.write('Random Forest Model')
knn_result = defaultdict(list)

for n in [1, 2, 3, 5, 7, 10, 13, 16, 21, 25, 29, 30]:
    knn_model = KNeighborsClassifier(n_neighbors=n)
    knn_model.fit(X_train, y_train) 
    knn_model_train_pred = knn_model.predict(X_train)
    knn_model_valid_pred = knn_model.predict(X_valid)
    knn_result['n_estimators'].append(n)
    knn_result['train_accuracy'].append(accuracy_score(y_train, knn_model_train_pred))
    knn_result['valid_accuracy'].append(accuracy_score(y_valid, knn_model_valid_pred))
    
st.write(pd.DataFrame(knn_result))
''')
kode(code33)

st.write('Random Forest Model')
knn_result = defaultdict(list)

for n in [1, 2, 3, 5, 7, 10, 13, 16, 21, 25, 29, 30]:
    knn_model = KNeighborsClassifier(n_neighbors=n)
    knn_model.fit(X_train, y_train) 
    knn_model_train_pred = knn_model.predict(X_train)
    knn_model_valid_pred = knn_model.predict(X_valid)
    knn_result['n_estimators'].append(n)
    knn_result['train_accuracy'].append(accuracy_score(y_train, knn_model_train_pred))
    knn_result['valid_accuracy'].append(accuracy_score(y_valid, knn_model_valid_pred))
    
st.write(pd.DataFrame(knn_result))


code34 = ('''
# create the best KNeighborsClassifier model
knn_model = KNeighborsClassifier(n_neighbors=30)
knn_model.fit(X_train, y_train)
''')
kode(code34)

knn_model = KNeighborsClassifier(n_neighbors=30)
knn_model.fit(X_train, y_train)


st.markdown('''Hasil dari **K-Nearest Neighbors Classifier Model** menghasilkan tingkat *accuracy* pada *training test* yang cukup baik sebesar **76.9%**, pada *validation testing* menghasilkan tingkat accuracy sebesar **75%**. Angka-angka tersebut telah memenuhi kriteria ambang batas dari nilai minimal tingkat *accuracy* yang ditentukan dan tidak terjadi *overfitting*, selanjutnya kita akan menerapkannya pada *test set*.
''')


code35 = ('''
# accuracy score knn model compare
st.write('K-Nearest Neighbors Model')
st.write('-------------------------')
y_train_knn_pred = knn_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_knn_pred))
y_valid_knn_pred = knn_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_knn_pred))
y_test_knn_pred = knn_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_knn_pred))
''')
kode(code35)

st.write('K-Nearest Neighbors Model')
st.write('-------------------------')
y_train_knn_pred = knn_model.predict(X_train)
st.write('Training set accuracy =', accuracy_score(y_train, y_train_knn_pred))
y_valid_knn_pred = knn_model.predict(X_valid)
st.write('Validation set accuracy =', accuracy_score(y_valid, y_valid_knn_pred))
y_test_knn_pred = knn_model.predict(X_test)
st.write('Test set accuracy',  accuracy_score(y_test, y_test_knn_pred))


st.markdown('''Pada *test set* menyajikan hasil yang tidak baik karena tingkat *accuracy* menurun cukup signifikan hingga berada dibawah ketentuan *threshold* dengan angka **71.8%** yang menjadikan model ini tidak bisa kita gunakan sebagai *machine learning*.
''')


code36 = ('''
# classification report the knn model of data test
st.write(classification_report(y_test, y_test_knn_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_knn_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)
''')
kode(code36)

st.write(classification_report(y_test, y_test_knn_pred, target_names=['Surf', 'Ultimate']))
st.write()
# menampilkan confusion matrix data test
cm = confusion_matrix(y_test, y_test_knn_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Tabel Confusion Matrix')
st.pyplot(fig)


st.markdown('''- Pada model ini hasil dari **classification report** hampir sama dengan hasil dari model **logistic regression**, **precision** pada paket ultimate lebih tinggi daripada paket surf, artinya prediksi pada paket ultimate lebih banyak yang benar daripada paket surf.

- Hasil yang mirip juga didaptkan pada nilai **recall** pada paket ultimate sangat rendah yang artinya dari semua data yang benar-benar ultimate hanya sedikit yang terprediksi dengan benar.

Model ini tidak mampu memprediksi paket ultimate dengan baik, data yang tidak seimbang bisa menjadi penyebab hal ini terjadi.

# Consclusions

**1. Data Preparation**
- Kita memulai dengan memauat dataset yang terdiri dari **4** kolom dan **3214** baris, ada 4 kolom dideskripsikan sebagai **float** yaitu kolom `calls`, `minutes`, `messages`, dan `mb_used`.
- Kita mengubah kolom `calls` yang memuat jumlah panggilan dan `messages` yang memuat jumlah pesan terkirim menjadi **int**.

**2. EDA and Data Visualization**
- Dari grafik banyaknya jumlah panggilan yang dilakukan, rata-rata pengguna melakukan panggilan sebanyak **30-80** dalam satu bulan.
- Rata-rata pengguna menghabiskan **250-500** menit panggilan dalam satu bulan.
- Para pengguna dari kedua paket rata-rata tidak menggunakan kuota pesan teks sama sekali, hal ini mungkin terjadi karena mereka sudah beralih pada aplikasi pesan *instant* atau *chatting* menggunakan data internet.
- Data yang digunakan oleh pengguna dalam kedua paket berada di sekitar angka **10.000-20.000 mb** dalam satu bulan.

**3. Split the Data**
- Kita menggunakan library `fast_ml.model_development` untuk membagi data langsung menjadi **3** jenis dengan proporsi **70%** *training set*, **15%** *validation set*, dan **15%** untuk *test set*.

**4. Models**

Kita menggunakan beberapa model dengan sejumlah pengaturan pada parameternya untuk mendapatkan hasil terbaik:
1. Hasil dari **Logistic Regression Model** menghasilkan tingkat *accuracy* sebesar **75%** untuk *training test* dan **74%** pada *validation test* dengan menetapkan **newton-cg** pada parameter `solver`. Hasil penerapan model pada data *test set* tidak mampu menaikkkan tingkat *accuracy* dan semakin terjadi penurunan pada *accuracy* menjadi **70.8%** yang menjadikan model ini tidak memenuh syarat untuk dijadikan sebagai model *machine learning*.
2. Dari beberapa proses *looping* yang kita gunakan dalam **Decision Tree Model** kita menetapkan parameter **max_dept** dengan *range* **1-11** dan mendapatkan beberapa hasil tingkat *accuracy* maka kita menetapkan menggunakan nilai **max_depth = 3**. Pada model ini tingkat *accuracy* pada *training set* menghasilkan angka **80%**, pada *validation set* menghasilkan angka **79.2%**, dan **76.6%** pada *test set* yang menjadikan model ini telah melewati *threshold* tingkat *accuracy* yang ditentukan.
3. Hasil dari **Random Forest Model** menunjukan terjadinya *overfitting* pada *training set* dengan angka **95%** terhadap *validation set* menghasilkan angka **77%**, dengan parameter `n-estimator` ditetapkan pada nilai **3**. Pada *test set* juga tidak menghasilkan perbedaan yang signifikan dengan menghasilkan tingkat *accuracy* sebesar **75%**. Dengan demikian model ini memang melewati nilai *threshold* yang ditentukan tetapi terjadinya *overfitting* menjadikan model ini tidak bisa kita gunakan.
4. Hasil dari **K-Nearest Neighbors Classifier Model** menghasilkan tingkat *accuracy* pada *training test* yang cukup baik sebesar **76.9%**, pada *validation testing* menghasilkan tingkat accuracy sebesar **75%**. Pada *test set* menyajikan hasil yang tidak baik karena tingkat *accuracy* menurun cukup signifikan hingga berada dibawah ketentuan *threshold* dengan angka **71.8%** yang menjadikan model ini tidak bisa kita gunakan sebagai *machine learning*.

**Main Consclusion**

Kita menemukan model terbaik dengan nilai *accuracy* paling tinggi dan *margin* paling rendah pada **Decision Tree Model** dengan parameter `max_dept` di-set ke angka **3**.
''')
