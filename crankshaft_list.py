import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import streamlit as st
import io


def kode(codes):
    st.code(codes, language='python')


st.title('Vehicles Price')

st.markdown('''- [Faktor Apakah yang Menjual Sebuah Mobil?](#scrollTo=8BMXdmov-5kP)

    - [Inisialisasi](#scrollTo=pMAsm9rv-5kb)

        - [Memuat Data](#scrollTo=oRmWivrF-5kg)

        - [Mengeksplorasi Data Awal](#scrollTo=guetWc9y-5ki)

        - [Kesimpulan dan Langkah-Langkah Selanjutnya](#scrollTo=RKrAFZA6-5k1)

    - [Mengatasi Nilai-Nilai yang Hilang](#scrollTo=qF0rWUa3-5k7)

        - [Mengisi kolom paint_color](#scrollTo=ozofcPcx-5k-)

        - [Mengisi kolom is_4wd](#scrollTo=9olSp3wm-5lB)

        - [Mengisi kolom cylinders](#scrollTo=r0ogB1---5lE)

        - [Mengisi kolom model_year](#scrollTo=fUkMszFH-5lJ)

        - [Mengisi kolom odometer](#scrollTo=e44YK0x_-5lN)

    - [Memperbaiki Tipe Data](#scrollTo=mdFBF69T-5lY)

        - [Memperbaki kolom is_4wd](#scrollTo=HUcuIJCR-5la)

        - [Memperbaki kolom model_year](#scrollTo=2WobaSRj-5ld)

        - [Memperbaki kolom cylinders](#scrollTo=TmCGtmLP-5lf)

        - [Mengubah kolom date_posted menjadi Timestamp](#scrollTo=6n8lXkOR-5lg)

        - [Memperbaiki register kolom type](#scrollTo=XCVQ4eNV-5li)

    - [Memperbaiki Kualitas Data](#scrollTo=cjHOvwTs-5ll)

        - [Menambahkan kolom dayofweek_posted](#scrollTo=zu641pH2-5ln)

        - [Menambahkan kolom month_posted](#scrollTo=KKdSK4b_-5lp)

        - [Menambahkan kolom year_posted](#scrollTo=5GWZbrRH-5lr)

        - [Menambahkan kolom vehicle_age](#scrollTo=nwVCTQtB-5lt)

        - [Menambahkan kolom avg_distance](#scrollTo=nXQdzvTI-5lv)

        - [Mengganti value kolom condition menjadi rating](#scrollTo=GLBlx2O_-5ly)

    - [Memeriksa Data yang Sudah Bersih](#scrollTo=P3eRup6K-5l9)

    - [Mempelajari Parameter Inti](#scrollTo=TK6Mpizd-5mF)

    - [Mempelajari dan Menangani Outlier](#scrollTo=1cAr6kDd-5mO)

    - [Mempelajari Parameter Inti tanpa Outlier](#scrollTo=mcjk9nPF-5mk)

    - [Masa Berlaku Iklan](#scrollTo=A7OnyoEp-5mr)

        - [Distribusi days_listed](#scrollTo=C00SUUgU-5mw)

        - [Average days_listed](#scrollTo=86xmifHJ-5m0)

        - [Filter days_listed](#scrollTo=T4GEWShc-5m6)

    - [Harga Rata-Rata Setiap Jenis Kendaraan](#scrollTo=OMlfjd2k-5nE)

        - [Jumlah masing-masing tipe](#scrollTo=deoCIZmw-5nG)

        - [Harga rata-rata tipe kendaraan](#scrollTo=9l072tTM-5nO)

        - [Kendaraan yang paling bergantung pada iklan](#scrollTo=fUS3NvXS-5nU)

    - [Faktor Harga](#scrollTo=cVY27jlD-5nd)

        - [Korelasi Harga Sedan dengan Variabel Numerik](#scrollTo=nB8MZFmH-5nf)

        - [Korelasi Harga Sedan dengan Variabel Kategorik](#scrollTo=F6G1YKt1-5nq)

        - [Korelasi Harga Sedan dengan  semua Variabel](#scrollTo=7QQo67SM-5oA)

        - [Korelasi Harga SUV dengan Variabel Numerik](#scrollTo=rX3K_oWz-5oE)

        - [Korelasi Harga SUV dengan Variabel Kategorik](#scrollTo=7uXlhIwE-5rO)

        - [Korelasi Harga SUV dengan  semua Variabel](#scrollTo=AgdnTgqc-5rV)

    - [Kesimpulan Umum](#scrollTo=-KYazlAt-5rY)

''')


st.markdown('''# Faktor Apakah yang Menjual Sebuah Mobil?

Sebagai seorang analis di Crankshaft List. Ratusan iklan kendaraan gratis ditayangkan di situs web Perusahaan setiap hari. Tim kami perlu mempelajari kumpulan data selama beberapa tahun terakhir dan menentukan faktor-faktor yang memengaruhi harga sebuah kendaraan.

**Goals**

Dalam project ini kita akan befokus pada Exploratory Data Analysis dan Data Visualization yang akan dapat membantu kita dalam mengidentifikasi outlier dalam data, menemukan pola dalam data, dan memberikan *insight* baru. Visualisasi data juga membantu menyampaikan cerita dengan menggambarkan data ke dalam bentuk visual yang lebih mudah dipahami, dan menyoroti tren.

Studi ini untuk menjawab 5 hipostesis :
1. Apakah terdapat korelasi antara harga dengan usia kendaraan
2. Apakah terdapat korelasi antara harga dengan jarak tempuh kendaraan
3. Apakah terdapat korelasi antara harga dengan kondisi kendaraan
4. Apakah terdapat korelasi antara harga dengan warna kendaraan
5. Apakah terdapat korelasi antara harga dengan tipe transmisi kendaraan

## 1. Inisialisasi

Kita akan memulai dengan memuat beberapa library yang akan dibutuhkan pada project ini seperti `pandas`, `numpy`, dan `matplotlib`.
''')


code1 = ('''# Muat semua library yang kita butuhkan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
''')
kode(code1)


st.markdown('''### 1.1. Memuat Data

Memuat informasi dari dateset dan mamuat beberapa baris dari dataset untuk mendapatkan informasi dari data
''')


code2 = ('''# Memuat file data menjadi DataFrame
df = pd.read_csv('/datasets/vehicles_us.csv')
''')
kode(code2)


df = pd.read_csv('https://code.s3.yandex.net/datasets/vehicles_us.csv')


st.markdown('''### 1.2. Mengeksplorasi Data Awal

*Dataset* berisi kolom-kolom berikut:
- `price` — harga kendaraan
- `model_year` — tahun pembuatan/keluaran mobil
- `model` — merk dan seri mobil
- `condition` — kondisi mobil
- `cylinders` — jumlah blok mesin
- `fuel` — gas, disel, dan lain-lain.
- `odometer` — jarak tempuh kendaraan saat iklan ditayangkan
- `transmission` — tipe perpindahan kecepatan/power
- `paint_color` — warna cat eksterior mobil
- `is_4wd` — apakah kendaraan memiliki penggerak 4 roda 
- `date_posted` — tanggal iklan ditayangkan
- `days_listed` — jumlah hari iklan ditayangkan hingga dihapus
''')


code3 = ('''# menampilkan informasi/rangkuman umum tentang DataFrame
df.info()
''')
kode(code3)


buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)


st.markdown('''Dari data diatas kita dapat melihat dalam dataset terdapat 51525 baris dan 13 kolom, kita juga dapat mengetahui beberapa kolom dalam dateset kita terdapat valui yang hilang di sana, seperti pada kolom `model_year`, `cylinders`, `odometer`, `paint_color`, dan `is_4wd`. 
''')


code4 = ('''# menampilkan sampel data
df.head(10)
''')
kode(code4)


st.write(df.head(10))


st.markdown('''Setelah kita melihat sampel dari data kita mendapat gambaran pada kolom `type` pada value `SUV` ditulis dengan huruf kapital, apakah karena itu merupakan singkatan, sepertinya tidak melihat value lainnya ditulis dengan huruf kecil.
''')


code5 = ('''# Checking type of data
df.dtypes
''')
kode(code5)


st.write(df.dtypes)


st.markdown('''Dari deskripsi data di atas bisa kita dapatkan beberapa kolom dengan tipe data yang tidak seharusnya seperti `model_year`, `cylinders`, dan `odometer` seharusnya `integer`. Kolom `is_4wd` bisa kita deskripsikan dengan angka `1` untuk yes dan `0` untuk no, atau dengan `true` or `false`.
''')


code6 = ('''# Memeriksa distribusi kolom dengan 'missing_value'
df.isna().sum()
''')
kode(code6)


st.write(df.isna().sum()
)


st.markdown('''Kita dapat melihat pada kolom `is_4wd` missing value terjadi pada separuh data.
''')


code7 = ('''# Memeriksa distribusi kolom 'is_4wd'
df['is_4wd'].value_counts(dropna=False)
''')
kode(code7)


st.write(df['is_4wd'].value_counts(dropna=False)
)


st.markdown('''Jika kita asumsikan nilai `1.0` sudah memuat semua mobil dengan penggerak 4WD apakah semua nilai pada `NaN` sudah tidak memuat mobil 4WD dan hanya memuat mobil 2WD atau lainnya. Jika seperti itu kita hanya perlu mengganti menjadi `0` atau no.
''')


code8 = ('''# Memeriksa distribusi kolom 'cylinders'
df['cylinders'].value_counts(dropna=False)
''')
kode(code8)


st.write(df['cylinders'].value_counts(dropna=False)
)


st.markdown('''Apakah mungkin untuk kita dapat mengisi value yang hilang pada `cylinders` atau cukup bagi kita menggantinya dengan `unknown`.
''')


code9 = ('''# Memeriksa distribusi kolom 'paint_color'
df['paint_color'].value_counts(dropna=False)
''')
kode(code9)


st.write(df['paint_color'].value_counts(dropna=False)
)


st.markdown('''Sepertinya tidak mungkin bagi kita untuk mengisinya dengan `custom` lebih masuk akal untuk mengisinya dengan `unknown`.
''')


code10 = ('''# Memeriksa statistik deskriptif dari data
df.describe()
''')
kode(code10)


st.write(df.describe()
)


st.markdown('''Nilai yang hilang terjadi pada beberapa kolom, kolom yang paling banyak terdapat *missing value* adalah `is_4wd` yang memiliki *missing value* setengah dari jumlah barisnya.

### Kesimpulan dan Langkah-Langkah Selanjutnya

Kita belum dapat mengetahui secara pasti penyebab nilai yangh hilang, apakah karena *human error* atau memang tidak memiliki akses data yang cukup dengan kendaraan tersebut megingat beberapa kendaraan memiliki usia yang sangat tua bisa lebih dari seratus tahun.

Kita akan menangani *missing value* dengan metode berikut :
1. Kita akan mengisi `paint_color` dengan `unknown` adalah pilihan yang paling *possible*.
2. Kita akan menggati nilai yang hilang pada `is_4wd` dengan angka `0`.
3. Kita dapat mengisi kolom `cylinders` dengan nilai rata-rata dari `model`.
4. Kita bisa mengganti *missing value* pada kolom `model_year` dengan rata-rata dari `model`.
5. Terakhir kita akan menangani kolom `odometer` dengan mengisi rata-rata berdasarkan `model_year` dan `condition` dengan metode `groupby`.

## 2. Data Preprocessing

### 2.1. Mengatasi Nilai-Nilai yang Hilang

Beberapa kolom yang akan kta perbaiki :
1. Kolom `paint_color`.
2. Kolom `is_4wd`.
3. Kolom `cylinders`.
4. Kolom `model_year`.
5. Kolom `odometer`.
''')


code11 = ('''# Memeriksa duplikat
df.duplicated().sum()
''')
kode(code11)


st.write(df.duplicated().sum()
)


st.markdown('''Tidak ada nilai duplikat

#### 2.1.1. Mengisi kolom `paint_color`
''')


code12 = ('''# Kita akan mengisi missing value di kolom 'paint_color' dengan 'unknown'
df['paint_color'].fillna('unknown', inplace=True)
df['paint_color'].isna().sum()
''')
kode(code12)


df['paint_color'].fillna('unknown', inplace=True)
st.write(df['paint_color'].isna().sum())


st.markdown('''Semua baris telah terisi
''')


code13 = ('''# Memeriksa distribusi kolom 'paint_color' 
df_paint_color = df.pivot_table(index='paint_color', values='price', aggfunc= 'count')
df_paint_color
''')
kode(code13)


df_paint_color = df.pivot_table(index='paint_color', values='price', aggfunc= 'count')
df_paint_color


st.markdown('''#### 2.1.2. Mengisi kolom `is_4wd`
''')


code14 = ('''# Kita akan mengisi missing value di kolom 'is_4wd' dengan '0'
df['is_4wd'].fillna(0, inplace=True)
df['is_4wd'].isna().sum()
''')
kode(code14)


df['is_4wd'].fillna(0, inplace=True)
st.write(df['is_4wd'].isna().sum())


st.markdown('''Semua baris telah terisi
''')


code15 = ('''# Memeriksa distribusi kolom 'is_4wd' 
df_is_4wd = df.pivot_table(index='is_4wd', values='price', aggfunc= 'count')
df_is_4wd
''')
kode(code15)


df_is_4wd = df.pivot_table(index='is_4wd', values='price', aggfunc= 'count')
df_is_4wd


st.markdown('''#### 2.1.3. Mengisi kolom `cylinders`
''')


code16 = ('''# Memeriksa distribusi missing valu pada 'cylinders'
df_cylinders_nan = df[df['cylinders'].isna()]
(df_cylinders_nan['model'].value_counts() / df['model'].value_counts()).sort_values(ascending=False)
''')
kode(code16)


df_cylinders_nan = df[df['cylinders'].isna()]
st.write((df_cylinders_nan['model'].value_counts() / df['model'].value_counts()).sort_values(ascending=False))


code17 = ('''# Memeriksa distribusi 'cylinders' berdasarkan 'model'
df.groupby(['model', 'cylinders'])['cylinders'].count()
''')
kode(code17)


st.write(df.groupby(['model', 'cylinders'])['cylinders'].count()
)


st.markdown('''Sepertinya masuk akal bagi kita untuk mengisi kolom `cylinders` dengan `median` dari `groupby` kolom `model` dan `cylinders`, karena setiap model memiliki `cylinders` yang hampir sama atau tidak memiliki perbedaan yang signifikan.
''')


code18 = ('''# Mengisi nilai yang hilang di kolom 'cylinders'
df['cylinders'] = df.groupby(['model'])['cylinders'].transform(lambda x: x.fillna(x.median()))
df['cylinders'].isna().sum()
''')
kode(code18)


df['cylinders'] = df.groupby(['model'])['cylinders'].transform(lambda x: x.fillna(x.median()))
st.write(df['cylinders'].isna().sum())


st.markdown('''Semua baris telah terisi
''')


code19 = ('''# Memerikasa distribusi kolom 'cylinders'
df_cylinders_pivot = df.pivot_table(index='cylinders', values='price', aggfunc= 'count')
df_cylinders_pivot
''')
kode(code19)


df_cylinders_pivot = df.pivot_table(index='cylinders', values='price', aggfunc= 'count')
df_cylinders_pivot


st.markdown('''#### 2.1.4. Mengisi kolom `model_year`
''')


code20 = ('''# menampilkan mode_year berdasarkan model
df.groupby(['model', 'model_year'])['model_year'].count()
''')
kode(code20)


st.write(df.groupby(['model', 'model_year'])['model_year'].count()
)


st.markdown('''Kita akan mengisi *missing value* pada `model_year` dengan median dari tahun keluaran kendaraan tersebut karena kita asumsikan berdasarkan data di atas bahwa suatu model mobil dikerluarkan dalam kurun waktu kurang lebih `5` tahun.
''')


code21 = ('''# Mengisi value 'model_year' yang hilang
df['model_year'] = df.groupby(['model'])['model_year'].transform(lambda x: x.fillna(x.median()))
df['model_year'].isna().sum()
''')
kode(code21)


df['model_year'] = df.groupby(['model'])['model_year'].transform(lambda x: x.fillna(x.median()))
st.write(df['model_year'].isna().sum())


st.markdown('''Semua baris telah terisi
''')


code22 = ('''# Memeriksa distribusi
df_model_year = df.pivot_table(index='model_year', values='price', aggfunc= 'count')
df_model_year
''')
kode(code22)


df_model_year = df.pivot_table(index='model_year', values='price', aggfunc= 'count')
df_model_year


st.markdown('''#### 2.1.5. Mengisi kolom `odometer`
''')


code23 = ('''# Memeriksa nilai 'median' pada 'odometer' berdasarkan 'model_year' dan 'condition'
df.groupby(['model_year', 'condition'])['odometer'].median()
''')
kode(code23)


st.write(df.groupby(['model_year', 'condition'])['odometer'].median()
)


code24 = ('''# Mengisi value 'model_year' yang hilang
df['odometer'] = df['odometer'].fillna(df.groupby(['model_year', 'condition'])['odometer'].transform('median'))
df['odometer'].isna().sum()
''')
kode(code24)


df['odometer'] = df['odometer'].fillna(df.groupby(['model_year', 'condition'])['odometer'].transform('median'))
st.write(df['odometer'].isna().sum())


st.markdown('''Masih terdapat `7` *missing value*, mari kita isi berdasarkan `model_year` saja.
''')


code25 = ('''# Mengisi missing value 'odometer' berdasarkan tahun
df['odometer'] = df['odometer'].fillna(df.groupby(['model_year'])['odometer'].transform('median'))
df['odometer'].isna().sum()
''')
kode(code25)


df['odometer'] = df['odometer'].fillna(df.groupby(['model_year'])['odometer'].transform('median'))
st.write(df['odometer'].isna().sum())


st.markdown('''Masih terdapat `1` baris yang belum terisi, mari kita isi dengan nilai terdekat, berdasarkan `model_year`.
''')


code26 = ('''df['odometer'] = df.groupby('model_year')['odometer'].apply(lambda x: x.interpolate(method='nearest')).ffill()          
df['odometer'].isna().sum()
''')
kode(code26)


df['odometer'] = df.groupby('model_year')['odometer'].apply(lambda x: x.interpolate(method='nearest')).ffill()          
st.write(df['odometer'].isna().sum())



code27 = ('''# Memeriksa nilai 'median' pada 'odometer' berdasarkan setelah diperbaiki
df.groupby(['model_year', 'condition'])['odometer'].median()
''')
kode(code27)


st.write(df.groupby(['model_year', 'condition'])['odometer'].median()
)


st.markdown('''Sudah, tidak ada nilai yang hilang, mari kita memeriksa informasi dataset.
''')


code30 = ('''# Menampilkan deskripsi data setelah mengisi missing value
df.info()
''')
kode(code30)


st.write(df.info()
)


st.markdown('''Semua kolom telah memiliki jumlah yang sama, artinya missing value telah ditangani.
''')


code31 = ('''# Memeriksa duplikat sekali lagi
df.duplicated().sum()
''')
kode(code31)


st.write(df.duplicated().sum()
)


st.markdown('''Tidak ada duplikat ditemukan. 

### 2.2. Memperbaiki Tipe Data

Setelah kita memiliki data yang lengkap, selanjutnya kita akan memperbaiki tipe data

#### 2.2.1. Memperbaki kolom `is_4wd`
''')


code32 = ('''# Transform float to int
df['is_4wd'] = df['is_4wd'].astype(np.int64)
''')
kode(code32)


df['is_4wd'] = df['is_4wd'].astype(np.int64)


code33 = ('''# Transform int to boolean
df['i(s_4wd'] = df['is_4wd'].astype(bool)
df['is_4wd'].dtypes
''')
kode(code33)


df['is_4wd'] = df['is_4wd'].astype(bool)
st.write(df['is_4wd'].dtypes
)


st.markdown('''Kolom `is_4wd` sudah menjadi `boolean`.
''')


code34 = ('''# Memeriksa distribusi kolom 'is_4wd' 
df['is_4wd'].value_counts()
''')
kode(code34)


st.write(df['is_4wd'].value_counts()
)


st.markdown('''Kolom `is_4wd` sudah memiliki value yang benar.

#### 2.2.2. Memperbaki kolom `model_year`
''')


code35 = ('''# Transform 'float' to 'int'
df['model_year'] = df['model_year'].astype(np.int64)
df['model_year'].dtypes
''')
kode(code35)


df['model_year'] = df['model_year'].astype(np.int64)
st.write(df['model_year'].dtypes
)


st.markdown('''Kolom `model_year` sudah menjadi `integer`.

#### 2.2.3. Memperbaki kolom `cylinders`
''')


code36 = ('''# Transform 'float' to 'int'
df['cylinders'] = df['cylinders'].astype(np.int64)
df['cylinders'].dtypes
''')
kode(code36)


df['cylinders'] = df['cylinders'].astype(np.int64)
st.write(df['cylinders'].dtypes
)


st.markdown('''#### 2.2.4. Mengubah kolom `date_posted` menjadi `Timestamp`
''')


code37 = ('''# Mengganti tipe ke data ke format timestamp
df['date_posted'] = pd.to_datetime(df.date_posted, format='%Y-%m-%d')
df['date_posted'].dtype
''')
kode(code37)


df['date_posted'] = pd.to_datetime(df.date_posted, format='%Y-%m-%d')
st.write(df['date_posted'].dtype
)


st.markdown('''#### 2.2.5. Memperbaiki register kolom `type`
''')


code38 = ('''# Mengganti value kolom 'type' menjadi lower
df['type'] = df['type'].str.lower()
''')
kode(code38)


df['type'] = df['type'].str.lower()


code39 = ('''# Memeriksa distribusi
df_type = df.pivot_table(index='type', values='price', aggfunc= 'count')
df_type
''')
kode(code39)


df_type = df.pivot_table(index='type', values='price', aggfunc= 'count')
df_type


st.markdown('''Register telah diperbaiki.
''')


code40 = ('''# Menampilkan deskripsi data
df.info()
''')
kode(code40)


buffer1 = io.StringIO()
df.info(buf=buffer1)
s1 = buffer1.getvalue()
st.text(s1)


st.markdown('''## 3. Memperbaiki Kualitas Data

Kita akan menambahkan beberapa kolom :
1. Kolom `dayofweek_posted` untuk mengetahui hari iklan ditayangkan.
2. Kolom `vehicle_age` untuk usia kendaraan saat diposting.
3. Kolom `avg_distance` untuk rata-rata jarak tempuh kendaraan per tahun.
4. Kita juga akan mengganti value pada kolom `condition` berdasarkan rating.

### 3.1. Menambahkan kolom `dayofweek_posted`
''')


code41 = ('''# Tambahkan nilai datetime pada saat iklan ditayangkan 
df['dayofweek_posted'] = df['date_posted'].dt.weekday
df.head()
''')
kode(code41)


df['dayofweek_posted'] = df['date_posted'].dt.weekday
st.write(df.head()
)


st.markdown('''Kolom `dayofweek_posted` telah ditambahkan.

### 3.2. Menambahkan kolom `month_posted`
''')


code42 = ('''# Tambahkan nilai datetime pada saat iklan ditayangkan 
df['month_posted'] = df['date_posted'].dt.month
df.head()
''')


df['month_posted'] = df['date_posted'].dt.month
st.write(df.head()
)


st.markdown('''Kolom `month_posted` telah ditambahkan.

### 3.3. Menambahkan kolom `year_posted`
''')


code43 = ('''# Tambahkan nilai datetime pada saat iklan ditayangkan 
df['year_posted'] = df['date_posted'].dt.year
df.head()
''')

df['year_posted'] = df['date_posted'].dt.year
st.write(df.head()
)


st.markdown('''Kolom `year_posted` telah ditambahkan.

### 3.4. Menambahkan kolom `vehicle_age`
''')


code44 = ('''# Tambahkan usia kendaraan saat iklan ditayangkan
df['vehicle_age'] = (df['year_posted'] + 1) - df['model_year']
df.head()
''')
kode(code44)


df['vehicle_age'] = (df['year_posted'] + 1) - df['model_year']
st.write(df.head()
)


st.markdown('''Kolom `vehicle_age` telah ditambahkan.

### 3.5. Menambahkan kolom `avg_distance`
''')


code45 = ('''# Tambahkan jarak tempuh rata-rata kendaraan per tahun
df['avg_distance'] = df['odometer'] / df['vehicle_age']
df['avg_distance'] = np.round(df['avg_distance'])
df.head()
''')
kode(code45)


df['avg_distance'] = df['odometer'] / df['vehicle_age']
df['avg_distance'] = np.round(df['avg_distance'])
st.write(df.head()
)



st.markdown('''Kolom `avg_distance` telah ditambahkan.

### 3.6. Mengganti value kolom `condition` menjadi rating
''')


code46 = ('''# Checking distribution of 'condition'
df['condition'].value_counts()
''')
kode(code46)


st.write(df['condition'].value_counts()
)


code47 = ('''# Mungkin membantu untuk mengganti nilai pada kolom 'condition' dengan sesuatu yang dapat dimanipulasi dengan lebih mudah
condition_rate = {
    'salvage' : 0, 
    'fair' : 1, 
    'good' : 2, 
    'excellent' : 3, 
    'like new' : 4, 
    'new' : 5
}
''')
kode(code47)


condition_rate = {
    'salvage' : 0, 
    'fair' : 1, 
    'good' : 2, 
    'excellent' : 3, 
    'like new' : 4, 
    'new' : 5
}


code48 = ('''# Menerapkan dictionary ke setiap baris
df['condition'] = df['condition'].map(condition_rate)
''')
kode(code48)

df['condition'] = df['condition'].map(condition_rate)


code49 = ('''# Memeriksa distribusi
df_condition = df.pivot_table(index='condition', values='price', aggfunc= 'count')
df_condition
''')
kode(code49)


df_condition = df.pivot_table(index='condition', values='price', aggfunc= 'count')
df_condition


st.markdown('''Kolom `condition` telah deperbarui.

## 3.2. Memeriksa Data yang Sudah Bersih

Melihat deskripsi dan menampilkan sampel.
''')


code50 = ('''# Menampilkan informasi/rangkuman umum tentang DataFrame
df.info()
''')


buffer2 = io.StringIO()
df.info(buf=buffer2)
s2 = buffer2.getvalue()
st.text(s2)


st.markdown('''Beberapa kolom baru telah ditambahkan, tipe data juga telah diperbaiki.
''')


code51 = ('''# Menampilkan sampel data
df.tail(10)
''')
kode(code51)


st.write(df.tail(10)
)


st.markdown('''## 4. Explanatory Data Analysis

### 4.1. Mempelajari Parameter Inti

Parameternya adalah :

- Harga
- Usia kendaraan ketika iklan ditayangkan
- Jarak tempuh
- Jumlah silinder
- Kondisi

Untuk mempermudah pekerjaan kita akan membuat beberapa fungsi untuk menghindari pengulangan yang tidak diperlukan.
''')


code52 = ('''# Menampilkan statistik deskriptif data
df.describe()
''')
kode(code52)


st.write(df.describe()
)


code53 = ('''# Membuat histogram
for column in ['price', 'vehicle_age', 'avg_distance']:
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], bins=50, kde=False)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)   
''')
kode(code53)


for column in ['price', 'vehicle_age', 'avg_distance']:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df[column], bins=50, kde=False)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)   


code54 = ('''# Menampilkan boxplot
for column in ['price', 'vehicle_age', 'avg_distance']:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x=column)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)  
''')
kode(code54)


for column in ['price', 'vehicle_age', 'avg_distance']:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df, x=column)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)  


code55 = ('''# menampilkan countplot
for column in ['cylinders', 'condition']:
    plt.figure(figsize=(10,6))
    sns.countplot(x=column, data=df)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)
''')
kode(code55)


for column in ['cylinders', 'condition']:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig)



st.markdown('''Gambaran yang bisa diambil :
1. Ada sekitar 75% mobil berada pada harga di bawah \\$16000, beberapa mobil memiliki harga \\$1 yang artinya kita bisa mendapatkan 3 mobil dengan menukar segelas kopi, kita melihat bahwa outlier ada di angka yang lebih dari sekitar \\$30000 dengan jumlah yang sangat besar hingga diatas \\$350000. 
2. Rata-rata usia kendaran adalah 13 tahun, 75% dibawah 17 tahun dan outlier pada usia melebihi 24 tahun, bebrapa mobil berusia sangat tua bahkan ratusan tahun.
3. 75% kendaraan memiliki jarak rata-rata dibawah 17000 miles, dengan rata-rata 13000 miles, sedangkan outlier pada rata-rata jarak ada di angka yang melebihi sekitar 30000 miles, beberapa mobil memiliki jarak rata-rata yang sangat tinggi hingga diatas 350000 miles per tahun.
4. Majority mobil memiliki 4, 6, dan 8 silinder mesin, beberapa mobil memiliki silinder lain daripada itu.
5. Majority mobil memiliki kondisi sangat baik, diikuti dengan kondis bagus, dan seperti baru.

### 4.2. Mempelajari dan Menangani Outlier

Kita akan menentukan *outiers* dari kolom price, usia, dan odometer.
''')


code56 = ('''# Menentukan batas bawah dan batas atas outlier
outlier_columns = ['price', 'vehicle_age', 'avg_distance']

Q1 = df[outlier_columns].quantile(0.25)
Q3 = df[outlier_columns].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR                  
''')
kode(code56)


outlier_columns = ['price', 'vehicle_age', 'avg_distance']

Q1 = df[outlier_columns].quantile(0.25)
Q3 = df[outlier_columns].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR                  


code57 = ('''# Menyimpan data tanpa outlier dalam DataFrame yang terpisah
good_df = df[ ~((df[outlier_columns] < lower_bound) | (df[outlier_columns] > upper_bound)).any(axis=1) ]      
''')
kode(code57)


good_df = df[ ~((df[outlier_columns] < lower_bound) | (df[outlier_columns] > upper_bound)).any(axis=1) ]      


code58 = ('''# Memfilter data, menghapus harga <$10 dan odometer <1
too_cheap = 10
too_low = 1

good_df = good_df.query('price >= @too_cheap')
good_df = good_df.query('avg_distance >= @too_low')
''')
kode(code58)


too_cheap = 10
too_low = 1

good_df = good_df.query('price >= @too_cheap')
good_df = good_df.query('avg_distance >= @too_low')


code59 = ('''# Menampilkan informasi dataset tanpa outliers
good_df.info()
''')
kode(code59)


buffer3 = io.StringIO()
good_df.info(buf=buffer3)
s3 = buffer3.getvalue()
st.text(s3)


st.markdown('''Setelah kita menghapus *outliers* baris yang tersisa dari dataset sebesar 46169. 

### 4.3. Mempelajari Parameter Inti tanpa Outlier

Membuat grafik baru dari kolom yang berisi *outliers* sebelumnya.
''')


code60 = ('''# Menampilkan statistik deskriptif dataset tanpa outliers
good_df.describe()
''')
kode(code60)


st.write(good_df.describe()
)


code61 = ('''# Membuat fungsi diagram
core_parameter = ['price', 'vehicle_age', 'avg_distance']

for column in core_parameter:
    plt.figure(figsize=(8,5))
    sns.histplot(good_df[column], bins=50, kde=True)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig) 
''')
kode(code61)


core_parameter = ['price', 'vehicle_age', 'avg_distance']

for column in core_parameter:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(good_df[column], bins=50, kde=True)
    plt.xlabel('')
    plt.ylabel('Ads')
    plt.title(column)
    st.pyplot(fig) 


st.markdown('''Kita bisa lihat distribusi dari kolom-kolom terlihat lebih baik dari sebelumnya tidak ada nilai yang terlalu jauh.

### 4.4. Masa Berlaku Iklan

Eksplorasi kolom days_listed untuk mengetahui berapa lama suatu iklan ditayangkan.

#### 4.4.1. Distribusi days_listed
''')


code62 = ('''# Menampilkan histogram 'days_listed'
plt.figure(figsize=(9,6))
sns.histplot(good_df['days_listed'], bins=50, kde=True)
plt.xlabel('Days')
plt.ylabel('Ads')
plt.title('Ads Duration')
st.pyplot(fig)
''')

kode(code62)


fig, ax = plt.subplots(figsize=(9,6))
sns.histplot(good_df['days_listed'], bins=50, kde=True)
plt.xlabel('Days')
plt.ylabel('Ads')
plt.title('Ads Duration')
st.pyplot(fig)


st.markdown('''Umumnya iklan ditayangkan selama 0-150 hari.

#### 4.4.2. Average days_listed
''')


code63 = ('''# Menampilkan statistik deskriptif 'days_listed'
good_df['days_listed'].describe()
''')
kode(code63)


st.write(good_df['days_listed'].describe()
)


code64 = ('''# Menampilkan histogram days_listed dengan mean dan median
mean=good_df['days_listed'].mean()
median=good_df['days_listed'].median()

plt.figure(figsize=(10,7))
ax_hist = sns.histplot(data=good_df, x='days_listed', bins=70, kde=True)
ax_hist.axvline(mean, color='r', linestyle='-', label='Mean')
ax_hist.axvline(median, color='g', linestyle='--', label='Median')
ax_hist.set(title='Days Listed', xlabel='', ylabel='Ads')
ax_hist.legend()
st.pyplot(fig)
''')
kode(code64)


mean=good_df['days_listed'].mean()
median=good_df['days_listed'].median()

fig, ax = plt.subplots(figsize=(10,7))
ax_hist = sns.histplot(data=good_df, x='days_listed', bins=70, kde=True)
ax_hist.axvline(mean, color='r', linestyle='-', label='Mean')
ax_hist.axvline(median, color='g', linestyle='--', label='Median')
ax_hist.set(title='Days Listed', xlabel='', ylabel='Ads')
ax_hist.legend()
st.pyplot(fig)


code65 = ('''# Menampilkan boxplot days_listed dengan mean dan median
plt.figure(figsize=(10,7))
ax_box = sns.boxplot(data=good_df, x='days_listed')
ax_box.axvline(mean, color='r', linestyle='-', label='Mean')
ax_box.axvline(median, color='g', linestyle='--', label='Median')
ax_box.set(title='Days Listed', xlabel='', ylabel='Ads')
ax_box.legend()
st.pyplot(fig)
''')
kode(code65)


fig, ax = plt.subplots(figsize=(10,7))
ax_box = sns.boxplot(data=good_df, x='days_listed')
ax_box.axvline(mean, color='r', linestyle='-', label='Mean')
ax_box.axvline(median, color='g', linestyle='--', label='Median')
ax_box.set(title='Days Listed', xlabel='', ylabel='Ads')
ax_box.legend()
st.pyplot(fig)


st.markdown('''Beberapa mobil di iklankan dalam waktu yang lama, rata-rata iklan diposting selama 33 hari dan 75% dibawah 53 hari.

#### 4.4.3. Filter days_listed

Kita akan memfilter data dari days listed agar tidak lebih dari 150 hari dan tidak kurang dari 1 hari.
''')


code66 = ('''# Memfilter data
too_fast = 1
too_long = 150

good_days_listed = good_df.query('@too_fast <= days_listed <= @too_long')
''')
kode(code66)

too_fast = 1
too_long = 150

good_days_listed = good_df.query('@too_fast <= days_listed <= @too_long')


code67 = ('''# Membuat diagram days_listed setelah difilter
plt.figure(figsize=(9,6))
sns.histplot(good_days_listed['days_listed'], bins=50, kde=True)
plt.xlabel('Days')
plt.ylabel('Ads')
plt.title('Ads Duration Good Data')
st.pyplot(fig)
''')
kode(code67)


fig, ax = plt.subplots(figsize=(9,6))
sns.histplot(good_days_listed['days_listed'], bins=50, kde=True)
plt.xlabel('Days')
plt.ylabel('Ads')
plt.title('Ads Duration Good Data')
st.pyplot(fig)


st.markdown('''Terlihat distribusi data sudah lebih baik dari sebelumnya.

### 4.5. Harga Rata-Rata Setiap Jenis Kendaraan

Melakukan explorasi untuk mengetahui kendaraan apa yang paling bergantung terhadapa iklan dan harga rata-rata dari tipe kendaraan paling popular.

#### 4.5.1. Jumlah masing-masing tipe
''')


code68 = ('''# Menampilkan jumlah masing-masing tipe
types_ads = good_df['type'].value_counts()
types_ads
''')
kode(code68)


types_ads = good_df['type'].value_counts()
types_ads


st.markdown('''Dua type mobil dengan jumlah iklan paling banyak yaitu, `suv` dan `sedan`.
''')


code69 = ('''# Membuat diagram
plt.figure(figsize=(10,6))
sns.countplot(x='type', data=good_df, order=good_df['type'].value_counts().index)
plt.xlabel('Type')
plt.ylabel('Ads')
plt.title('Count of Type')
st.pyplot(fig)
''')
kode(code69)


fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(x='type', data=good_df, order=good_df['type'].value_counts().index)
plt.xlabel('Type')
plt.ylabel('Ads')
plt.title('Count of Type')
st.pyplot(fig)


st.markdown('''#### 4.5.2. Harga rata-rata tipe kendaraan
''')


code70 = ('''# Menampilkan rata-rata harga setiap tipe kendaraan
type_pivot_table = good_df.pivot_table(index='type',
                                      values='price',
                                      aggfunc=['mean', 'median'], 
                                      ).reset_index()
type_pivot_table.sort_values(('median', 'price'), ascending=False)
''')
kode(code70)


type_pivot_table = good_df.pivot_table(index='type',
                                      values='price',
                                      aggfunc=['mean', 'median'], 
                                      ).reset_index()
st.write(type_pivot_table.sort_values(('median', 'price'), ascending=False)
)


code71 = ('''# Membuat barplot
type_pivot_table.sort_values(('mean', 'price'), ascending=False).plot(kind='bar', 
                                                                      figsize=(9, 6), 
                                                                      x='type', 
                                                                      y=['mean', 'median'], 
                                                                      label=['Mean','Median']
                                                                     )

plt.xlabel('Type')
plt.ylabel('Average Price')
plt.title('Average Price of Type')
st.pyplot(fig)
''')
kode(code71)


type_pivot_table.sort_values(('mean', 'price'), ascending=False).plot(kind='bar', 
                                                                      figsize=(9, 6), 
                                                                      x='type', 
                                                                      y=['mean', 'median'], 
                                                                      label=['Mean','Median']
                                                                     )

plt.xlabel('Type')
plt.ylabel('Average Price')
plt.title('Average Price of Type')
st.pyplot(fig)


st.markdown('''Overview :
1. Tipe bus, truck, dan pick-up memiliki rata-rata harga tertinggi artinya, mobil tipe angkutan umum dan angkutan masal memiliki average harga di atas mobil tipe lain.
2. Mobil tipe offroad, coupe, convertible, suv, dan other, memiliki harga rata-rata dibawah tipe angkutan umum, artinya mobil dengan tipe sport memiliki haraga rata-rata tertinggi kedua.
3. Tipe kendaran perkotaan yang berisi wagon, van, mini-van, sedan, dan hatchback memiliki harga rata-rata terendah dari jenis mobil lainnya.

#### 4.5.3. Kendaraan yang paling bergantung pada iklan
''')


code72 = ('''# Filter most two type
type_sedan = good_df.query('type == "sedan"')
type_suv = good_df.query('type == "suv"')
''')
kode(code72)


type_sedan = good_df.query('type == "sedan"')
type_suv = good_df.query('type == "suv"')


code73 = ('''# Membuat fungsi masing-masing type
def hist_plot(ct, col, title):
    plt.figure(figsize=(9,6))
    sns.histplot(ct[col], bins=30, kde=True)
    plt.xlabel('Price')
    plt.ylabel('Ads')
    plt.title(title)
    st.pyplot(fig)
''')
kode(code73)


def hist_plot(ct, col, title):
    fig, ax = plt.subplots(figsize=(9,6))
    sns.histplot(ct[col], bins=30, kde=True)
    plt.xlabel('Price')
    plt.ylabel('Ads')
    plt.title(title)
    st.pyplot(fig)


code74 = ('''# Statistik deskriptif type sedan
type_sedan['price'].describe()
''')
kode(code74)


st.write(type_sedan['price'].describe()
)


code75 = ('''# Distribusi harga sedan
hist_plot(type_sedan, 'price', 'Sedan Price')
''')
kode(code75)


st.write(hist_plot(type_sedan, 'price', 'Sedan Price')
)


st.markdown('''Rata-rata harga sedan dibawah \\$6000, 75% sedan berada pada harga di bawah \\$9000.
''')


code76 = ('''# Statistik deskriptif type suv
type_suv['price'].describe()
''')
kode(code76)


st.write(type_suv['price'].describe()
)


code77 = ('''# Distribusi harga suv
hist_plot(type_suv, 'price', 'SUV Price')
''')
kode(code77)


st.write(hist_plot(type_suv, 'price', 'SUV Price')
)


st.markdown('''Rata-rata harga SUV dibawah \\$9000, 75% sedan berada pada harga di bawah \\$14500.

### 4.6. Faktor Harga

Mengeksplor data untuk mencari faktor-faktor yang dapat memengaruhi harga dari kendaraan dengan beberapa parameter seperti usia mobil, jarak tempuh, warna, tipe transmisi, dan kondisi, dengan menggunakan matrik korelasi dan scatterplot.

#### 4.6.1. Korelasi Harga Sedan dengan Variabel Numerik
''')


code78 = ('''# Menghitung jumlah masing-masing kategori dari sedan
for col in ['paint_color', 'transmission', 'condition']:
    print()
    print(type_sedan[col].value_counts())
    print()
''')
kode(code78)


for col in ['paint_color', 'transmission', 'condition']:
    st.write()
    st.write(type_sedan[col].value_counts())
    st.write()


code79 = ('''# Menghapus kategori yang tidak valid dari sedan
good_type_sedan = type_sedan.query('condition != 0 and condition != 5')
good_type_sedan = good_type_sedan.query('paint_color != "purple" and paint_color != "yellow" and paint_color != "orange"')
''')
kode(code79)


good_type_sedan = type_sedan.query('condition != 0 and condition != 5')
good_type_sedan = good_type_sedan.query('paint_color != "purple" and paint_color != "yellow" and paint_color != "orange"')


code80 = ('''# Membuat fungsi scatterplot data numerikal dari sedan
num_variable = ['vehicle_age', 'avg_distance', 'condition']

for column in num_variable:
    print()
    print('Correlation between Price and', column, ':', good_type_sedan['price'].corr(good_type_sedan[column]))
    plt.figure(figsize=(9,6))
    sns.scatterplot(x=good_type_sedan[column], y=good_type_sedan['price'], data=good_type_sedan, alpha=0.2)
    plt.xlabel('')
    plt.ylabel('Price')
    plt.title('Sedan ' + column)
    st.pyplot(fig) 
    print()
''')
kode(code80)


num_variable = ['vehicle_age', 'avg_distance', 'condition']

for column in num_variable:
    st.write()
    st.write('Correlation between Price and', column, ':', good_type_sedan['price'].corr(good_type_sedan[column]))
    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(x=good_type_sedan[column], y=good_type_sedan['price'], data=good_type_sedan, alpha=0.2)
    plt.xlabel('')
    plt.ylabel('Price')
    plt.title('Sedan ' + column)
    st.pyplot(fig) 
    st.write()


st.markdown('''Kita hampir melihat adanya korelasi negatif antara harga sedan dengan kolom usia, meskipun titik-titiknya tidak terlalu jelas, kita tidak melihat korelasi antara harga dan rata-rata jarak, kita bisa melihat tabel korelasi untuk mengetahui secara pasti angkanya.
''')


code81 = ('''# Menampilkan tabel korelasi data numerikal pada sedan
type_sedan_metric_num = good_type_sedan[['price', 'vehicle_age', 'avg_distance', 'condition']]
type_sedan_metric_num.corr()
''')
kode(code81)


type_sedan_metric_num = good_type_sedan[['price', 'vehicle_age', 'avg_distance', 'condition']]
st.write(type_sedan_metric_num.corr()
)


st.markdown('''Kita melihat adanya koneksi negatif yang tidak terlalu kuat antara harga terhadap usia sebesar -0,65, artinya semakin tinggi usia dan odometer maka akan semakin rendah harga suatu mobil.

Kita juga bisa melihat korelasi yang sangat lemah antara harga dengan rata-rata jarak sebesar 0.08.
''')


code82 = ('''# Menampilkan heatmap korelasi data ta pada sedan
plt.figure(figsize=(9,7))
sns.heatmap(type_sedan_metric_num.corr(), annot=True)
st.pyplot(fig)
''')
kode(code82)


fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(type_sedan_metric_num.corr(), annot=True)
st.pyplot(fig)


st.markdown('''Kita juga mengetahui nilai koneksi yang kecil antara harga mobil terhadap kondisi dengan angka 0.31.
''')


code83 = ('''# Menampilkan diagram korelasi data kategorikal pada sedan
sns.pairplot(type_sedan_metric_num)
st.pyplot(fig)
''')
kode(code83)


fig = sns.pairplot(type_sedan_metric_num)
st.pyplot(fig)


st.markdown('''Kita bisa melihat persebaran dari data numerikal.

#### 4.6.2. Korelasi Harga Sedan dengan Variabel Kategorik
''')


code84 = ('''# Membuat kolom baru untuk data kategorikal menjadi integer
good_type_sedan['paint_color_num'] = good_type_sedan['paint_color'].astype('category').cat.codes
good_type_sedan['transmission_num'] = good_type_sedan['transmission'].astype('category').cat.codes
''')
kode(code84)


good_type_sedan['paint_color_num'] = good_type_sedan['paint_color'].astype('category').cat.codes
good_type_sedan['transmission_num'] = good_type_sedan['transmission'].astype('category').cat.codes


code85 = ('''# Menampilkan sampel data dari sedan
good_type_sedan.head()
''')
kode(code85)


st.write(good_type_sedan.head()
)


code86 = ('''# Membuat fungsi boxplot data kategorikal pada sedan dan suv
def box_plot(data, num, col, title):
    print('Correlation between Price and', num, ':', data['price'].corr(data[num]))
    plt.figure(figsize=(9,6))
    sns.boxplot(x=col, y='price', data=data, showfliers = False)
    plt.xlabel('')
    plt.ylabel('Price')
    plt.title(title)
    st.pyplot(fig) 
    print()
''')
kode(code86)


def box_plot(data, num, col, title):
    st.write('Correlation between Price and', num, ':', data['price'].corr(data[num]))
    fig, ax = plt.subplots(figsize=(9,6))
    sns.boxplot(x=col, y='price', data=data, showfliers = False)
    plt.xlabel('')
    plt.ylabel('Price')
    plt.title(title)
    st.pyplot(fig) 
    st.write()


code87 = ('''# Menampilkan boxplot kolom warna pada sedan
box_plot(good_type_sedan, 'paint_color_num', 'paint_color', 'Sedan Paint Color' )
''')
kode(code87)


st.write(box_plot(good_type_sedan, 'paint_color_num', 'paint_color', 'Sedan Paint Color' )
)


st.markdown('''Kita mengetahui bahwa mobil dengan warna hitam dan putih akan memiliki harga jual lebih tinggi dari warna lainnya, dengan warna hijau harga jual paling rendah, artinya kebanyakan konsumen lebih menyukai kendaraan dengan warna netral dibandingkan dengan warna yang terlihat mencolok.
''')


code88 = ('''# Menampilkan boxplot kolom transmisi pada sedan
box_plot(good_type_sedan, 'transmission_num', 'transmission', 'Sedan Transmisson')
''')
kode(code88)


st.write(box_plot(good_type_sedan, 'transmission_num', 'transmission', 'Sedan Transmisson')
)


st.markdown('''Mobil matic memiliki harga yang lebih tinggi dari mobil manual.
''')


code89 = ('''# Menampilkan tabel korelasi data kategorikal pada sedan
type_sedan_metric_cat = good_type_sedan[['price', 'paint_color_num', 'transmission_num']]
type_sedan_metric_cat.corr()
''')
kode(code89)


type_sedan_metric_cat = good_type_sedan[['price', 'paint_color_num', 'transmission_num']]
st.write(type_sedan_metric_cat.corr()
)


st.markdown('''Kita melihat tidak adanya koneksi harga terhadap warna dan transisi, mungkin jika warna dan transmisi kita bagi kedalam kelompok-kelompok kita akan menemukan sebuah korelasi.
''')


code90 = ('''# Menampilkan heatmap korelasi data kategorikal pada sedan
plt.figure(figsize=(7,5))
sns.heatmap(type_sedan_metric_cat.corr(), annot=True)
st.pyplot(fig)
''')
kode(code90)


fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(type_sedan_metric_cat.corr(), annot=True)
st.pyplot(fig)


st.markdown('''#### 4.6.3. Korelasi Harga Sedan dengan  semua Variabel 
''')


code91 = ('''# Menampilkan tabel korelasi semua data pada sedan
corr_sedan = good_type_sedan[['price','vehicle_age', 'avg_distance', 'condition', 'paint_color_num', 'transmission_num']]
corr_sedan.corr()
''')
kode(code91)


corr_sedan = good_type_sedan[['price','vehicle_age', 'avg_distance', 'condition', 'paint_color_num', 'transmission_num']]
st.write(corr_sedan.corr()
)


code92 = ('''# Menampilkan tabel korelasi seluruh data pada sedan
plt.figure(figsize=(11,9))
sns.heatmap(corr_sedan.corr(), annot=True)
st.pyplot(fig)
''')
kode(code92)


fig, ax = plt.subplots(figsize=(11,9))
sns.heatmap(corr_sedan.corr(), annot=True)
st.pyplot(fig)


st.markdown('''#### 4.6.4. Korelasi Harga SUV dengan Variabel Numerik
''')


code93 = ('''# Menghitung jumlah masing-masing kategori pada suv
for col in ['paint_color', 'transmission', 'condition']:
    print()
    print(type_suv[col].value_counts())
    print()
''')
kode(code93)


for col in ['paint_color', 'transmission', 'condition']:
    st.write()
    st.write(type_suv[col].value_counts())
    st.write()


code94 = ('''# Menghapus kategori yang tidak valid pada suv
good_type_suv = type_suv.query('condition != 0 and condition != 5')
good_type_suv = good_type_suv.query('paint_color != "purple" and paint_color != "yellow"')
''')
kode(code94)


good_type_suv = type_suv.query('condition != 0 and condition != 5')
good_type_suv = good_type_suv.query('paint_color != "purple" and paint_color != "yellow"')


code95 = ('''# Membuat fungsi scatterplot pada suv
for column in num_variable:
    print()
    print('Correlation between Price and', column, ':', good_type_suv['price'].corr(good_type_suv[column]))
    plt.figure(figsize=(9,6))
    sns.scatterplot(x=good_type_suv[column], y=good_type_suv['price'], data=good_type_suv, alpha=0.2)
    plt.xlabel('')
    plt.ylabel('Price')
    plt.title('SUV ' + column)
    st.pyplot(fig) 
    print()
''')
kode(code95)


for column in num_variable:
    st.write()
    st.write('Correlation between Price and', column, ':', good_type_suv['price'].corr(good_type_suv[column]))
    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(x=good_type_suv[column], y=good_type_suv['price'], data=good_type_suv, alpha=0.2)
    plt.xlabel('')
    plt.ylabel('Price')
    plt.title('SUV ' + column)
    st.pyplot(fig) 
    st.write()


st.markdown('''Kita juga hampir melihat adanya korelasi negatif antara harga suv dengan kolom usia, meskipun titik-titiknya tidak terlalu jelas kita tidak melihat korelasi harga dengan rata-rata jarak, kita bisa melihat tabel korelasi untuk mengetahui lebih detail.
''')


code96 = ('''# Menampilkan tabel korelasi data numerikal pada suv
type_suv_metric_num = good_type_suv[['price', 'vehicle_age', 'avg_distance', 'condition']]
type_suv_metric_num.corr()
''')
kode(code96)


type_suv_metric_num = good_type_suv[['price', 'vehicle_age', 'avg_distance', 'condition']]
st.write(type_suv_metric_num.corr()
)


st.markdown('''Kita bisa melihat adanya koneksi negatif yang tidak terlalu kuat antara harga suv terhadap usia sebesar -0,64  artinya semakin tinggi usia maka akan semakin rendah harga suatu mobil.

Kita tidak menemukan korelasi antara harga dan rata-rata-jarak hanya 0.16.
''')


code97 = ('''# Menampilkan heatmap korelasi data numerikal pada suv
plt.figure(figsize=(9,7))
sns.heatmap(type_suv_metric_num.corr(), annot=True)
st.pyplot(fig)
''')
kode(code97)


fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(type_suv_metric_num.corr(), annot=True)
st.pyplot(fig)


st.markdown('''Sama seperti sedan, kita juga mengetahui hampir tidak ada koneksi antara harga mobil terhadap kondisi dengan angka 0.29.
''')


code98 = ('''# Menampilkan diagram korelasi data numerikal pada suv
sns.pairplot(type_suv_metric_num)
st.pyplot(fig)
''')
kode(code98)


fig = sns.pairplot(type_suv_metric_num)
st.pyplot(fig)


st.markdown('''Distribusi variabel numerikal terhadap harga suv.

#### 4.6.5. Korelasi Harga SUV dengan Variabel Kategorik
''')


code99 = ('''# Membuat kolom baru untuk kategori suv menjadi integer
good_type_suv['paint_color_num'] = good_type_suv['paint_color'].astype('category').cat.codes
good_type_suv['transmission_num'] = good_type_suv['transmission'].astype('category').cat.codes
''')
kode(code99)


good_type_suv['paint_color_num'] = good_type_suv['paint_color'].astype('category').cat.codes
good_type_suv['transmission_num'] = good_type_suv['transmission'].astype('category').cat.codes


code100 = ('''# Menampilkan sampel data tipe suv
good_type_suv.head()
''')
kode(code100)


st.write(good_type_suv.head()
)


code101 = ('''# Menampilkan boxplot kolom warna dari suv
box_plot(good_type_sedan, 'paint_color_num', 'paint_color', ' SUV Paint Color')
''')
kode(code101)


st.write(box_plot(good_type_sedan, 'paint_color_num', 'paint_color', ' SUV Paint Color')
)


st.markdown('''Kita juga melihat warna mobil warna hitam dan putih lebih disukai konsumen dari warna lain.
''')


code102 = ('''# Menampilkan boxplot kolom warna dari transmisi
box_plot(good_type_suv, 'transmission_num', 'transmission', 'SUV Transmisson')
''')
kode(code102)


st.write(box_plot(good_type_suv, 'transmission_num', 'transmission', 'SUV Transmisson')
)


st.markdown('''Menarik disini bahwa mobil suv dengan tipe manual memiliki harga yang lebih tinggi dari versi matic.
''')


code103 = ('''# Menampilkan tabel korelasi data kategorikal pada suv
type_suv_metric_cat = good_type_suv[['price', 'paint_color_num', 'transmission_num']]
type_suv_metric_cat.corr()
''')
kode(code103)


type_suv_metric_cat = good_type_suv[['price', 'paint_color_num', 'transmission_num']]
st.write(type_suv_metric_cat.corr()
)


st.markdown('''Tidak ada korelasi antara harga suv dengan warna dan transmisi.
''')


code104 = ('''# Menampilkan heatmap korelasi data kategori pada suv
plt.figure(figsize=(7,5))
sns.heatmap(type_suv_metric_cat.corr(), annot=True)
st.pyplot(fig)
''')
kode(code104)


fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(type_suv_metric_cat.corr(), annot=True)
st.pyplot(fig)


st.markdown('''#### 4.6.6. Korelasi Harga SUV dengan  semua Variabel 
''')


code105 = ('''# Menampilkan tabel korelasi terhadap harga suv
corr_suv = good_type_suv[['price','vehicle_age', 'avg_distance', 'condition', 'paint_color_num', 'transmission_num']]
corr_suv.corr()
''')
kode(code105)


corr_suv = good_type_suv[['price','vehicle_age', 'avg_distance', 'condition', 'paint_color_num', 'transmission_num']]
st.write(corr_suv.corr()
)


code106 = ('''# Menampilkan heatmap korealsi dari harga suv
plt.figure(figsize=(11,9))
sns.heatmap(corr_suv.corr(), annot=True)
st.pyplot(fig)
''')
kode(code106)


fig, ax = plt.subplots(figsize=(11,9))
sns.heatmap(corr_suv.corr(), annot=True)
st.pyplot(fig)


st.markdown('''Heatmap dari semua variabel terhdap harga suv.

## Kesimpulan Umum

Kita telah melakukan beberapa tahap dalam memproses data mobil untuk mendapatkan kesimpulan.

**A. Tahap Praproses**

Dari eksplorasi yang kita lakukan kita mendapatkan beberapa konsklusi:
1. Kita memulai dengan ukuran dataset sebanyak **51525** baris dan **13** kolom, ada 5 kolom yang terdapat *missing value* yaitu model_year, cylinders, odometer, paint_color, dan is_4wd.
2. Langkah-langkah yang kita lakukan berikutnya adalah mengisi nilai dari kolom-kolom yang terdapat *missing value*, memperbaiki tipe data, memperbaiki kualitas data, dan menambahkan beberapa kolom.

Penyebab nilai yang hilang, bisa diakibatkan karena *human error* atau memang tidak memiliki akses data yang cukup dengan kendaraan tersebut megingat beberapa kendaraan memiliki usia yang sangat tua bisa lebih dari seratus tahun


**B. Tahap Esksplorasi**

Setelah tahap prapemrosesan data kita melakukan beberapa ekplorasi:
1. Menetapkan batas *outliers* dari kolom harga, usia, dan odometer, dan membuat dataset baru dengan jumlah baris sebanyak **46169**.
2. Kita juga memfilter untuk mendapatkan waktu iklan dengan rentang **1 - 150 hari**.
3. Kita mendapati bahwa tipe mobil yang paling populer adalah sedan dan SUV.


**C. Konsklusi**

Dari eksplorasi yang kita lakukan kita mendapatkan beberapa konsklusi:
1. Harga mobil terhadap usia memiliki koneksi negatif meskipun nilainya tidak terlalu tinggi, artinya mobil yang lebih baru akan memiliki harga yang lebih tinggi.
2. Harga mobil dan rata-rata jarak memiliki korelasi yang sangat lemah.
3. Harga dan kondisi menunjukan korelasi yang rendah.
4. Mobil dengan warna hitam dan putih memiliki harga yang lebih tinggi dari warna lainnya.
5. Sedangkan tipe transmisi tidak selalu menunjukan bahwa mobil matic akan lebih mahal dari mobil manual.
''')
