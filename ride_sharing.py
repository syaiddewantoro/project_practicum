import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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


st.markdown('''# Zuber

Kami bekerja sebagai analis untuk Zuber, sebuah perusahaan berbagi tumpangan (ride-sharing) baru yang diluncurkan di Chicago. Tugas Kami adalah untuk menemukan pola pada informasi yang tersedia. Kami ingin memahami preferensi penumpang dan dampak faktor eksternal terhadap perjalanan.
Dengan menggunakan basis data, Kami akan menganalisis data dari kompetitor dan menguji hipotesis terkait pengaruh cuaca terhadap frekuensi perjalanan.

**Goals:**
- Mengidentifikasi top 10 wilayah yang menjadi tujuan pengantaran
- Menampilkan diagram perusahaan yang memiliki jumlah pengantaran 
- Menampilkan grafik 10 wilayah teratas yang menjadi destinasi penumpang  

**Uji Hipotesis:**
- Durasi rata-rata perjalanan dari Loop ke Bandara Internasional O'Hare berubah pada hari-hari Sabtu yang hujan 

## 1. Initialization
''')


code1 = ('''
# load all libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
''')
kode(code1)


st.markdown('''## 2. Data Preprocessing
''')


code2 = ('''
# load all datas

df_company = pd.read_csv('/datasets/project_sql_result_01.csv')
df_trips = pd.read_csv('/datasets/project_sql_result_04.csv')
df_loop_ohare = pd.read_csv('/datasets/project_sql_result_07.csv')
''')
kode(code2)

df_company = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/learning-materials/data-analyst-eng/moved_project_sql_result_01.csv')
df_trips = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/learning-materials/data-analyst-eng/moved_project_sql_result_04.csv')
df_loop_ohare = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/learning-materials/data-analyst-eng/moved_project_sql_result_07.csv')


st.markdown('''### 2.1. Company Data

**Deskripsi Data**
- `company_name`: nama perusahaan taksi
- `trips_amount`: jumlah perjalanan untuk setiap perusahaan taksi pada tanggal 15-16 November 2017.
''')


code3 = ('''
# menampilkan sample data company
df_company
''')
kode(code3)

df_company


code4 = ('''
# menampilkan nilai unique
df_company['company_name'].sort_values().unique()
''')
kode(code4)

st.write(df_company['company_name'].sort_values().unique()
)


code5 = ('''
# menampilkan informasi dataset company
df_company.info()
''')
kode(code5)

buffer(df_company)


code6 = ('''
# menampilkan statisktik deskriptif data
df_company.describe()
''')
kode(code6)

st.write(df_company.describe()
)


st.markdown('''Dataset terdiri dari **2** kolom dan **64** baris, tidak terdapat *missing value* pada data, type data juga sudah sesuai dimana kolom `company_name` didefinisikan sebagai **object** dan kolom `trips_amount` didefinisikan sebagai **int64**.

Tabel deskripsi juga menunjukkan bahwa suatu perusahaan dengan jumlah perjalanan tertinggi memiliki angka **19558** dan perusahaan dengan jumlah terendah memiliki angka perjalanan **2**, dengan rata-rata **2145** pada tanggal 15-16 November 2017, artinya ada perusahaan yang hanya memiliki 1 kali perjalanan dalam satu hari menunjukkan adanya perbedaan yang sangat signifikan antara satu perusahaan dan perusahaan lainnya.

Tetapi kita melihat ada beberapa **register** dimulai dengan angka maka kita akan mempelajarinya lebih lanjut apakah memengaruh.

### 2.2. Location Data

**Deskripsi Data:**
- `dropoff_location_name`: nama wilayah di Chicago tempat perjalanan berakhir
- `average_trips`: jumlah rata-rata perjalanan yang berakhir di setiap wilayah pada bulan November 2017.
''')


code7 = ('''
# menampilkan sample dataset location 
df_trips
''')
kode(code7)

df_trips


code8 = ('''
# menampilkan nilai unique
df_trips['dropoff_location_name'].sort_values().unique()
''')
kode(code8)

st.write(df_trips['dropoff_location_name'].sort_values().unique()
)


code9 = ('''
# menampilkan informasi dataset location 
df_trips.info()
''')
kode(code9)

buffer(df_trips)


st.markdown('''Kolom `average_trips` didefinisikan sebagai **float**, maka kita akan mengubahnya karena seharusnya typenya adalah **integer**, karena menunjukkan rata-rata perjalanan. 
''')


code10 = ('''
# convert average_trips column
df_trips['average_trips'] = df_trips['average_trips'].apply(np.int64)
df_trips.info()
''')
kode(code10)

buffer(df_trips)


code11 = ('''
# menampilkan datanya lagi setelah dilakukan pembersihan
df_trips
''')
kode(code11)

df_trips


code12 = ('''
# menampilkan statistik deskriptif data
df_trips.describe()
''')
kode(code12)

st.write(df_trips.describe()
)


st.markdown('''Dataset diatas terdiri dari **2** kolom dan **94** baris, tidak terdapat *missing value* pada data diatas, tipe data seharusnya sudah benar, kolom `dropoff_location_name` didefinisikan sebagai **object** dan kolom `average_trips` didefinisikan sebagai **int**.  

Perjalanan tertinggi terjadi pada wilayah dengan jumlah **10727** pada bulan November 2017 dan perjalanan dengan jumlah **1** menjadi wilayah yang paling sedikit menjadi tujuan pada bulan November 2017. Dengan rata-rata **599** perjalanan ke setiap wilayah pada bulan tersebut.

## 3. Data Visualization

### 3.1. Top 10 Company
''')


code13 = ('''
# Menampilkan 10 perusahaan dengan jumlah traffic terbanyak
top_company = df_company.sort_values(by='trips_amount', ascending=False).head(10)
top_company
''')
kode(code13)

top_company = df_company.sort_values(by='trips_amount', ascending=False).head(10)
top_company


code14 = ('''
# menampilkan grafik
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.barplot(data=top_company, x='trips_amount', y='company_name')
ax.set_xlabel('Trips')
ax.set_ylabel('Company')
ax.set_title('Top 10 Company and Number of Trips')
st.pyplot(fig)
''')
kode(code14)

fig, ax = plt.subplots(figsize=(12,6))
ax = sns.barplot(data=top_company, x='trips_amount', y='company_name')
ax.set_xlabel('Trips')
ax.set_ylabel('Company')
ax.set_title('Top 10 Company and Number of Trips')
st.pyplot(fig)


st.markdown('''Dari diagram diatas kita mengetahui bahwa **Flash Cab** menjadi perusahaan dengan jumlah perjalanan terbanyak pada 15-16 November 2017. Dengan jumlah **19558** atau mendekati angka 20000, dengan jumlah ini **Flash Cab** secara signifikan mengungguli perusahaan lainnya dengan margin sekitar **42%** dari perusahaan **Taxi Affiliation Service** yang menempati posisi kedua.

### 3.2. Top 10 Destination
''')


code15 = ('''
# Menampilkan 10 destinasi dengan jumlah tertinggi
top_trips = df_trips.sort_values(by='average_trips', ascending=False).head(10)
top_trips
''')
kode(code15)

top_trips = df_trips.sort_values(by='average_trips', ascending=False).head(10)
top_trips


code16 = ('''
# menampilkan grafik
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.barplot(data=top_trips, x='average_trips', y='dropoff_location_name')
ax.set_xlabel('Avg Trips')
ax.set_ylabel('Location')
ax.set_title('Top 10 Dropoff Location')
st.pyplot(fig)
''')
kode(code16)

fig, ax = plt.subplots(figsize=(12,6))
ax = sns.barplot(data=top_trips, x='average_trips', y='dropoff_location_name')
ax.set_xlabel('Avg Trips')
ax.set_ylabel('Location')
ax.set_title('Top 10 Dropoff Location')
st.pyplot(fig)


st.markdown('''Dari grafik diatas kita dapat mengambil kesimpulan bahwa **Loop** menjadi tujuan pengantaran paling tinggi di bulan November 2017.

## 4. Uji Hipotesis

### 4.1. Enrich Data
''')


code17 = ('''
# menampilkan sample dataset duration
df_loop_ohare
''')
kode(code17)

df_loop_ohare


code18 = ('''
# Menampilkan informasi data
df_loop_ohare.info()
''')
kode(code18)

buffer(df_loop_ohare
)


code19 = ('''
# menampilkan statistik deskriptif data
df_loop_ohare.describe()
''')
kode(code19)

st.write(df_loop_ohare.describe()
)


st.markdown('''Data diatas memiliki jumlah **3** kolom dan **1068** baris, kolom `start_ts` yang berisi waktu perjalanan tidak didefinisikan sebagai **timestamp** melainkan **float** maka kita akan mengubahnya terlebih dahulu. Untuk memudahkan kita akan menambah kolom menit. 

Waktu terlama suatu perjalan ditempuh dalam waktu **124** menit atat lebih dari 2 jam, waktu tersingkat adalah **0** detik yang mungkin terjadi karena perjalanan dibatalkan, dengan rata-rata waktu tempuh **34** menit setiap perjalanan atau kurang lebih setengan jam. 
''')


code20 = ('''
# Mengubah type kolom start_ts
df_loop_ohare['start_ts'] =  pd.to_datetime(df_loop_ohare['start_ts'], infer_datetime_format=True)
df_loop_ohare.info()
''')
kode(code20)

df_loop_ohare['start_ts'] =  pd.to_datetime(df_loop_ohare['start_ts'], infer_datetime_format=True)
buffer(df_loop_ohare)


code21 = ('''
# Menambahkan kolom menit
df_loop_ohare['duration_minutes'] = np.round(df_loop_ohare['duration_seconds'] / 60)
df_loop_ohare
''')
kode(code21)

df_loop_ohare['duration_minutes'] = np.round(df_loop_ohare['duration_seconds'] / 60)
df_loop_ohare


st.markdown('''Kita akan mengelompokkan data menjadi dua kondisi yaitu `good` dan `bad`.
''')


code22 = ('''
# Memeriksa distribusi good dan bad weather
df_loop_ohare.groupby('weather_conditions').agg(amount=('start_ts','count'), avg_sec=('duration_seconds', 'mean'),
                                             avg_min=('duration_minutes','mean'), total_min=('duration_minutes',
                                            'sum')).reset_index()
''')
kode(code22)

st.write(df_loop_ohare.groupby('weather_conditions').agg(amount=('start_ts','count'), avg_sec=('duration_seconds', 'mean'),
                                             avg_min=('duration_minutes','mean'), total_min=('duration_minutes',
                                            'sum')).reset_index()
)


st.markdown('''Dari tabel diatas kita mendapatkan *overview* bahwa terdapat perbedaan jumlah yang signifikan antara  kondisi **bad** dan **good** pada cuaca saat perjalanan berlangsung. Terdapat **180** jumlah pengantaran pada saat kondisi **bad** dan **888** pengantaran pada kondisi **good**. 

Selanjutnya kita akan memeriksa dan mengatasi apabila terdapat **outliers** dalam data kemudian kita akan membagi menjadi dua dataset berdasarkan kondisi untuk pengujian hipotesis.
''')


code23 = ('''
# memeriksa dan mengatasi outliers
Q1 = df_loop_ohare['duration_seconds'].quantile(0.25)
Q3 = df_loop_ohare['duration_seconds'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR                  

df_loop_ohare = df_loop_ohare[ ~((df_loop_ohare['duration_seconds'] < lower_bound) | (df_loop_ohare['duration_seconds'] > upper_bound)) ]
''')
kode(code23)

Q1 = df_loop_ohare['duration_seconds'].quantile(0.25)
Q3 = df_loop_ohare['duration_seconds'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR                  

df_loop_ohare = df_loop_ohare[ ~((df_loop_ohare['duration_seconds'] < lower_bound) | (df_loop_ohare['duration_seconds'] > upper_bound)) ]


code24 = ('''
# Memeriksa distribusi good dan bad weather
df_loop_ohare.groupby('weather_conditions').agg(amount=('start_ts','count'), avg_sec=('duration_seconds', 'mean'),
                                             avg_min=('duration_minutes','mean'), total_min=('duration_minutes',
                                            'sum')).reset_index()
''')
kode(code24)

st.write(df_loop_ohare.groupby('weather_conditions').agg(amount=('start_ts','count'), avg_sec=('duration_seconds', 'mean'),
                                             avg_min=('duration_minutes','mean'), total_min=('duration_minutes',
                                            'sum')).reset_index()
)


st.markdown('''Setelah kita memeriksa dan mengatasi adanya *outliers* terdapat sedikit perubuhan dalam data, yaitu jumlah pengantaran menjadi **179** saat cuaca buruk dan **883** saat cuaca baik, serta rata-rata waktu pengantaran juga terdapat sedikit perubahan.
''')


code25 = ('''
# Memfilter dataset berdasarkan genre yang akan diuji
df_weather_good = df_loop_ohare.query('weather_conditions == "Good"').reset_index(drop=True)
df_weather_bad = df_loop_ohare.query('weather_conditions == "Bad"').reset_index(drop=True)
''')
kode(code25)

df_weather_good = df_loop_ohare.query('weather_conditions == "Good"').reset_index(drop=True)
df_weather_bad = df_loop_ohare.query('weather_conditions == "Bad"').reset_index(drop=True)


st.markdown('''### 4.3. Test Hypothesis

Hypothesis :

- H₀ : Durasi rata-rata perjalanan dari Loop ke Bandara Internasional O'Hare tidak berubah pada hari-hari Sabtu yang hujan
- H₁ : Durasi rata-rata perjalanan dari Loop ke Bandara Internasional O'Hare berubah pada hari-hari Sabtu yang hujan
''')


code26 = ('''
# Menampilkan diagram untuk mengetahui distribusi good weather
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df_weather_good['duration_seconds'], bins=70, kde=True)
plt.xlabel('Seconds')
plt.ylabel('Freq')
plt.title('Distribusi Durasi Pengantaran pada saat Cuaca Baik')
st.pyplot(fig)
''')
kode(code26)

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df_weather_good['duration_seconds'], bins=70, kde=True)
plt.xlabel('Seconds')
plt.ylabel('Freq')
plt.title('Distribusi Durasi Pengantaran pada saat Cuaca Baik')
st.pyplot(fig)


code27 = ('''
# Menampilkan diagram untuk mengetahui distribusi bad weather
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df_weather_bad['duration_seconds'], bins=70, kde=True)
plt.xlabel('Seconds')
plt.ylabel('Freq')
plt.title('Distribusi Durasi Pengantaran pada saat Cuaca Buruk')
st.pyplot(fig)
''')
kode(code27)

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df_weather_bad['duration_seconds'], bins=70, kde=True)
plt.xlabel('Seconds')
plt.ylabel('Freq')
plt.title('Distribusi Durasi Pengantaran pada saat Cuaca Buruk')
st.pyplot(fig)


st.markdown('''Untuk menentukan `equal_var` True atau False kita akan menggunakan **Levene Test** karena seperti yang kita lihat salah satu diagram menunjukan distribusi yang miring ke kanan atau **Non-normal Distribution**, kita menetapkan jika nilai `p-value` lebih dari **0.05** maka bisa kita asumsikan bahwa kedua sampel memiliki `equal variance`.
''')


code28 = ('''
# Determine if the two samples have equal or unequal variance
stats.levene(df_weather_good['duration_seconds'], df_weather_bad['duration_seconds'])
''')
kode(code28)

stats.levene(df_weather_good['duration_seconds'], df_weather_bad['duration_seconds'])


st.markdown('''Nilai `pvalue` menunjukkan angka **0.61**, maka kita bisa tetapkan kedua populasi memiliki varians yang sama, karena lebih tinggi dari nilai **Alpha** yang ditentukan.
''')


code29 = ('''
# Test the hypothesis

alpha = 0.05
results = stats.ttest_ind(df_weather_good['duration_seconds'], df_weather_bad['duration_seconds'], equal_var=True)
st.write('p-value:', results.pvalue)

if results.pvalue < alpha:
    st.write('We reject the null hypothesis')
else:
    st.write("We can't reject the null hypothesis")
''')
kode(code29)

alpha = 0.05
results = stats.ttest_ind(df_weather_good['duration_seconds'], df_weather_bad['duration_seconds'], equal_var=True)
st.write('p-value:', results.pvalue)

if results.pvalue < alpha:
    st.write('We reject the null hypothesis')
else:
    st.write("We can't reject the null hypothesis")


st.markdown('''Dari pengujian diatas kita mendapatkan hasil bahwa `Durasi rata-rata perjalanan dari Loop ke Bandara Internasional O'Hare berubah pada hari-hari Sabtu yang hujan` maka kita dapat meyimpulkan bahwa kondisi cuaca memengaruhi durasi perjalanan, yang mana pada kondisi cuaca sedang hujan akan membutuhkan waktu rata-rata 7-8 menit lebih lama.

# Consclusion

**1. Pengumpulan Data dan Tahap Preprocessing**
- Project ini dimulai dengan pengambilan data eksternal melalui situs web, mempelajari basis data, dan menganalisa data dari kompetitor.
- Kita memiliki **3** dataset untuk dianalisis, dataset pertama mamuat tentang informasi perusahaan taxi, data kedua memuat tentang informasi lokasi pengantaran, dan data ketiga memuat informasi tentang waktu yang dibutuhkan untuk pengantaran pada saat cuaca sedang hujan dan tidak hujan.
- Pertama-tama kita memuat **2** dataset *company* dan *dropoff location*, kita mempelajari isi data, memastikan bahwa type data sesuai.

**2. Explanatory Data Analysis dan Uji Hipotesis**
- Selajutnya kita menganalisis mengenai perusahaan yang memiliki perjalanan terbanyak, dan tujuan terpopuler kemudian kita menampilkan grafiknya.
- Tahap selanjutnya kita memuat data mengenai cuaca dan menganalisa apakah cuaca memengaruhi durasi pengantaran dengan metode **t-test**.

**3. Result and Rocommendation**
- Perusahaan **Flash Cab** secara signifikan memiliki jumlah pengantaran lebih banyak dibandingkan para kompetitor memiliki selisih diatas **40%** dibandingkan peringkat dua, data lainnya juga menunjukkan bahwa **Loop** dan **River North** menjadi tujuan terpopuler dari lokasi lainnya.
- Hasil uji hipotesis menunjukan bahwa `Durasi rata-rata perjalanan dari Loop ke Bandara Internasional O'Hare berubah pada hari-hari Sabtu yang hujan` maka kita dapat meyimpulkan bahwa kondisi cuaca dapat memengaruhi waktu penumpang tiba di lokasi tujuan dangan selisih waktu tempuh rata-rata 7 menit lebih lama jika cuaca hujan.
- Kita perlu memepelajari lebih lanjut mengapa perusahaan **Flash Cab** memiliki jumlah perjalanan tertinggi, kita juga mengetahui bahwa **Loop** dan **River North** menjadi tujuan tepopuler jadi para *driver* dapat memilih untuk menunggu penumpang di area tersebut, pada saat cuaca hujan kita dapat memberitahukan *customer* bahwa mungkin akan terjadi keterlambatan waktu penjemputan dan pengantaran sehingga para penumpang dapat memesan taxi lebih awal.
''')
