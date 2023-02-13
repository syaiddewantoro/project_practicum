#!/usr/bin/env python
# coding: utf-8

# # Menganalisis risiko peminjam gagal membayar
# 
# Sebagai kredit analyst proyek kami ialah menyiapkan laporan untuk bank bagian kredit. Kami mencari tahu pengaruh status perkawinan seorang nasabah dan jumlah anak terhadap probabilitas ketepatan waktu dalam melunasi pinjaman. Bank sudah memiliki beberapa data mengenai kelayakan kredit nasabah.
# 
# Laporan Anda akan dipertimbangkan pada saat membuat **penilaian kredit** untuk calon nasabah. **Penilaian kredit** digunakan untuk mengevaluasi kemampuan calon peminjam untuk melunasi pinjaman mereka.
# 
# Tujuan utama dari poject ini adalah untuk mengetahui kelayakan seorang klien untuk mendapatkan kredit berdasarkan status dan keadaan mereka yang tersimpan dalam data kita. Kita juga menguji kapasitas nasabah berdasarkan karakteristik mereka yang kita rangkum berdasarkan kategori-kategori sehingga diperoleh *pattern* untuk memberikan lampu kuning kepada nasabah yang masuk ke dalam kategori tertentu.
# 
# Hipotesis project :
# <br>1. Apakah terdapat korelasi antara jumlah anak dengan kemampuan melunasi pinjaman tepat waktu?
# <br>2. Apakah terdapat korelasi antara status keluarga dengan kemampuan melunasi pinjaman tepat waktu?
# <br>3. Apakah terdapat korelasi antara kelas ekonomi dengan kemampuan melunasi pinjaman tepat waktu?
# <br>4. Apakah terdapat korelasi antara tujuan kredit dengan kemampuan melunasi pinjaman tepat waktu?

# ## Membuka *file* data dan menampilkan informasi umumnya. 
# 
# Kita akan mulai dengan mengimport library dan memuat data.

# In[1]:


# Memuat semua perpustakaan
import pandas as pd


# In[2]:


# muat data
df = pd.read_csv('/datasets/credit_scoring_eng.csv')


# ## Soal 1. Eksplorasi Data
# 
# **Deskripsi Data**
# - *`children`* - jumlah anak dalam keluarga
# - *`days_employed`* - pengalaman kerja dalam hari
# - *`dob_years`* - usia klien dalam tahun
# - *`education`* - pendidikan klien
# - *`education_id`* - tanda pengenal pendidikan
# - *`family_status`* - status perkawinan
# - *`family_status_id`* - tanda pengenal status perkawinan
# - *`gender`* - jenis kelamin klien
# - *`income_type`* - jenis pekerjaan
# - *`debt`* - apakah klien memiliki hutang pembayaran pinjaman
# - *`total_income`* - pendapatan bulanan
# - *`purpose`* - tujuan mendapatkan pinjaman

# In[3]:


# Memeriksa jumlah baris dan kolom dalam dataset 
df.shape


# In[4]:


# Menampilkan 10 baris pertama dalam dataset
df.head(10)


# Dari sampel data yang ditampilkan, terdapat beberapa masalah yang bisa dideteksi yaitu tedapat value negatif dan value yang saya rasa sangat tinggi dari kolom `days_employed` yang saya rasa tidak masuk akal karena pada kolom menampilkan pengalaman kerja dalam hari , serta penulisan huruf kapital yang tidak tidak teratur pada kolom `education`.

# In[5]:


# Mendapatkan informasi seluruh kolom dalam data
df.info()


# Terdapat nilai yang hilang pada kolom `days_employed` dan `total_income`.

# In[6]:


# Menampilkan nilai yang hilang dalam dataset
df[df['days_employed'].isna()].head(10)


# Sejauh ini, dari yang saya lihat saya asumsikan bahwa value yang hilang memang tampak simetris karena berada pada baris yang sama, tapi untuk dapat menyimpulkan apakah nilai yang hilang memang bebar-benar berada pada baris yang sama perlu dilakukan investigasi lebih lanjut.

# In[7]:


# Melakukan beberapa pemfilteran untuk mengetahui jumlah baris dari value yang hilang
df_filtered_nan = df[df['days_employed'].isna()]
df_filtered_nan = df_filtered_nan[df_filtered_nan['total_income'].isna()]
df_filtered_nan.shape[0]


# Data yang haliang dalam dataset kita berjumlah `2174` baris.

# In[8]:


# Sekarang kta akan menghitung rasio value yang hilang dari seluruh dataframe
df_distribution_nan = df_filtered_nan.shape[0] / df.shape[0] 
print(f'Distribusi nilai yang hilang sebesar: {df_distribution_nan:0%}')


# **Kesimpulan menengah**
# 
# Kita bisa simpulkan jumlah nilai hilang sama dengan jumlah tabel yang difilter. Artinya nilai yang hilang dari tabel yang difilter simetris.
# 
# Persentase data yang hilang dari dataframe sebesar `10%`, cukup berpengaruh terhadap data kita bukan?
# 
# Selanjutnya saya akan menghitung jumlah persentase dari value yang hilang apakah memiliki dampak yang signifikan terhadap data dalam dataset, sehingga kita akan mengetahui langkah yang tepat untuk memproses data yang hilang tersebut.

# In[9]:


# Menerapkan filter untuk menampilkan baris yang hilang dari data
df_filtered_nan.head(10)


# Value yang hilang tampak simetris dari table yang ditampilkan.

# In[10]:


# Memeriksa distribusi
print(df_filtered_nan['income_type'].value_counts())
print()
df_filtered_nan['income_type'].value_counts() / df['income_type'].value_counts() * 100


# Cukup terpola disini tetapi memang ada beberapa kategori yang valuenya tidak bisa dikatakan valid untuk dilakukan perhitungan mengingat jumlahnya hanya sedikit sekitar `1 - 2` data saja jumlahnya seperti kategori `unemployed, paternity / maternity leave, student,` dan `entrepreneur`. Secara overall data yang hilang terdistribusi sebesar `10%` pada setiap kategori.

# **Kemungkinan penyebab hilangnya nilai dalam data**
# 
# Belum dapat ditentukan penyebab nilai yang hilang, kita harus mempertimbangkan kemungkinan-kemungkinan lain penyebab nilai yang hilai apakah nilai yang memiliki karakteristik tertentu seperti dari klien yang sudah menikah, jumlah anak, ataupun klien yang memiliki tunggakan pembayaran kredit.

# In[11]:


# Memeriksa distribusi di seluruh dataset
df.isna().sum() / df.shape[0] * 100


# **Kesimpulan menengah**
# 
# Jumlah nilai yang hilang dalam dataset mirip dengan tabel yang difilter.
# <br>Dan jumlah data yang hilang pada kolom `days_employed` sama dengan `total_income` yang artinya error hanya terjadi pada dua kolom tersebut.
# 
# Hal ini menimbulkan pertanyaan apakah nilai yang hilang membentuk suatu pola atau terjadi secara acak?
# <br>Untuk mejawab pertannyaan ini mari kita lakukan analisa lebih lanjut.

# In[12]:


# Memeriksa apakah ada pola lain yang menyebabkan hilangnya data dari kolom 'family_status'
print(df_filtered_nan['family_status'].value_counts())
print()
df_filtered_nan['family_status'].value_counts() / df['family_status'].value_counts() * 100


# **Kesimpulan menengah**
# 
# Cukup menarik bahwa nilai yang hilang terdistribusi secara merata dari kategori di kolom `family_status` yaitu sekitar `10%`. Tetapi juga dikatahui tidak ada ketegori khusus yang mengakibatkan data hilang disebabkan oleh salah satu kategori saja.

# In[13]:


# Check for relation both 'gender' and missing value
print(df_filtered_nan['gender'].value_counts())
print()
df_filtered_nan['gender'].value_counts() / df['gender'].value_counts() * 100


# Pada kategori gender data yang hilang masing-masing terdistribusi sebesar `9%` dan `10%` artinya data yang hilang tidak hanya terdapat pada satu kategori saja. 

# In[14]:


# Memeriksa pendapatan dari beberapa pekerjaan yang kemungkinan tidak memiliki 'income'
df[df['income_type'].isin(['unemployed', 'paternity / maternity leave', 'student', 'entrepreneur'])]


# Dari tabel diatas kita bisa mendapatkan informasi bahwa nilai yang hilang tidak selalu diakibatkan oleh pekerjaan yang biasanya tidak memiliki penghasilan, mungkin ada beberapa alasan bagaimana cara mereka mendapatkan penghasilan meskipun tidak sedang memiliki pekerjaan yang regular. Seperti pada baris `3133` klien yang `unemployed` tetapi mengajukan pinjaman untuk membeli properti untuk desewakan, meskipun belum diketahui darimana penghasilannya, Klien `9410` seorang `student` apakah dia seorang pekerja part time atau memiliki *rich parents* tetapi cukup aneh mengingat kegunaannya tidak untuk melanjutkan pendidikan melainkan membangun properti. Selain itu di beberapa negara juga memilki peraturan bahwa `paternity / maternity leave` *still getting paid*.

# **Kesimpulan**
# 
# Konklusi yang bisa kita ambil mengenai penyebab data yang hilang adalah *human error* karena data yang hilang tidak terjadi pada kategori tertentu saja melainkan pada beberapa kategori.
# 
# Dari beberapa pengujian yang sudah saya lakukan saya menemukan bahwa dari kolom `family_status` dan `income_type` saya menemukan bahwa setiap kategori dalam masing-masing kolom tersebut memiliki nilai yang hilang hampir sama di setiap kategori yang sekitar `10%` artinya nilai yang hilang terdistribusi secara merata di setiap kategori dan nilainya simetris dengan pengujian yang kita lakukan sebelumnya dengan mencari distribusi nilai yang hilang di seluruh dataset yang menghasilkan angka juga sebesar `10%`.
# 
# Untuk beberapa masalah seperti:
# <br>1. Untuk nilai yang hilang saya akan membuat beberapa kateghori berdasarkan usia untuk mencari rata-rata `days_employed` dan `total_income` yang akan digunakan untuk mengisi value yang hilang.
# <br>2. Untuk mengatasi register yang berbeda pada kolom `education` saya akan mengubah semua huruf menjadi `lower`.
# <br>3. Kita bisa melakukan `drop` untuk nilai duplikat.

# ## Transformasi data

# **Memeriksa kolom 'education'**

# In[15]:


# Memeriksa ejaan pada kolom 'education' yang terindikasi memiliki perbebedaan penulisan dengan makna yang sama
df_education_value = df.pivot_table(index='education', values='days_employed', aggfunc= 'count')
df_education_value


# In[16]:


# Kemudian kita akan memeriksa nilai duplikat sebelum kita mengangani register yang tidak teratur
df.duplicated().sum()


# In[17]:


# Memperbaiki penulisan
df['education'] = df['education'].str.lower()


# In[18]:


# Memeriksa kembali nilai duplikat setelah dilakukan perbaikan
df.duplicated().sum()


# Nilai duplikat bertambah dari sebelumnya `54` baris menjadi `71` baris, artinya ada baris yang memilki nilai yang sama dalam data dengan input `education` yang berbeda. 

# In[19]:


# Memeriksa kembali apakah penulisan kolom 'education' telah diperbaiki
df_education_value = df.pivot_table(index='education', values='days_employed', aggfunc= 'count')
df_education_value


# Masalah pada kolom `education` telah kita atasi selanjutnya kita akan memeriksa kolom `children`.

# **Memeriksa kolom `children`**

# In[20]:


# Menampilkan distribusi nilai pada kolom `children`
df_children_value = df.pivot_table(index='children', values='days_employed', aggfunc= 'count')
df_children_value


# Terdapat value yang menunjukkan angka -1 dan 20 yang saya nilai tidak mungkin.
# <br>Kita asumsikan nilai yang bermasalah terjadi karena kesalahan input/typo.
# <br>Kita akan me-replace nilai -1 menjadi 1 dan 20 menjadi 2.

# In[21]:


# Memeriksa jumlah dan persentase data yang bermasalah pada kolom 'children'
df_child_problem_value = (df['children'] == -1).sum() + (df['children'] == 20).sum()
print('Jumlah nilai yang tidak wajar:', df_child_problem_value)
df_child_prob_percent = df_child_problem_value / df.shape[0]
print(f'Persentase data yang bermasalah adalah: {df_child_prob_percent:.2%}')


# Data dengan nilai yang tidak wajar tidak sampai `1%` dalam dataset kita meskipun begitu lebih baik kita perbaiki alih-alih melakukan *drop* pada data tersebut.

# In[22]:


# Memperbaiki nilai yang bermasalah
df.loc[df['children'] == -1, 'children'] = 1 
df.loc[df['children'] == 20, 'children'] = 2 
df.duplicated().sum()


# In[23]:


# Periksa kembali kolom `children` untuk memastikan semua telah diperbaiki
df_children_value = df.pivot_table(index='children', values='days_employed', aggfunc= 'count')
df_children_value


# In[24]:


# Mari kita periksa kembali nilai duplikat
df.duplicated().sum()


# Belum ada perubahan pada nilai duplikat.

# **Memeriksa kolom `days_employed`**

# In[25]:


# Kita akan melihat nilai yang tidak wajar pada kolom 'days_employed'
df[df['income_type'] == 'retiree']


# In[26]:


# Menampilkan statistik deskriptif pada kolom 'days_employed' 
df['days_employed'].describe()


# Terdapat angka-angka negatif dan nilai yang tidak wajar, perlu diingat kolom ini menampilkan jumlah hari klien telah bekerja.

# In[27]:


# Menampilkan nilai rata-rata 'days_employed'
df_days_employed_median = df.groupby(['income_type'])['days_employed'].median()
df_days_employed_median


# Kita tidak bisa melakukan proses uji kelayakan credit sebelum memperbaiki nilai pada kolom ini, nilai negatif sama dengan nilai yang hilang.

# In[28]:


df_days_employed_problem_value = (df['days_employed'] < 0).sum() + (df['days_employed'] > 21600).sum()
print(df_days_employed_problem_value)
df_days_employed_prob_percent = df_days_employed_problem_value / df.shape[0]
print(f'Persentase data yang bermasalah adalah: {df_days_employed_prob_percent:.2%}')


# Terdapat masalah yang sangat tinggi pada kolom `days_employed`, jika kita melihat nilai negatif sudah berdampak besar dalam dataset karena itu sama saja dengan nilai yang hilang, lalu bagaimana dengan nilai yang terlampau tinggi, tidak ada seorangpun bekerja melebihi usianya sendiri.
# <br>Karena nilai mungkin terjadi karena kesalah teknis dalam penambahan tanda `-` jadi solusi yang dapat saya lakukan adalah dengan dengan mengganti angka negatif menjadi positif dengan fungsi `absolut`.
# <br>Kemudian untuk nilai yang terlampau tinggi yang saya asumsikan hanya terjadi pada pada `kategori`, `retiree` dan `unemployed` maka saya putuskan untuk mengganti nilai tersebut dengan `0` dengan batas `21600 hari atau 59 tahun`.

# In[29]:


# Menerapkan fungsi 'absolut' untuk nilai negatif dan me-'replace' nilai yang tidak wajar dengan angka '0'
df['days_employed'] = df['days_employed'].abs()
df.loc[df['days_employed'] > 21600, 'days_employed'] = 0


# In[30]:


# Memastikan nilai telah diperbaiki
print(df['days_employed'].describe())
print()
print((df['days_employed'] < 0).sum())


# Terdapat masalah pada kolom `usia` yaitu tidak seseorang yang bekerja sejak usia `0`

# **Memeriksa kolom `dob_years`**

# In[31]:


# Memeriksa kolom 'dob_years' apakah memiliki nilai anomali
df_dob_years_value = df.pivot_table(index='dob_years', values='days_employed', aggfunc= 'count')
df_dob_years_value


# In[32]:


# Melihat persentase data anomali
df_days_employed_prob_percent = (df['dob_years'] == 0).sum() / df.shape[0]
print(f'Persentase data yang bermasalah adalah: {df_days_employed_prob_percent:.2%}')


# Saya akan mengganti value `0` di kolom `age`, dengan `median` dari usia.
# <br>Karena tiap klien yang memiliki usia `0` memiliki karakteristik yang berbeda.

# In[33]:


# Mengganti angka '0' dengan rata-rata
avg_dob_years = df['dob_years'].median() 
df.loc[df['dob_years'] == 0, 'dob_years'] = avg_dob_years


# In[34]:


# Memastikai nilai telah diperbaiki
df[df['dob_years'] == 0].shape[0]


# **Memeriksa kolom `family_status`**

# In[35]:


# Memeriksa kolom 'family_status'
df_family_status_value = df.pivot_table(index='family_status', values='days_employed', aggfunc= 'count')
df_family_status_value


# Tidak ditemukan masalah pada kolom `family_status`

# **Memeriksa kolom `gender`**

# In[36]:


# Check 'gender' column
df_gender_value = df.pivot_table(index='gender', values='days_employed', aggfunc= 'count')
df_gender_value


# Terdapat satu masalah yaitu pada value `XNA`
# <br>Sulit untuk mengidentifikasi untuk mengganti value tersebut, dan saya putuskan untuk menghapus 1 baris tersebut

# In[37]:


# Menghapus value `XNA`
df = df.loc[df["gender"] != 'XNA']


# In[38]:


# Memastikan kolom telah diperbaiki
df_gender_value = df.pivot_table(index='gender', values='days_employed', aggfunc= 'count')
df_gender_value


# **Memeriksa kolom `income_type`**

# In[39]:


# Mari kita lihat nilai dalam kolom
df_income_type_value = df.pivot_table(index='income_type', values='days_employed', aggfunc= 'count')
df_income_type_value


# Tidak ada masalah pada kolom `income_type`

# Terdapat nilai duplikat sebesar `72` atau hanya berdampak `0.3%` dalam dataset sehingga saya putuskan untuk melakukan `drop` untuk nilai duplikat, karena nilai tersebut masih dapat diterima.

# In[40]:


# Memeriksa duplikat
print(df.duplicated().sum())
print()
df_duplicated_percent = df.duplicated().sum() / df.shape[0] 
print(f'Distribusi nilai duplikat sebesar: {df_duplicated_percent:.2%}')


# In[41]:


# Atasi duplikat, jika ada
df = df.drop_duplicates().reset_index(drop=True)


# In[42]:


# Terakhir periksa apakah kita memiliki duplikat
df.duplicated().sum()


# In[43]:


# Periksa ukuran dataset yang sekarang Anda miliki setelah manipulasi pertama yang Anda lakukan
df.shape


# In[44]:


old_df_shape = 21525
df_shape_percentage = (old_df_shape - df.shape[0]) / old_df_shape
print(f'Persentase dari perubahan dalam dataset adalah: {df_shape_percentage:.2%}')


# Kita telah memperbaiki beberapa masalah dalam dataset seperti:
# <br>1. Nilai negatif dan nilai yang terlampau tinggi dalam kolom `days_employed`.
# <br>2. Register yang tidak teratur dalam kolom `education`.
# <br>3. Menghapus nilai duplikat.
# <br>4. Beberapa masalah yang terjadi di kolom `children` dan `gender`.
# 
# Sehingga kita mendapatkan perubahan sebesar `0.34%` yang artinya nilai tersebut tidak begitu berdampak dalam dataset karena kurang dari `1%`.

# # Bekerja dengan nilai yang hilang

# Saya memasukkan dictionary `numpy` untuk mempercepat pekerjaan saya yang akan digunakan untuk me-`replace` nilai `0` pada kolom `days_employed` setelah membuat beberapa kategori usia.

# In[45]:


# Import dictionary
import numpy as np


# ### Memperbaiki nilai yang hilang di `total_income`

# Kita akan mulai untuk mengatasi nilai yang hilang pada kolom`total_income`.
# 
# Pertama-tama kita akan membuat beberapa kategori berdasarkan rentang usia dari kolom `dob_years`, yang diharapkan dapat membantu kita dalam mencari nilai rata-rata dari setiap rentang usia yang akan kita gunakan untuk mengisi nilai yang hilang dari masing-masing kategori.

# In[46]:


# Menulis fungsi untuk menghitung kategori usia
def age_group(age):
 
    if age <= 30:
        return '19-30'
    elif age <= 40:
        return '31-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    else:
        return '+60'


# In[47]:


# Melakukan pengujian apakah fungsi bekerja atau tidak
print(age_group(23))
print(age_group(38))
print(age_group(46))
print(age_group(55))
print(age_group(70))


# In[48]:


# Membuat kolom baru berdasarkan fungsi
df['age_group'] = df['dob_years'].apply(age_group)


# In[49]:


# Memeriksa bagaimana nilai di dalam kolom baru
df.tail(10)


# Beberapa faktor yang dapat memengaruhi pendapatan diantaranya `education`, `days_employed` (pengalaman kerja dalam hari), dan `income_type` (jenis pekerjaan).
# <br>Untuk menentukan apakah kita akan menggunakan `mean` atau `median` untuk mengisi nilai yang hilang, kita melakukan eksplorasi lebih dalam untuk mengetahui bagaimana distribusi nilai `mean` dan `median` berdasarkan kategori yang kita kelompokkan terdistribusi secara normal atau tidak.

# In[50]:


# Membuat tabel tanpa nilai yang hilang dan menampilkan beberapa barisnya
df_clean = df[df.notna()]
df_clean.head(10)


# In[51]:


# Menampilkan nilai 'mean' dari 'total_income' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['total_income'].mean()


# In[52]:


# Menampilkan nilai 'median' dari 'total_income' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['total_income'].median()


# Pendapatan rata-rata berdasarkan kategori dari `mean` memiliki rata-rata yang lebih besar untuk setiap kategori, dengan demikian saya akan memilih `median` karena lebih merepresentasikan rata-rata berdasarkan kategori kelompok umur.

# In[53]:


# Saatnya kita mengisi nilai yang hilang
df['total_income'] = df.groupby(['age_group'])['total_income'].transform(lambda x: x.fillna(x.median()))


# In[54]:


# Memeriksa statistik deskriptif dari 'total_income'
df['total_income'].describe()


# In[55]:


# Memeriksa apakah nilai hilang sudah terisi
df['total_income'].isna().sum()


# Diketahui data yang hilang belum terisi pada kolom `total_income` itu karena kita belum memasukkan nilai dari kolom baru yang kita buat untuk mengganti nilai yang hilang dari kolom tersebut. Untuk mengisi kolom tersebut kita akan menggantinya dengan metode `fillna` dari kolom baru yang kita buat yaitu `total_revenue`.

# Seperti yang kita lihat sudah tidak terdapat nilai yang hilang dari kolom `total_income`.
# Saatnya memeriksa apakah jumlah kolom `total_income` sama dengan jumlah kolom lainnya.

# In[56]:


# Memeriksa jumlah entri kolom dari dataset
df.info()


# Dari informasi di atas kita menemukan bahwa jumlah pada kolom `total_income` sudah memiliki nilai yang sama dengan kolom lainnya.
# <br>Selanjutnya kita akan memperbaiki nilai pada kolom `days_employed`.

# ###  Memperbaiki nilai di `days_employed`

# Tentunya faktor yang paling berpengaruh di dalam kolom `days_employed` adalah `age_category` bayangkan seseorang tidak mungkin berkerja melebihi usia minimal seseorang untuk bekerja secara legal atau bahkan pengalaman kerja seseorang melebihi jumlah usianya sendiri, itu merupakan hal yang mustahil.

# In[57]:


# Menampilkan nilai 'median' dari 'days_employed' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['days_employed'].median()


# In[58]:


# Menampilkan nilai 'median' dari 'days_employed' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['days_employed'].mean()


# Saya rasa `mean` bisa mewakili nilai untuk kolom `days_employed`, selain karena `median` memiliki nilai `0` yang mungkin terjadi karena kita mengubah nilai yang trelampau tinggi pada pengujian sebelumnya, tetapi apakah seseorang yang telah mencapai usia pensiun tidak memiliki pengalaman bekerja sama sekali? Atau mungkin sesorang yang telah pensiun pengalamannya akan dihitung sebagai nilai `0`.

# In[59]:


# Waktunya untuk kita mengganti nilai yang hilang dan nilai '0' dengan rata-rata
df.loc[df['days_employed'] == 0, 'days_employed'] = np.NaN
df['days_employed'] = df.groupby(['age_group'])['days_employed'].transform(lambda x: x.fillna(x.mean()))


# In[60]:


# Menampilkan statistik deskriptif dari kolom 'days_employed'
df['days_employed'].describe()


# In[61]:


# Memeriksa apakah nilai yang hilang telah teratasi
df['days_employed'].isna().sum()


# Memeriksa jumlah kolom dari seluruh dataset.

# In[62]:


# Memeriksa informasi seluruh dataset
df.info()


# Semua kolom termasuk `days_employed` telah memiliki nilai yang sama, saya rasa pekerjaan kita untuk *data cleansing* telah selesai. Sekarang mari kita coba mengeksplorasi data lebih lanjut untuk mendapatkan hal menarik dalam data maupun sebagai acuan kita dalam mengambil keputusan.

# ## Pengkategorian Data
# 
# Sepertinya saya menemukan hal menarik di kolom `purpose` yaitu banyak sekali peng-kategorian data yang saya rasa bisa kita sederhanakan menjadi lebih general, sehingga kita akan lebih mudah dalam melakukan investigasi dalam mengambil keputusan.

# In[63]:


# Menampilkan kolom 'purpose' untuk dikategorikan dengan lebih umum
df['purpose'].value_counts()


# Memeriksa value `unique`.

# In[64]:


# Memeriksa nilai unik
df['purpose'].sort_values().unique()


# Dari pengamatan saya sebelumnya memang benar bahwa peng-kategorian yang terjadi di kolom `purpose` bisa generalisasikan ke dalam beberapa kategori yang umum dan mudah dipahami, sperti `car`, `property`, `education`, dan `wedding`. Mari kita buat fungsinya di bawah ini:

# In[65]:


# Menulis fungsi untuk mengkategorikan data berdasarkan topik umum
def purpose_common(purpose):
    
    if 'property' in purpose:
        return 'property'
    elif 'estate' in purpose:
        return 'property'
    elif 'hous' in purpose:
        return 'property'
    elif 'car' in purpose:
        return 'car'
    elif 'educ' in purpose:
        return 'education'
    elif 'univ' in purpose:
        return 'education'
    elif 'wedd' in purpose:
        return 'wedding'


# In[66]:


# Memuat kolom dengan kategori dan menghitung nilainya
df['general_purpose'] = df['purpose'].apply(purpose_common)
print(df['general_purpose'].value_counts())
print()
df['general_purpose'].count()


# Kita akan membuat kategori berdasarkan `total_income` ke dalam beberapa kelas.

# In[67]:


# Melihat kolom 'total_income' untuk dikategorikan
df['total_income'].value_counts().sort_index()


# In[68]:


# Mendapatkan kesimpulan statistik untuk kolom 'total_income'
df['total_income'].describe()


# Kita mengkategorikan `total_income` ke dalam beberapa kelas kategori berdasarkan klasifikasi tingkat ekonomi. Hal ini akan memudahkan kita dalamn proses pengambilan keputusan kedepannya. Mari kita buat fungsinya di bawah ini:

# In[69]:


# Membuat fungsi untuk pengkategorian menjadi kelompok kelas
def income_class(total_income):
 
    if total_income <= 32000:
        return 'poor'
    elif total_income <= 53000:
        return 'lower middle class'
    elif total_income <= 106000:
        return 'middle class'
    elif total_income <= 373000:
        return 'upper middle class'
    else:
        return 'rich'


# In[70]:


# Membuat kolom baru dengan kategori
df['economic_class'] = df['total_income'].apply(income_class)


# In[71]:


# Menghitung distribusi
df['economic_class'].value_counts()


# ## Memeriksa Hipotesis
# 

# **Apakah terdapat korelasi antara memiliki anak dengan membayar kembali tepat waktu?**

# In[72]:


# Memeriksa apakah kolom 'children' berpengaruh terhadap 'debt'
pivot_table_children = df.pivot_table(index='children', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_children


# In[73]:


# Memeriksa persentase untuk mendapatkan konklusi
pivot_table_children['percent_1'] = pivot_table_children[1] / (pivot_table_children[1] + pivot_table_children[0]) * 100
pivot_table_children


# **Kesimpulan**
# 
# Dari data di atas kita temukan bahwa:
# <br>1. Klien yang memliki `1` sampai `4` anak memiliki persentase yang hampir sama di angka `8%` sampai `9%`.
# <br>2. Klien yang memiliki `5` anak tidak memiliki hutang pembayaran pinjaman tatapi tidak bisa kita jadikan acuan karena jumlah data terlalu sedikit.
# <br>3. Klien yang **tidak** memilki anak memiliki rasio yang paling kecil untuk hutang pembayaran pinjaman sebesar `7%` hal ini mungkin terjadi karena mereka memiliki tanggungan yang lebih sedikit dibandingkan dengan klien yang  telah memilki anak.
# 
# Hal ini tentu akan memudahkan dalam pengambilan keputusan kita dalam memberikan kredit kepada klien yang belum memiliki anak karena kemampuan mereka dalam melunasi kredit mereka.

# **Apakah terdapat korelasi antara status keluarga dengan membayar kembali tepat waktu?**

# In[74]:


# Memeriksa data 'family_status' memengaruhi kolom 'debt'
pivot_table_family_status = df.pivot_table(index='family_status', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_family_status


# In[75]:


# Menghitung rasio untuk mencari kseimpulan
pivot_table_family_status['percent_1'] = pivot_table_family_status[1] / (pivot_table_family_status[1] + pivot_table_family_status[0]) * 100
pivot_table_family_status


# **Kesimpulan**
# 
# Ada beberapa hal menarik yang kita temukan disini:
# <br>1. Klien yang `unmarried` dan `civil partnership` memiliki persentase yang cukup tinggi sebesar `9%`.
# <br>2. Klien yang `divorced` dan `married` memiliki rasio sebesar `7%` yang artinya lebih kecil daripada klien yang belum menikah apakah karena mereka dapat menggabungkan penghasilan dengan pasangan mereka, hal ini tentu berbanding terbalik dengan pengujian sebelumnya bahwa klien yang belum memiliki anak memiliki persentase hutang lebih kecil mengingat seseorang yang telah menikah punya kecenderungan untuk memiliki anak.
# <br>3. Klien dengan status `widow / widower` memiliki pensentase hutang paling kecil sebesar `6%`. Apakah kita akan mempertimbangkan status sebagai dasar dalam pengambilan keputusan?

# **Apakah terdapat korelasi antara tingkat pendapatan dengan membayar kembali tepat waktu?**

# In[76]:


# Memeriksa apakah 'economic_class' memiliki korelasi dengan 'debt'
pivot_table_economic_class = df.pivot_table(index='economic_class', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_economic_class


# In[77]:


# Menghitung persentase distribusi untuk menarik kesimpulan
pivot_table_economic_class['percent_1'] = pivot_table_economic_class[1] / (pivot_table_economic_class[1] + pivot_table_economic_class[0]) * 100
pivot_table_economic_class


# **Kesimpulan**
# 
# Beberapa hal yang kita temukan dari manipulasi data di atas diantaranya:
# <br>1. Klien dengan tingkat ekonomi `lower middle class` dan `middle class` memilki persentase yang seimbang yaitu sebesar `7%` untuk klien yang memiliki hutang pembayaran kredit.
# <br>2. Klien dengan tingkat ekonomi `poor` memiliki kemungkinan untuk menunggak lebih besar yaitu `8%`. Apakah ini dapat memengaruhi keputusan kita untuk tidak memberikan kredit mengingat jumlah mereka yang paling banyak.
# <br>3. Klien dengan pendapatan `upper middle class` yang memiliki risiko paling kecil sebesar `6%`.

# **Bagaimana tujuan kredit memengaruhi tarif otomatis?**

# In[78]:


# Memeriksa persentase 'general_purpose' terhadap 'debt' untuk menggali konklusi
pivot_table_general_purpose = df.pivot_table(index='general_purpose', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_general_purpose['percent_1'] = pivot_table_general_purpose[1] / (pivot_table_general_purpose[1] + pivot_table_general_purpose[0]) * 100
pivot_table_general_purpose


# **Kesimpulan**
# 
# Berdasarkan penganganan yang kita lakukan hal-hal yang bisa kita dapatkan:
# <br>1. Klien yang melakukan pinjaman untuk keperluan `car` dan `education` memiliki risiko gagal bayar paling tinggi sebesar `9%`.
# <br>2. Klien yang menggunakan pinjamannya untuk `wedding` memiliki persentase hutang pinjaman yang lebih rendah dibandigkan kriteria sebelumnya.
# <br>3. Klien yang membayar tepat waktu yaitu untuk kegunaan `property` dengan risiko hutang tak lancar sebesar `7%`.

# # Kesimpulan Umum 
# 
# Kita telah melakukan proses *cleansing data* untuk memperbaiki data-data yang bermasalah dalam dataset kita. Pembersihan yang kita lakukan meliputi mengisi value yang hilang, menghapus nilai duplikat, memperbaiki register yang tak beraturan, nilai yang terlalu besar, hingga mengganti nilai yang tidak wajar, sehingga kita mendapati dataset yang dapat kita olah untuk proses analisa kredit.
# 
# Temuan yang kita dapatkan setelah melakukan beberapa eksplorasi kita mendapati bahwa terdapat korelasi antara jumlah anak dan status perkawinan dalam risiko pemayaran kredit, klien yang tidak memiliki anak akan lebih mudah dalam melunasi hutangnya dibandingkan dengan klien yang memiliki anak. Klien yang menikah atau pernah memiliki pasangan memiliki risiko lebih rendah gagal bayar daripada klien dengan status *single* maupun tinggal bersama.
# Klien yang memiliki penghasilan lebih rendah akan lebih tinggi untuk memiliki hutang pinjaman, dan klien yang menggunakan uangnya untuk keperluan rumah akan lebih besar persentase mereka untuk dapat melunasi hutangnya.
# 
# Tetapi apakah semua manipulasi data yang kita lakukan dapat kita gunakan dalam proses *decision making* sehingga akan meminimalisir risiko yang akan terjadi di kemudian hari?
