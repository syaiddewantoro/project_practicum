import pandas as pd
import streamlit as st
import io

def kode(codes):
    st.code(codes, language='python')

st.markdown('''# Table of Contents

- [Menganalisis risiko peminjam gagal membayar](#scrollTo=6kdAtWqC9kE3)

    - [Membuka file data dan menampilkan informasi umumnya.](#scrollTo=p9dvgiKR9kE8)

    - [Soal 1. Eksplorasi Data](#scrollTo=dbanB5G29kFB)

    - [Transformasi data](#scrollTo=fgZe0KOS9kFS)

    - [Bekerja dengan nilai yang hilang](#scrollTo=xGaK_ajF9kFu)
        
        - [Memperbaiki nilai yang hilang di total_income](#scrollTo=OpEvpBYt9kFw)

        - [Memperbaiki nilai di days_employed](#scrollTo=gpNi37l-9kF8)

    - [Pengkategorian Data](#scrollTo=zjrvASkB9kGC)

    - [Memeriksa Hipotesis](#scrollTo=WKevJhcT9kGH)

- [Kesimpulan Umum](#scrollTo=QzRT9lG99kGT)

''')

st.title('Credit Scoring')

st.markdown('''# Menganalisis risiko peminjam gagal membayar

Sebagai kredit analyst proyek kami ialah menyiapkan laporan untuk bank bagian kredit. Kami mencari tahu pengaruh status perkawinan seorang nasabah dan jumlah anak terhadap probabilitas ketepatan waktu dalam melunasi pinjaman. Bank sudah memiliki beberapa data mengenai kelayakan kredit nasabah.

Laporan Kita akan dipertimbangkan pada saat membuat **penilaian kredit** untuk calon nasabah. **Penilaian kredit** digunakan untuk mengevaluasi kemampuan calon peminjam untuk melunasi pinjaman mereka.

Tujuan utama dari poject ini adalah untuk mengetahui kelayakan seorang klien untuk mendapatkan kredit berdasarkan status dan keadaan mereka yang tersimpan dalam data kita. Kita juga menguji kapasitas nasabah berdasarkan karakteristik mereka yang kita rangkum berdasarkan kategori-kategori sehingga diperoleh *pattern* untuk memberikan lampu kuning kepada nasabah yang masuk ke dalam kategori tertentu.

Hipotesis project :
1. Apakah terdapat korelasi antara jumlah anak dengan kemampuan melunasi pinjaman tepat waktu?
2. Apakah terdapat korelasi antara status keluarga dengan kemampuan melunasi pinjaman tepat waktu?
3. Apakah terdapat korelasi antara kelas ekonomi dengan kemampuan melunasi pinjaman tepat waktu?
4. Apakah terdapat korelasi antara tujuan kredit dengan kemampuan melunasi pinjaman tepat waktu?

## Membuka *file* data dan menampilkan informasi umumnya. 

Kita akan mulai dengan mengimport library dan memuat data.
''')

code1 = ('''# Memuat semua perpustakaan
import pandas as pd
''')
kode(code1)

code2 = ('''# memuat data
df = pd.read_csv('/datasets/credit_scoring_eng.csv')
''')
kode(code2)

df = pd.read_csv('https://github.com/syaiddewantoro/project_practicum/blob/main/credit_scoring_eng.csv')

st.markdown('''## Tahap 1. Eksplorasi Data

**Deskripsi Data**
- *`children`* - jumlah anak dalam keluarga
- *`days_employed`* - pengalaman kerja dalam hari
- *`dob_years`* - usia klien dalam tahun
- *`education`* - pendidikan klien
- *`education_id`* - tanda pengenal pendidikan
- *`family_status`* - status perkawinan
- *`family_status_id`* - tanda pengenal status perkawinan
- *`gender`* - jenis kelamin klien
- *`income_type`* - jenis pekerjaan
- *`debt`* - apakah klien memiliki hutang pembayaran pinjaman
- *`total_income`* - pendapatan bulanan
- *`purpose`* - tujuan mendapatkan pinjaman
''')

code3 = ('''# Memeriksa jumlah baris dan kolom dalam dataset 
df.shape
''')
kode(code3)

df_shape = df.shape
df_shape

code4 = ('''# Menampilkan 10 baris pertama dalam dataset
df.head(10)
''')
kode(code4)

df

st.markdown('''Dari sampel data yang ditampilkan, terdapat beberapa masalah yang bisa dideteksi yaitu tedapat value negatif dan value yang kita nilai sangat tinggi dari kolom `days_employed` yang kita nilai tidak masuk akal karena pada kolom menampilkan pengalaman kerja dalam hari , serta penulisan huruf kapital yang tidak tidak teratur pada kolom `education`.
''')

code5 = ('''# Mendapatkan informasi seluruh kolom dalam data
df.info()
''')
kode(code5)

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.markdown('''Terdapat nilai yang hilang pada kolom `days_employed` dan `total_income`.
''')

code6 = ('''# Menampilkan nilai yang hilang dalam dataset
df[df['days_employed'].isna()].head(10)
''')
kode(code6)

df_days_employed_isna = df[df['days_employed'].isna()].head(10)
df_days_employed_isna

st.markdown('''Sejauh ini, dari yang kita lihat, kita asumsikan bahwa value yang hilang memang tampak simetris karena berada pada baris yang sama, tapi untuk dapat menyimpulkan apakah nilai yang hilang memang bebar-benar berada pada baris yang sama perlu dilakukan investigasi lebih lanjut.
''')

code7 = ('''# Melakukan beberapa pemfilteran untuk mengetahui jumlah baris dari value yang hilang
df_filtered_nan = df[df['days_employed'].isna()]
df_filtered_nan = df_filtered_nan[df_filtered_nan['total_income'].isna()]
df_filtered_nan.shape[0]
''')
kode(code7)

df_filtered_nan = df[df['days_employed'].isna()]
df_filtered_nan = df_filtered_nan[df_filtered_nan['total_income'].isna()]
df_filtered_nan_shape = df_filtered_nan.shape[0]
df_filtered_nan_shape

st.markdown('''Data yang hilang dalam dataset kita berjumlah `2174` baris.
''')

code8 = ('''# Sekarang kta akan menghitung rasio value yang hilang dari seluruh dataframe
df_distribution_nan = df_filtered_nan.shape[0] / df.shape[0] 
st.write(f'Distribusi nilai yang hilang sebesar: {df_distribution_nan:0%}')
''')
kode(code8)

df_distribution_nan = df_filtered_nan.shape[0] / df.shape[0] 
df_distribution_nan_print = st.write(f'Distribusi nilai yang hilang sebesar: {df_distribution_nan:0%}')
df_distribution_nan_print

st.markdown('''**Kesimpulan menengah**

Kita bisa simpulkan jumlah nilai hilang sama dengan jumlah tabel yang difilter. Artinya nilai yang hilang dari tabel yang difilter simetris.

Persentase data yang hilang dari dataframe sebesar `10%`, cukup berpengaruh terhadap data kita bukan?

Selanjutnya kita akan menghitung jumlah persentase dari value yang hilang apakah memiliki dampak yang signifikan terhadap data dalam dataset, sehingga kita akan mengetahui langkah yang tepat untuk memproses data yang hilang tersebut.
''')

code9 = ('''# Menerapkan filter untuk menampilkan baris yang hilang dari data
df_filtered_nan.head(10)
''')
kode(code9)

df_filtered_nan_head = df_filtered_nan.head(10)
df_filtered_nan_head

st.markdown('''Value yang hilang tampak simetris dari table yang ditampilkan.
''')

code10 = ('''# Memeriksa distribusi
st.write(df_filtered_nan['income_type'].value_counts())
st.write()
df_filtered_nan['income_type'].value_counts() / df['income_type'].value_counts() * 100
''')
kode(code10)

cetak_df_filtered = st.write(df_filtered_nan['income_type'].value_counts())
cetak_df_filtered
cetak1 = st.write()
cetak1
cetak_df_filtered1 =  df_filtered_nan['income_type'].value_counts() / df['income_type'].value_counts() * 100
cetak_df_filtered1

st.markdown('''Cukup terpola disini tetapi memang ada beberapa kategori yang valuenya tidak bisa dikatakan valid untuk dilakukan perhitungan mengingat jumlahnya hanya sedikit sekitar `1 - 2` data saja jumlahnya seperti kategori `unemployed, paternity / maternity leave, student,` dan `entrepreneur`. Secara overall data yang hilang terdistribusi sebesar `10%` pada setiap kategori.

**Kemungkinan penyebab hilangnya nilai dalam data**

Belum dapat ditentukan penyebab nilai yang hilang, kita harus mempertimbangkan kemungkinan-kemungkinan lain penyebab nilai yang hilai apakah nilai yang memiliki karakteristik tertentu seperti dari klien yang sudah menikah, jumlah anak, ataupun klien yang memiliki tunggakan pembayaran kredit.
''')

code11 = ('''# Memeriksa distribusi di seluruh dataset
df.isna().sum() / df.shape[0] * 100
''')
kode(code11)

cetak_df_isna1 = df.isna().sum() / df.shape[0] * 100
cetak_df_isna1

st.markdown('''**Kesimpulan Sementara**

Jumlah nilai yang hilang dalam dataset mirip dengan tabel yang difilter.
<br>Dan jumlah data yang hilang pada kolom `days_employed` sama dengan `total_income` yang artinya error hanya terjadi pada dua kolom tersebut.

Hal ini menimbulkan pertanyaan apakah nilai yang hilang membentuk suatu pola atau terjadi secara acak?
<br>Untuk mejawab pertannyaan ini mari kita lakukan analisa lebih lanjut.
''')

code12 = ('''# Memeriksa apakah ada pola lain yang menyebabkan hilangnya data dari kolom 'family_status'
st.write(df_filtered_nan['family_status'].value_counts())
st.write()
df_filtered_nan['family_status'].value_counts() / df['family_status'].value_counts() * 100
''')
kode(code12)

cetak_df_value_counts = st.write(df_filtered_nan['family_status'].value_counts())
cetak_df_value_counts
st.write()
cetak_df_filtered2 =  df_filtered_nan['family_status'].value_counts() / df['family_status'].value_counts() * 100
cetak_df_filtered2

st.markdown('''**Kesimpulan Sementara**

Cukup menarik bahwa nilai yang hilang terdistribusi secara merata dari kategori di kolom `family_status` yaitu sekitar `10%`. Tetapi juga dikatahui tidak ada ketegori khusus yang mengakibatkan data hilang disebabkan oleh salah satu kategori saja.
''')

code13 = ('''# Check for relation both 'gender' and missing value
st.write(df_filtered_nan['gender'].value_counts())
st.write()
df_filtered_nan['gender'].value_counts() / df['gender'].value_counts() * 100
''')
kode(code13)

st.write(df_filtered_nan['gender'].value_counts())
st.write()
cetak_df_filtered4 =  df_filtered_nan['gender'].value_counts() / df['gender'].value_counts() * 100
cetak_df_filtered4

st.markdown('''Pada kategori gender data yang hilang masing-masing terdistribusi sebesar `9%` dan `10%` artinya data yang hilang tidak hanya terdapat pada satu kategori saja. 
''')

code14 = ('''# Memeriksa pendapatan dari beberapa pekerjaan yang kemungkinan tidak memiliki 'income'
df[df['income_type'].isin(['unemployed', 'paternity / maternity leave', 'student', 'entrepreneur'])]
''')
kode(code14)

df_isin = df[df['income_type'].isin(['unemployed', 'paternity / maternity leave', 'student', 'entrepreneur'])]
df_isin

st.markdown('''Dari tabel diatas kita bisa mendapatkan informasi bahwa nilai yang hilang tidak selalu diakibatkan oleh pekerjaan yang biasanya tidak memiliki penghasilan, mungkin ada beberapa alasan bagaimana cara mereka mendapatkan penghasilan meskipun tidak sedang memiliki pekerjaan yang regular. Seperti pada baris `3133` klien yang `unemployed` tetapi mengajukan pinjaman untuk membeli properti untuk desewakan, meskipun belum diketahui darimana penghasilannya, Klien `9410` seorang `student` apakah dia seorang pekerja part time atau memiliki *rich parents* tetapi cukup aneh mengingat kegunaannya tidak untuk melanjutkan pendidikan melainkan membangun properti. Selain itu di beberapa negara juga memilki peraturan bahwa `paternity / maternity leave` *still getting paid*.

**Kesimpulan**

Konklusi yang bisa kita ambil mengenai penyebab data yang hilang adalah *human error* karena data yang hilang tidak terjadi pada kategori tertentu saja melainkan pada beberapa kategori.

Dari beberapa pengujian yang sudah kita lakukan kita menemukan bahwa dari kolom `family_status` dan `income_type` kita menemukan bahwa setiap kategori dalam masing-masing kolom tersebut memiliki nilai yang hilang hampir sama di setiap kategori yang sekitar `10%` artinya nilai yang hilang terdistribusi secara merata di setiap kategori dan nilainya simetris dengan pengujian yang kita lakukan sebelumnya dengan mencari distribusi nilai yang hilang di seluruh dataset yang menghasilkan angka juga sebesar `10%`.

Untuk beberapa masalah seperti:
1. Untuk nilai yang hilang kita akan membuat beberapa kateghori berdasarkan usia untuk mencari rata-rata `days_employed` dan `total_income` yang akan digunakan untuk mengisi value yang hilang.
2. Untuk mengatasi register yang berbeda pada kolom `education` kita akan mengubah semua huruf menjadi `lower`.
3. Kita bisa melakukan `drop` untuk nilai duplikat.

## Transformasi data

**Memeriksa kolom 'education'**
''')

code15 = ('''# Memeriksa ejaan pada kolom 'education' yang terindikasi memiliki perbebedaan penulisan dengan makna yang sama
df_education_value = df.pivot_table(index='education', values='days_employed', aggfunc= 'count')
df_education_value
''')
kode(code15)

df_education_value = df.pivot_table(index='education', values='days_employed', aggfunc= 'count')
df_education_value

code16 = ('''# Kemudian kita akan memeriksa nilai duplikat sebelum kita mengangani register yang tidak teratur
df.duplicated().sum()
''')
kode(code16)

df_duplicated = df.duplicated().sum()
df_duplicated

code17 = ('''# Memperbaiki penulisan
df['education'] = df['education'].str.lower()
''')
kode(code17)

df['education'] = df['education'].str.lower()

code18 = ('''# Memeriksa kembali nilai duplikat setelah dilakukan perbaikan
df.duplicated().sum()
''')
kode(code18)

df_duplicated1 = df.duplicated().sum()
df_duplicated1

st.markdown('''Nilai duplikat bertambah dari sebelumnya `54` baris menjadi `71` baris, artinya ada baris yang memilki nilai yang sama dalam data dengan input `education` yang berbeda. 
''')

code19 = ('''# Memeriksa kembali apakah penulisan kolom 'education' telah diperbaiki
df_education_value = df.pivot_table(index='education', values='days_employed', aggfunc= 'count')
df_education_value
''')
kode(code19)

df_education_value = df.pivot_table(index='education', values='days_employed', aggfunc= 'count')
df_education_value

st.markdown('''Masalah pada kolom `education` telah kita atasi selanjutnya kita akan memeriksa kolom `children`.

**Memeriksa kolom `children`**
''')

code20 = ('''# Menampilkan distribusi nilai pada kolom `children`
df_children_value = df.pivot_table(index='children', values='days_employed', aggfunc= 'count')
df_children_value
''')
kode(code20)

df_children_value = df.pivot_table(index='children', values='days_employed', aggfunc= 'count')
df_children_value

st.markdown('''- Terdapat value yang menunjukkan angka -1 dan 20 yang kita nilai tidak mungkin terjadi pada keolompok usia.
- Kita asumsikan nilai yang bermasalah terjadi karena kesalahan input/typo.
- Kita akan me-replace nilai -1 menjadi 1 dan 20 menjadi 2.
''')

code21 = ('''# Memeriksa jumlah dan persentase data yang bermasalah pada kolom 'children'
df_child_problem_value = (df['children'] == -1).sum() + (df['children'] == 20).sum()
st.write('Jumlah nilai yang tidak wajar:', df_child_problem_value)
df_child_prob_percent = df_child_problem_value / df.shape[0]
st.write(f'Persentase data yang bermasalah adalah: {df_child_prob_percent:.2%}')
''')
kode(code21)

df_child_problem_value = (df['children'] == -1).sum() + (df['children'] == 20).sum()
st.write('Jumlah nilai yang tidak wajar:', df_child_problem_value)
df_child_prob_percent = df_child_problem_value / df.shape[0]
st.write(f'Persentase data yang bermasalah adalah: {df_child_prob_percent:.2%}')

st.markdown('''Data dengan nilai yang tidak wajar tidak sampai `1%` dalam dataset kita meskipun begitu lebih baik kita perbaiki alih-alih melakukan *drop* pada data tersebut.
''')

code22 = ('''# Memperbaiki nilai yang bermasalah
df.loc[df['children'] == -1, 'children'] = 1 
df.loc[df['children'] == 20, 'children'] = 2 
df.duplicated().sum()
''')
kode(code22)

df.loc[df['children'] == -1, 'children'] = 1 
df.loc[df['children'] == 20, 'children'] = 2 
df_duplicated2 = df.duplicated().sum()
df_duplicated2

code23 = ('''# Periksa kembali kolom `children` untuk memastikan semua telah diperbaiki
df_children_value = df.pivot_table(index='children', values='days_employed', aggfunc= 'count')
df_children_value
''')
kode(code23)

df_children_value = df.pivot_table(index='children', values='days_employed', aggfunc= 'count')
df_children_value

code24 =('''# Mari kita periksa kembali nilai duplikat
df.duplicated().sum()
''')
kode(code24)

df_duplicated3 = df.duplicated().sum()
df_duplicated3

st.markdown('''Belum ada perubahan pada nilai duplikat.

**Memeriksa kolom `days_employed`**
''')

code25 = ('''# Kita akan melihat nilai yang tidak wajar pada kolom 'days_employed'
df[df['income_type'] == 'retiree']
''')
kode(code25)

df_income_retiree = df[df['income_type'] == 'retiree']
df_income_retiree

code26 = ('''# Menampilkan statistik deskriptif pada kolom 'days_employed' 
df['days_employed'].describe()
''')
kode(code26)

df_days_employed_describe = df['days_employed'].describe()
df_days_employed_describe

st.markdown('''Terdapat angka-angka negatif dan nilai yang tidak wajar, perlu diingat kolom ini menampilkan jumlah hari klien telah bekerja.
''')

code27 = ('''# Menampilkan nilai rata-rata 'days_employed'
df_days_employed_median = df.groupby(['income_type'])['days_employed'].median()
df_days_employed_median
''')
kode(code27)

df_days_employed_median = df.groupby(['income_type'])['days_employed'].median()
df_days_employed_median

st.markdown('''Kita tidak bisa melakukan proses uji kelayakan credit sebelum memperbaiki nilai pada kolom ini, nilai negatif sama dengan nilai yang hilang.
''')

code28 = ('''# persentase data yang bermasalah
df_days_employed_problem_value = (df['days_employed'] < 0).sum() + (df['days_employed'] > 21600).sum()
st.write(df_days_employed_problem_value)
df_days_employed_prob_percent = df_days_employed_problem_value / df.shape[0]
st.write(f'Persentase data yang bermasalah adalah: {df_days_employed_prob_percent:.2%}')
''')
kode(code28)

df_days_employed_problem_value = (df['days_employed'] < 0).sum() + (df['days_employed'] > 21600).sum()
st.write(df_days_employed_problem_value)
st.write()
df_days_employed_prob_percent = df_days_employed_problem_value / df.shape[0]
st.write(f'Persentase data yang bermasalah adalah: {df_days_employed_prob_percent:.2%}')

st.markdown('''- Terdapat masalah yang sangat tinggi pada kolom `days_employed`, jika kita melihat nilai negatif sudah berdampak besar dalam dataset karena itu sama saja dengan nilai yang hilang, lalu bagaimana dengan nilai yang terlampau tinggi, tidak ada seorangpun bekerja melebihi usianya sendiri.
- Karena nilai mungkin terjadi karena kesalah teknis dalam penambahan tanda `-` jadi solusi yang dapat kita lakukan adalah dengan dengan mengganti angka negatif menjadi positif dengan fungsi `absolut`.
- Kemudian untuk nilai yang terlampau tinggi yang kita asumsikan hanya terjadi pada pada `kategori`, `retiree` dan `unemployed` maka kita putuskan untuk mengganti nilai tersebut dengan `0` dengan batas `21600 hari atau 59 tahun`.
''')

code29 = ('''# Menerapkan fungsi 'absolut' untuk nilai negatif dan me-'replace' nilai yang tidak wajar dengan angka '0'
df['days_employed'] = df['days_employed'].abs()
df.loc[df['days_employed'] > 21600, 'days_employed'] = 0
''')
kode(code29)

df['days_employed'] = df['days_employed'].abs()
variable2 = df.loc[df['days_employed'] > 21600, 'days_employed'] = 0
variable2

code30 = ('''# Memastikan nilai telah diperbaiki
st.write(df['days_employed'].describe())
st.write()
st.write((df['days_employed'] < 0).sum())
''')

st.write(df['days_employed'].describe())
st.write()
st.write((df['days_employed'] < 0).sum())

st.markdown('''Terdapat masalah pada kolom `usia` yaitu tidak seseorang yang bekerja sejak usia `0`

**Memeriksa kolom `dob_years`**
''')

code31 = ('''# Memeriksa kolom 'dob_years' apakah memiliki nilai anomali
df_dob_years_value = df.pivot_table(index='dob_years', values='days_employed', aggfunc= 'count')
df_dob_years_value
''')
kode(code31)

df_dob_years_value = df.pivot_table(index='dob_years', values='days_employed', aggfunc= 'count')
df_dob_years_value

code32 = ('''# Melihat persentase data anomali
df_days_employed_prob_percent = (df['dob_years'] == 0).sum() / df.shape[0]
st.write(f'Persentase data yang bermasalah adalah: {df_days_employed_prob_percent:.2%}')
''')
kode(code32)

df_days_employed_prob_percent = (df['dob_years'] == 0).sum() / df.shape[0]
st.write(f'Persentase data yang bermasalah adalah: {df_days_employed_prob_percent:.2%}')

st.markdown('''Kita akan mengganti value `0` di kolom `age`, dengan `median` dari usia.
<br>Karena tiap klien yang memiliki usia `0` memiliki karakteristik yang berbeda.
''')

code33 = ('''# Mengganti angka '0' dengan rata-rata
avg_dob_years = df['dob_years'].median() 
df.loc[df['dob_years'] == 0, 'dob_years'] = avg_dob_years
''')
kode(code33)

avg_dob_years = df['dob_years'].median() 
variable1 = df.loc[df['dob_years'] == 0, 'dob_years'] = avg_dob_years
variable1

code34 = ('''# Memastikai nilai telah diperbaiki
df[df['dob_years'] == 0].shape[0]
''')
kode(code34)

st.write(df[df['dob_years'] == 0].shape[0])

st.markdown('''**Memeriksa kolom `family_status`**
''')

code35 = ('''# Memeriksa kolom 'family_status'
df_family_status_value = df.pivot_table(index='family_status', values='days_employed', aggfunc= 'count')
df_family_status_value
''')
kode(code35)

df_family_status_value = df.pivot_table(index='family_status', values='days_employed', aggfunc= 'count')
df_family_status_value

st.markdown('''Tidak ditemukan masalah pada kolom `family_status`

**Memeriksa kolom `gender`**
''')

code36 = ('''# Check 'gender' column
df_gender_value = df.pivot_table(index='gender', values='days_employed', aggfunc= 'count')
df_gender_value
''')
kode(code36)

df_gender_value = df.pivot_table(index='gender', values='days_employed', aggfunc= 'count')
df_gender_value

st.markdown(''''Terdapat satu masalah yaitu pada value `XNA`
<br>Sulit untuk mengidentifikasi untuk mengganti value tersebut, dan kita putuskan untuk menghapus 1 baris tersebut
''')

code37 = ('''# Menghapus value `XNA`
df = df.loc[df["gender"] != 'XNA']
''')
kode(code37)

df = df.loc[df["gender"] != 'XNA']

code38 = ('''# Memastikan kolom telah diperbaiki
df_gender_value = df.pivot_table(index='gender', values='days_employed', aggfunc= 'count')
df_gender_value
''')
kode(code38)

df_gender_value1 = df.pivot_table(index='gender', values='days_employed', aggfunc= 'count')
df_gender_value1

st.markdown('''**Memeriksa kolom `income_type`**
''')

code39 = ('''# Mari kita lihat nilai dalam kolom
df_income_type_value = df.pivot_table(index='income_type', values='days_employed', aggfunc= 'count')
df_income_type_value
''')
kode(code39)

df_income_type_value = df.pivot_table(index='income_type', values='days_employed', aggfunc= 'count')
df_income_type_value

st.markdown('''Tidak ada masalah pada kolom `income_type`

Terdapat nilai duplikat sebesar `72` atau hanya berdampak `0.3%` dalam dataset sehingga kita putuskan untuk melakukan `drop` untuk nilai duplikat, karena nilai tersebut masih dapat diterima.
''')

code40 = ('''# Memeriksa duplikat
st.write(df.duplicated().sum())
st.write()
df_duplicated_percent = df.duplicated().sum() / df.shape[0] 
st.write('Distribusi nilai duplikat sebesar:', df_duplicated_percent:.2%)
''')
kode(code40)

st.write(df.duplicated().sum())
st.write()
df_duplicated_percent = df.duplicated().sum() / df.shape[0] 
st.write(f'Distribusi nilai duplikat sebesar: {df_duplicated_percent:.2%}')

code41 = ('''# Atasi duplikat, jika ada
df = df.drop_duplicates().reset_index(drop=True)
''')
kode(code41)

df = df.drop_duplicates().reset_index(drop=True)

code42 = ('''# Terakhir periksa apakah kita memiliki duplikat
df.duplicated().sum()
''')
kode(code42)

st.write(df.duplicated().sum())


code43 = ('''# Periksa ukuran dataset yang sekarang Anda miliki setelah manipulasi pertama yang Anda lakukan
df.shape
''')
kode(code43)

st.write(df.shape)

code44 = ('''# rasio old_data_shape dan new_data_shape
old_df_shape = 21525
df_shape_percentage = (old_df_shape - df.shape[0]) / old_df_shape
st.write(f'Persentase dari perubahan dalam dataset adalah: {df_shape_percentage:.2%}')
''')
kode(code44)

old_df_shape = 21525
df_shape_percentage = (old_df_shape - df.shape[0]) / old_df_shape
st.write(f'Persentase dari perubahan dalam dataset adalah: {df_shape_percentage:.2%}')

st.markdown('''Kita telah memperbaiki beberapa masalah dalam dataset seperti:
1. Nilai negatif dan nilai yang terlampau tinggi dalam kolom `days_employed`.
2. Register yang tidak teratur dalam kolom `education`.
3. Menghapus nilai duplikat.
4. Beberapa masalah yang terjadi di kolom `children` dan `gender`.

Sehingga kita mendapatkan perubahan sebesar `0.34%` yang artinya nilai tersebut tidak begitu berdampak dalam dataset karena kurang dari `1%`.

# Bekerja dengan nilai yang hilang

Kita memasukkan dictionary `numpy` untuk mempercepat pekerjaan kita yang akan digunakan untuk me-`replace` nilai `0` pada kolom `days_employed` setelah membuat beberapa kategori usia.
''')

code45 = ('''# Import dictionary
import numpy as np
''')
kode(code45)

import numpy as np

st.markdown('''### Memperbaiki nilai yang hilang di `total_income`

Kita akan mulai untuk mengatasi nilai yang hilang pada kolom`total_income`.

Pertama-tama kita akan membuat beberapa kategori berdasarkan rentang usia dari kolom `dob_years`, yang diharapkan dapat membantu kita dalam mencari nilai rata-rata dari setiap rentang usia yang akan kita gunakan untuk mengisi nilai yang hilang dari masing-masing kategori.
''')

code46 = ('''# Menulis fungsi untuk menghitung kategori usia
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
''')
kode(code46)

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

code47 = ('''# Melakukan pengujian apakah fungsi bekerja atau tidak
st.write(age_group(23))
st.write(age_group(38))
st.write(age_group(46))
st.write(age_group(55))
st.write(age_group(70))
''')
kode(code47)

st.write(age_group(23))
st.write(age_group(38))
st.write(age_group(46))
st.write(age_group(55))
st.write(age_group(70))

code48 = ('''# Membuat kolom baru berdasarkan fungsi
df['age_group'] = df['dob_years'].apply(age_group)
''')
kode(code48)

df['age_group'] = df['dob_years'].apply(age_group)

code49 = ('''# Memeriksa bagaimana nilai di dalam kolom baru
df.tail(10)
''')
kode(code49)

cetak_df_tail = df.tail(10)
cetak_df_tail

st.markdown('''- Beberapa faktor yang dapat memengaruhi pendapatan diantaranya `education`, `days_employed` (pengalaman kerja dalam hari), dan `income_type` (jenis pekerjaan).
- Untuk menentukan apakah kita akan menggunakan `mean` atau `median` untuk mengisi nilai yang hilang, kita melakukan eksplorasi lebih dalam untuk mengetahui bagaimana distribusi nilai `mean` dan `median` berdasarkan kategori yang kita kelompokkan terdistribusi secara normal atau tidak.
''')

code50 = ('''# Membuat tabel tanpa nilai yang hilang dan menampilkan beberapa barisnya
df_clean = df[df.notna()]
df_clean.head(10)
''')
kode(code50)

df_clean = df[df.notna()]
df_clean = df_clean.head(10)
df_clean

code51 = ('''# Menampilkan nilai 'mean' dari 'total_income' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['total_income'].mean()
''')
kode(code51)

df_clean_groupby = df_clean.groupby(['age_group'])['total_income'].mean()
df_clean_groupby 

code52 =('''# Menampilkan nilai 'median' dari 'total_income' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['total_income'].median()
''')
kode(code52)

df_clean_groupby1 = df_clean.groupby(['age_group'])['total_income'].median()
df_clean_groupby1

st.markdown('''Pendapatan rata-rata berdasarkan kategori dari `mean` memiliki rata-rata yang lebih besar untuk setiap kategori, dengan demikian kita akan memilih `median` karena lebih merepresentasikan rata-rata berdasarkan kategori kelompok umur.
''')

code53 = ('''# Saatnya kita mengisi nilai yang hilang
df['total_income'] = df.groupby(['age_group'])['total_income'].transform(lambda x: x.fillna(x.median()))
''')
kode(code53)

df['total_income'] = df.groupby(['age_group'])['total_income'].transform(lambda x: x.fillna(x.median()))

code54 = ('''# Memeriksa statistik deskriptif dari 'total_income'
df['total_income'].describe()
''')
kode(code54)

df_total_income_describe = df['total_income'].describe()
df_total_income_describe

code55 = ('''# Memeriksa apakah nilai hilang sudah terisi
df['total_income'].isna().sum()
''')
kode(code55)

df_total_income_isna_sum = df['total_income'].isna().sum()
df_total_income_isna_sum

st.markdown('''Diketahui data yang hilang belum terisi pada kolom `total_income` itu karena kita belum memasukkan nilai dari kolom baru yang kita buat untuk mengganti nilai yang hilang dari kolom tersebut. Untuk mengisi kolom tersebut kita akan menggantinya dengan metode `fillna` dari kolom baru yang kita buat yaitu `total_revenue`.

Seperti yang kita lihat sudah tidak terdapat nilai yang hilang dari kolom `total_income`.
Saatnya memeriksa apakah jumlah kolom `total_income` sama dengan jumlah kolom lainnya.
''')

code56 = ('''# Memeriksa jumlah entri kolom dari dataset
df.info()
''')
kode(code56)

buffer1 = io.StringIO()
df.info(buf=buffer1)
s1 = buffer1.getvalue()
st.text(s1)

st.markdown('''Dari informasi di atas kita menemukan bahwa jumlah pada kolom `total_income` sudah memiliki nilai yang sama dengan kolom lainnya.
<br>Selanjutnya kita akan memperbaiki nilai pada kolom `days_employed`.

###  Memperbaiki nilai di `days_employed`

Tentunya faktor yang paling berpengaruh di dalam kolom `days_employed` adalah `age_category` bayangkan seseorang tidak mungkin berkerja melebihi usia minimal seseorang untuk bekerja secara legal atau bahkan pengalaman kerja seseorang melebihi jumlah usianya sendiri, itu merupakan hal yang mustahil.
''')

code57 = ('''# Menampilkan nilai 'median' dari 'days_employed' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['days_employed'].median()
''')
kode(code57)

df_clean_groupby2 = df_clean.groupby(['age_group'])['days_employed'].median()
df_clean_groupby2

code58 = ('''# Menampilkan nilai 'median' dari 'days_employed' berdasarkan 'age_group'
df_clean.groupby(['age_group'])['days_employed'].mean()
''')
kode(code58)

df_clean_groupby3 = df_clean.groupby(['age_group'])['days_employed'].mean()
df_clean_groupby3

st.markdown('''Kita merasa `mean` bisa mewakili nilai untuk kolom `days_employed`, selain karena `median` memiliki nilai `0` yang mungkin terjadi karena kita mengubah nilai yang trelampau tinggi pada pengujian sebelumnya, tetapi apakah seseorang yang telah mencapai usia pensiun tidak memiliki pengalaman bekerja sama sekali? Atau mungkin sesorang yang telah pensiun pengalamannya akan dihitung sebagai nilai `0`.
''')

code59 = ('''# Waktunya untuk kita mengganti nilai yang hilang dan nilai '0' dengan rata-rata
df.loc[df['days_employed'] == 0, 'days_employed'] = np.NaN
df['days_employed'] = df.groupby(['age_group'])['days_employed'].transform(lambda x: x.fillna(x.mean()))
''')
kode(code59)

df.loc[df['days_employed'] == 0, 'days_employed'] = np.NaN
df['days_employed'] = df.groupby(['age_group'])['days_employed'].transform(lambda x: x.fillna(x.mean()))

code60 = ('''# Menampilkan statistik deskriptif dari kolom 'days_employed'
df['days_employed'].describe()
''')
kode(code60)

df_days_employed_describe1 = df['days_employed'].describe()
df_days_employed_describe1

code61 = ('''# Memeriksa apakah nilai yang hilang telah teratasi
df['days_employed'].isna().sum()
''')
kode(code61)

df_days_employed_isna1 = df['days_employed'].isna().sum()
df_days_employed_isna1

st.markdown('''Memeriksa jumlah kolom dari seluruh dataset.
''')

code62 = ('''# Memeriksa informasi seluruh dataset
df.info()
''')
kode(code62)

buffer2 = io.StringIO()
df.info(buf=buffer2)
s2 = buffer2.getvalue()
st.text(s2)

st.markdown('''Semua kolom termasuk `days_employed` telah memiliki nilai yang sama, Kita nilai pekerjaan kita untuk *data cleansing* telah selesai. Sekarang mari kita coba mengeksplorasi data lebih lanjut untuk mendapatkan hal menarik dalam data maupun sebagai acuan kita dalam mengambil keputusan.

## Pengkategorian Data

Sepertinya kita menemukan hal menarik di kolom `purpose` yaitu banyak sekali peng-kategorian data yang kita nilai bisa kita sederhanakan menjadi lebih general, sehingga kita akan lebih mudah dalam melakukan investigasi dalam mengambil keputusan.
''')

code63 = ('''# Menampilkan kolom 'purpose' untuk dikategorikan dengan lebih umum
df['purpose'].value_counts()
''')
kode(code63)

df_purpose_value_counts = df['purpose'].value_counts()
df_purpose_value_counts

st.markdown('''Memeriksa value `unique`.
''')

code64 = ('''# Memeriksa nilai unik
df['purpose'].sort_values().unique()
''')
kode(code64)

df_purpose_unique = df['purpose'].sort_values().unique()
df_purpose_unique

st.markdown('''Dari pengamatan kita sebelumnya memang benar bahwa peng-kategorian yang terjadi di kolom `purpose` bisa generalisasikan ke dalam beberapa kategori yang umum dan mudah dipahami, sperti `car`, `property`, `education`, dan `wedding`. Mari kita buat fungsinya di bawah ini:
''')

code65 = ('''# Menulis fungsi untuk mengkategorikan data berdasarkan topik umum
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
''')
kode(code65)

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

code66 = ('''# Memuat kolom dengan kategori dan menghitung nilainya
df['general_purpose'] = df['purpose'].apply(purpose_common)
st.write(df['general_purpose'].value_counts())
st.write()
df['general_purpose'].count()
''')
kode(code66)

df['general_purpose'] = df['purpose'].apply(purpose_common)
st.write(df['general_purpose'].value_counts())
st.write()
st.write(df['general_purpose'].count())


st.markdown('''Kita akan membuat kategori berdasarkan `total_income` ke dalam beberapa kelas.
''')

code67 = ('''# Melihat kolom 'total_income' untuk dikategorikan
df['total_income'].value_counts().sort_index()
''')
kode(code67)

df_total_income_value_counts = df['total_income'].value_counts().sort_index()
df_total_income_value_counts

code68 = ('''# Mendapatkan kesimpulan statistik untuk kolom 'total_income'
df['total_income'].describe()
''')
kode(code68)

df_total_income_describe1 = df['total_income'].describe()
df_total_income_describe1

st.markdown('''Kita mengkategorikan `total_income` ke dalam beberapa kelas kategori berdasarkan klasifikasi tingkat ekonomi. Hal ini akan memudahkan kita dalamn proses pengambilan keputusan kedepannya. Mari kita buat fungsinya di bawah ini:
''')

code69 = ('''# Membuat fungsi untuk pengkategorian menjadi kelompok kelas
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
''')
kode(code69)

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

code70 = ('''# Membuat kolom baru dengan kategori
df['economic_class'] = df['total_income'].apply(income_class)
''')
kode(code70)

df['economic_class'] = df['total_income'].apply(income_class)

code71 = ('''# Menghitung distribusi
df['economic_class'].value_counts()
''')
kode(code71)

df_economic_class_value_counts = df['economic_class'].value_counts()
df_economic_class_value_counts

st.markdown('''## Memeriksa Hipotesis

**Apakah terdapat korelasi antara memiliki anak dengan membayar kembali tepat waktu?**
''')

code72 = ('''# Memeriksa apakah kolom 'children' berpengaruh terhadap 'debt'
pivot_table_children = df.pivot_table(index='children', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_children
''')
kode(code72)

pivot_table_children = df.pivot_table(index='children', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_children

code73 = ('''# Memeriksa persentase untuk mendapatkan konklusi
pivot_table_children['percent_1'] = pivot_table_children[1] / (pivot_table_children[1] + pivot_table_children[0]) * 100
pivot_table_children
''')
kode(code73)

pivot_table_children['percent_1'] = pivot_table_children[1] / (pivot_table_children[1] + pivot_table_children[0]) * 100
pivot_table_children

st.markdown('''**Kesimpulan**

Dari data di atas kita temukan bahwa:
1. Klien yang memliki `1` sampai `4` anak memiliki persentase yang hampir sama di angka `8%` sampai `9%`.
2. Klien yang memiliki `5` anak tidak memiliki hutang pembayaran pinjaman tatapi tidak bisa kita jadikan acuan karena jumlah data terlalu sedikit.
3. Klien yang **tidak** memilki anak memiliki rasio yang paling kecil untuk hutang pembayaran pinjaman sebesar `7%` hal ini mungkin terjadi karena mereka memiliki tanggungan yang lebih sedikit dibandingkan dengan klien yang  telah memilki anak.

Hal ini tentu akan memudahkan dalam pengambilan keputusan kita dalam memberikan kredit kepada klien yang belum memiliki anak karena kemampuan mereka dalam melunasi kredit mereka.

**Apakah terdapat korelasi antara status keluarga dengan membayar kembali tepat waktu?**
''')

code74 = (''' # Memeriksa data 'family_status' memengaruhi kolom 'debt'
pivot_table_family_status = df.pivot_table(index='family_status', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_family_status
''')
kode(code74)

pivot_table_family_status = df.pivot_table(index='family_status', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_family_status

code75 = ('''# Menghitung rasio untuk mencari kseimpulan
pivot_table_family_status['percent_1'] = pivot_table_family_status[1] / (pivot_table_family_status[1] + pivot_table_family_status[0]) * 100
pivot_table_family_status
''')
kode(code75)

pivot_table_family_status['percent_1'] = pivot_table_family_status[1] / (pivot_table_family_status[1] + pivot_table_family_status[0]) * 100
pivot_table_family_status

st.markdown('''**Kesimpulan**

Ada beberapa hal menarik yang kita temukan disini:
1. Klien yang `unmarried` dan `civil partnership` memiliki persentase yang cukup tinggi sebesar `9%`.
2. Klien yang `divorced` dan `married` memiliki rasio sebesar `7%` yang artinya lebih kecil daripada klien yang belum menikah apakah karena mereka dapat menggabungkan penghasilan dengan pasangan mereka, hal ini tentu berbanding terbalik dengan pengujian sebelumnya bahwa klien yang belum memiliki anak memiliki persentase hutang lebih kecil mengingat seseorang yang telah menikah punya kecenderungan untuk memiliki anak.
3. Klien dengan status `widow / widower` memiliki pensentase hutang paling kecil sebesar `6%`. Apakah kita akan mempertimbangkan status sebagai dasar dalam pengambilan keputusan?

**Apakah terdapat korelasi antara tingkat pendapatan dengan membayar kembali tepat waktu?**
''')

code76 = ('''# Memeriksa apakah 'economic_class' memiliki korelasi dengan 'debt'
pivot_table_economic_class = df.pivot_table(index='economic_class', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_economic_class
''')
kode(code76)

pivot_table_economic_class = df.pivot_table(index='economic_class', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_economic_class

code77 = ('''# Menghitung persentase distribusi untuk menarik kesimpulan
pivot_table_economic_class['percent_1'] = pivot_table_economic_class[1] / (pivot_table_economic_class[1] + pivot_table_economic_class[0]) * 100
pivot_table_economic_class
''')
kode(code77)

pivot_table_economic_class['percent_1'] = pivot_table_economic_class[1] / (pivot_table_economic_class[1] + pivot_table_economic_class[0]) * 100
pivot_table_economic_class

st.markdown('''**Kesimpulan**

Beberapa hal yang kita temukan dari manipulasi data di atas diantaranya:
1. Klien dengan tingkat ekonomi `lower middle class` dan `middle class` memilki persentase yang seimbang yaitu sebesar `7%` untuk klien yang memiliki hutang pembayaran kredit.
2. Klien dengan tingkat ekonomi `poor` memiliki kemungkinan untuk menunggak lebih besar yaitu `8%`. Apakah ini dapat memengaruhi keputusan kita untuk tidak memberikan kredit mengingat jumlah mereka yang paling banyak.
3. Klien dengan pendapatan `upper middle class` yang memiliki risiko paling kecil sebesar `6%`.

**Bagaimana tujuan kredit memengaruhi tarif otomatis?**
''')

code78 = ('''# Memeriksa persentase 'general_purpose' terhadap 'debt' untuk menggali konklusi
pivot_table_general_purpose = df.pivot_table(index='general_purpose', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_general_purpose['percent_1'] = pivot_table_general_purpose[1] / (pivot_table_general_purpose[1] + pivot_table_general_purpose[0]) * 100
pivot_table_general_purpose
''')
kode(code78)

pivot_table_general_purpose = df.pivot_table(index='general_purpose', columns= 'debt', values='days_employed', aggfunc='count')
pivot_table_general_purpose['percent_1'] = pivot_table_general_purpose[1] / (pivot_table_general_purpose[1] + pivot_table_general_purpose[0]) * 100
pivot_table_general_purpose

st.markdown('''**Kesimpulan**

Berdasarkan penganganan yang kita lakukan hal-hal yang bisa kita dapatkan:
1. Klien yang melakukan pinjaman untuk keperluan `car` dan `education` memiliki risiko gagal bayar paling tinggi sebesar `9%`.
2. Klien yang menggunakan pinjamannya untuk `wedding` memiliki persentase hutang pinjaman yang lebih rendah dibandigkan kriteria sebelumnya.
3. Klien yang membayar tepat waktu yaitu untuk kegunaan `property` dengan risiko hutang tak lancar sebesar `7%`.

# Kesimpulan Umum 

- Kita telah melakukan proses *cleansing data* untuk memperbaiki data-data yang bermasalah dalam dataset kita. Pembersihan yang kita lakukan meliputi mengisi value yang hilang, menghapus nilai duplikat, memperbaiki register yang tak beraturan, nilai yang terlalu besar, hingga mengganti nilai yang tidak wajar, sehingga kita mendapati dataset yang dapat kita olah untuk proses analisa kredit.

- Temuan yang kita dapatkan setelah melakukan beberapa eksplorasi kita mendapati bahwa terdapat korelasi antara jumlah anak dan status perkawinan dalam risiko pemayaran kredit, klien yang tidak memiliki anak akan lebih mudah dalam melunasi hutangnya dibandingkan dengan klien yang memiliki anak. 
    - Klien yang menikah atau pernah memiliki pasangan memiliki risiko lebih rendah gagal bayar daripada klien dengan status *single* maupun tinggal bersama.
    - Klien yang memiliki penghasilan lebih rendah akan lebih tinggi untuk memiliki hutang pinjaman, dan klien yang menggunakan uangnya untuk keperluan rumah akan lebih besar persentase mereka untuk dapat melunasi hutangnya.

- Tetapi apakah semua manipulasi data yang kita lakukan dapat kita gunakan dalam proses *decision making* sehingga akan meminimalisir risiko yang akan terjadi di kemudian hari?
''')
