import pandas as pd
import streamlit as st
import io

def kode(codes):
    st.code(codes, language='python')


code1 = ('import pandas as pd')
kode(code1)

code2 = ('df.info()')
kode(code2)

data = pd.read_csv('F:\Download Ubuntu\Project 2\credit_scoring_eng.csv')

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

## Inisialisasi

Kita akan memulai dengan memuat beberapa library yang akan dibutuhkan pada project ini seperti `pandas`, `numpy`, dan `matplotlib`.
''')