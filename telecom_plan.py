import pandas as pd
import numpy as np 
from matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math as mt
import streamlit as st
import io

def kode(codes):
    st.code(codes, language='python')

def buffer(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    detail = buffer.getvalue()
    return st.text(detail)

sns.set()

st.title('Megaline Telecom')

st.markdown(''' # Table of Contents
- [Megaline Telecom](#scrollTo=WvTQ-2Oqj4q_)
    - [Initialization](#scrollTo=bg9VNXf6j4rH)
    - [Load all Data](#scrollTo=jDBGBh50j4rL)
        - [Users](#scrollTo=WnIOyk0gj4rO)
            - [Fix Data](#scrollTo=fcTItMFqj4rS)
        - [Calls](#scrollTo=fVd2fz5ej4rT)
            - [Fix Data](#scrollTo=gl726Qczj4rW)
            - [Enrich Data](#scrollTo=BKQO4BM6j4rX)
        - [Messages](#scrollTo=k8qQGJlAj4rd)
            - [Fix Data](#scrollTo=iUU4Yctyj4rm)
            - [Enrich Data](#scrollTo=g3PlF8Q9j4rn)
        - [Internet](#scrollTo=MNB9faJij4rq)
            - [Fix Data](#scrollTo=Ot0t5pdUj4rt)
            - [Enrich Data](#scrollTo=z2XqfesTj4ru)
        - [Plans](#scrollTo=b3DWeyiFj4rw)
    - [Study plan conditions](#scrollTo=8OqPU_VSj4ry)
        - [Calls](#scrollTo=IpeCQxE-j4rz)
        - [Messages](#scrollTo=g35ocEe2j4r1)
        - [Internet](#scrollTo=C9_jOz0Ij4r3)
        - [Create new dataframe](#scrollTo=wgavlRIrj4r6)
        - [Calculate monthly revenue](#scrollTo=wUzzyCNZj4r_)
    - [Study Costumer Behavior](#scrollTo=kzGMBq04j4sC)
        - [Calls Duration](#scrollTo=t7JHgxdaj4sE)
        - [Messages Sent](#scrollTo=-qgP6x-rj4sQ)
        - [Data Used](#scrollTo=rbet-dwWj4sY)
        - [Revenue](#scrollTo=EOId3l7Oj4sh)
    - [Test statistical hyposthesis](#scrollTo=T_Nz0mNmj4sq)
- [Conclusion](#scrollTo=UH9sd6tOj4s0)

''')

st.markdown('''# Telecom Plans Anlysis

As an analyst for the telecom operator Megaline, the company offers its clients two prepaid plans, Surf and Ultimate. The marketing department wants to discover which plans generate more revenue to adjust the advertising budget.
We Will conduct a preliminary analysis of the plans based on a relatively small client selection. 

We will have the data on 500 Megaline clients: who they are, where they're from, which plan they use, and the number of calls they made and text messages they sent in 2018. Our job is to analyze the clients' behavior and determine which prepaid plan generates more revenue.

**The purpose of this project is:**
1. Analyze users behavior
2. Calculate the mean, variance and standard deviation
3. Visualize the data and describe the distribution

**Test the hypothesis:**
- The average income of users of Ultimate and Surf phone plans is different.
- The average income of users in the NY-NJ area differs from that of users from other regions.

## 1. Initialization
''')

code1 = ('''
# Loading libraries

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats as st
import math as mt
''')
kode(code1)

st.markdown('''## 2. Load all Data
''')

code2 = ('''
# Load all the DataFrames

try:
    calls = pd.read_csv('/datasets/megaline_calls.csv') 
    internet = pd.read_csv('/datasets/megaline_internet.csv')
    messages = pd.read_csv('/datasets/megaline_messages.csv')
    plans = pd.read_csv('/datasets/megaline_plans.csv')
    users = pd.read_csv('/datasets/megaline_users.csv')
    
except:
    calls = pd.read_csv('/media/syaid/32D6E870D6E8362F/Download Ubuntu/Project 4/megaline_calls.csv') 
    internet = pd.read_csv('/media/syaid/32D6E870D6E8362F/Download Ubuntu/Project 4/megaline_internet.csv')
    messages = pd.read_csv('/media/syaid/32D6E870D6E8362F/Download Ubuntu/Project 4/megaline_messages.csv')
    plans = pd.read_csv('/media/syaid/32D6E870D6E8362F/Download Ubuntu/Project 4/megaline_plans.csv')
    users = pd.read_csv('/media/syaid/32D6E870D6E8362F/Download Ubuntu/Project 4/megaline_users.csv')
''')
kode(code2)

calls = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/moved_megaline_calls.csv') 
internet = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/moved_megaline_internet.csv')
messages = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/moved_megaline_messages.csv')
plans = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/moved_megaline_plans.csv')
users = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/moved_megaline_users.csv')
    

st.markdown('''We have 5 dataframes. Next step we are going to describe our datasets.

### 2.1. Users

Tabel users (data pengguna):
- `user_id` — ID pengguna
- `first_name` — nama depan pengguna
- `last_name` — nama belakang pengguna
- `age` — usia pengguna (tahun)
- `reg_date` — tanggal mulai berlangganan (dd, mm, yy)
- `churn_date` — tanggal pengguna berhenti menggunakan layanan (jika nilainya hilang atau tidak ada, paket 
                   layanan sedang digunakan saat data ini dibuat)
- `city` — kota tempat tinggal pengguna
- `plan` — nama paket telepon
''')

code3 = ('''
# Show the sample of data
users.head()
''')
kode(code3)

st.write(users.head()
)


code4 = ('''
# Show information from users dataset
users.info()
''')
kode(code4)

buffer(users)


st.markdown('''- The dataframe has 500 entries and 8 columns.
- The churn_date and the reg_date columns are currently represent as an object.
- There are some missing values on the churn_date columns.
''')


code5 = ('''
# Checking missing value
users.isna().sum()
''')
kode(code5)

st.write(users.isna().sum()
)


st.markdown('''They are 466 missing values in churn_date, but fill the missing values it's not necessary because the missing values indicate that clients still using our service.

#### 2.1.1. Fix Data
''')


code6 = ('''
# convert data to timestamp
for i in ('reg_date', 'churn_date'):
    users[i] = pd.to_datetime(users[i], format='%Y-%m-%d')
    st.write(users[i].dtype)
''')
kode(code6)

for i in ('reg_date', 'churn_date'):
    users[i] = pd.to_datetime(users[i], format='%Y-%m-%d')
    st.write(users[i].dtype)


st.markdown('''Convert `reg_date` and `churn_date` column to timestamp format

### 2.2. Calls

Tabel calls (data panggilan):
- `id` — ID sesi web unik
- `call_date` — tanggal panggilan
- `duration` — durasi panggilan (dalam menit)
- `user_id` — ID pengguna yang melakukan panggilan
''')


code7 = ('''
# Show the sample of data
calls.head()
''')
kode(code7)

st.write(calls.head()
)


code8 = ('''
# Show information from calls dataset
calls.info()
''')
kode(code8)

buffer(calls)


st.markdown('''- There are 137735 rows and 4 columns on the calls dataframe.
- The call_date columns are currently displayed as an object.
- There is no missing value on the calls dataframe.

Kita akan mengubah type dari kolom call_date ke format Timestamp dan menambahkan beberapa kolom baru untuk untuk melakukan analisis lanjutan.

#### 2.2.1. Fix Data
''')

code9 = ('''
# Convert data to timestamp
calls['call_date'] = pd.to_datetime(calls['call_date'], format='%Y-%m-%d')
calls['call_date'].dtype
''')
kode(code9)

calls['call_date'] = pd.to_datetime(calls['call_date'], format='%Y-%m-%d')
st.write(calls['call_date'].dtype
)


st.markdown('''Convert date in the column call_date to timestamp format.

#### 2.2.2. Enrich Data
''')


code9 = ('''
# Create 'call_month' column from 'call_date'
calls['month'] = calls['call_date'].dt.month
calls['year'] = calls['call_date'].dt.year                    
calls['year_month'] = calls['year'].astype('str') + '_' + calls['month'].astype('str')
calls.head()
''')
kode(code9)


calls['month'] = calls['call_date'].dt.month
calls['year'] = calls['call_date'].dt.year                    
calls['year_month'] = calls['year'].astype('str') + '_' + calls['month'].astype('str')
st.write(calls.head()
)


st.markdown('''We will add a new column called month, based on the call_date column to indicate when the call occurred. Kita juga membuat kolom `year_month` untuk mengantisipasi jika ada tahun lain dalam data untuk meminimalisir kesalahan.
''')


code10 = ('''
# Check the amount of missed calls
calls[calls['duration'] == 0.0]
''')
kode(code10)

st.write(calls[calls['duration'] == 0.0]
)


st.markdown('''We decided to remove the value on the duration column with 0 value.
''')


code11 = ('''
# Remove duration with value 0.0
calls = calls.query('duration != 0.0')
calls[calls['duration'] == 0.0]
''')
kode(code11)

calls = calls.query('duration != 0.0')
st.write(calls[calls['duration'] == 0.0]
)


st.markdown('''The null values has been removed from the calls dataframe.
''')


code12 = ('''
# Membuat fungsi untuk membulatkan durasi panggilan menjadi menit
calls['duration'] = np.ceil(calls['duration']).astype(int)
calls.head()
''')
kode(code12)

calls['duration'] = np.ceil(calls['duration']).astype(int)
st.write(calls.head()
)


st.markdown('''Sesuai dengan ketentuan yang ditetapkan dari operator setiap panggilan akan dibulatkan ke atas meskipun panggilan tersebut hanya berdurasi 1 detik, panggilan tersebut akan tetap dihitung selama menjadi 1 menit.

### 2.3. Messages

Tabel messages (data SMS):
- `id` — ID SMS unik
- `message_date` — tanggal SMS dikirim
- `user_id` — ID pengguna yang mengirim SMS
''')


code13 = ('''
# Show the sample of data
messages.head()
''')
kode(code13)

st.write(messages.head()
)


code14 = ('''
# Show information from messages dataset
messages.info()
''')
kode(code14)

buffer(messages)


st.markdown('''- The dataframe has 76051 rows and 3 columns.
- The message_date is currently shown as object.
- The dataframe has no missing value.

Kita akan mengubah type dari kolom message_date ke format Timestamp dan menambahkan beberapa kolom baru untuk untuk melakukan analisis lanjutan.

#### 2.3.1. Fix Data
''')


code15 = ('''
# Convert data to timestamp
messages['message_date'] = pd.to_datetime(messages['message_date'], format='%Y-%m-%d')
messages['message_date'].dtype
''')
kode(code15)

messages['message_date'] = pd.to_datetime(messages['message_date'], format='%Y-%m-%d')
st.write(messages['message_date'].dtype
)


st.markdown('''The message_date column on the dataframe represent as object we will convert the its column to timestamp.

#### 2.3.2. Enrich Data
''')


code16 = ('''
# Create 'message_month' column from 'message_date'
messages['month'] = messages['message_date'].dt.month
messages['year'] = messages['message_date'].dt.year                    
messages['year_month'] = messages['year'].astype('str') + '_' + messages['month'].astype('str')
messages.head()
''')
kode(code16)

messages['month'] = messages['message_date'].dt.month
messages['year'] = messages['message_date'].dt.year                    
messages['year_month'] = messages['year'].astype('str') + '_' + messages['month'].astype('str')
st.write(messages.head()
)


st.markdown('''We will add a new column called month, its represent the month based the message_date column. Kita juga membuat kolom `year_month` untuk mengantisipasi jika ada tahun lain dalam data untuk meminimalisir kesalahan.

### 2.4. Internet

Tabel internet (data sesi web):
- `id` — ID sesi web unik
- `mb_used` — volume data yang dihabiskan selama sesi (dalam megabita)
- `session_date` — tanggal sesi web
- `user_id` — ID pengguna
''')


code17 = ('''
# Show the sample of data
internet.head()
''')
kode(code17)

st.write(internet.head()
)


code18 = ('''
# Show information from internet dataset
internet.info()
''')
kode(code18)

buffer(internet)


st.markdown('''- The dataframe have 104825 rows and 4 columns.
- The session_date columns are currently printed as an object
- The dataframe has no missing value.

Kita akan mengubah type dari kolom session_date ke format Timestamp dan menambahkan beberapa kolom baru untuk untuk melakukan analisis lanjutan.

#### 2.4.1.  Fix Data
''')


code19 = ('''
# Convert data to timestamp
internet['session_date'] = pd.to_datetime(internet['session_date'], format='%Y-%m-%d')
internet['session_date'].dtype
''')
kode(code19)

internet['session_date'] = pd.to_datetime(internet['session_date'], format='%Y-%m-%d')
st.write(internet['session_date'].dtype
)


st.markdown('''We convert the data on the session_date column as object to timestamp format.

#### 2.4.2. Enrich Data
''')


code20 = ('''
# Create 'session_month' column from 'session_date'
internet['month'] = internet['session_date'].dt.month
internet['year'] = internet['session_date'].dt.year                    
internet['year_month'] = internet['year'].astype('str') + '_' + internet['month'].astype('str')
internet.head()
''')
kode(code20)

internet['month'] = internet['session_date'].dt.month
internet['year'] = internet['session_date'].dt.year                    
internet['year_month'] = internet['year'].astype('str') + '_' + internet['month'].astype('str')
st.write(internet.head()
)


st.markdown('''Because the analysis needs the month column, we will create a new column named month to capture the month of the internet access. Kita juga membuat kolom `year_month` untuk mengantisipasi jika ada tahun lain dalam data untuk meminimalisir kesalahan.
''')


code21 = ('''
# Menampilkan informasi dataset internet
internet.info()
''')
kode(code21)

buffer(internet)


st.markdown('''### 2.5. Plans

Tabel plans (data paket telepon):
- `plan_name` — nama paket telepon
- `usd_monthly_fee` — biaya bulanan dalam dolar AS
- `minutes_included` — alokasi menit panggilan bulanan
- `messages_included` — alokasi SMS bulanan
- `mb_per_month_included` — alokasi volume data bulanan (dalam megabita)
- `usd_per_minute` — harga per menit jika telah melebihi batas alokasi paket (misalnya, jika paket memiliki 
                       alokasi 100 menit, maka penggunaan mulai dari menit ke-101 akan dikenakan biaya)
- `usd_per_message` — harga per SMS jika telah melebihi batas alokasi paket
- `usd_per_gb` — harga per ekstra gigabita data jika telah melebihi batas alokasi paket (1 GB = 1024 megabita)
''')


code22 = ('''
# Show the sample of data
plans.head()
''')
kode(code22)

st.write(plans.head()
)


code23 = ('''
# Show information from plans dataset
plans.info()
''')
kode(code23)

buffer(plans)


st.markdown('''## 3. Study plan conditions

### 3.1. Calls
''')


code24 = ('''
# Show the amount of each user number of calls

calls_per_user_per_month = calls.pivot_table(index=['user_id', 'year_month'], 
                                  values='duration', 
                                  aggfunc=['sum', 'count'])

calls_per_user_per_month = calls_per_user_per_month.reset_index()
calls_per_user_per_month.columns = ['user_id', 'year_month', 'length_call', 'num_call',]
calls_per_user_per_month
''')
kode(code24)

calls_per_user_per_month = calls.pivot_table(index=['user_id', 'year_month'], 
                                  values='duration', 
                                  aggfunc=['sum', 'count'])

calls_per_user_per_month = calls_per_user_per_month.reset_index()
calls_per_user_per_month.columns = ['user_id', 'year_month', 'length_call', 'num_call',]
calls_per_user_per_month


st.markdown('''Kita telah membuat pivot table per panggilan dalam satu bulan yang dilakukan dari masing-masing pengguna.

### 3.2. Messages
''')


code25 = ('''
# Calculate the number of messages sent per month

messages_per_user_per_month = messages.pivot_table(index = ['user_id', 'year_month'], 
                                  values = 'message_date', 
                                  aggfunc= 'count')

messages_per_user_per_month = messages_per_user_per_month.reset_index()
messages_per_user_per_month.columns = ['user_id', 'year_month', 'num_message']
messages_per_user_per_month
''')
kode(code25)

messages_per_user_per_month = messages.pivot_table(index = ['user_id', 'year_month'], 
                                  values = 'message_date', 
                                  aggfunc= 'count')

messages_per_user_per_month = messages_per_user_per_month.reset_index()
messages_per_user_per_month.columns = ['user_id', 'year_month', 'num_message']
messages_per_user_per_month


st.markdown('''Kita telah membuat pivot tabel dari jumlah pesan yang dikirimkan oleh masing-masing pengguna dalam satu bulan.

### 3.3. Internet
''')


code26 = ('''
# Calculate the data volume used per month

internet_volume_per_user_per_month = internet.pivot_table(index = ['user_id', 'year_month'], 
                                  values = 'mb_used', 
                                  aggfunc= 'sum')

internet_volume_per_user_per_month = internet_volume_per_user_per_month.reset_index()

internet_volume_per_user_per_month.columns = ['user_id', 'year_month', 'gb_used']

internet_volume_per_user_per_month
''')
kode(code26)

internet_volume_per_user_per_month = internet.pivot_table(index = ['user_id', 'year_month'], 
                                  values = 'mb_used', 
                                  aggfunc= 'sum')

internet_volume_per_user_per_month = internet_volume_per_user_per_month.reset_index()

internet_volume_per_user_per_month.columns = ['user_id', 'year_month', 'gb_used']

internet_volume_per_user_per_month


st.markdown('''Kita menampilkan pivot table dari penggunaan data internet oleh masing-masing pengguna dalam satu bulan.
''')


code27 = ('''
internet_volume_per_user_per_month['gb_used'] = np.ceil(internet_volume_per_user_per_month['gb_used']/1024)

internet_volume_per_user_per_month
''')
kode(code27)

internet_volume_per_user_per_month['gb_used'] = np.ceil(internet_volume_per_user_per_month['gb_used']/1024)

internet_volume_per_user_per_month


st.markdown('''Kita telah mengubah satuan penggunaan dari `mb` menjadi `gb`, kita juga telah membulatkan jumlah pemakaian keatas.

### 3.4. Create new dataframe
''')


code28 = ('''
# Merge the data for calls, minutes, messages, internet based on user_id and month

df = calls_per_user_per_month.merge(messages_per_user_per_month, on=['user_id', 'year_month'], how='outer') 
                                
df = df.merge(internet_volume_per_user_per_month, on=['user_id', 'year_month'], how='outer') 

df
''')
kode(code28)

df = calls_per_user_per_month.merge(messages_per_user_per_month, on=['user_id', 'year_month'], how='outer') 
                                
df = df.merge(internet_volume_per_user_per_month, on=['user_id', 'year_month'], how='outer') 

st.write(df
)


st.markdown('''Kita membuat dataframe baru yang akan kita gunakan untuk melakukan analisis.
''')


code29 = ('''
# Add the plan information

df = df.merge(users, on='user_id', how='left')

df
''')
kode(code29)

df = df.merge(users, on='user_id', how='left')

st.write(df
)


st.markdown('''Kita menggabungkan tabel df dan tabel plans untuk mengetahui paket apa yang digunakan oleh tiap pengguna.
''')


code30 = ('''
# Remove first_name and last_name collumns
df = df.drop(['first_name', 'last_name', 'age'], axis=1)
df.head()
''')
kode(code30)

df = df.drop(['first_name', 'last_name', 'age'], axis=1)
st.write(df.head()
)


st.markdown('''Kita menghapus kolom yang tidak diperlukan untuk analisis
''')


code31 = ('''
# print the information of data
df.info()
''')
kode(code31)

buffer(df)


st.markdown('''There is some column that has missing values, then lets us fill the missing value with 0 except the churn_date column.
''')


code32 = ('''
# Fillna missing values with 0
for i in ('num_call', 'length_call', 'num_message', 'gb_used'):
    df[i] = df[i].fillna(0)
''')
kode(code32)

for i in ('num_call', 'length_call', 'num_message', 'gb_used'):
    df[i] = df[i].fillna(0)


code33 = ('''
# print the dataframe infromation
df.info()
''')
kode(code33)

buffer(df)


st.markdown('''We have already fill the missing values in the datasets.

### 3.5. Calculate monthly revenue
''')

code34 = ('''
# Menampilkan dataframe plans
plans
''')
kode(code34)

plans


code35 = ('''
# Create function to calculate monthly revenue

def revenue(row):
    messages = row['num_message']  # We define rows and columns from data on our measurements
    duration = row['length_call']
    gb = row['gb_used']
    plan = row['plan']
    
    package_overlimit = 0  # Define some variables to count income 
    message_overlimit = 0
    internet_overlimit = 0
    calls_overlimit = 0
   
    # In the next block of the function we need to describe a logic expression to calculate the traffic 
    # If it exceeds the usage limit, then the user who is charged the tariff exceeds the limit
    if plan == 'surf':
        package_cost = 20
        if duration > 500:
            calls_overlimit = (duration - 500) * 0.3
        if messages > 50:
            message_overlimit = (messages - 50) * 0.3
        if gb > 15:
            internet_overlimit = (gb - 15) * 10
              
    
    elif plan == 'ultimate':
        package_cost = 70
        if duration > 3000:
            calls_overlimit = (duration - 3000) * 0.1
        if messages > 1000:
            message_overlimit = (messages - 1000) * 0.1    
        if gb > 30:
            internet_overlimit = (gb - 30) * 7        

    # Finally, add up all those values to get a total revenue         
    total_bill = package_cost + calls_overlimit + message_overlimit + internet_overlimit
    
    return total_bill 
''')
kode(code35)

def revenue(row):
    messages = row['num_message']  # We define rows and columns from data on our measurements
    duration = row['length_call']
    gb = row['gb_used']
    plan = row['plan']
    
    package_overlimit = 0  # Define some variables to count income 
    message_overlimit = 0
    internet_overlimit = 0
    calls_overlimit = 0
   
    # In the next block of the function we need to describe a logic expression to calculate the traffic 
    # If it exceeds the usage limit, then the user who is charged the tariff exceeds the limit
    if plan == 'surf':
        package_cost = 20
        if duration > 500:
            calls_overlimit = (duration - 500) * 0.3
        if messages > 50:
            message_overlimit = (messages - 50) * 0.3
        if gb > 15:
            internet_overlimit = (gb - 15) * 10
              
    
    elif plan == 'ultimate':
        package_cost = 70
        if duration > 3000:
            calls_overlimit = (duration - 3000) * 0.1
        if messages > 1000:
            message_overlimit = (messages - 1000) * 0.1    
        if gb > 30:
            internet_overlimit = (gb - 30) * 7        

    # Finally, add up all those values to get a total revenue         
    total_bill = package_cost + calls_overlimit + message_overlimit + internet_overlimit
    
    return total_bill 


st.markdown('''We will calculate the total volume of data used by each customer every month.
''')

code36 = ('''
# Apply function to our data and create a new column to return the result
df['revenue'] = df.apply(revenue, axis=1)
df.head()
''')
kode(code36)

df['revenue'] = df.apply(revenue, axis=1)
st.write(df.head()
)


st.markdown('''We added a new column called revenue to calculate monthly revenue from every user.

## 4. Study Costumer Behavior
''')


code37 = ('''
# print the amount of each plan
plan_pivot = df.pivot_table(index='plan', values='user_id', aggfunc=['count']).reset_index()
plan_pivot
''')
kode(code37)

plan_pivot = df.pivot_table(index='plan', values='user_id', aggfunc=['count']).reset_index()
plan_pivot


code38 = ('''
# Filter data by plans
df_surf = df.query('plan == "surf"')
df_ultimate = df.query('plan == "ultimate"')
''')
kode(code38)

df_surf = df.query('plan == "surf"')
df_ultimate = df.query('plan == "ultimate"')


st.markdown('''We will create two new dataframes to split ultimate plan users and surf plan users.

### 4.1. Calls Duration
''')


code39 = ('''
# print the amount of calls duration each plan per month
call_duration_pivot = df.pivot_table(index=['plan', 'year_month'], values='length_call', aggfunc='mean').reset_index()
call_duration_pivot
''')
kode(code39)

call_duration_pivot = df.pivot_table(index=['plan', 'year_month'], values='length_call', aggfunc='mean').reset_index()
call_duration_pivot


st.markdown('''Show the total revenue from each month from two plans.
''')


code40 = ('''
# Create barplot function

def barplot (a, b, c, d):
    colors = ['#69b3a2', '#4374B3']
    sns.set_palette(sns.color_palette(colors))
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='year_month', y=b, hue='plan', data=a)
    plt.xlabel('Month')
    plt.ylabel(c)
    plt.title(d)
    st.pyplot(fig)
''')
kode(code40)

def barplot (a, b, c, d):
    colors = ['#69b3a2', '#4374B3']
    sns.set_palette(sns.color_palette(colors))
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='year_month', y=b, hue='plan', data=a)
    plt.xlabel('Month')
    plt.ylabel(c)
    plt.title(d)
    st.pyplot(fig)



code41 = ('''
# Show barplot from call duration
barplot(call_duration_pivot, 'length_call', 'Avg Minutes of Call', 'Average Monthly Call Durations each Plan per Month')
''')
kode(code41)

barplot(call_duration_pivot, 'length_call', 'Avg Minutes of Call', 'Average Monthly Call Durations each Plan per Month')


st.markdown('''We can see based on the histogram; we get overviews:
- The histogram has increased in the first few months, and the histogram gets stable after half a year.
- The average users spent 400 minutes on calls every month.
- One of the plans is not always be higher than the other plan.
''')


code42 = ('''
# Create histogram function
def histplot (a, b, c):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(a[b], bins=70, kde=True)
    plt.xlabel(b)
    plt.ylabel('users')
    plt.title(c)
    st.pyplot(fig)
''')
kode(code42)

def histplot (a, b, c):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(a[b], bins=70, kde=True)
    plt.xlabel(b)
    plt.ylabel('users')
    plt.title(c)
    st.pyplot(fig)


code43 = ('''
# print histogram monthly call duration for surf user
histplot(df_surf, 'length_call', 'Histogram Monthly Minutes of Call for Surf User')
''')
kode(code43)

histplot(df_surf, 'length_call', 'Histogram Monthly Minutes of Call for Surf User')


code44 = ('''
# print histogram monthly call duration for ultimate user
histplot(df_ultimate, 'length_call', 'Histogram Monthly Minutes of Call for Ultimate User')
''')
kode(code44)

histplot(df_ultimate, 'length_call', 'Histogram Monthly Minutes of Call for Ultimate User')


st.markdown('''- The minutes of calls by surf plan users and ultimate plan users are right skewed.
- Both two plans has the same shape.
- The majority of users of both two plans spent 200 to 600 minutes of call in every month.
''')


code45 = ('''
# Calculate the mean and the variance of the surf plan monthly call duration

avg_monthly_call_duration_surf = df_surf['length_call'].mean()
st.write('Call duration Surf Mean', round(avg_monthly_call_duration_surf,2))

var_monthly_call_duration_surf = np.var(df_surf['length_call'])
st.write('Call duration Surf Variance', round(var_monthly_call_duration_surf,2))

std_monthly_call_duration_surf = np.sqrt(var_monthly_call_duration_surf)
st.write('Call duration Surf Standard Deviation', round(std_monthly_call_duration_surf,2))
''')
kode(code45)

avg_monthly_call_duration_surf = df_surf['length_call'].mean()
st.write('Call duration Surf Mean', round(avg_monthly_call_duration_surf,2))

var_monthly_call_duration_surf = np.var(df_surf['length_call'])
st.write('Call duration Surf Variance', round(var_monthly_call_duration_surf,2))

std_monthly_call_duration_surf = np.sqrt(var_monthly_call_duration_surf)
st.write('Call duration Surf Standard Deviation', round(std_monthly_call_duration_surf,2))


code46 = ('''
# Calculate the mean and the variance of the ultimate plan monthly call duration

avg_monthly_call_duration_ultimate = df_ultimate['length_call'].mean()
st.write('Call duration Ultimate Mean', round(avg_monthly_call_duration_ultimate,2))

var_monthly_call_duration_ultimate = np.var(df_ultimate['length_call'])
st.write('Call duration Ultimate Variance', round(var_monthly_call_duration_ultimate,2))

std_monthly_call_duration_ultimate = np.sqrt(var_monthly_call_duration_ultimate)
st.write('Call duration Ultimate Standard Deviation', round(std_monthly_call_duration_ultimate,2))
''')
kode(code46)

avg_monthly_call_duration_ultimate = df_ultimate['length_call'].mean()
st.write('Call duration Ultimate Mean', round(avg_monthly_call_duration_ultimate,2))

var_monthly_call_duration_ultimate = np.var(df_ultimate['length_call'])
st.write('Call duration Ultimate Variance', round(var_monthly_call_duration_ultimate,2))

std_monthly_call_duration_ultimate = np.sqrt(var_monthly_call_duration_ultimate)
st.write('Call duration Ultimate Standard Deviation', round(std_monthly_call_duration_ultimate,2))

st.markdown('''The mean, variance, and standard deviation of call duration of both two plans have similiar values.
''')


code47 = ('''
# Create boxplot function
def boxplot(a, b, c):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x=a[b])
    plt.ylabel('Users')
    plt.title(c)
    st.pyplot(fig)
''')
kode(code47)

def boxplot(a, b, c):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x=a[b])
    plt.ylabel('Users')
    plt.title(c)
    st.pyplot(fig)


code48 = ('''
# print boxplot monthly call duration for surf user
boxplot(df_surf, 'length_call', 'Boxplot Monthly Minutes of Call for Surf User')
''')
kode(code48)

boxplot(df_surf, 'length_call', 'Boxplot Monthly Minutes of Call for Surf User')


code49 = ('''
# print boxplot monthly call duration for ultimate user
boxplot(df_ultimate, 'length_call', 'Boxplot Monthly Minutes of Call for Ultimate User')
''')
kode(code49)

boxplot(df_ultimate, 'length_call', 'Boxplot Monthly Minutes of Call for Ultimate User')

st.markdown('''There were outliers from both of the two plans when users spent over 1000 minutes of calls.

### 4.2. Messages Sent
''')

code50 = ('''
# print the amount of messages sent each plan per month
message_sent_pivot = df.pivot_table(index=['plan', 'year_month'], values='num_message', aggfunc='median').reset_index()
message_sent_pivot
''')
kode(code50)

message_sent_pivot = df.pivot_table(index=['plan', 'year_month'], values='num_message', aggfunc='median').reset_index()
message_sent_pivot


code51 = ('''
# Show barplot from messages sent
barplot(message_sent_pivot, 'num_message', 'Avg Messages Sent', 'Average Monthly Messages Sent each Plan per Month')
''')
kode(code51)


st.markdown('''We can see based on the histogram; we get overviews:
- The histogram has increased in every single month.
- The users of the ultimate plan send more messages than users of the surf plan.
''')

code52 = ('''
# print histogram monthly messages sent for surf user
histplot(df_surf, 'num_message', 'Histogram Monthly Sent Messages for Surf User')
''')
kode(code52)

histplot(df_surf, 'num_message', 'Histogram Monthly Sent Messages for Surf User')


code53 = ('''
# print histogram monthly messages sent for ultimate user
histplot(df_ultimate, 'num_message', 'Histogram Monthly Sent Messages for Ultimate User')
''')
kode(code53)

histplot(df_ultimate, 'num_message', 'Histogram Monthly Sent Messages for Ultimate User')


st.markdown('''- The average text messages histogram for both plans is highly skewed to the right
- Most of the users do not require text messages at all.
- The majority of users from both of the two plans send less than 100 messages in a month.
''')


code54 = ('''
# Calculate the mean and the variance of the surf plan monthly messages sent

avg_monthly_sent_message_surf = df_surf['num_message'].mean()
st.write('Message sent Surf Mean', round(avg_monthly_sent_message_surf,2))

var_monthly_sent_message_surf = np.var(df_surf['num_message'])
st.write('Message sent Surf Variance', round(var_monthly_sent_message_surf,2))

std_monthly_sent_message_surf = np.sqrt(var_monthly_sent_message_surf)
st.write('Message sent Surf Standard Deviation', round(std_monthly_sent_message_surf,2))
''')
kode(code54)

avg_monthly_sent_message_surf = df_surf['num_message'].mean()
st.write('Message sent Surf Mean', round(avg_monthly_sent_message_surf,2))

var_monthly_sent_message_surf = np.var(df_surf['num_message'])
st.write('Message sent Surf Variance', round(var_monthly_sent_message_surf,2))

std_monthly_sent_message_surf = np.sqrt(var_monthly_sent_message_surf)
st.write('Message sent Surf Standard Deviation', round(std_monthly_sent_message_surf,2))


code55 = ('''
# Calculate the mean and the variance of the the ultimate plan monthly messages sent

avg_monthly_sent_message_ultimate = df_ultimate['num_message'].mean()
st.write('Message sent Ultimate Mean', round(avg_monthly_sent_message_ultimate,2))

var_monthly_sent_message_ultimate = np.var(df_ultimate['num_message'])
st.write('Message sent Ultimate Variance', round(var_monthly_sent_message_ultimate,2))

std_monthly_sent_message_ultimate = np.sqrt(var_monthly_sent_message_ultimate)
st.write('Message sent Ultimate Standard Deviation', round(std_monthly_sent_message_ultimate,2))
''')
kode(code55)

avg_monthly_sent_message_ultimate = df_ultimate['num_message'].mean()
st.write('Message sent Ultimate Mean', round(avg_monthly_sent_message_ultimate,2))

var_monthly_sent_message_ultimate = np.var(df_ultimate['num_message'])
st.write('Message sent Ultimate Variance', round(var_monthly_sent_message_ultimate,2))

std_monthly_sent_message_ultimate = np.sqrt(var_monthly_sent_message_ultimate)
st.write('Message sent Ultimate Standard Deviation', round(std_monthly_sent_message_ultimate,2))


st.markdown('''The mean, variance, and standard deviation of the number of sent messages in both plans have similar values.
''')

code56 = ('''
# print boxplot monthly messages sent for ultimate user
boxplot(df_surf, 'num_message', 'Boxplot Monthly Sent Messages for Surf User')
''')
kode(code56)

boxplot(df_surf, 'num_message', 'Boxplot Monthly Sent Messages for Surf User')


code57 = ('''
# print boxplot monthly messages sent for ultimate user
boxplot(df_ultimate, 'num_message', 'Boxplot Monthly Sent Messages for Ultimate User')
''')
kode(code57)

boxplot(df_ultimate, 'num_message', 'Boxplot Monthly Sent Messages for Ultimate User')


st.markdown('''- The ultimate users sent more messages in a month than surf plan users.
- Most users of the surf plan sent less than 50 messages a month, and the majority of the ultimate users sent less than 65 messages in a month.
- There are some outliers from both of the two plans when the users sends more than 120 messages in a month.

### 4.3. Data Used
''')

code58 = ('''
# print the amount of data used each plan per month
data_used_pivot = df.pivot_table(index=['plan', 'year_month'], values='gb_used', aggfunc='median').reset_index()
data_used_pivot
''')
kode(code58)

data_used_pivot = df.pivot_table(index=['plan', 'year_month'], values='gb_used', aggfunc='median').reset_index()
data_used_pivot


code59 = ('''
# Show barplot from data used
barplot(data_used_pivot, 'gb_used', 'Avg Data Used', 'Average Monthly Data Used each Plan per Month')
''')
kode(code59)

barplot(data_used_pivot, 'gb_used', 'Avg Data Used', 'Average Monthly Data Used each Plan per Month')


st.markdown('''We can see based on the histogram; we get overviews:
- The histogram has increased in the first few months, and the histogram gets stable after half a year.
- The average users used over than 15 GB of data every month.
- One of the plans is not always be higher than the other plan.
''')


code60 = ('''
# print histogram monthly Data Used for surf user
histplot(df_surf, 'gb_used', 'Histogram Monthly Data Used for Surf User')
''')
kode(code60)

histplot(df_surf, 'gb_used', 'Histogram Monthly Data Used for Surf User')


code61 = ('''
# print histogram monthly Data Used for ultimate user
histplot(df_ultimate, 'gb_used', 'Histogram Monthly Data Used for Ultimate User')
''')
kode(code61)

histplot(df_ultimate, 'gb_used', 'Histogram Monthly Data Used for Ultimate User')


st.markdown('''- The average internet volume histogram for both plans has a normal distribution.
- Most users need around 10 to 20 gb of data in a month.
''')


code62 = ('''
# Calculate the mean and the variance of the surf plan monthly data used

avg_monthly_data_used_surf = df_surf['gb_used'].mean()
st.write('Data used Surf Mean', round(avg_monthly_data_used_surf,2))

var_monthly_data_used_surf = np.var(df_surf['gb_used'])
st.write('Data used Surf Variance', round(var_monthly_data_used_surf,2))

std_monthly_data_used_surf = np.sqrt(var_monthly_data_used_surf)
st.write('Data used Surf Standard Deviation', round(std_monthly_data_used_surf,2))
''')
kode(code62)

avg_monthly_data_used_surf = df_surf['gb_used'].mean()
st.write('Data used Surf Mean', round(avg_monthly_data_used_surf,2))

var_monthly_data_used_surf = np.var(df_surf['gb_used'])
st.write('Data used Surf Variance', round(var_monthly_data_used_surf,2))

std_monthly_data_used_surf = np.sqrt(var_monthly_data_used_surf)
st.write('Data used Surf Standard Deviation', round(std_monthly_data_used_surf,2))


code63 = ('''
# Calculate the mean and the variance of the ultimate plan monthly data used

avg_monthly_data_used_ultimate = df_ultimate['gb_used'].mean()
st.write('Data used Ultimate Mean', round(avg_monthly_data_used_ultimate,2))

var_monthly_data_used_ultimate = np.var(df_ultimate['gb_used'])
st.write('Data used Ultimate Variance', round(var_monthly_data_used_ultimate,2))

std_monthly_data_used_ultimate = np.sqrt(var_monthly_data_used_ultimate)
st.write('Data used Ultimate Standard Deviation', round(std_monthly_data_used_ultimate,2))
''')
kode(code63)

avg_monthly_data_used_ultimate = df_ultimate['gb_used'].mean()
st.write('Data used Ultimate Mean', round(avg_monthly_data_used_ultimate,2))

var_monthly_data_used_ultimate = np.var(df_ultimate['gb_used'])
st.write('Data used Ultimate Variance', round(var_monthly_data_used_ultimate,2))

std_monthly_data_used_ultimate = np.sqrt(var_monthly_data_used_ultimate)
st.write('Data used Ultimate Standard Deviation', round(std_monthly_data_used_ultimate,2))


st.markdown('''The mean, variance, and standard deviation of data used in both of the two plans have similar values.
''')


code64 = ('''
# print boxplot monthly data used for surf user
boxplot(df_surf, 'gb_used', 'Boxplot Monthly Data Used for Ultimate User')
''')
kode(code64)

boxplot(df_surf, 'gb_used', 'Boxplot Monthly Data Used for Ultimate User')


code65 = ('''
# print boxplot monthly data used for ultimate user
boxplot(df_ultimate, 'gb_used', 'Boxplot Monthly Data Used for Ultimate User')
''')
kode(code65)

boxplot(df_ultimate, 'gb_used', 'Boxplot Monthly Data Used for Ultimate User')


st.markdown('''There were outliers when users use more than 30 GB of data in a month.

### 4.4. Revenue
''')

code66 = ('''
# print the amount of revenue each plan per month
revenue_pivot = df.pivot_table(index=['plan', 'year_month'], values='revenue', aggfunc='mean').reset_index()
revenue_pivot
''')
kode(code66)

revenue_pivot = df.pivot_table(index=['plan', 'year_month'], values='revenue', aggfunc='mean').reset_index()
revenue_pivot


code67 = ('''
# Show barplot from revenue
barplot(revenue_pivot, 'revenue', 'Avg Revenue', 'Average Monthly Revenue each Plan per Month')
''')
kode(code67)

barplot(revenue_pivot, 'revenue', 'Avg Revenue', 'Average Monthly Revenue each Plan per Month')


st.markdown(''''We can see based on the histogram; we get overviews:
- The histogram has increased in the first few months, and the histogram gets stable after half a year.
- The average users from the surf plan are paying extra for their bill.
''')


code68 = ('''
# print histogram monthly revenue for surf user
histplot(df_surf, 'revenue', 'Histogram Monthly Revenue for Ultimate User')
''')
kode(code68)

histplot(df_surf, 'revenue', 'Histogram Monthly Revenue for Ultimate User')


code69 = ('''
# print histogram monthly revenue for ultimate user
histplot(df_ultimate, 'revenue', 'Histogram Monthly Revenue for Ultimate User')
''')
kode(code69)

histplot(df_ultimate, 'revenue', 'Histogram Monthly Revenue for Ultimate User')


st.markdown('''- Average revenue histogram for both plans are skewed to the right.
''')


code70 = ('''
# Calculate the mean and the variance of the surf plan monthly revenue

avg_monthly_revenue_surf = df_surf['revenue'].mean()
st.write('Monthly Revenue Surf Mean', round(avg_monthly_revenue_surf,2))

var_monthly_revenue_surf = np.var(df_surf['revenue'])
st.write('Monthly Revenue Surf Variance', round(var_monthly_revenue_surf,2))

std_monthly_revenue_surf = np.sqrt(var_monthly_revenue_surf)
st.write('Monthly Revenue Surf Standard Deviation', round(std_monthly_revenue_surf,2))
''')
kode(code70)

avg_monthly_revenue_surf = df_surf['revenue'].mean()
st.write('Monthly Revenue Surf Mean', round(avg_monthly_revenue_surf,2))

var_monthly_revenue_surf = np.var(df_surf['revenue'])
st.write('Monthly Revenue Surf Variance', round(var_monthly_revenue_surf,2))

std_monthly_revenue_surf = np.sqrt(var_monthly_revenue_surf)
st.write('Monthly Revenue Surf Standard Deviation', round(std_monthly_revenue_surf,2))


code71 = ('''
# Calculate the mean and the variance of the ultimate plan monthly revenue

avg_monthly_revenue_ultimate = df_ultimate['revenue'].mean()
st.write('Monthly Revenue Ultimate Mean', round(avg_monthly_revenue_ultimate,2))

var_monthly_revenue_ultimate = np.var(df_ultimate['revenue'])
st.write('Monthly Revenue Ultimate Variance', round(var_monthly_revenue_ultimate,2))

std_monthly_revenue_ultimate = np.sqrt(var_monthly_revenue_ultimate)
st.write('Monthly Revenue Ultimate Standard Deviation', round(std_monthly_revenue_ultimate,2))
''')
kode(code71)

avg_monthly_revenue_ultimate = df_ultimate['revenue'].mean()
st.write('Monthly Revenue Ultimate Mean', round(avg_monthly_revenue_ultimate,2))

var_monthly_revenue_ultimate = np.var(df_ultimate['revenue'])
st.write('Monthly Revenue Ultimate Variance', round(var_monthly_revenue_ultimate,2))

std_monthly_revenue_ultimate = np.sqrt(var_monthly_revenue_ultimate)
st.write('Monthly Revenue Ultimate Standard Deviation', round(std_monthly_revenue_ultimate,2))


st.markdown('''- The mean from both two plans has a similar value.
- The variance of the surf plan is very higher than the ultimate plan.
- The standard deviation from the surf plan is higher than the ultimate plan.
''')


code72 = ('''
# print boxplot monthly revenue for surf user
boxplot(df_surf, 'revenue', 'Boxplot Monthly Revenue for Ultimate User')
''')
kode(code72)

boxplot(df_surf, 'revenue', 'Boxplot Monthly Revenue for Ultimate User')


code73 = ('''
# print boxplot monthly revenue for ultimate user
boxplot(df_ultimate, 'revenue', 'Boxplot Monthly Revenue for Ultimate User')
''')
kode(code73)

boxplot(df_ultimate, 'revenue', 'Boxplot Monthly Revenue for Ultimate User')


st.markdown('''- There are outliers from the surf plan when the users spend more than \\$200 in a month.
- The ultimate users had a lot of outliers when their spent more than \\$70 in a month.

## 5. Test statistical hyposthesis

- The average income of users of Ultimate and Surf phone plans is different.
- The average income of users in the NY-NJ area differs from that of users from other regions.

### 5.1. **First Hypothesis :**
- H₀ : Rata-rata pendapatan dari pengguna paket telepon Ultimate dan Surf tidak berbeda.
- H₁ : Rata-rata pendapatan dari pengguna paket telepon Ultimate dan Surf berbeda.


Alpha value - 5%

Untuk menentukan `equal_var` True atau False kita akan menggunakan **Levene Test**, dimana jika nilai `p=value` lebih dari `0.05` maka bisa kita asumsikan bahwa kedua sampel memiliki `equal variance`, kita juga menetapkan argumen `median` pada fungsi center karena seperti yang kita ketahui pada diagram kedua paket miring ke kanan.
''')


code74 = ('''
# Determine if the two samples have equal variance
plans_var_levene = st.levene(df_surf['revenue'], df_ultimate['revenue'], center='median')
plans_var_levene
''')
kode(code74)

plans_var_levene = stats.levene(df_surf['revenue'], df_ultimate['revenue'], center='median')
plans_var_levene


st.markdown('''Nilai `p-value` dari paket Surf dan paket Ultimate menghasilkan angka `5.0` yang artinya `equal`.
''')


code75 = ('''
# Test the hypothesis

alpha = 0.05
results = st.ttest_ind(df_surf['revenue'], df_ultimate['revenue'], equal_var=True)
st.write('p-value:', results.pvalue)

if results.pvalue < alpha:
    st.write('We reject the null hypothesis')
else:
    st.write("We can't reject the null hypothesis")
''')
kode(code75)

alpha = 0.05
results = stats.ttest_ind(df_surf['revenue'], df_ultimate['revenue'], equal_var=True)
st.write('p-value:', results.pvalue)

if results.pvalue < alpha:
    st.write('We reject the null hypothesis')
else:
    st.write("We can't reject the null hypothesis")


st.markdown('''Hasil dari uji-t yang kita lakukan adalah menolak Hipotesis 0 yang artinya bahwa rata-rata pendapatan dari kedua paket berbeda atau tidak sama. Jadi kita bisa mengajukan kepada departemen marketing untuk bisa menerapkan strategi iklan yang berbeda pada kedua paket tersebut.
''')


code76 = ('''
# Filter City by New York and New Jersey
df_ny_nj = df[df['city'].str.contains('NY-NJ')].reset_index(drop=True)
df_ny_nj.head()
''')
kode(code76)

df_ny_nj = df[df['city'].str.contains('NY-NJ')].reset_index(drop=True)
st.write(df_ny_nj.head()
)


code77 = ('''
# Filter city other of New York and New Jersey
df_city_other = df[~df['city'].str.contains('NY-NJ')].reset_index(drop=True)
df_city_other.head()
''')
kode(code77)

df_city_other = df[~df['city'].str.contains('NY-NJ')].reset_index(drop=True)
st.write(df_city_other.head()
)


code78 = ('''
# Memeriksa kolom city
df_ny_nj['city'].value_counts()
''')
kode(code78)

st.write(df_ny_nj['city'].value_counts()
)


code79 = ('''
# print histogram monthly revenue for NY-NJ
histplot(df_ny_nj, 'revenue', 'Histogram Monthly Revenue of NY-NJ')
''')
kode(code79)

histplot(df_ny_nj, 'revenue', 'Histogram Monthly Revenue of NY-NJ')


code80 = ('''
# print histogram monthly revenue for Other City
histplot(df_city_other, 'revenue', 'Histogram Monthly Revenue of Other City')
''')
kode(code80)

histplot(df_city_other, 'revenue', 'Histogram Monthly Revenue of Other City')


code81 = ('''
# Calculate the mean and the variance of the NY-NJ users

avg_ny_nj = df_ny_nj['revenue'].mean()
st.write('NY-NJ mean:', round(avg_ny_nj,2))

var_ny_nj = np.var(df_ny_nj['revenue'])
st.write('NY-NJ var:', round(var_ny_nj,2))

std_ny_nj = np.sqrt(var_ny_nj)
st.write('NY-NJ std', round(std_ny_nj,2))
''')
kode(code81)

avg_ny_nj = df_ny_nj['revenue'].mean()
st.write('NY-NJ mean:', round(avg_ny_nj,2))

var_ny_nj = np.var(df_ny_nj['revenue'])
st.write('NY-NJ var:', round(var_ny_nj,2))

std_ny_nj = np.sqrt(var_ny_nj)
st.write('NY-NJ std', round(std_ny_nj,2))


code82 = ('''
# Calculate the mean and the variance of the NY-NJ users

avg_other_city = df_city_other['revenue'].mean()
st.write('Other city mean:', round(avg_other_city,2))

var_other_city = np.var(df_city_other['revenue'])
st.write('Other city var:', round(var_other_city,2))

std_other_city = np.sqrt(var_ny_nj)
st.write('Other city std', round(std_other_city,2))
''')
kode(code82)

avg_other_city = df_city_other['revenue'].mean()
st.write('Other city mean:', round(avg_other_city,2))

var_other_city = np.var(df_city_other['revenue'])
st.write('Other city var:', round(var_other_city,2))

std_other_city = np.sqrt(var_ny_nj)
st.write('Other city std', round(std_other_city,2))


st.markdown('''### 5.1. **Second Hypothesis :**

- H₀ : Rata-rata pendapatan dari pengguna di wilayah NY-NJ tidak berbeda dengan pendapatan pengguna dari wilayah lain.
- H₁ : Rata-rata pendapatan dari pengguna di wilayah NY-NJ berbeda dengan pendapatan pengguna dari wilayah lain.

Alpha value - 5%

Untuk menentukan `equal_var` True atau False kita akan menggunakan **Levene Test**, dimana jika nilai `p=value` lebih dari `0.05` maka bisa kita asumsikan bahwa kedua sampel memiliki `equal variance`, kita juga menetapkan argumen `median` pada fungsi center karena seperti yang kita ketahui pada diagram kedua dataframe miring ke kanan.
''')


code83 = ('''
# Determine if the two samples have equal variance
city_var_levene = st.levene(df_ny_nj['revenue'], df_city_other['revenue'], center='median')
city_var_levene
''')
kode(code83)

city_var_levene = stats.levene(df_ny_nj['revenue'], df_city_other['revenue'], center='median')
city_var_levene


st.markdown('''Nilai rasio varians dari NY-NJ dan kota lain menghasilkan angka `1.2` maka kita akan mengatur variable `equal_var` menjadi True.
''')


code84 = ('''
# Test the hypothesis

alpha = 0.05
results = st.ttest_ind(df_ny_nj['revenue'], df_city_other['revenue'], equal_var=True)
st.write('p-value:', results.pvalue)

if results.pvalue < alpha:
    st.write('We reject the null hypothesis')
else:
    st.write("We can't reject the null hypothesis")
''')
kode(code84)

alpha = 0.05
results = stats.ttest_ind(df_ny_nj['revenue'], df_city_other['revenue'], equal_var=True)
st.write('p-value:', results.pvalue)

if results.pvalue < alpha:
    st.write('We reject the null hypothesis')
else:
    st.write("We can't reject the null hypothesis")


st.markdown('''Dari uji-t yang kita lakukan menghasilkan bahwa kita tidak dapat menolak Hipotesis 0, yang artinya rata-rata pendapatan dari kota NY-NJ dan kota lainnya adalah sama, jadi kita dapat merekomendasikan untuk menerapkan strategi iklan yang sama pada setiap kota.

# Conclusion

We have done some steps in processing the mobile plan data to get conclusions:

**1. Preprocessing Data**
- We have five datasets, and each data contains some information about the users using our service. That 5 dataset includes the information of 500 users using the service, minutes of call they spent, number of messages they sent, total data they used, and the information about the price of each plan.
- There are some missing values in the data but to fill the missing values isn't always needed.
- We have to convert some columns, especially the column with the object type, to the timestamp format to make it easier for us to do the following analysis.

**2. Transformation Data**
- We created new columns like the month based on the date columns in the calls, messages, and internet dataset to calculate the amount of traffic from each user every month.
- We created the function to calculate the revenue from each user every month, applied the process, and created a new column called revenue.

**3. Visualisation and Analysis**
- The calling behaviour from users of each plan is very similar.
- The ultimate users sent more messages than surf plan users, but there is no significant difference between the two plans. Kebanyakan pengguna tidak menggunakan kuota pesan mereka sama sekali, hal ini mungkin terjadi karena mereka lebih memilih menggunakan aplikasi pesan instan dengan internet, kita bisa menyarankan untuk mengubah kuota sms menjadi kuota aplikasi instant messaging.
- The usage of the internet for ultimate plan users is slightly higher than users of the surf plan, but the variance of the users from ultimate plan is less than the surf plan users.
- The users from surf plans brought in more revenue than ultimate plans, it's because the number of ultimate programs users is more than the surf plans users.

**4. Test the Hypothesis**
- Hasil dari analisis yang telah kita lakukan menetapkan bahwa rata-rata pendapatan dari surf plan dan ultimate plan bebeda, hal ini tentu terlalu mengejutkan karena kita melihat ada sebagian pengguna pada paket surf membayar ekstra untuk tagihan mereka untuk kelebihan penggunaan, dengan hasil ini tentu akan menjadi pertimbangan bagi departemen marketing dalam menyesuaikan budget iklan pada setiap paket.
- Hasil analisis lain juga menetapkan bahwa pendapatan rata-rata dari pengguna di NY-NJ dan kota lainnya tidak berbeda, dengan hasil ini tentu akan menjadi pertimbangan bagi departemen marketing dalam menyesuaikan budget iklan pada satu kota dan kota lainnya, yang artinya kita bisa merekomendasikan departemen periklanan untuk tidak perlu membeda-bedakan anggaran iklan pada setiap kota.

Kita bisa merekomendasikan bagi pengguna yang sering mengalami kelebihan penggunaan untuk meng-upgrade paket mereka, atau jika jumlah mereka cukup banyak kita bisa mempertimbangkan untuk menambah paket baru sebagai opsi diantara paket surf dan paket ultimate.
''')
