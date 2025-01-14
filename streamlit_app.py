import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Judul Aplikasi
st.title('Prediksi Harga Properti')
st.write("Aplikasi untuk memprediksi harga properti berdasarkan fitur-fitur yang diberikan.")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('house_prices.csv')
    return data

data = load_data()
st.write("## Dataset")
st.dataframe(data.head())

# Preprocessing Dataset
selected_features = ['Price (in rupees)', 'Carpet Area', 'location', 'Status', 'Transaction', 'Furnishing', 'facing']
data = data[selected_features]

# Konversi kolom numerik
data['Carpet Area'] = data['Carpet Area'].str.extract(r'(\d+)').astype(float)

# Isi nilai kosong
data['Price (in rupees)'] = data['Price (in rupees)'].fillna(data['Price (in rupees)'].median())

categorical_columns = ['location', 'Status', 'Transaction', 'Furnishing', 'facing']
for col in categorical_columns:
    data[col] = data[col].fillna('Unknown')
    data[col] = data[col].astype('category').cat.codes

# Pisahkan fitur dan target
X = data.drop(columns=['Price (in rupees)'])
y = data['Price (in rupees)']

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

st.write("## Input Data untuk Prediksi")

# Input Fitur
carpet_area = st.number_input("Carpet Area (sqft)", min_value=0.0, step=1.0)
location = st.selectbox("Location", options=data['location'].unique())
status = st.selectbox("Status", options=data['Status'].unique())
transaction = st.selectbox("Transaction", options=data['Transaction'].unique())
furnishing = st.selectbox("Furnishing", options=data['Furnishing'].unique())
facing = st.selectbox("Facing", options=data['facing'].unique())

# Encode Input Data
input_data = {
    'Carpet Area': carpet_area,
    'location': data['location'].astype('category').cat.codes[data['location'] == location].iloc[0],
    'Status': data['Status'].astype('category').cat.codes[data['Status'] == status].iloc[0],
    'Transaction': data['Transaction'].astype('category').cat.codes[data['Transaction'] == transaction].iloc[0],
    'Furnishing': data['Furnishing'].astype('category').cat.codes[data['Furnishing'] == furnishing].iloc[0],
    'facing': data['facing'].astype('category').cat.codes[data['facing'] == facing].iloc[0],
}

input_df = pd.DataFrame([input_data])

if st.button("Prediksi Harga"):
    prediction = model.predict(input_df)[0]
    st.write(f"### Prediksi Harga Properti: Rp {prediction:,.2f}")

st.write("---")
st.write("Aplikasi ini menggunakan model Random Forest untuk memprediksi harga properti berdasarkan data yang telah dilatih.")
