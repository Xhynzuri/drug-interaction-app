import streamlit as st
import base64

# Fungsi untuk mengonversi gambar ke base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path gambar
logo_path = "logo.png"
logo_base64 = get_base64_of_image(logo_path)

# CSS untuk posisi tengah dan spacing lebih baik
st.markdown(
    f"""
    <style>
    .container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 30px;  /* Beri jarak dari atas */
    }}
    .logo {{
        width: 180px; /* Ukuran lebih proporsional */
        margin-bottom: 20px; /* Tambah jarak ke teks */
    }}
    .title {{
        font-size: 42px;
        font-weight: bold;
    }}
    </style>
    <div class="container">
        <img class="logo" src="data:image/png;base64,{logo_base64}" />
        <h1 class="title">INTERAXIN</h1>
    </div>
    """,
    unsafe_allow_html=True
)


import pandas as pd
import joblib  # Untuk memuat model Machine Learning
from sklearn.preprocessing import LabelEncoder

# Load dataset dan model
@st.cache_data
def load_data():
    df = pd.read_csv("drug_interactions_extended.csv")
    le_drug1 = LabelEncoder()
    le_drug2 = LabelEncoder()
    le_severity = LabelEncoder()
    df['Drug1_encoded'] = le_drug1.fit_transform(df['Drug 1'])
    df['Drug2_encoded'] = le_drug2.fit_transform(df['Drug 2'])
    df['Severity_encoded'] = le_severity.fit_transform(df['Severity'])
    model = joblib.load("drug_interaction_model.pkl")  # Pastikan model sudah dibuat
    return df, le_drug1, le_drug2, le_severity, model

df, le_drug1, le_drug2, le_severity, model = load_data()

# Streamlit UI
st.title("💊 Prediksi Interaksi Obat dengan AI🤖")
st.write("Masukkan dua nama obat untuk mengetahui interaksi dan tingkat keparahannya.")

# Menggabungkan semua obat dari kedua kolom menjadi satu daftar unik
drugs = list(set(df['Drug 1'].unique()).union(set(df['Drug 2'].unique())))
drugs.sort()  # Urutkan agar rapi

# Input pengguna
drug1 = st.selectbox("Pilih Obat Pertama", drugs)
drug2 = st.selectbox("Pilih Obat Kedua", drugs)

if st.button("Prediksi Interaksi"):
    try:
        d1_encoded = le_drug1.transform([drug1])[0]
        d2_encoded = le_drug2.transform([drug2])[0]
        input_data = pd.DataFrame([[d1_encoded, d2_encoded]], columns=['Drug1_encoded', 'Drug2_encoded'])
        severity_encoded = model.predict(input_data)[0]
        severity = le_severity.inverse_transform([severity_encoded])[0]
        interaction_row = df[(df['Drug 1'] == drug1) & (df['Drug 2'] == drug2)]
        
        if not interaction_row.empty:
            description = interaction_row['Interaction Description'].values[0]
        else:
            description = "Tidak ada deskripsi interaksi untuk kombinasi ini."
        
        st.success(f"### ⚠️ Interaksi {drug1} + {drug2}")
        st.write(f"**Keparahan:** {severity}")
        st.write(f"**Deskripsi:** {description}")
    except ValueError:
        st.error("Obat tidak ditemukan dalam dataset. Coba lagi!")
