import os
import streamlit as st
import base64
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 🔍 Menentukan path absolut agar bisa digunakan di Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path gambar
logo_path = os.path.join(BASE_DIR, "logo.png")

# Fungsi untuk mengonversi gambar ke base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

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
        margin-top: 30px;  
    }}
    .logo {{
        width: 180px; 
        margin-bottom: 20px;
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

# 🔍 Path model dan LabelEncoder
model_path = os.path.join(BASE_DIR, "drug_interaction_model.pkl")
le_drug1_path = os.path.join(BASE_DIR, "le_drug1.pkl")
le_drug2_path = os.path.join(BASE_DIR, "le_drug2.pkl")
le_severity_path = os.path.join(BASE_DIR, "le_severity.pkl")

# Load model dan encoders yang sudah disimpan
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    le_drug1 = joblib.load(le_drug1_path)
    le_drug2 = joblib.load(le_drug2_path)
    le_severity = joblib.load(le_severity_path)

    return model, le_drug1, le_drug2, le_severity

model, le_drug1, le_drug2, le_severity = load_model()

# 🔍 Path dataset
dataset_path = os.path.join(BASE_DIR, "drug_interactions_extended_fixed.csv")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(dataset_path)
    df["Drug 1"] = df["Drug 1"].str.title()  # Huruf pertama besar
    df["Drug 2"] = df["Drug 2"].str.title()
    return df

df = load_data()

# Streamlit UI
st.title("💊 Prediksi Interaksi Obat dengan AI 🤖")
st.write("Masukkan dua nama obat untuk mengetahui interaksi dan tingkat keparahannya.")

# Menggabungkan semua obat dari kedua kolom menjadi satu daftar unik
drugs = sorted(set(df['Drug 1'].unique()).union(set(df['Drug 2'].unique())))

# Input pengguna (tetap menampilkan huruf kapital sesuai dengan dataset)
drug1 = st.selectbox("Pilih Obat Pertama", drugs)
drug2 = st.selectbox("Pilih Obat Kedua", drugs)

if st.button("Prediksi Interaksi"):
    try:
        # ✅ Pastikan obat dikenali oleh LabelEncoder sebelum transformasi
        if drug1 not in le_drug1.classes_ or drug2 not in le_drug2.classes_:
            st.error(f"Obat tidak dikenali dalam model: {drug1}, {drug2}")
            st.stop()

        # Transformasi input ke format model
        d1_encoded = le_drug1.transform([drug1])[0]
        d2_encoded = le_drug2.transform([drug2])[0]

        # 🚀 Pastikan urutan input model selalu konsisten
        input_data = pd.DataFrame([[min(d1_encoded, d2_encoded), max(d1_encoded, d2_encoded)]], 
                                  columns=['Drug1_encoded', 'Drug2_encoded'])

        # Prediksi keparahan
        severity_encoded = model.predict(input_data)[0]
        severity = le_severity.inverse_transform([severity_encoded])[0]

        # 🔍 Cari interaksi dalam dataset tanpa terpengaruh urutan
        interaction_row = df[((df["Drug 1"] == drug1) & (df["Drug 2"] == drug2)) |
                             ((df["Drug 1"] == drug2) & (df["Drug 2"] == drug1))]

        description = interaction_row["Interaction Description"].values[0] if not interaction_row.empty else "Tidak ada deskripsi interaksi."

        # ✅ Tampilan hasil tetap kapital
        st.success(f"### ⚠️ Interaksi {drug1} + {drug2}")
        st.write(f"**Keparahan:** {severity}")
        st.write(f"**Deskripsi:** {description}")

    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")
