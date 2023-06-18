import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# Load the pre-trained model
model = load_model('best_model.h5')

# Preprocess the dataset
relevant_columns = ['Open', 'High', 'Low', 'Volume', 'Sentiment', 'Close']  # Ganti dengan kolom-kolom yang relevan dari dataset Anda
scaler = MinMaxScaler(feature_range=(0, 1))

# Define the time steps
time_steps = 3

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Simulator TA Kelompok 27",
                   page_icon=":arrow_double_up:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")

# ---- LOAD ASSETS ----
animasi1 = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_rMIWDc0fSB.json")
animasi2 = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_raiw2hpe.json")
gambar1 = Image.open("images/Tasya.jpg")
gambar2 = Image.open("images/Montana.jpg")


# Functions for each page
def show_beranda():
    # ---- HEADER SECTION ----
    with st.container():
        st.subheader("HALO! SELAMAT DATANG DI WEBSITE INI :wave:")
        st.markdown('<p class="title-justify" style="font-size: 32px; font-weight: bold;">Prediksi Harga Saham dengan Pendekatan Algoritma Long Short Term Memory (LSTM) dan Gated Recurrent Unit (GRU) dengan Analisis Sentimen Twitter</p>', unsafe_allow_html=True)
        st.markdown('<p class="text-justify">Teknologi machine learning dapat dimanfaatkan untuk memprediksi pergerakan harga saham. Dengan mengumpulkan data historis dan informasi lainnya, model machine learning dapat dilatih untuk mempelajari pola dan tren dalam data dan menggunakan informasi tersebut untuk memprediksi pergerakan harga saham. Namun, perlu diingat bahwa pasar saham sangat kompleks dan dipengaruhi oleh banyak faktor. Oleh karena itu, prediksi harga saham selalu memiliki tingkat ketidakpastian yang tinggi dan bukan merupakan sesuatu yang dapat diandalkan secara mutlak.</p>', unsafe_allow_html=True)
        #st.write("[Artefak Tugas Akhir >>>](https://drive.google.com/drive/folders/11iqhOTdJxsybn8x1jLwVRErZ4Qn68p8U)")

    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("TENTANG ALGORITMA")
            st.write("##")
            st.markdown(
                """
                <ul class="text-justify">
                    LSTM memiliki tiga gerbang (gate) utama: gerbang input, gerbang output, dan gerbang lupa (forget gate), sementara GRU hanya memiliki dua gerbang utama: gerbang reset dan gerbang update.
                    <li>Gerbang input LSTM dapat memutuskan seberapa banyak informasi baru yang harus disimpan dalam memori jangka panjang, sementara gerbang reset GRU dapat digunakan untuk mengatur ulang memori jangka pendek yang ada.</li>
                    <li>Gerbang lupa LSTM memungkinkan model untuk "menghapus" informasi yang tidak lagi diperlukan dari memori jangka panjang, sedangkan gerbang update GRU dapat digunakan untuk menentukan seberapa banyak informasi baru yang harus diambil dari memori jangka pendek.</li>
                    <li>GRU memiliki jumlah parameter yang lebih sedikit dibandingkan LSTM, sehingga GRU dapat lebih cepat dikompilasi dan lebih cepat dilatih daripada LSTM.</li>
                    <li>Pilihan antara LSTM dan GRU untuk tugas tertentu biasanya tergantung pada sifat data sekuensial yang dihadapi, serta kompleksitas dan kebutuhan pemrosesan model yang diperlukan.</li>
                </ul>
                """,
                    unsafe_allow_html=True
            )

            #st.write("[Penjelasan Perbedaan Kedua Algoritma >](https://www.youtube.com/watch?v=8HyCNIVRbSU&ab_channel=TheA.I.Hacker-MichaelPhi)")
        with right_column:
            st_lottie(animasi2, height=300, key="coding2")
            st_lottie(animasi2, height=300, key="coding1")
           # st_lottie(animasi1, height=300, key="coding1")

def show_prediksi():
    #st.subheader("Halaman Prediksi")
    st.markdown('<p class="title-justify" style="font-size: 32px; font-weight: bold;">Halaman Prediksi</p>', unsafe_allow_html=True)
    st.write("---")

    # Input form
    input_values = []
    for i in range(time_steps):
        st.markdown(f"### Data Ke- {i+1}")
        input_row = []
        for idx, column in enumerate(relevant_columns):
            input_key = f"{column}_t-{i}"
            value = st.number_input(f'{column} (t{i})', value=0.0, key=input_key)
            input_row.append(value)
        input_values.append(input_row)

    # Predict button
    if st.button('Prediksi'):
        # Prepare input data
        input_data = np.array(input_values)
        input_data = scaler.fit_transform(input_data)
        input_data = input_data.reshape(1, time_steps, len(relevant_columns))

        # Make prediction
        predicted_data = model.predict(input_data)
        predicted_data = scaler.inverse_transform(predicted_data)

        # Create dataframe for predicted values
        df_predicted = pd.DataFrame(predicted_data, columns=relevant_columns)

        # Display predicted values as dataframe
        st.subheader('Hasil Prediksi')
        st.dataframe(df_predicted)

def show_kontak():
    # ---- PROJECTS ----
    with st.container():
        st.header("Kelompok TA - 2022_2023 - 27")
        st.write("---")
        st.write(" ")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(gambar2)
        with text_column:
            st.subheader("Montana Gurning")
            st.write(
                """
                - NIM       : 11S19017
                - PRODI     : S1 INFORMATIKA
                - ANGKATAN  : 2019
                """
            )
            st.write("Mengeluh karena revisi skripsi hanya membuat pikiran semakin stress, jadikan revisi sebagai penyemangat dari sebuah kesalahan.")
            st.markdown(
                "[Akun Linked-in](https://www.linkedin.com/in/montana-gurning-178bb0200/)")

    with st.container():
        st.write(" ")
        st.write(" ")
        st.write(" ")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(gambar1)
        with text_column:
            st.subheader("Tasya Juli Chantika Gurning")
            st.write(
                """
                - NIM       : 11S19068
                - PRODI     : S1 INFORMATIKA
                - ANGKATAN  : 2019
                """
            )
            st.write(
                "Teruntuk dirimu yang sedang berjuang menyelesaikan skripsi, jangan pernah ada kata nanti.")
            st.markdown(
                "[Akun Linked-in](https://www.linkedin.com/in/tasyagurning/)")
            
    # ---- CONTACT ----
    with st.container():
        st.write(" ")
        st.write(" ")
        st.write("---")
        st.write(" ")
        st.write(" ")
        st.header("Beri Kami Kesan dan Pesan Anda!")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/montanagurning913@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Nama" required>
            <input type="email" name="email" placeholder="Email" required>
            <textarea name="message" placeholder="Kesan dan Pesan" required></textarea>
            <button type="submit">Kirim</button>
        </form>
        """

        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()

# Main function
def main():
    st.sidebar.header("Aplikasi Prediksi Harga Saham") 

    menu = ["Beranda", "Prediksi", "Kontak"]
    choice = st.sidebar.selectbox("#", menu)

    if choice == "Beranda":
        show_beranda()
    elif choice == "Prediksi":
        show_prediksi()
    elif choice == "Kontak":
        show_kontak()

if __name__ == '__main__':
    main()
