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
relevant_columns = ['Open (Rp)', 'High (Rp)', 'Low (Rp)', 'Volume', 'Sentiment (-1/0/1)', 'Close (Rp)']  # Ganti dengan kolom-kolom yang relevan dari dataset Anda
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
                    1. Neural Network (NN):
                    <li>Model matematis yang terinspirasi oleh cara kerja otak manusia.</li>
                    <li>Terdiri dari jaringan neuron buatan terhubung dalam lapisan-lapisan.</li>
                    <li>Digunakan untuk mempelajari pola-pola kompleks dalam data.</li>
                    2. Recurrent Neural Network (RNN):
                    <li>Jenis Neural Network yang dirancang untuk data berurutan.</li>
                    <li>Menggunakan informasi sebelumnya melalui hubungan maju-mundur antar langkah waktu.</li>
                    <li>Efektif dalam memodelkan ketergantungan kontekstual dalam urutan data.</li>
                    3. Long Short-Term Memory (LSTM):
                    <li>Varian RNN yang mengatasi masalah gradien yang menghilang/meledak.</li>
                    <li>Memiliki gate: forget gate, input gate, dan output gate.</li>
                    <li>Memungkinkan kontrol aliran informasi dalam memori internal.</li>                
                    4. Gated Recurrent Unit (GRU):
                    <li>Varian RNN dengan arsitektur yang lebih sederhana dibandingkan LSTM.</li>
                    <li>Memiliki gate: update gate dan reset gate.</li>
                    <li>Efisien dalam kasus-kasus di mana LSTM terlalu kompleks.</li>                        
                </ul>
                """,
                    unsafe_allow_html=True
            )

            #st.write("[Penjelasan Perbedaan Kedua Algoritma >](https://www.youtube.com/watch?v=8HyCNIVRbSU&ab_channel=TheA.I.Hacker-MichaelPhi)")
        with right_column:
            st_lottie(animasi2, height=300, key="coding2")
            st_lottie(animasi2, height=300, key="coding1")
           # st_lottie(animasi1, height=300, key="coding1")

    # ---- XX ----
    with st.container():
        st.write("---")
        st.header("PENJELASAN ATRIBUT")
        st.write("##")
        st.markdown(
                """
                <ul class="text-justify">
                    <li>Open : Harga pembukaan (open price) adalah harga perdagangan pertama kali suatu saham dibeli atau dijual pada suatu sesi perdagangan. Ini merupakan harga yang ditetapkan saat perdagangan dimulai pada waktu tertentu, misalnya saat pasar saham dibuka pada pagi hari.</li>
                    <li>High : Harga tertinggi (high price) adalah harga tertinggi yang dicapai oleh suatu saham selama suatu periode tertentu, misalnya selama satu hari perdagangan. Ini menunjukkan harga paling tinggi yang pembeli bersedia bayar selama periode tersebut.</li>
                    <li>Low : Harga terendah (low price) adalah harga terendah yang dicapai oleh suatu saham selama suatu periode tertentu. Ini menunjukkan harga paling rendah yang penjual bersedia terima selama periode tersebut.</li>     
                    <li>Volume : Volume perdagangan (trading volume) adalah jumlah total saham yang diperdagangkan selama suatu periode tertentu. Ini mencerminkan tingkat likuiditas suatu saham dan dapat memberikan indikasi tentang minat dan partisipasi pasar.</li>
                    <li>Sentiment : Sentimen pasar (market sentiment) mengacu pada persepsi umum atau sentimen yang dimiliki oleh pelaku pasar terhadap suatu saham atau pasar secara keseluruhan. Sentimen bisa positif, negatif, atau netral, dan dapat mempengaruhi harga saham secara langsung atau tidak langsung.</li>  
                    <li>Close : Harga penutupan (closing price) adalah harga terakhir yang diperdagangkan untuk suatu saham pada akhir suatu sesi perdagangan. Ini adalah harga yang digunakan untuk menghitung perubahan harga relatif dan kinerja saham selama periode tersebut.</li>                    
                </ul>
                """,
                    unsafe_allow_html=True
            )

def show_prediksi():
    # st.subheader("Halaman Prediksi")
    st.markdown('<p class="title-justify" style="font-size: 32px; font-weight: bold;">Halaman Prediksi</p>', unsafe_allow_html=True)
    st.write("---")

    # Input form
    input_values = []
    for i in range(time_steps):
        st.markdown(f"### Data Ke- {i+1}")
        input_row = []
        for idx, column in enumerate(relevant_columns):
            input_key = f"{column}_t-{i}"
            
            if column == 'Sentiment (-1/0/1)':
                options = [-1, 0, 1]
                value = st.selectbox(f'{column}', options, index=1, key=input_key)
            else:
                # Ubah label kolom menjadi lebih singkat
                label = column.split(' ')[0]
                value = st.number_input(f'{label}', value=np.nan, key=input_key)
                if np.isnan(value):
                    value = 0.0
            
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

        # Remove index from dataframe
        df_predicted_no_index = df_predicted.copy()
        df_predicted_no_index.reset_index(drop=True, inplace=True)

        # Restrict values for Sentiment column
        #df_predicted_no_index['Sentiment'] = df_predicted_no_index['Sentiment'].apply(lambda x: -1 if x < -0.5 else (1 if x > 0.5 else 0))
        df_predicted_no_index['Sentiment (-1/0/1)'] = df_predicted_no_index['Sentiment (-1/0/1)'].apply(lambda x: -1 if x < 0 else (0 if x == 0 else 1))

        st.write("##")
        
        # Display predicted values as table without index
        st.subheader('Hasil Prediksi')
        st.write(df_predicted_no_index.to_html(index=False, justify='center'), unsafe_allow_html=True)

        st.write("##")
        st.write("##")
        st.write("---")
        st.write("---")

        # Get the predicted close price for the next day
        predicted_close = df_predicted_no_index['Close (Rp)'].iloc[0]

        # Display the predicted close price
        st.markdown('<p style="font-size: 32px; text-align: center;">Hasil Prediksi Harga Pada Hari Berikutnya (Rp) :</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 48px; text-align: center;"> {predicted_close}</p>', unsafe_allow_html=True)

        st.write("---")
        st.write("---")

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
