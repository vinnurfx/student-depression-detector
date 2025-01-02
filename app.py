from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
import os

app = Flask(__name__)

# Inisialisasi label encoders
label_encoders = {
    'Gender': LabelEncoder(),
    'Sleep Duration': LabelEncoder(),
    'Dietary Habits': LabelEncoder(),
    'Suicidal Thoughts': LabelEncoder(),
    'Family History of Mental Illness': LabelEncoder()
}

# Fit label encoders dengan data yang sesuai
label_encoders['Gender'].fit(['Male', 'Female'])
label_encoders['Sleep Duration'].fit(['Less than 5 hours', '5 - 6 hours', '6 - 7 hours', '7 - 8 hours', 'More than 8 hours'])
label_encoders['Dietary Habits'].fit(['Unhealthy', 'Moderate', 'Healthy'])
label_encoders['Suicidal Thoughts'].fit(['Yes', 'No'])
label_encoders['Family History of Mental Illness'].fit(['Yes', 'No'])

# Load model
model = joblib.load('rfc_model.pkl')

# Path ke file CSV
csv_file_path = 'prediction.csv'
# Fungsi untuk menulis header ke file CSV jika file belum ada
def write_csv_header():
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 
                      'Dietary Habits', 'Suicidal Thoughts', 'Study Hours', 'Financial Stress', 
                      'Family History', 'Prediction']
            writer.writerow(header)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari form
        form_data = request.form
        input_data = []

        # Memproses setiap input
        input_data.append(label_encoders['Gender'].transform([form_data['gender']])[0])
        input_data.append(int(form_data['age']))
        input_data.append(int(form_data['academic_pressure']))
        input_data.append(int(form_data['study_satisfaction']))
        input_data.append(label_encoders['Sleep Duration'].transform([form_data['sleep_duration']])[0])
        input_data.append(label_encoders['Dietary Habits'].transform([form_data['dietary_habits']])[0])
        input_data.append(label_encoders['Suicidal Thoughts'].transform([form_data['suicidal_thoughts']])[0])
        input_data.append(int(form_data['study_hours']))
        input_data.append(int(form_data['financial_stress']))
        input_data.append(label_encoders['Family History of Mental Illness'].transform([form_data['family_history']])[0])

        # Melakukan prediksi
        prediction = model.predict([input_data])[0]
        result = "Depressed" if prediction == 1 else "Not Depressed"

        if result == "Depressed":
            message =  "Eh Ya Ampun! Jangan bunuh diri yaa! Telepon Ambulan Ninuninu!"
            saran = [
                "Tetaplah hidup dan jangan menyerah. Anda tidak sendirian dalam perjuangan ini.",
                "Berkonsultasilah dengan seorang profesional, seperti psikolog atau psikiater!",
                "Cari dukungan dari orang-orang terdekat Anda, seperti keluarga atau teman."
            ]
        else:
            message = "Yeay! Tetap hidup yaa dan Jangan depresi ya deck yaa!"
            saran = [
                "Tetaplah hidup dengan optimis",
                "Jangan lupa bercerita kalau ada masalah",
                "Semangat Yaa!"
            ]

         # Menyimpan data dan hasil prediksi ke file CSV
        write_csv_header()
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([form_data['gender'], form_data['age'], form_data['academic_pressure'], form_data['study_satisfaction'], 
                             form_data['sleep_duration'], form_data['dietary_habits'], 
                             form_data['suicidal_thoughts'], form_data ['study_hours'], form_data ['financial_stress'], 
                             form_data['family_history'], result])

        # Menampilkan hasil di halaman baru
        return render_template('result.html', result=result, message=message, saran=saran)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)