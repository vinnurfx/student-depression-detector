import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# 1. Load Dataset
data = pd.read_csv('D:\KULIAH\SEMESTER 3\Matematika Diskrit\SereniTrack\data\depressionstudentdataset.csv')

# 2. Preprocessing
data = data.dropna()  # Tidak ada data yang dihapus karena data semuanya terisi

# 3. Encode Data
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 4. Split Data
X = data.drop('Depression', axis=1)  # Fitur
y = data['Depression']  # (0: Tidak Depresi, 1: Depresi)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# 7. Evaluasi model
y_pred = rfc.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Simpan model
joblib.dump(rfc, 'rfc_model.pkl')