import joblib

model = joblib.load("model.pkl")

def predict_disease(symptoms):
    prediction = model.predict([symptoms])
    return prediction[0]
