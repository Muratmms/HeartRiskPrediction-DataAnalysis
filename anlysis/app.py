from keras.models import load_model
import numpy as np
import tensorflow as tf
print(10*"*","Heart Risk Prediction",10*"*")
def get_user_input():

    Age = float(input("Age: "))
    Dizziness = 1 if input("Dizziness (y/n): ").strip().lower() == 'y' else 0
    Cold_Sweats_Nausea = 1 if input("Cold Sweats/Nausea (y/n): ").strip().lower() == 'y' else 0
    Palpitations = 1 if input("Palpitations (y/n): ").strip().lower() == 'y' else 0
    Shortness_of_Breath = 1 if input("Shortness of Breath (y/n): ").strip().lower() == 'y' else 0
    Fatigue = 1 if input("Fatigue (y/n): ").strip().lower() == 'y' else 0
    Swelling = 1 if input("Swelling (y/n): ").strip().lower() == 'y' else 0
    Pain_Arms_Jaw_Back = 1 if input("Pain in Arms/Jaw/Back (y/n): ").strip().lower() == 'y' else 0
    Chest_Pain = 1 if input("Chest Pain (y/n): ").strip().lower() == 'y' else 0

    
    return np.array([[Age, Dizziness, Cold_Sweats_Nausea, Palpitations, Shortness_of_Breath, Fatigue, Chest_Pain, Swelling, Pain_Arms_Jaw_Back]])

def main():
    model_path = "heart_risk_model.h5"  # Model dosyanızın ismi
    model = load_model(model_path)
    
    user_input = get_user_input()
    prediction = model.predict(user_input)
    if prediction > 0.5:
        print("Riskli")
    else:
        print("Sağlıklı")
    print(f"Sağlık tahmin sonucu: {prediction[0][0]:.4f}")

if __name__ == "__main__":
    main()
