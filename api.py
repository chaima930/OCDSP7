from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import  MinMaxScaler
from lightgbm import LGBMClassifier

app = Flask(__name__)

# Charger le modèle et le scaler à partir des fichiers pickle
with open('/Users/chaima/Downloads/Projet+Mise+en+prod+-+home-credit-default-risk/OCDSP7/data/model.pkl', 'rb') as file:
    model = pickle.load(file)


with open('/Users/chaima/Downloads/Projet+Mise+en+prod+-+home-credit-default-risk/OCDSP7/data/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home_page():
    return 'Bienvenue sur l\'API de score de crédit'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON envoyées dans la requête POST
        data = request.get_json(force=True)
        df_test = np.array(data['df_test'])

        # Effectuer la mise à l'échelle des données
        scaled_data = scaler.transform(df_test)

        # Faire les prédictions
        prediction = model.predict_proba(scaled_data)[:, 1]  # Assuming a binary classification task

        # Renvoyer les prédictions sous forme de JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # Gérer les erreurs potentielles et renvoyer un message d'erreur
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Lancer l'application Flask en mode debug
    app.run(debug=True,port=5003)