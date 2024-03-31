import streamlit as st
import requests
import pandas as pd
import pyarrow.parquet as pq

# Function to make API request and get prediction
def get_prediction(data):
    api_url = "http://127.0.0.1:5005/predict"  # Update with your API URL
    df_test = {'df_test': data.drop(columns=['SK_ID_CURR']).values.tolist()}
    response = requests.post(api_url, json=df_test)

    try:
        result = response.json()
        prediction_score = result['prediction'][0]

        # Classify as 'Credit accepted' if probability of class 0 is greater than 0.5
        if prediction_score > 0.5:
            prediction_result = 'Credit accepted'
        else:
            prediction_result = 'Credit denied'
            return prediction_result, prediction_score

    except Exception as e :
        st.error(f"Error getting prediction: {e}")
        return None, None

# Charger les données parquet d'exemple
parquet_file = 'OCDSP7/data/df_test.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)

# Application Streamlit
st.title('Credit Scoring Prediction')

# Dropdown pour les IDs des clients sélectionnés
selected_client_id = st.selectbox('Select Client ID', df['SK_ID_CURR'].unique())

# Obtenir les données du client sélectionné
selected_client_data = df[df['SK_ID_CURR'] == selected_client_id]

# Afficher les données du client sélectionné
st.write(selected_client_data)

# Bouton pour déclencher la prédiction
if st.button('Predict'):
    # Obtenir la prédiction
    prediction_result, prediction_score = get_prediction(selected_client_data)
    
    # Afficher le résultat de la prédiction
    st.subheader('Prediction Result:')
    if prediction_result is not None:
        st.write(f"The credit status is: {prediction_result}")
        st.write(f"The prediction score is: {prediction_score:.2%}")
