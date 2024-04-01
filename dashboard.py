import streamlit as st
import requests
import pandas as pd
import pyarrow.parquet as pq

# Function to make API request and get prediction
def get_prediction(data):
    api_url = "https://card-4873eb75da10.herokuapp.com/predict"  # Update with your API URL
    df_test = {'df_test': data.drop(columns=['SK_ID_CURR','index']).values.tolist()}
    response = requests.post(api_url, json=df_test)

    try:
        result = response.json()
        # Vérifiez la structure de la réponse JSON
        print("Response JSON:", result)
        if 'prediction' in result:
            prediction_score = result['prediction'][0]
            # Classify as 'Credit accepted' if probability of class 0 is greater than 0.5
            if prediction_score > 0.5:
                prediction_result = 'Credit accepted'
            else:
                prediction_result = 'Credit denied'
            return prediction_result, prediction_score
        else:
            print("La clé 'prediction' n'est pas présente dans la réponse JSON.")
            return None, None
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return None, None
# Load example parquet data
parquet_file = 'OCDSP7/data/df_test.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)


# Streamlit application
st.title('My Streamlit App - Selecting Client IDs')

# Dropdown for selecting client IDs
selected_client_id = st.selectbox('Select Client ID:', df['SK_ID_CURR'].unique())

# Get data of the selected client
selected_client_data = df[df['SK_ID_CURR'] == selected_client_id]

# Display data of the selected client
st.write(selected_client_data)

# Button to trigger prediction
if st.button('Predict'):
    # Get prediction
    prediction_result, prediction_score = get_prediction(selected_client_data)
    
    # Display prediction result
    st.subheader('Prediction Result:')
    if prediction_result is not None:
        st.write(f"The credit status is: {prediction_result}")
        st.write(f"The prediction score is: {prediction_score:.2%}")