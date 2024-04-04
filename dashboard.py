import streamlit as st
import requests
import pandas as pd
import pyarrow.parquet as pq

# Function to make API request and get prediction
def get_prediction(data):
    api_url = "https://card-4873eb75da10.herokuapp.com/predict"  # Update with your API URL
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

    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None, None

# Load sample parquet data (replace this with your actual data loading)
parquet_file = 'data/df_test.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

# Streamlit app
st.title('Credit Scoring Prediction ')

# Dropdown for client IDs
selected_client_id = st.selectbox('Select Client ID:', df['SK_ID_CURR'].unique())

# Display selected client's data
st.subheader('Selected Client Data:')
selected_client_data = df.loc[df['SK_ID_CURR'] == selected_client_id]
st.write(selected_client_data)

# Button to trigger prediction
if st.button('Predict'):
    # Make API request and get prediction
    prediction_result, prediction_score = get_prediction(selected_client_data)

    # Display prediction result
    st.subheader('Prediction Result:')
    if prediction_result is not None:
        st.write(f"The credit status is: {prediction_result}")
        st.write(f"The prediction score is: {prediction_score:.2%}")