import streamlit as st
import pandas as pd
import joblib


##@st.cache_resource(allow_output_mutation=True)
def load_model():
    model_path = 'best_model.joblib'
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f'Model file not found at {model_path}. Please check the file path and try again.')
        return None
    except Exception as e:
        st.error(f'An error occurred while loading the model: {e}')
        return None

model = load_model()

# Function to make predictions
def make_prediction(input_data):
    if model:
        return model.predict(input_data)[0]
    else:
        return None

# Streamlit UI
st.title('MSP Prediction')

# Input features from the user
st.header('Please enter the input values:')
C = st.number_input('C', value=0.0, format='%f')
H = st.number_input('H', value=0.0, format='%f')
N = st.number_input('N', value=0.0, format='%f')
O = st.number_input('O', value=0.0, format='%f')
S = st.number_input('S', value=0.0, format='%f')
VM = st.number_input('VM', value=0.0, format='%f')
Ash = st.number_input('Ash', value=0.0, format='%f')
FC = st.number_input('FC', value=0.0, format='%f')
Cel = st.number_input('Cel', value=0.0, format='%f')
Hem = st.number_input('Hem', value=0.0, format='%f')
Lig = st.number_input('Lig', value=0.0, format='%f')
Plantcapa = st.number_input('Plantcapacity(kg/hr', value=0.0, format='%f')

# Select location
location_options = ['China', 'UK', 'US']
location = st.selectbox('Location', options=location_options)
location_data = [int(loc == location) for loc in location_options]

if st.button('Predict MSP'):
    # Create a dataframe with the input features
    input_features = [C, H, N, O, S, VM, Ash, FC, Cel, Hem, Lig, Plantcapa] + location_data
    feature_names = ['C', 'H', 'N', 'O', 'S', 'VM', 'Ash', 'FC', 'Cel', 'Hem', 'Lig', 'Plantcapacity(kg/hr'] + [f'Location_{loc}' for loc in location_options]
    input_df = pd.DataFrame([input_features], columns=feature_names)
    
    # Display the input DataFrame
    st.write('Input features:')
    st.dataframe(input_df)

    # Get prediction
    prediction = make_prediction(input_df)
    if prediction is not None:
        st.success(f'The predicted MSP is: {prediction:.3f}')
    else:
        st.error('Unable to make prediction. Ensure that the model is loaded correctly.')
