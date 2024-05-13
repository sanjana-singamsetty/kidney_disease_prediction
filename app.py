import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import shap

# Load the logistic regression model
logreg = joblib.load('logreg_model.pkl')

# Function to preprocess input data
def preprocess_input(age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell, pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia):
    # Encode categorical features
    red_blood_cells = 1 if red_blood_cells == 'abnormal' else 0
    pus_cell = 1 if pus_cell == 'abnormal' else 0
    pus_cell_clumps = 1 if pus_cell_clumps == 'present' else 0
    bacteria = 1 if bacteria == 'present' else 0
    hypertension = 1 if hypertension == 'yes' else 0
    diabetes_mellitus = 1 if diabetes_mellitus == 'yes' else 0
    coronary_artery_disease = 1 if coronary_artery_disease == 'yes' else 0
    appetite = 1 if appetite == 'good' else 0
    pedal_edema = 1 if pedal_edema == 'yes' else 0
    anemia = 1 if anemia == 'yes' else 0

    # Concatenate input features
    input_features = np.array([age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell, pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia]).reshape(1, -1)

    return input_features

def main():
    st.title('Kidney Disease Predictor')

    # Input fields with default values
    age = st.number_input('Age', min_value=0, max_value=150, step=1, value=30)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=300, step=1, value=120)
    specific_gravity = st.number_input('Specific Gravity', min_value=1.0, max_value=1.1, step=0.001, value=1.005)
    albumin = st.number_input('Albumin', min_value=0, max_value=5, step=1, value=0)
    sugar = st.number_input('Sugar', min_value=0, max_value=5, step=1, value=0)
    red_blood_cells = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
    pus_cell = st.selectbox('Pus Cell', ['normal', 'abnormal'])
    pus_cell_clumps = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
    bacteria = st.selectbox('Bacteria', ['present', 'notpresent'])
    blood_glucose_random = st.number_input('Blood Glucose Random', min_value=0, max_value=500, step=1, value=100)
    blood_urea = st.number_input('Blood Urea', min_value=0, max_value=250, step=1, value=40)
    serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0, max_value=15.0, step=0.1, value=1.0)
    sodium = st.number_input('Sodium', min_value=100, max_value=200, step=1, value=135)
    potassium = st.number_input('Potassium', min_value=2.0, max_value=10.0, step=0.1, value=4.0)
    haemoglobin = st.number_input('Haemoglobin', min_value=0.0, max_value=20.0, step=0.1, value=14.0)  # Corrected here
    packed_cell_volume = st.number_input('Packed Cell Volume', min_value=0, max_value=100, step=1, value=40)
    white_blood_cell_count = st.number_input('White Blood Cell Count', min_value=0, max_value=25000, step=100, value=8000)
    red_blood_cell_count = st.number_input('Red Blood Cell Count', min_value=0.0, max_value=10.0, step=0.1, value=4.5)
    hypertension = st.selectbox('Hypertension', ['yes', 'no'])
    diabetes_mellitus = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
    coronary_artery_disease = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
    appetite = st.selectbox('Appetite', ['good', 'poor'])
    pedal_edema = st.selectbox('Pedal Edema', ['yes', 'no'])
    anemia = st.selectbox('Anemia', ['yes', 'no'])

    # Button to trigger prediction
    if st.button('Predict'):
        # Preprocess input data
        input_features = preprocess_input(age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell, pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia)

        # Make prediction using the loaded model
        prediction = logreg.predict(input_features)
        if prediction[0] == 1:
            st.write('Prediction: Kidney Disease (Class 1)')
            st.table({"Class": [0, 1], "Prediction": ["No Kidney Disease", "Kidney Disease"]})
        else:
            st.write('Prediction: No Kidney Diseawse (Class 0)')
            st.table({"Class": [0, 1], "Prediction": ["No Kidney Disease", "Kidney Disease"]})
         # Explain the prediction using SHAP
        st.set_option('deprecation.showPyplotGlobalUse', False)
        explainer = shap.Explainer(logreg, np.zeros((1, 24)))  # Generate an artificial dataset
        shap_values = explainer.shap_values(input_features)

        # Plot the SHAP summary plot
        st.title('SHAP Summary Plot')
        shap.summary_plot(shap_values, input_features, show=False)
        st.pyplot()

if __name__ == '__main__':
    main()
