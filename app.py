import streamlit as st
import pandas as pd
import pickle

# Load the preprocessor
def load_preprocess():
    with open("model/preprocessor.pkl", "rb") as file:
        preprocessor = pickle.load(file)
    return preprocessor

# Load the model
def load_model():
    with open("model/xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Main App
def main():
    st.title("Bank Marketing Prediction App")
    st.write("Provide the inputs below to predict customer response.")

    # Input form
    st.sidebar.header("Input Features")
    with st.form("input_form"):
        # Categorical columns
        job = st.selectbox("Job", ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"])
        marital = st.selectbox("Marital Status", ["married","divorced","single"])
        education = st.selectbox("Education", ["unknown","secondary","primary","tertiary"])
        default = st.selectbox("Has Credit in Default?", ['yes', 'no', 'unknown'])
        housing = st.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])
        loan = st.selectbox("Has Personal Loan?", ['yes', 'no', 'unknown'])
        contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone', 'unknown'])
        month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        poutcome = st.selectbox("Outcome of the Previous Marketing Campaign", ['failure', 'nonexistent', 'success'])

        # Numerical columns
        day_of_week = st.slider("Day of the Week (Numeric)", 1, 30, 5)
        age = st.number_input("Age", value=30, step=1)
        balance = st.number_input("Balance", value=1000, step=100)
        campaign = st.slider("Number of Contacts during Campaign", 1, 50, 2)
        duration = st.slider("Duration of Last Contact (seconds)", 0, 5000, 200)

        # Submit button
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Create a DataFrame for the inputs
        input_data = pd.DataFrame({
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [month],
            'poutcome': [poutcome],
            'day_of_week': [day_of_week],
            'age': [age],
            'balance': [balance],
            'campaign': [campaign],
            'duration': [duration],
        })
        input_data[input_data.select_dtypes(['object']).columns] = input_data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

        # Load the pipeline
        load_preprocessor = load_preprocess()
        final_model = load_model()

        # Make predictions
        final_data = load_preprocessor.transform(input_data)
        prediction = final_model.predict(final_data)
        result = "Positive" if prediction[0] == 1 else "Negative"

        # Display the result
        st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()
