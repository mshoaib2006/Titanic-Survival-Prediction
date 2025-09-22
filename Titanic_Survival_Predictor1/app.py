import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
feature_names = model_package.get("features", [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'Title', 'HadCabin', 'FamilySize', 'IsAlone'
])

MAPPINGS = {
    'class': {"First": 1, "Second": 2, "Third": 3},
    'gender': {"Male": "male", "Female": "female"},
    'embarked': {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"},
    'title': {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Other": "Other"}
}

def preprocess_input(data: dict) -> pd.DataFrame:
    row = data.copy()
    row['Sex'] = 0 if row['Sex'] == "male" else 1
    row['Embarked'] = {'S': 0, 'C': 1, 'Q': 2}[row['Embarked']]
    row['Title'] = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}[row['Title']]
    row['HadCabin'] = int(row['HadCabin'])
    df = pd.DataFrame([row], columns=feature_names)
    return df

st.title("Titanic Survival Prediction")
st.markdown("Predict whether a passenger would have survived the Titanic disaster.")

with st.form("prediction_form"):
    st.header("Passenger Information")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Class", options=list(MAPPINGS['class'].keys()))
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0,
                              step=0.5, format="%.1f")
    with col2:
        gender = st.selectbox("Gender", options=list(MAPPINGS['gender'].keys()))
        embarked = st.selectbox("Embarked", options=list(MAPPINGS['embarked'].keys()))

    title = st.selectbox("Title", options=list(MAPPINGS['title'].keys()))
    had_cabin = st.checkbox("Had Cabin", value=False)

    st.header("Family Details")
    col3, col4 = st.columns(2)
    with col3:
        sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
    with col4:
        parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)

    st.header("Ticket Information")
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0,
                           step=1.0, format="%.2f")

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    input_data = {
        'Pclass': MAPPINGS['class'][pclass],
        'Sex': MAPPINGS['gender'][gender],
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Fare': float(fare),
        'Embarked': MAPPINGS['embarked'][embarked],
        'Title': title,
        'HadCabin': had_cabin,
        'FamilySize': sibsp + parch + 1,
        'IsAlone': bool(sibsp + parch == 0)
    }

    start_time = datetime.now()
    input_df = preprocess_input(input_data)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    response_time = (datetime.now() - start_time).total_seconds()

    st.divider()
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if prediction == 1:
            st.success("## Survived")
            st.balloons()
        else:
            st.error("## Did Not Survive")

    with col_res2:
        prob = probability if prediction == 1 else 1 - probability
        st.metric("Confidence", f"{prob:.1%}")
        st.caption(f"Response time: {response_time:.2f}s")

    with st.expander("Technical Details"):
        st.json({
            "input_data": input_data,
            "prediction": int(prediction),
            "probability": round(probability, 4),
            "processed_features": input_df.to_dict(orient="records")[0]
        })

with st.expander("About this app"):
    st.markdown("""
    ### Titanic Survival Predictor  
    This app predicts whether a passenger would have survived the Titanic disaster 
    based on their characteristics using a machine learning model.
    
    **Features:**
    - Streamlit-only (no FastAPI required)
    - Detailed passenger information collection
    - Probability-based predictions
    - Local model loading for deployment on Streamlit Cloud
    """)

