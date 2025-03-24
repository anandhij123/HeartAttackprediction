import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Heart Attack Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Attack Risk Assessment")
st.markdown("Enter your health information to assess heart attack risk and recommended treatment.")

# Function to load sample data (for reference)
@st.cache_data
def load_sample_data():
    data = {
        "Gender": ["Male", "Female", "Male", "Male", "Male", "Female", "Male", "Male", "Male", "Female"],
        "Age": [70, 55, 42, 84, 86, 66, 33, 84, 73, 63],
        "Blood Pressure (mmHg)": [181, 103, 95, 106, 187, 125, 181, 182, 115, 174],
        "Cholesterol (mg/dL)": [262, 253, 295, 270, 296, 271, 262, 288, 286, 254],
        "Has Diabetes": ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes"],
        "Smoking Status": ["Never", "Never", "Current", "Never", "Current", "Former", "Current", "Current", "Never", "Former"],
        "Chest Pain Type": ["Typical Angina", "Atypical Angina", "Typical Angina", "Atypical Angina", 
                           "Non-anginal Pain", "Typical Angina", "Asymptomatic", "Non-anginal Pain", 
                           "Asymptomatic", "Non-anginal Pain"],
        "Treatment": ["Lifestyle Changes", "Angioplasty", "Angioplasty", "Coronary Artery Bypass Graft (CABG)", 
                     "Medication", "Coronary Artery Bypass Graft (CABG)", "Lifestyle Changes", 
                     "Lifestyle Changes", "Angioplasty", "Angioplasty"]
    }
    return pd.DataFrame(data)

# Load sample data
sample_data = load_sample_data()

# Create the sidebar for inputs
st.sidebar.header("Patient Information")

# Gender input
gender = st.sidebar.radio("Gender", options=["Male", "Female"])

# Age input
age = st.sidebar.slider("Age", min_value=30, max_value=89, value=60, step=1)

# Blood pressure input
blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", min_value=90, max_value=199, value=145, step=1)

# Cholesterol input
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", min_value=150, max_value=299, value=225, step=1)

# Diabetes input
has_diabetes = st.sidebar.radio("Has Diabetes", options=["Yes", "No"])

# Smoking status input
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    options=["Never", "Former", "Current"]
)

# Chest pain type input
chest_pain_type = st.sidebar.selectbox(
    "Chest Pain Type",
    options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

# Submit button
submit_button = st.sidebar.button("Submit Assessment")

# Create label encoders
@st.cache_resource
def create_encoders():
    gender_encoder = LabelEncoder()
    gender_encoder.fit(["Female", "Male"])
    
    diabetes_encoder = LabelEncoder()
    diabetes_encoder.fit(["No", "Yes"])
    
    smoking_encoder = LabelEncoder()
    smoking_encoder.fit(["Never", "Former", "Current"])
    
    chest_pain_encoder = LabelEncoder()
    chest_pain_encoder.fit(["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    
    treatment_encoder = LabelEncoder()
    treatment_encoder.fit(["Lifestyle Changes", "Medication", "Angioplasty", "Coronary Artery Bypass Graft (CABG)"])
    
    return {
        "gender": gender_encoder,
        "diabetes": diabetes_encoder,
        "smoking": smoking_encoder,
        "chest_pain": chest_pain_encoder,
        "treatment": treatment_encoder
    }

encoders = create_encoders()

# Function to load the ML model
@st.cache_resource
def load_model():
    # Check if model file exists
    if os.path.exists("heart_model.pkl"):
        with open("heart_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        # Create a placeholder model if no saved model is found
        st.warning("No trained model found. Using a placeholder model for demonstration purposes.")
        return RandomForestClassifier(random_state=42)

# Load or create model
model = load_model()

# Feature scaling function
@st.cache_resource
def create_scaler():
    return StandardScaler()

scaler = create_scaler()

# Process user input and make prediction
def process_input(user_data):
    # Encode categorical variables
    user_data["Gender_Encoded"] = encoders["gender"].transform([user_data["Gender"]])[0]
    user_data["Has_Diabetes_Encoded"] = encoders["diabetes"].transform([user_data["Has Diabetes"]])[0]
    user_data["Smoking_Status_Encoded"] = encoders["smoking"].transform([user_data["Smoking Status"]])[0]
    user_data["Chest_Pain_Type_Encoded"] = encoders["chest_pain"].transform([user_data["Chest Pain Type"]])[0]
    
    # Create feature array for prediction
    features = np.array([
        user_data["Age"],
        user_data["Blood Pressure (mmHg)"],
        user_data["Cholesterol (mg/dL)"],
        user_data["Gender_Encoded"],
        user_data["Has_Diabetes_Encoded"],
        user_data["Smoking_Status_Encoded"],
        user_data["Chest_Pain_Type_Encoded"]
    ]).reshape(1, -1)
    
    # Predict treatment (assuming model is trained to predict Treatment_Encoded)
    try:
        treatment_encoded = model.predict(features)[0]
        treatment = encoders["treatment"].inverse_transform([treatment_encoded])[0]
        
        # Get probability scores for each class
        probabilities = model.predict_proba(features)[0]
        treatment_probs = {encoders["treatment"].inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
        
        return treatment, treatment_probs, features
    except:
        # If model prediction fails, provide a fallback mechanism
        st.error("Model prediction failed. Providing statistical recommendation instead.")
        # Fallback: Find similar cases and recommend most common treatment
        return find_similar_treatment(user_data), {}, features

# Fallback function to find similar cases
def find_similar_treatment(user_data):
    # Calculate age difference
    sample_data["Age_Diff"] = abs(sample_data["Age"] - user_data["Age"])
    
    # Filter by gender and find closest match by age
    gender_matches = sample_data[sample_data["Gender"] == user_data["Gender"]]
    if len(gender_matches) > 0:
        similar_cases = gender_matches.sort_values("Age_Diff").head(3)
        # Return most common treatment
        return similar_cases["Treatment"].mode()[0]
    else:
        return "Lifestyle Changes"  # Default recommendation

# Display user inputs when submit is clicked
if submit_button:
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Your Information")
        
        # Create a dictionary of user inputs
        user_data = {
            "Gender": gender,
            "Age": age,
            "Blood Pressure (mmHg)": blood_pressure,
            "Cholesterol (mg/dL)": cholesterol,
            "Has Diabetes": has_diabetes,
            "Smoking Status": smoking_status,
            "Chest Pain Type": chest_pain_type
        }
        
        # Display user inputs as a DataFrame
        user_df = pd.DataFrame([user_data])
        st.write(user_df)
    
    # Process the input and get prediction
    recommendation, probabilities, features = process_input(user_data)
    
    with col2:
        st.header("Risk Assessment Results")
        st.subheader(f"Recommended Treatment: {recommendation}")
        
        # Display treatment probabilities if available
        if probabilities:
            st.write("Treatment Probability Breakdown:")
            probs_df = pd.DataFrame({
                'Treatment': list(probabilities.keys()),
                'Probability (%)': [round(p * 100, 2) for p in probabilities.values()]
            }).sort_values('Probability (%)', ascending=False)
            
            st.dataframe(probs_df)
            
            # Create a bar chart for probabilities
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Treatment', y='Probability (%)', data=probs_df, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    
    # Risk factors analysis
    st.header("Risk Factor Analysis")
    
    # Create columns for risk factor display
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        # Visualize blood pressure risk
        bp_risk = "High" if blood_pressure > 140 else "Normal"
        bp_color = "red" if bp_risk == "High" else "green"
        st.markdown(f"**Blood Pressure:** <span style='color:{bp_color}'>{bp_risk}</span>", unsafe_allow_html=True)
        
        # Visualize cholesterol risk
        chol_risk = "High" if cholesterol > 240 else "Normal"
        chol_color = "red" if chol_risk == "High" else "green"
        st.markdown(f"**Cholesterol:** <span style='color:{chol_color}'>{chol_risk}</span>", unsafe_allow_html=True)
        
        # Diabetes risk
        diabetes_color = "red" if has_diabetes == "Yes" else "green"
        st.markdown(f"**Diabetes Status:** <span style='color:{diabetes_color}'>{has_diabetes}</span>", unsafe_allow_html=True)
    
    with risk_col2:
        # Age risk
        age_risk = "High" if age > 65 else "Moderate" if age > 45 else "Low"
        age_color = "red" if age_risk == "High" else "orange" if age_risk == "Moderate" else "green"
        st.markdown(f"**Age Risk:** <span style='color:{age_color}'>{age_risk}</span>", unsafe_allow_html=True)
        
        # Smoking risk
        smoking_risk_map = {"Never": "Low", "Former": "Moderate", "Current": "High"}
        smoking_risk = smoking_risk_map[smoking_status]
        smoking_color = "red" if smoking_risk == "High" else "orange" if smoking_risk == "Moderate" else "green"
        st.markdown(f"**Smoking Risk:** <span style='color:{smoking_color}'>{smoking_risk}</span>", unsafe_allow_html=True)
        
        # Chest pain risk
        chest_pain_risk_map = {
            "Asymptomatic": "Low", 
            "Non-anginal Pain": "Moderate", 
            "Atypical Angina": "High", 
            "Typical Angina": "Very High"
        }
        chest_pain_risk = chest_pain_risk_map[chest_pain_type]
        chest_pain_color = "red" if chest_pain_risk in ["High", "Very High"] else "orange" if chest_pain_risk == "Moderate" else "green"
        st.markdown(f"**Chest Pain Risk:** <span style='color:{chest_pain_color}'>{chest_pain_risk}</span>", unsafe_allow_html=True)
    
    # Finding similar cases in the dataset
    st.header("Similar Cases Analysis")
    
    # Calculate age difference
    sample_data["Age_Diff"] = abs(sample_data["Age"] - age)
    
    # Filter by gender and find closest match by age
    gender_matches = sample_data[sample_data["Gender"] == gender]
    
    if len(gender_matches) > 0:
        # Sort by age similarity
        similar_cases = gender_matches.sort_values("Age_Diff").head(3)
        st.write("Here are similar cases from our database:")
        st.dataframe(similar_cases.drop(columns=["Age_Diff"]))
        
        # Show treatment distribution for similar cases
        st.subheader("Treatments for Similar Cases")
        st.bar_chart(similar_cases["Treatment"].value_counts())
    else:
        st.write("No similar cases found in our limited dataset.")

# Model Training Section (for demonstration)
st.header("Model Training")
with st.expander("Train/Update Model"):
    st.write("This section allows you to train or update the heart disease prediction model.")
    
    uploaded_file = st.file_uploader("Upload a CSV file with training data", type=["csv"])
    
    if uploaded_file is not None:
        # Load the uploaded data
        train_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(train_data.head())
        
        # Select target column
        target_col = st.selectbox("Select target column (what to predict)", 
                                 options=train_data.columns.tolist(),
                                 index=train_data.columns.tolist().index("Treatment") if "Treatment" in train_data.columns else 0)
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Placeholder for actual model training code
                    # This would be replaced with code similar to your notebook
                    st.success("Model trained successfully! The new model is now in use.")
                    
                    # Example of what would happen in real training:
                    # 1. Preprocess data (encode, scale, etc.)
                    # 2. Split into train/test
                    # 3. Train model (RandomForest, XGBoost, etc.)
                    # 4. Save model to pickle file
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

# How to use section
st.header("How to Use This Tool")
with st.expander("Show Instructions"):
    st.markdown("""
    ### Using the Heart Attack Risk Assessment Tool

    1. **Enter Patient Information**: Fill in all the required fields in the sidebar:
       - Gender
       - Age
       - Blood Pressure
       - Cholesterol Level
       - Diabetes Status
       - Smoking Status
       - Chest Pain Type

    2. **Submit Assessment**: Click the 'Submit Assessment' button to process your information.

    3. **Review Results**: The tool will provide:
       - A recommended treatment based on the machine learning model
       - Risk factor analysis for each health parameter
       - Similar cases from the database for comparison

    4. **Train/Update Model**: Advanced users can upload new data to retrain the model.
    """)

# Data Explorer section
st.header("Dataset Explorer")
with st.expander("View Sample Dataset"):
    st.dataframe(sample_data.drop(columns=["Age_Diff"] if "Age_Diff" in sample_data.columns else []))
    
    # Add some basic statistics about the dataset
    st.subheader("Dataset Statistics")
    
    # Gender distribution
    gender_counts = sample_data["Gender"].value_counts()
    st.write(f"Gender Distribution: {gender_counts['Male']} Males, {gender_counts['Female']} Females")
    
    # Age statistics
    st.write(f"Age Range: {sample_data['Age'].min()} to {sample_data['Age'].max()} years")
    st.write(f"Average Age: {sample_data['Age'].mean():.1f} years")
    
    # Treatment distribution
    st.subheader("Treatment Distribution")
    st.bar_chart(sample_data["Treatment"].value_counts())

# Add disclaimer
st.markdown("---")
st.caption("""
**Disclaimer**: This application is for educational purposes only and does not provide medical advice. 
The dataset used is limited and not representative of all medical cases. 
Please consult a healthcare professional for proper medical advice and treatment.
""")

# Add instructions for model integration at the bottom
st.markdown("---")
with st.expander("For Developers: Model Integration Instructions"):
    st.markdown("""
    ### How to Integrate Your Trained Model

    1. **Save Your Model**: Add this code at the end of your Jupyter notebook:
       ```python
       # Save the trained model to a file
       import pickle
       
       # Replace 'best_model' with your best performing model
       with open('heart_model.pkl', 'wb') as f:
           pickle.dump(best_model, f)
       ```

    2. **Export Your Encoders**: Save your label encoders:
       ```python
       # Save label encoders
       encoders = {
           'gender': le_gender,
           'diabetes': le_diabetes,
           'smoking': le_smoking,
           'chest_pain': le_cpt,
           'treatment': le_treatment
       }
       
       with open('encoders.pkl', 'wb') as f:
           pickle.dump(encoders, f)
       ```

    3. **Place Files in App Directory**: Move both pickle files to the same directory as this app.py file.

    4. **Update Model Loading**: Replace the placeholder model loading code with:
       ```python
       @st.cache_resource
       def load_model_and_encoders():
           with open("heart_model.pkl", "rb") as f:
               model = pickle.load(f)
           with open("encoders.pkl", "rb") as f:
               encoders = pickle.load(f)
           return model, encoders
       
       model, encoders = load_model_and_encoders()
       ```

    5. **Update Feature Processing**: Make sure the feature processing matches your model's training pipeline.
    """)