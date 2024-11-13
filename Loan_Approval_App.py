import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        padding-bottom: 2rem;
    }
    .stSubheader {
        font-size: 1.5rem !important;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .insights-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("LogisticRegression_pipeline.pkl")

model = load_model()

def calculate_range(value, bins, labels):
    try:
        result = pd.cut([value], bins=bins, labels=labels, include_lowest=True)[0]
        return str(result) if pd.notnull(result) else labels[0]
    except:
        return labels[0]

def create_app():
    # Title with emoji and description
    st.title("Loan Eligibility Predictor 💰")
    st.markdown("""
        <div style="background-color: #f0f2f6; color: #333333; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            Welcome to our Loan Eligibility Predictor! Fill in your details below to check your loan eligibility.
            We use advanced analytics to provide you with an instant assessment.
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📝 Personal Information", "💵 Financial Details", "🏠 Loan & Property Details"])
    
    with tab1:
        st.markdown("<div class='section-title'>Personal Details</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("What is your gender?", ["Male", "Female"])
            married = st.selectbox("What is your marital status?", ["Yes", "No"])
            dependents = st.selectbox("How many dependents do you have?", ["0", "1", "2", "3+"])
        with col2:
            education = st.selectbox("What is your education level?", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    
    with tab2:
        st.markdown("<div class='section-title'>Income Information</div>", unsafe_allow_html=True)
        input_type = st.radio("Choose input method:", ["Slider", "Manual Entry"])
        
        if input_type == "Slider":
            applicant_income = st.slider(
                "What is your monthly income ($)?",
                min_value=150, max_value=81000, value=3800, step=100,
                help="Drag the slider to set your monthly income"
            )
            coapplicant_income = st.slider(
                "Co-applicant's monthly income ($)?",
                min_value=0, max_value=41667, value=0, step=100,
                help="Drag the slider to set co-applicant's income (if any)"
            )
        else:
            applicant_income = st.number_input(
                "What is your monthly income ($)?",
                min_value=150, max_value=81000, value=3800, step=100
            )
            coapplicant_income = st.number_input(
                "Co-applicant's monthly income ($)?",
                min_value=0, max_value=41667, value=0, step=100
            )
    
    with tab3:
        st.markdown("<div class='section-title'>Loan Details</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if input_type == "Slider":
                loan_amount = st.slider(
                    "Loan amount needed (in thousands $)",
                    min_value=9, max_value=700, value=128, step=1
                )
            else:
                loan_amount = st.number_input(
                    "Loan amount needed (in thousands $)",
                    min_value=9, max_value=700, value=128, step=1
                )
            
            credit_history = st.selectbox(
                "Do you have a clean credit history?", 
                options=[1.0, 0.0],
                format_func=lambda x: "Yes" if x == 1.0 else "No"
            )
        
        with col2:
            loan_terms = [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0]
            loan_amount_term = st.selectbox(
                "Preferred loan repayment period (months)",
                options=loan_terms
            )
            property_area = st.selectbox(
                "Property area type",
                ["Urban", "Rural", "Semiurban"]
            )

    # Feature Engineering
    has_coapplicant = "Y" if coapplicant_income > 0 else "N"
    monthly_loan_burden_percentage = (loan_amount * 1000 / loan_amount_term) / applicant_income * 100
    total_application_income = applicant_income + coapplicant_income
    
    # Define bins and labels
    burden_bins = [0, 6.855608, 8.843264, 467.414530]
    burden_labels = ['Low Burden (0-6.86%)', 'Medium Burden (6.86-8.84%)', 'High Burden (8.84%+)']
    coapplicant_bins = [0, 1188.5, 2297.25, 41667]
    coapplicant_labels = ['Low Income', 'Medium Income', 'High Income']
    loan_amount_bins = [0, 100, 168, 700]
    loan_amount_labels = ['Low Loan Amount', 'Medium Loan Amount', 'High Loan Amount']
    
    # Calculate ranges
    loan_burden_range = calculate_range(monthly_loan_burden_percentage, burden_bins, burden_labels)
    coapplicant_income_range = calculate_range(coapplicant_income, coapplicant_bins, coapplicant_labels)
    loan_amount_range = calculate_range(loan_amount, loan_amount_bins, loan_amount_labels)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area],
        "HasCoapplicant": [has_coapplicant],
        "Monthly_Loan_Burden_Percentage": [monthly_loan_burden_percentage],
        "Total_Application_Income": [total_application_income],
        "Coapplicant_Income_Range": [coapplicant_income_range],
        "LoanAmount_Range": [loan_amount_range],
        "Loan_Burden_Range": [loan_burden_range]
    })
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Check Loan Eligibility", use_container_width=True)
    
    if predict_button:
        try:
            prediction = model.predict(input_data)
            loan_status = "Approved" if prediction[0] == 1 else "Not Approved"
            
            # Display prediction in a styled container
            st.markdown("<div class='insights-card'>", unsafe_allow_html=True)
            if loan_status == "Approved":
                st.success("🎉 Congratulations! Your loan is likely to be approved!")
            else:
                st.error("We're sorry, but the loan is likely to be not approved at this time.")
            
            # Display insights in columns
            st.subheader("Application Insights")
            col1, col2, col3 = st.columns(3)
            
            monthly_payment = (loan_amount * 1000) / loan_amount_term
            with col1:
                st.metric("Monthly Payment", f"${monthly_payment:,.2f}")
            with col2:
                st.metric("Monthly Loan Burden", f"{monthly_loan_burden_percentage:.2f}%")
            with col3:
                st.metric("Total Household Income", f"${total_application_income:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    create_app()