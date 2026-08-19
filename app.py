import streamlit as st
import pandas as pd
from model import load_and_train_model
from openai import OpenAI

# Page Configuration
st.set_page_config(page_title="Predictive Churn & Revenue Risk Advisor", layout="wide")

st.title("🛡️ Predictive Churn & Revenue Risk Advisor")
st.markdown("Identify high-risk accounts and leverage Generative AI to generate personalized retention strategies.")

# Load Data and Model
@st.cache_resource
def get_data():
    return load_and_train_model()

df, model = get_data()

# Sidebar Filters
st.sidebar.header("Filter Accounts")
risk_threshold = st.sidebar.slider("Minimum Churn Risk % to Flag", 0, 100, 50)

# Filter high-risk customers
high_risk_df = df[df['ChurnProbability'] >= risk_threshold].sort_values(by="ChurnProbability", ascending=False)

# Main Layout Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers Tracked", len(df))
col2.metric("High-Risk Accounts Flagged", len(high_risk_df))
col3.metric("Potential Revenue at Risk", f"${(high_risk_df['MonthlyCharges'].sum()):,.2f}")

st.markdown("---")
st.subheader("⚠️ High-Risk Customer Queue")

if high_risk_df.empty:
    st.success("No customers exceed the selected risk threshold!")
else:
    # Display table of risky customers
    st.dataframe(high_risk_df[['CustomerID', 'Tenure', 'MonthlyCharges', 'SupportTickets', 'UsageFrequency', 'ChurnProbability']], use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 AI-Powered Account Diagnostics & Action Plan")
    
    selected_customer_id = st.selectbox("Select a Customer ID to Generate AI Playbook:", high_risk_df['CustomerID'])
    
    if selected_customer_id:
        cust_data = high_risk_df[high_risk_df['CustomerID'] == selected_customer_id].iloc[0]
        
        # Display individual card metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tenure (Months)", cust_data['Tenure'])
        c2.metric("Monthly Spend", f"${cust_data['MonthlyCharges']}")
        c3.metric("Support Tickets", cust_data['SupportTickets'])
        c4.metric("Churn Risk Score", f"{cust_data['ChurnProbability']:.1f}%")

        if st.button("Generate AI Retention Playbook"):
            # Set up your OpenAI client (or use a local LLM via Ollama)
            # api_key = st.text_input("Enter OpenAI API Key", type="password")
            
            prompt = f"""
            You are an expert Enterprise Account Strategist. Analyze this customer profile and provide:
            1. Root cause hypothesis for why they are likely to churn.
            2. Recommended business action (e.g., discount tier, product training, executive check-in).
            3. A draft retention email template tailored to their metrics.

            Customer Profile:
            - Customer ID: {cust_data['CustomerID']}
            - Tenure: {cust_data['Tenure']} months
            - Monthly Spend: ${cust_data['MonthlyCharges']}
            - Support Tickets Logged: {cust_data['SupportTickets']}
            - Usage Frequency Score: {cust_data['UsageFrequency']} (Low score means low engagement)
            - Predicted Churn Probability: {cust_data['ChurnProbability']:.1f}%
            """

            # Placeholder response if API key is not hardcoded/provided during dev, 
            # or standard OpenAI completion call:
            try:
                client = OpenAI(api_key="YOUR_OPENAI_API_KEY") # Replace or use st.secrets
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.info("To see live AI output, configure your OpenAI API key in the code or use environment secrets. Here is a sample structure of what the AI generates:")
                st.markdown("""
                **1. Root Cause Analysis:** High support ticket volume combined with a drop in usage frequency indicates frustration with onboarding or unresolved technical blocks.
                **2. Recommended Action:** Assign a Customer Success Manager for an immediate check-in call and offer a 15% discount for a 3-month contract extension.
                **3. Draft Email:** *'Hi [Customer], we noticed you've faced a few hiccups recently...'*
                """)