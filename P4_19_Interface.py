import os
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Before running the interface code:
# 1. Fine-tune the model and save it
# 2. Put saved model folder with the app.py

# To run the interface code:
# 1. Paste this command in the terminal: streamlit run P4_19_Interface.py

# Set up page styling and configurations
st.set_page_config(
    page_title="FIES Income Bracket Predictor",
    layout="centered"
)


# Cache the model loading so it only loads into memory once
@st.cache_resource
def load_prediction_assets(model_dir="./final_fies_model"):
    if not os.path.exists(model_dir):
        return None, None, None, None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, label_encoder, device


# Initialize assets
tokenizer, model, label_encoder, device = load_prediction_assets()

# Header
st.title("Household Income Bracket Predictor")
st.markdown("""
This app utilizes a fine-tuned DistilBERT Language Model to predict a household's income quintile 
based on Household Profile, Housing & Living Conditions, Assets and Annual Household Expenditures.
""")

if model is None:
    st.error(
        "⚠️ **Model files not found!** Please ensure the `./final_fies_model` directory exists right next to this script.")
    st.stop()

# User Input Form
with st.form("household_form"):
    st.subheader("Household Profile")
    col1, col2 = st.columns(2)
    with col1:
        family_members = st.number_input("Total Number of Family Members", min_value=1, max_value=30, value=4)
    with col2:
        employed_members = st.number_input("Family Members Employed", min_value=0, max_value=10, value=1)

    st.divider()
    st.subheader("Housing & Living Conditions")
    col3, col4, col5 = st.columns(3)
    with col3:
        building_type = st.selectbox("Type of Building/House", [
            "Single house",
            "Duplex",
            "Commercial/industrial/agricultural building",
            "Multi-unit residential",
            "Institutional living quarter",
            "Other building unit (e.g. cave, boat)"
        ])
    with col4:
        roof_type = st.selectbox("Type of Roof", [
            "Strong material(galvanized,iron,al,tile,concrete,brick,stone,asbestos)",
            "Light material (cogon,nipa,anahaw)",
            "Mixed but predominantly strong materials",
            "Mixed but predominantly light materials",
            "Salvaged/makeshift materials",
            "Mixed but predominantly salvaged materials",
            "Not Applicable"
        ])
    with col5:
        wall_type = st.selectbox("Type of Walls", [
            "Strong",
            "Light",
            "Quite Strong",
            "Very Light",
            "Salvaged",
            "NOt applicable"
        ])

    bedrooms = st.slider("Number of bedrooms", min_value=0, max_value=10, value=2)

    st.divider()
    st.subheader("Amenities & Assets")
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        electricity_select = st.selectbox("Electricity", ["Yes", "No"])
        # Map back to exact numeric representations found in the dataset
        electricity = "1" if electricity_select == "Yes" else "0"
    with col7:
        tv_count = st.number_input("Number of Television", min_value=0, max_value=10, value=1)
    with col8:
        aircon_count = st.number_input("Number of Airconditioner", min_value=0, max_value=10, value=0)
    with col9:
        car_count = st.number_input("Number of Car, Jeep, Van", min_value=0, max_value=10, value=0)

    st.divider()
    st.subheader("Annual Household Expenditures (PHP)")

    col10, col11 = st.columns(2)
    with col10:
        food_exp = st.number_input("Total Food Expenditure", min_value=0, max_value=1000000, value=115000, step=5000)
        housing_exp = st.number_input("Housing and water Expenditure", min_value=0, max_value=2500000, value=60000,
                                      step=5000)
        edu_exp = st.number_input("Education Expenditure", min_value=0, max_value=1000000, value=15000, step=2000)
    with col11:
        trans_exp = st.number_input("Transportation Expenditure", min_value=0, max_value=1000000, value=5000, step=1000)
        med_exp = st.number_input("Medical Care Expenditure", min_value=0, max_value=1500000, value=3500, step=1000)
        comm_exp = st.number_input("Communication Expenditure", min_value=0, max_value=2000000, value=3000, step=500)

    submit_btn = st.form_submit_button("Predict Income Bracket", use_container_width=True)

# Process inference
if submit_btn:
    # 1. Map values into the exact format
    food = f"{food_exp:,.0f}"
    housing = f"{housing_exp:,.0f}"
    edu = f"{edu_exp:,.0f}"
    trans = f"{trans_exp:,.0f}"
    medical = f"{med_exp:,.0f}"
    comm = f"{comm_exp:,.0f}"

    # 2. Re-creating the text prompt string used during fine-tuning
    templated_text = (
        f"Household of {family_members} members ({employed_members} employed). "
        f"Type: {building_type}, Roof: {roof_type}, Walls: {wall_type}. "
        f"House has {bedrooms} bedrooms. "
        f"Amenities: Electricity={electricity}, TV={tv_count}, Aircon={aircon_count}, Car/Jeep={car_count}. "
        f"Expenditures: Food: {food} PHP, Housing/Water: {housing} PHP, Edu: {edu} PHP, "
        f"Trans: {trans} PHP, Medical: {medical} PHP, Communication: {comm} PHP."
    )

    # 3. Model Tokenization
    with st.spinner("Analyzing profile metrics..."):
        inputs = tokenizer(
            templated_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_id = outputs.logits.argmax(-1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

    # 4. Display Results
    st.success("### Prediction Complete!")

    if "Q5" in predicted_label or "Q4" in predicted_label:
        st.metric(label="Estimated Target Class", value=predicted_label, delta="High Income Tier")
    elif "Q3" in predicted_label:
        st.metric(label="Estimated Target Class", value=predicted_label, delta="Middle Income Tier", delta_color="off")
    else:
        st.metric(label="Estimated Target Class", value=predicted_label, delta="- Low Income Tier",
                  delta_color="inverse")

    # Generated text string sequence sent to NLP
    with st.expander("View underlying text string fed to NLP Model"):
        st.code(templated_text, language="markdown")