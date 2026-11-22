# PythonProject
By fine-tuning DistilBERT on natural language summaries of household attributes, we automated the classification of
households into five socio-economic quintiles (Q1 to Q5) with 64.9% accuracy. The pipeline
concludes with an AI Financial Consultant function and an interactive Streamlit web application
that provides automated spending stability insights.

## Problem Statement

The core task is multi-class text classification. Given a natural language summary of a
household’s demographics, housing characteristics, amenities, and expenditures, the
model must assign it to one of five income quintiles:

- Q1 (Lowest) - Bottom 20% of income distribution
- Q2 - Lower-middle income (20th-40th percentile)
- Q3 - Middle income (40th-60th percentile)
- Q4 - Upper-middle income (60th-80th percentile)
- Q5 (Highest) - Top 20% of income distribution

## Input

A structured natural language string encoding household attributes. Each row in
the FIES dataset is converted into a sentence of the form:

“Household of 5 members (2 employed). Type: Single house, Roof: Strong material,
Walls: Strong. House has 3 bedrooms. Amenities: Electricity=1, TV=2, Aircon=0,
Car/Jeep=0. Expenditures: Food: 85,000 PHP, Housing/Water: 20,000 PHP, Edu: 12,000
PHP, Trans: 8,000 PHP, Medical: 3,000 PHP, Communication: 2,500 PHP.”

## Output

The predicted quintile label (Q1-Q5) along with a derived financial insight based
on the household’s spending profile.

## Dataset

[Filipino Family Income and Expenditure](https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure)

## Project structure

- `P4_19.ipynb` (Source code & Model Notebook)
- `P4_19_Interface.py` (Interactive Interface Code)

## How to run

- Fine-tune the model in Google Colab using `P4_19.ipynb` and save it
- Put saved model folder (final_fies_model) with the `P4_19_Interface.py`
- Open `P4_19_Interface.py` Paste this command in the terminal: streamlit run P4_19_Interface.py

<img width="736" height="857" alt="image" src="https://github.com/user-attachments/assets/210d60d8-06ce-4b69-b7f4-9e65561c5cb0" />
<img width="722" height="441" alt="image" src="https://github.com/user-attachments/assets/6d1e5d1e-0fca-49c1-b139-f233ff3f5a3c" />
