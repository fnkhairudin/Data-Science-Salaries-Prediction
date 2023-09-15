import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

model = pickle.load(open('ds-salary-predictor-1.sav', 'rb'))

st.title('Data Science Salary Prediction')

data = pd.read_csv('modeling_used.csv')

valCol = {}
cols = data.columns.drop('salary_in_usd')
for col in cols:
    valCol.update({
        col: tuple(data[col].unique())
    })

# FUNCTION
def user_report():
    work_year = st.sidebar.selectbox(cols[0], valCol[cols[0]])
    experience_level = st.sidebar.selectbox(cols[1], valCol[cols[1]])
    employment_type = st.sidebar.selectbox(cols[2], valCol[cols[2]])
    employee_residence = st.sidebar.selectbox(cols[3], valCol[cols[3]])
    remote_ratio = st.sidebar.selectbox(cols[4], valCol[cols[4]])
    company_location = st.sidebar.selectbox(cols[5], valCol[cols[5]])
    company_size = st.sidebar.selectbox(cols[6], valCol[cols[6]])
    job_position = st.sidebar.selectbox(cols[7], valCol[cols[7]])
    job_scope = st.sidebar.selectbox(cols[8], valCol[cols[8]])
    company_region = st.sidebar.selectbox(cols[9], valCol[cols[9]])
    aboard = st.sidebar.selectbox(cols[10], valCol[cols[10]])

    user_report_data = {
        cols[0]:work_year,
        cols[1]:experience_level,
        cols[2]:employment_type,
        cols[3]:employee_residence,
        cols[4]:remote_ratio,
        cols[5]:company_location,
        cols[6]:company_size,
        cols[7]:job_position,
        cols[8]:job_scope,
        cols[9]:company_region,
        cols[10]:aboard,
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

if __name__  == "__main__":
    user_data = user_report()
    st.header('Your Data')
    st.write(user_data)

    st.subheader('Salary Prediction')
    if st.button('Predict your salary here!'):
        salary = model.predict(user_data)
        st.subheader('$'+str(np.round(salary[0], 2)))