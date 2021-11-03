import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image


# --- Get infos --- #

def fetch_train(path):
    return pd.read_csv(path)

def fetch_data(path, n_rows=1):
    df = pd.read_csv(path)
    return df.sample(n=n_rows)

def get_prediction_client(df):
    to_send = df.to_json()
    res = requests.post(f"http://localhost:8080/predict_client", json=to_send)
    return res

def get_prediction_db(df):
    to_send = df.to_json()
    res = requests.post(f"http://localhost:8080/predict_db", json=to_send)
    return res

# --- Display --- #

def display_intro_page():
    st.markdown('# Welcome on this dashboard !')
    st.markdown('## Credit default scoring app')
    st.markdown(f'<p style="font-size: 20px">This project is a part of my Data Scientist Degree where we were tasked to deploy\
        a machine learning model online. </p>', True)
    
    st.markdown('## Main goals of the project')
    st.markdown('<ul style="list-style-type: circle;"> \
    <li style="font-size: 20px">Build a machine learning model that predict the probability of a customer defaulting on their loan </li> \
    <li style="font-size: 20px">Make this ML model available through an API</li> \
    <li style="font-size: 20px">Create an interactive dashboard for the bank relationship managers</li> \
    </ul>', True)

    st.markdown('## How to use it ?')
    st.markdown('<p style="font-size: 20px">To begin, you have to choose on the sidebar a database to work with, by default you can find two already available (Paris and Marseille\'s clients).\
        <br>It is also possible to import a custom database, an expander at the bottom is available.<br>\
        Main informations on the database are displayed automatically.</p>', True)
    st.markdown('<p style="font-size: 20px">To predict the score of a specific client, you have to choose the client ID.\
        To better understand the score, you can compare some informations of the client versus the values of all the others clients.<br>\
        The multiselect bow allows you to chose which features to compare</p>', True)

    st.markdown('## Some information before you go !')

    st.markdown('<p style="font-size: 20px">The data for the kaggle competition was provided by Home Credit Group they operate mostly in Asia. \
    Our training dataset is mostly Cash loans. </p>', True)

    st.markdown('<p style="font-size: 20px">We do not know if the data is from one country or multiple, \
        so we don\'t know if the currency value is homogenous throughout the dataset. \
        A reverse google image search of the picture in the KAGGLE competition, seems to indicate it\'s\
        from their Vietnamese branch. If the entire dataset is in Vietnamese dong , \
        then the maximum loan of our dataset is approximately 23 USD$. \
        Most of the information on the loans purposes is missing, but when they were indicated they were mostly for consumer goods. \
        Most common: Mobile phones, electronics, computers, furniture.</p>', True)
    st.markdown('<p style="font-size: 20px">The dataset is coming from this \
        <a href="https://www.kaggle.com/c/home-credit-default-risk/overview)">Kaggle competition !</a></p>',True)

def display_centered_icon(path: str):
    image = Image.open(path)
    col1, col2, col3 = st.sidebar.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width = 'always')

    with col3:
        st.write("")


# --- Preproc utils --- #

def handle_outliers(df):
    ### DAYS_EMPLOYED_ANOM ###
    # Create an anomalous flag column
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    df.DAYS_EMPLOYED = df.DAYS_EMPLOYED.abs()
    print(f'(DAYS_EMPLOYED) {df["DAYS_EMPLOYED_ANOM"].sum()} anomalous values found.')

    ### DAYS columns ###
    for col in df.columns[df.columns.str.contains('DAYS')]:
        df[col] = df[col].abs()

    ### Categorical XNA ###
    for col_name in df.select_dtypes(include='object').columns:
        if 'XNA' in (df[col_name].unique()):
            nb_xna = df[col_name].value_counts().loc['XNA']
            print(f'{nb_xna} XNAs found in {col_name}')
            df[col_name].replace('XNA', np.nan, inplace=True)

    return df

def adding_features(df):
    age = df.DAYS_BIRTH/365
    
    # AMT_CREDIT: Credit amount of the loan ; AMT_ANNUITY: Loan annuity
    credit_annuity_ratio = df.AMT_CREDIT /  df.AMT_ANNUITY
    credit_goods_price_ratio =  df.AMT_CREDIT / df.AMT_GOODS_PRICE
    credit_income_ratio =  df.AMT_CREDIT / df.AMT_INCOME_TOTAL
    annuity_income_ratio = df.AMT_ANNUITY / df.AMT_INCOME_TOTAL
    days_employed_life = df.DAYS_EMPLOYED / df.DAYS_BIRTH
    
    
    tmp = df[['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1']]
    tmp = pd.concat([tmp, credit_annuity_ratio.rename('CREDIT_ANNUITY_RATIO')], axis=1) 
            
    # Adding new features to the dataframe
    df['AGE'] = age
    df['CREDIT_ANNUITY_RATIO'] = credit_annuity_ratio
    df['CREDIT_GOODS_PRICE_RATIO'] = credit_goods_price_ratio
    df['CREDIT_INCOME_RATIO'] = credit_income_ratio
    df['ANNUITY_INCOME_RATIO'] = annuity_income_ratio
    df['DAYS_EMPLOYED_LIFE'] = days_employed_life
        
    print('Features added.')
    
    return df