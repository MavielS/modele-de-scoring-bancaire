import streamlit as st
import streamlit.components.v1 as components # For shap

st.set_page_config(page_title='Credit Scoring App', layout='wide')

import numpy as np
import pandas as pd 
from pandas.api.types import is_numeric_dtype, is_string_dtype

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt

import shap
import json
import os

from front_func import * 

# --- if first time, launch the API --- 
# if 'api' not in st.session_state:
#     st.session_state.api = True

# if st.session_state.api == True:
#     st.session_state.api = False
#     print('Launching API')
#     os.system('python ./backend/main.py')

# --- set variables in session state ---

if 'db_train' not in st.session_state:
    st.session_state.db_train = fetch_train('sample_train.csv') # Load a sample of the train set

if 'intro_page' not in st.session_state:
    st.session_state.intro_page = 0

if 'tmp_db_choosen' not in st.session_state:
    st.session_state.tmp_db_choosen = None

if 'client_info_placeholder' not in st.session_state:
    st.session_state.client_info_placeholder = st.empty()

if 'client_score_placeholder' not in st.session_state:
    st.session_state.client_score_placeholder = st.empty()

if 'client_compare_placeholder' not in st.session_state:
    st.session_state.client_compare_placeholder = st.empty()

if 'dbs_names' not in st.session_state:
    st.session_state.dbs_names = [
        'Paris\' clients', 
        'Marseille\'s clients'
    ]

if 'dbs' not in st.session_state:
    st.session_state.dbs = {
        'Paris\' clients': fetch_data(PATH_CSV, n_rows=SIZE_PARIS), 
        'Marseille\'s clients': fetch_data(PATH_CSV, n_rows=SIZE_MARSEILLE)
    }


## SIDEBAR
display_centered_icon(PATH_ICON)

db_choosen = st.sidebar.radio(
    'Select a database to work with:',
    st.session_state.dbs_names,
)
# st.info(st.session_state.tmp_db_choosen)

# st.write(db_choosen)
if st.session_state.intro_page == 0 or db_choosen is None:
    st.session_state.intro_page = 1
    st.session_state.tmp_db_choosen = db_choosen 
    display_intro_page()
elif st.session_state.tmp_db_choosen != db_choosen and db_choosen is not None: # Change on the radio widget
    st.session_state.current_db = st.session_state.dbs[db_choosen]
    display_info_db(st.session_state.current_db, db_choosen)

    st.sidebar.markdown('---')
    id_client = st.sidebar.selectbox(label='Client ID', options=st.session_state.current_db['SK_ID_CURR'])

    features_to_comp = st.sidebar.multiselect(label='Features to compare', options=st.session_state.current_db.columns, 
    default=['CODE_GENDER', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_EMPLOYED'])

    st.sidebar.button('Results', on_click=display_results, args=(int(id_client), features_to_comp,  ))

    st.session_state.tmp_db_choosen = db_choosen 
elif db_choosen is not None:
    st.session_state.current_db = st.session_state.dbs[db_choosen]
    # display_info_db(st.session_state.current_db, db_choosen)
    # st.sidebar.button('Display infos database', on_click=display_info_db, args=(st.session_state.current_db, db_choosen, ))

    st.sidebar.markdown('---')
    id_client = st.sidebar.selectbox(label='Client ID', options=st.session_state.current_db['SK_ID_CURR'])

    features_to_comp = st.sidebar.multiselect(label='Features to compare', options=st.session_state.current_db.columns, 
    default=['CODE_GENDER', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_EMPLOYED'])

    st.sidebar.button('Results', on_click=display_results, args=(int(id_client), features_to_comp,  ))

    st.session_state.tmp_db_choosen = db_choosen 

## Sidebar -- Adding a client database
st.sidebar.markdown('---')
with st.sidebar.expander("Add clients data", True):
    # ADDING A DATABASE
    uploaded_file = st.file_uploader('Choose a database to add', type=['csv'],
                        key='input_new_db_name')

    def submit_add_db(db_name: str, df: pd.DataFrame):
        """ Callback function during adding a new db. """
        db_name = uploaded_file.name
        
        # display a warning if the user entered an existing name
        real_name = db_name.replace('.csv', '')
        if real_name in st.session_state.dbs_names:
            st.warning(f'The "{real_name}" already exists.')
        else:
            st.session_state.dbs_names.append(real_name)
            st.session_state.dbs[real_name] = df

    if uploaded_file is not None:
        st.button('Add', key='button_add_db',
                on_click=submit_add_db, args=(uploaded_file.name, pd.read_csv(uploaded_file)))

    if db_choosen is not None:

        # DELETING A DATABASE
        def submit_delete_db(db_name: str):
            """ Callback function during deleting an existing db. """
            st.session_state.dbs_names.remove(db_name)

        to_delete = st.selectbox('Choose a database to delete:', st.session_state.dbs_names)
        st.button('Delete', key='button_delete_db',
                on_click=submit_delete_db, args=(to_delete, ))
 
