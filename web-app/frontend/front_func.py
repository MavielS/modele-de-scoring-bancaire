import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image

import numpy as np
import pandas as pd 
from pandas.api.types import is_numeric_dtype, is_string_dtype

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt

import shap
import json

# Some constants
THRESHOLD_MODEL = 0.45
PATH_CSV = 'application_test.csv'
PATH_ICON = 'icon.png'
SIZE_PARIS = 10000
SIZE_MARSEILLE = 7000

from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',['#12c43a', 'w',  '#e23f3f'], N=256) 
colors = ['#12c43a', '#e23f3f'] # Red - Green
paper_color = '#0e1117' ; plot_color = '#32363E' # Color of streamlit bg
font_color = 'white' ; font_family='Lato' ; font_size=20

url_prediction = 'http://fastapi:8080'


# --- Get infos --- #

def fetch_train(path):
    return pd.read_csv(path)

def fetch_data(path, n_rows=1):
    df = pd.read_csv(path)
    return df.sample(n=n_rows)

def get_prediction_client(df):
    to_send = df.to_json()
    res = requests.post(url_prediction + '/predict_client', json=to_send)
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
        Main informations on the database are displayed automatically when clicking on it.</p>', True)
    st.markdown('<p style="font-size: 20px">To predict the score of a specific client, you have to choose the client ID.\
        To better understand the score, you can compare some informations of the client versus the values of all the others clients.<br>\
        The multiselect bow allows you to chose which features to compare.</p>', True)

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

def display_results(id_client: int, features_to_comp: list):
    """ Callback function triggered when results of a prediction are wanted """

    valids_ids = st.session_state.current_db['SK_ID_CURR'].tolist()
    if id_client not in valids_ids:
        st.warning('This ID does not exist')
    else:
        client = st.session_state.current_db[st.session_state.current_db['SK_ID_CURR'] == id_client]#.iloc[0]
        
        ## INFO CLIENT ##
        with st.session_state.client_info_placeholder.container():
            # with st.expander("Client main infos", expanded=True):
            with st.spinner(text='Loading...'):
                display_info_client(client) 
            with st.expander('Find features compared here'):
                display_compare_client(id_client, st.session_state.current_db, features_to_comp)

        ## SCORE RESULT CLIENT ##
        # with st.session_state.client_score_placeholder.container():
        #     # with st.expander("Client score", expanded=True):
        #     with st.spinner(text='Loading...'):
        #         display_score_client(client.iloc[0])


        ## COMPARISON WITH OTHERS CLIENTS ## 
        # with st.session_state.client_compare_placeholder.container(): 

def display_info_client(client):
    
    to_disp = client.iloc[0]

    st.markdown(f'## Client n°{to_disp.SK_ID_CURR}')
    st.markdown(f'<p style="text-align: center; font-size: 60px ; line-height: 30px "><br/>{to_disp.AMT_CREDIT}<br/></p>', True)
    st.markdown(f'<p style="text-align: center; font-size: 20px ; font-family: Candara">requested<br/></p>', True)

    res = get_prediction_client(to_disp)
    expected_value = res.json()['expected'] ; shap_value = res.json()['shap_val']
    feature_names = res.json()['col'] ; client_values = res.json()['client']

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = res.json()['1']*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default risk %"},
        gauge = {'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD_MODEL*100}}))
    st.plotly_chart(fig, True)

    if res.json()['1'] >= THRESHOLD_MODEL:
        st.markdown('<p style="text-align:center; font_size:30px; line-height: 10px ; font-family:Candara;">Credit refused</p>',True)
    else:
        st.markdown('<p style="text-align:center; font_size:30px; line-height: 10px ; font-family:Candara;">Credit accepted</p>',True)

    with st.expander('Find here all the infos about this client', False):
        st.write(client.append(pd.Series(), ignore_index=True))
    
    with st.expander('Find here more details about the decision'):
        # st_shap(shap.force_plot(expected_value, np.array(json.loads(shap_value)), plot_cmap=colors, feature_names=json.loads(feature_names)))
        # shap.plots._waterfall.waterfall_legacy(expected_value, np.array(json.loads(shap_value)),feature_names=json.loads(feature_names), show=True)

        shap.decision_plot(base_value = expected_value, shap_values = np.array(json.loads(shap_value)), 
        features=np.array(json.loads(client_values)), feature_names=json.loads(feature_names), 
        axis_color='white', y_demarc_color='white', # y_demarc_color='white',
        show=False, 
        plot_color=cmap, new_base_value=THRESHOLD_MODEL)
        fig=plt.gcf()
        ax = plt.gca()
        ax.set_xlabel('Model output value', fontsize=15, color='white')
        ax.spines['bottom'].set_color('white')
        # ax.set_facecolor('white')
        # fig.set_edgecolor('white')
        fig.set_facecolor(paper_color)
        fig.set_figheight(6)
        fig.set_figwidth(10)
        fig.suptitle('Explanation of the score', fontsize=12, color='white')
        
        st.pyplot(fig, clear_figure=True, transparent=True, bbox_inches='tight' )

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     ...
    #     # st.write('')
    #     # st.markdown(f'**Contract type:** {client.NAME_CONTRACT_TYPE}')
    #     # st.markdown(f'**Gender:** {client.CODE_GENDER}')
    #     # st.markdown(f'**Income type:** {client.NAME_INCOME_TYPE}')
        

    # with col2:
    #     ...
    #     # st.markdown(f'## {int(client.AMT_CREDIT)}')
    #     # st.caption('Total credit')
    #     # st.markdown(f'**Age:** {int(client.DAYS_BIRTH/-365)}')
    #     # st.markdown(f'**Education:** {client.NAME_EDUCATION_TYPE}')

    # with col3:
    #     ...
    #     # st.write('')
    #     # st.markdown(f'**Annuity:** {client.AMT_ANNUITY}')
    #     # st.markdown(f'**Nb of childs:** {client.CNT_CHILDREN}')
    #     # st.markdown(f'**Family status:** {client.NAME_FAMILY_STATUS}')

def display_score_client(client):

    return
    #TODO Change to correspond to the new format received


def display_compare_client(client_id: int, db_selected: pd.DataFrame, features_to_comp: list = None):

    with st.spinner(text='Loading...'):
        
        client = db_selected[db_selected.SK_ID_CURR == client_id].copy(deep=True)
        res = get_prediction_client(client.iloc[0])

        if res.json()['1'] >= THRESHOLD_MODEL:
            client['TARGET'] = 1
        else:
            client['TARGET'] = 0

        db = st.session_state.db_train

        # db = add_predicted_target_to_db(db_selected.copy(deep=True))
        # client = db_selected[db_selected.SK_ID_CURR == client_id].copy(deep=True)

        # Some preproc used for the prediction
        # Handling outliers
        client = handle_outliers(client.copy(deep=True))
        db = handle_outliers(db.copy(deep=True))

        # Add features
        client = adding_features(client.copy(deep=True))
        db = adding_features(db.copy(deep=True))
        
        if(features_to_comp == None):
            features_to_comp = ['CODE_GENDER', 'AMT_ANNUITY', 'AMT_CREDIT']
        
        db_accepted = db[db.TARGET==0]
        db_refused = db[db.TARGET==1]

        nb_rows = len(features_to_comp)//2 + len(features_to_comp)%2
        nb_cols = 2
        fig = make_subplots(rows= nb_rows, cols=nb_cols, 
                            horizontal_spacing=0.25/nb_cols, vertical_spacing=0.5/nb_rows,
                            figure= go.Figure(
                                layout=go.Layout(width = 600*len(features_to_comp)/2, height = 500*len(features_to_comp)/2,
                                font=dict(color=font_color, family=font_family, size=font_size),
                                # barmode='overlay'
                                )))

        for ind, col in enumerate(features_to_comp):

            # Managing the subplot grid
            row_number = ind//2 +1 ; col_number = ind%2 +1
            if row_number == 1 and col_number  == 1:
                showlegend = True
            else:
                showlegend = False

            # In case of a NaN in the for this client feature
            if client[col].isna().iloc[0]:
                st.warning(f'No info for {col} for client {client_id}')
                continue

            # In case of NaN in the database for the specific feature
            to_show_accepted = db_accepted[col].dropna(axis=0)
            to_show_refused = db_refused[col].dropna(axis=0)          

            if is_numeric_dtype(db[col]):
                bins = int(1 + 3.322*np.log(len(db[col].unique())))
                bin_size = (db[col].max() - db[col].min())/bins

                fig_tmp = ff.create_distplot([to_show_accepted, to_show_refused],
                             show_rug = False, show_hist=False, bin_size = bin_size, histnorm='probability',
                             colors=colors, group_labels=['Accepted', 'Refused'])

                fig.add_trace(
                    go.Scatter(fig_tmp['data'][0], showlegend=showlegend),
                    row=row_number, col=col_number,
                    
                )

                fig.add_trace(
                    go.Scatter(fig_tmp['data'][1], showlegend=showlegend),
                    row=row_number, col=col_number,
                )

                fig.add_vline(
                            x=client[col].values[0],
                            line_width = 3,
                            line_dash = 'dash',
                            annotation_text="Client", annotation_position="top",
                            line_color='white',
                            row=row_number, col=col_number,
                        )

                fig.update_xaxes(title_text=col, gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number,)
                fig.update_yaxes(title_text='Count', gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number,)

            elif is_string_dtype(db[col]):

                fig.add_trace(
                    go.Histogram(
                        histfunc='count', 
                        x=db_accepted[col], 
                        name='Accepted',
                        marker={'color':colors[0]},
                        bingroup=1, # Allow same binarization
                        showlegend=showlegend
                    ),
                    row=row_number, col=col_number,
                )
                fig.add_trace(
                    go.Histogram(
                        histfunc='count', 
                        x=db_refused[col], 
                        name='Refused',
                        marker={'color':colors[1]},
                        bingroup=1,
                        showlegend=showlegend
                    ),
                    row=row_number, col=col_number,
                )

                fig.add_vline(
                    x=client[col].values[0],
                    line_width = 3,
                    line_dash = 'dash', 
                    line_color='white',
                    row=row_number, col=col_number,
                )

                fig.update_xaxes(title_text=col, gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number,)
                fig.update_yaxes(title_text='Count', gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number,)
        
        st.plotly_chart(fig, True)

def display_info_db(df: pd.DataFrame, name_db: str):
    st.markdown(f'<p style="text-align: center; font-size: 60px ; line-height: 100px">Main infos of this database<br/></p>', True)

    st.markdown(f'<p style=" font-size: 30px ; line-height: 35px">The database {name_db} contains: <br> \
    <ul style="list-style-type: circle;"> \
    <li style="font-size: 20px"> {df.shape[0]} clients </li> \
    <li style="font-size: 20px"> {df.shape[1]} differents features for each</li> \
    </ul> \
    <br/></p>', True)

    st.markdown('## Sample of the database')
    with st.expander("Click to show/hide the samples", True):
        st.write(df.sample(5))

    st.markdown('## Some clients features')

    with st.expander("Click to show/hide the infos", False):

        db = handle_outliers(df.copy(deep=True))
        db['AGE'] = db.DAYS_BIRTH/(365)

        infos_to_disp = ['AGE', 'CODE_GENDER', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']

        nb_rows = len(infos_to_disp)//2 + len(infos_to_disp)%2
        nb_cols = 2

        fig = make_subplots(rows= nb_rows, cols=nb_cols, 
                            horizontal_spacing=0.3/nb_cols, vertical_spacing=0.3/nb_rows,
                            figure= go.Figure(
                                layout=go.Layout(width = 600*len(infos_to_disp)/2, height = 500*len(infos_to_disp)/2,
                                font=dict(color=font_color, size=font_size, family='serif'),
                                # barmode='overlay'
                                )))

        for ind, col in enumerate(infos_to_disp):
            to_disp = db[col]
            row_number = ind//2 +1 ; col_number = ind%2 +1

            if is_numeric_dtype(to_disp):
                bins = int(1 + 3.322*np.log(len(db[col].unique())))
                bin_size = (db[col].max() - db[col].min())/bins

                fig_tmp = ff.create_distplot([to_disp.dropna()],
                                        show_rug = False, show_hist=False, bin_size = bin_size, histnorm='probability',
                                        colors=['#5f9fe3'], group_labels = ['Info'])

                fig.add_trace(
                    go.Scatter(fig_tmp['data'][0], showlegend=False),
                    row=row_number, col=col_number,
                )
                
                try:
                    dataMean = round(to_disp.dropna().mean(), 2)
                    dataMedian = round(to_disp.dropna().median(), 2)
                    dataStd = round(to_disp.dropna().std(),  2)
                except Exception as e:
                    print(f'Erreur: {e}.')


                fig.add_annotation(
                    go.layout.Annotation(
                        text=f'Mean: {dataMean}<br>Median {dataMedian}<br>Std: {dataStd}',
                        align='left',
                        showarrow=False,
                        xref='x domain',
                        yref='y domain',
                        y=0.98,x=0.99,
                        bgcolor='#65666a',
                        bordercolor='#65666a',
                        borderwidth=1, font_family='serif'
                    ),
                    row=row_number, col=col_number
                )
                
                fig.update_xaxes(title_text=col, gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number)
                fig.update_yaxes(title_text='Count', gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number)

            elif is_string_dtype(to_disp):
                    fig.add_trace(
                        go.Histogram(
                            histfunc='count', 
                            x=to_disp, 
                            # name='Accepted',
                            marker={'color':'#5f9fe3'},
                            bingroup=1, # Allow same binarization
                            showlegend=False
                        ),
                        row=row_number, col=col_number,
                    )
                    fig.update_xaxes(title_text=col, gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number,)
                    fig.update_yaxes(title_text='Count', gridcolor='#65666a', zerolinecolor='#65666a', gridwidth=2, row=row_number, col=col_number,)
            
            
        st.plotly_chart(fig,True)

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

# --- Not used --- #


def get_prediction_db(df):
    to_send = df.to_json()
    res = requests.post(f"http://localhost:8080/predict_db", json=to_send)
    return res

def add_predicted_target_to_db(original_db):
    db = original_db.copy(deep=True)
    
    res = get_prediction_db(db)
    target = []
    dict_res = res.json() # Our json results in a dict format

    for ind in dict_res.keys(): # We go through the result of each line
        target.append(1 if dict_res[ind]['1'] >= THRESHOLD_MODEL else 0)

    original_db['target'] = target
    return original_db

import shap
import streamlit.components.v1 as components # For shap


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)