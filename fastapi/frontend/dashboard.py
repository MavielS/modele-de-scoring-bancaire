import streamlit as st
import streamlit.components.v1 as components # For shap

st.set_page_config(page_title='Credit Scoring App', layout='wide')

import numpy as np
import pandas as pd 
import os

import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap, to_rgba
cmap=LinearSegmentedColormap.from_list('rg',['#12c43a', 'w',  '#e23f3f'], N=256) 

import shap

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

import json

from pandas.api.types import is_numeric_dtype, is_string_dtype


from front_func import *

THRESHOLD_MODEL = 0.45
PATH_CSV = 'frontend/application_test.csv'
PATH_ICON = 'frontend/icon.png'
SIZE_PARIS = 10000
SIZE_MARSEILLE = 7000

colors = ['#12c43a', '#e23f3f'] # Red - Green
paper_color = '#0e1117' ; plot_color = '#32363E'
font_color = 'white' ; font_family='Lato' ; font_size=20

# Load the train dataset

# set variables in session state

if 'db_train' not in st.session_state:
    st.session_state.db_train = fetch_train('frontend/sample_train.csv')


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

def add_predicted_target_to_db(original_db):
    db = original_db.copy(deep=True)
    
    res = get_prediction_db(db)
    target = []
    dict_res = res.json() # Our json results in a dict format

    for ind in dict_res.keys(): # We go through the result of each line
        target.append(1 if dict_res[ind]['1'] >= THRESHOLD_MODEL else 0)

    original_db['target'] = target
    return original_db

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
    default=['CODE_GENDER', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL'])

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
 
