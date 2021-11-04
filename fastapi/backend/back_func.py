import pandas as pd
import numpy as np
import pickle
import shap

from sklearn.preprocessing import LabelEncoder

# Load all the neccessary objects
path_rf = './backend/rf_objects/'

model = pickle.load(open(path_rf+'rf_best', 'rb'))
scaler_rf = pickle.load(open(path_rf+'scaler_rf', 'rb'))
imputer_rf = pickle.load(open(path_rf+'imputer_rf', 'rb'))
le_rf = pickle.load(open(path_rf+'le_rf', 'rb'))
one_hot_rf = pickle.load(open(path_rf+'one_hot_rf', 'rb'))
to_drop = pickle.load(open(path_rf+'to_drop', 'rb')) 

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

def get_prediction(sample_client, explained = False):

    # Preprocess the data

    # 1. Handle Outliers
    sample_client = handle_outliers(sample_client)

    # 2. Encode
    # Label encoding
    le = LabelEncoder()
    for col in le_rf:
        le.fit(sample_client[col])
        sample_client[col] = le.transform(sample_client[col])

    # One-Hot encoding of categorical variables
    for col in one_hot_rf.feature_names_in_: # Check if all the type of the columns are corrects
        sample_client[col] = sample_client[col].astype('object')

    cat_df = one_hot_rf.feature_names_in_

    cat_df_oh = pd.DataFrame(data = one_hot_rf.transform(sample_client.select_dtypes('object')).astype('int'), columns=one_hot_rf.get_feature_names_out())
    cat_df_oh.drop(cat_df_oh.columns[cat_df_oh.columns.str.contains('nan')], axis=1, inplace=True) # The one hot considers the value 'nan' as a category, we delete them

    sample_client.drop(cat_df, axis=1, inplace=True) # We remove the object features
    # sample_client = pd.concat([sample_client, cat_df_oh], axis=1) # We add the one hot-encoded features
    
    sample_client[cat_df_oh.columns] = cat_df_oh.values  # pd.concat have unexpected behavior

    # Quick and Dirty solution ; TODO: Find why the pd.concat change the type of DAYS_EMPLOYED_ANOM from bool to object
    # Not the first bug I see using pd.concat
    # sample_client.DAYS_EMPLOYED_ANOM = sample_client.DAYS_EMPLOYED_ANOM.astype('bool')

    # 3. Impute NaNs 
    imputer = imputer_rf

    col_num = sample_client.select_dtypes(include=np.number).columns
    sample_client[col_num] = imputer.transform(sample_client[col_num])
    
    # 4. Add features
    sample_client = adding_features(sample_client)
    
    # 5. Delete low variance features
    sample_client.drop(to_drop, axis=1, inplace=True)

    before_scaling = sample_client.copy(deep=True)

    # 6. Scale datas
    col_num = sample_client.select_dtypes(include=np.number).columns
    sample_client[col_num] = scaler_rf.transform(sample_client[col_num])

    # Make the prediction
    prediction = model.predict_proba(sample_client)

    if explained:
        

        #Explainability
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(sample_client)
        feature_names = sample_client.columns
        return prediction,  explainer, shap_value, feature_names, before_scaling
    else:
        return prediction