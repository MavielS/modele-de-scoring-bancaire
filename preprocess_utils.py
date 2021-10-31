import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def encoding(df, unique_categ=2):
    '''
    Label encode the columns with unique_categ unique categories.
    One-hot encoded the others.
    '''
    
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    
    col_to_le = []

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
          if len(list(df[col].unique())) <= unique_categ: # Label encode columns with unique categories < unique_categ
            le.fit(df[col]) # Train & transform the data
            df[col] = le.transform(df[col])
            # Keep track of how many columns were label encoded
            le_count += 1
            # Keep track of the names of the columns
            col_to_le.append(col)

    print('%d columns were label encoded.' % le_count)

    tmp_shape = df.shape[1]

    # one-hot encoding of categorical variables
    cat_df = df.select_dtypes(include='object')

    oh = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh.fit(cat_df)
    values = oh.transform(cat_df)

    cat_df_oh = pd.DataFrame(data = values.astype('int'), columns=oh.get_feature_names_out())
    cat_df_oh.drop(cat_df_oh.columns[cat_df_oh.columns.str.contains('nan')], axis=1, inplace=True) # The one hot considers the value 'nan' as a category, we delete them

    df.drop(cat_df.columns, axis=1, inplace=True) # We remove the object features
    df = pd.concat([df, cat_df_oh], axis=1) # We add the one hot-encoded features

    print(f'{df.shape[1] - tmp_shape} columns were one-hot encoded.')

    return df, col_to_le, oh

def handle_nans(df, imput=SimpleImputer, strategy='median', **params_imputer):
    
    print(f'Imputing with {imput} \nStrategy: {strategy} ; Params: {params_imputer}')
    imputer = imput(strategy=strategy, **params_imputer)

    col_num = df.select_dtypes(include=np.number).columns
    imputer.fit(df[col_num])
    df[col_num] = imputer.transform(df[col_num])

    return df, imputer

def scaling(train, test, s=MinMaxScaler, **params_scaler):
    print(f'Scaling with {s} \nParams: {params_scaler}')
    if s is None:
        return train, test
    else:
        scaler = s(**params_scaler)

        col_num = train.select_dtypes(include=np.number).columns
        scaler.fit(train[col_num])

        train[col_num] = scaler.transform(train[col_num])
        test[col_num] = scaler.transform(test[col_num])


        return train, test, scaler

        
def sample_data(X_train, y_train, stratify='TARGET', nb_samples=30000):

    stacked = pd.concat([X_train, y_train], axis=1)
    arr_sampled, _ = train_test_split(stacked, train_size=nb_samples, stratify=stacked[stratify])
    
    arr_sampled_df = pd.DataFrame(arr_sampled, columns=stacked.columns)
    X_train_s = arr_sampled_df.drop(stratify, axis=1) # Delete Target columns
    y_train_s = arr_sampled_df.TARGET

    # Checking stratification
    print(X_train_s.shape)
    print(y_train_s.shape)
    print(y_train_s.value_counts()/len(y_train_s))

    return X_train_s, y_train_s