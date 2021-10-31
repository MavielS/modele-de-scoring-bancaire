from fastapi import FastAPI, Request
import uvicorn
import pickle
import json 
import shap

from pydantic import BaseModel

from back_func import *


app = FastAPI()

# Load all the neccessary objects
path_rf = './backend/rf_objects/'

model = pickle.load(open(path_rf+'rf_best', 'rb'))
scaler_rf = pickle.load(open(path_rf+'scaler_rf', 'rb'))
imputer_rf = pickle.load(open(path_rf+'imputer_rf', 'rb'))
le_rf = pickle.load(open(path_rf+'le_rf', 'rb'))
one_hot_rf = pickle.load(open(path_rf+'one_hot_rf', 'rb'))
to_drop = pickle.load(open(path_rf+'to_drop', 'rb')) 


@app.get("/")
def hello_world():
    return {"msg": "Bienvenue !"}

@app.post('/predict_client')
async def predict_client(request: Request):
    received = await request.json()
    json_data = json.loads(received)
    client = pd.DataFrame(data=np.array(list(json_data.values())).reshape(1,-1), columns=list(json_data.keys()))

    to_change_type = pd.read_csv('./frontend/application_test.csv') # Load examples to have to right 

    # Changing type to make it corresponds to the type of the value seen during fit
    for col in client.columns:
        client[col] = client[col].astype(to_change_type[col].dtype)   

    # Score
    res, explainer, shap_value, feature_names, client = get_prediction(client, True)
    # res = get_prediction(client)


    return {'0': res[0][0], '1':res[0][1],
    'expected':explainer.expected_value[1],
    'shap_val':json.dumps(shap_value[1][0].tolist()),
    'col':json.dumps(feature_names.tolist()),
    'client': json.dumps(client.values.tolist())
    }

@app.post('/predict_db')
async def predict_db(request: Request):
    received = await request.json()
    db = pd.read_json(received)

    to_change_type = pd.read_csv('./frontend/application_test.csv') # Load examples to have the right type 

    # Changing type to make it corresponds to the type of the value seen during fit
    for col in db.columns:
        db[col] = db[col].astype(to_change_type[col].dtype) 

    res = get_prediction(db)

    to_post = {}

    for ind, score in enumerate(res):
        to_post[str(ind)] = {'0':score[0], '1':score[1]}

    return to_post

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


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8080)

    