from fastapi import FastAPI, Request
import uvicorn
import json

# from pydantic import BaseModel

from back_func import *


app = FastAPI()

@app.get("/")
def hello_world():
    return {"msg": "Welcome on this dashboard !"}

@app.post('/predict_client')
async def predict_client(request: Request):
    received = await request.json()
    json_data = json.loads(received)
    client = pd.DataFrame(data=np.array(list(json_data.values())).reshape(1,-1), columns=list(json_data.keys()))

    to_change_type = pd.read_csv('./backend/sample.csv') # Load examples to have to right 

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



# @app.post('/predict_db')
# async def predict_db(request: Request):
#     received = await request.json()
#     db = pd.read_json(received)

#     to_change_type = pd.read_csv('./frontend/application_test.csv') # Load examples to have the right type 

#     # Changing type to make it corresponds to the type of the value seen during fit
#     for col in db.columns:
#         db[col] = db[col].astype(to_change_type[col].dtype) 

#     res = get_prediction(db)

#     to_post = {}

#     for ind, score in enumerate(res):
#         to_post[str(ind)] = {'0':score[0], '1':score[1]}

#     return to_post

if __name__ == "__main__":
    uvicorn.run("main:app", reload=False, host="0.0.0.0", port=8080)

    