from fastapi import FastAPI

app = FastAPI()


# De quoi j'ai besoin ?
# Renvoie une prédiction + son interprétation pour un client défini (Un endpoint pour chaque id)
# # 
# # Base de données stockée dans streamlit ?

@app.get("/")
def hello_world():
    return {"msg": "Bienvenue !"}

@app.get("/ping")
def pong():
    return {"ping": "pong!"}