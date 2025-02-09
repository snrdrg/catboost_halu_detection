from fastapi import FastAPI, Request, Response, status
from pydantic import BaseModel, ValidationError, root_validator
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats

import torch
import torch.nn.functional as F

import json
import os
import sqlite3
import uuid

def dists(x):
    return np.absolute(np.pad(x[1:], (0, 1), mode='constant') - x)

def entropy_out(x):
    t = 0.65 #0.65
    return np.unique(np.absolute(x) > t, return_counts = True)[1][1]/x.shape[0]

def compose_features_means(x):
    return np.array((np.mean(x[1]), np.mean(x[2]), np.mean(x[3]), np.mean(x[4]), np.mean(x[5]), np.mean(x[6]), entropy_out(x[5])))

def compose_features_medians(x):
    return np.array((np.median(x[1]), np.median(x[2]), np.median(x[3]), np.median(x[4]), np.median(x[5]), np.median(x[6])))

def compose_features_stds(x):
    return np.array((np.std(x[1]), np.std(x[2]), np.std(x[3]), np.std(x[4]), np.std(x[5]), np.std(x[6])))

def compose_features(x):
    result = np.concatenate((compose_features_means(x), compose_features_medians(x)))
    return np.concatenate((result, compose_features_stds(x)))

def prep_data(uid):

    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()    
    cursor.execute("SELECT vector FROM cache WHERE session_uuid = ?", (uid,))
    conn.commit()
    conn.close()
    
    logits = [np.array(json.loads(row[0])) for row in cursor.fetchall()]
    print(f'Data for uid {uid} loaded')

    log_max = []
    log_med = []
    log_mean = []
    log_std = []
    log_mode = []
    log_entropy = []
    log_similarity = []
    prev_logits = None
    
    for row in logits:
        probs = F.softmax(torch.tensor(row), dim=-1).numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-12))        
        if prev_logits is not None:
            dist = 1 - cosine(torch.tensor(row), prev_logits[0])
        else:
            dist = 0
        prev_logits = torch.tensor(row)
        
        log_max.append(row.argmax())
        log_med.append(np.median(row))
        log_mean.append(row.mean())
        log_std.append(row.std())
        log_mode.append(stats.mode(row)[0])
        log_entropy.append(entropy)
        log_similarity.append(dist)

    return compose_features([log_max, log_med, log_mean, log_std, log_mode, log_entropy, log_similarity])


class RequestBaseModel(BaseModel):
    @root_validator(pre=True)
    def body_params_case_insensitive(cls, values: dict):
        for field in cls.__fields__:
            in_fields = list(filter(lambda f: f.lower() == field.lower(), values.keys()))
            for in_field in in_fields:
                values[field] = values.pop(in_field)

        return values

class LogVec(BaseModel):
    data: str
    uid: str


class SessionUID(BaseModel):
    uid: str


app = FastAPI()

@app.get("/")
def get_root():
    return ""


@app.get("/predict")
def predict(uid:SessionUID, response: Response):
    model = CatBoostClassifier()
    treshold = 0.3915 #hardcoded, will be moved into config
    try:
        model.load_model('cyber_llama3238b.cbm') #hardcoded, will be moved into config
    except Exception:
        print('Model error')
        return None

    try:
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS(SELECT 1 FROM sessions WHERE session_uuid = ?)", (uid.uid,))
        conn.commit()
        conn.close()
    except Exception:
        print('Error in loading cache')
        return None
    exists = cursor.fetchone()[0]
    if exists:
        X = np.array([prep_data(uid.uid)])
        results = (model.predict_proba(X)[:,1] > treshold).astype(int)[0]
    else:
        return None


@app.post("/update")
def update(vec:LogVec, response: Response):
    try:
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()  
        cursor.execute("INSERT INTO cache (session_uuid, vector) VALUES (?, ?)", (vec.uid, vec.data))
        conn.commit()
        conn.close()        
        return "Success"
    except Exception:
        return "Failed"


@app.get("/startSession")
def startSession(response: Response):
    try:
        new_uuid = str(uuid.uuid4())
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (session_uuid) VALUES (?)", (new_uuid,))
        conn.commit()
        conn.close()
        return new_uuid
    except Exception:
        return "Failed"


@app.get("/endSession")
def endSession(uid:SessionUID, response: Response):
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cache WHERE session_uuid = ?", (uid.uid,))
    cursor.execute("DELETE FROM sessions WHERE session_uuid = ?", (uid.uid,))
    conn.commit()
    conn.close()
    return "Success"




