import gradio as gr
import requests

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data}

def call_endpoint(dataset, history):
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/pcl_catalog_rag_chatbot/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    input=[{"query": dataset}]
    ds_dict = {'dataframe_split': input.to_dict(orient='split')} if isinstance(input, pd.DataFrame) else create_tf_serving_json(input)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    result = response.json()
    return ''.join(result['predictions'])

app = gr.ChatInterface(call_endpoint)

if __name__ == '__main__':

    app.launch()