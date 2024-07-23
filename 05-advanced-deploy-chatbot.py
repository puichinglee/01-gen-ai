# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Section 5: Deploy our Chatbot Model and enable Online Evaluation Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-0.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's now deploy our model as an endpoint to be able to send real-time queries.
# MAGIC
# MAGIC Once our model is live, we will need to monitor its behaviour to detect potential anomaly and drift over time.
# MAGIC
# MAGIC We won't be able to measure correctness as we don't have ground truth, but we can track model perpelxity and other metrics like professionalism over time.
# MAGIC
# MAGIC This can easily be done by turning on your Model Endpoint inference table, automatically saving every query input and output as one of your Delta Lake tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0/ Install required external libraries, set parameters, define functions

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.18.0 mlflow==2.10.1 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 transformers==4.30.2 torch==1.13.1 "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.4-py3-none-any.whl"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="pcl_vs_endpoint"
catalog = "pcl_catalog"
dbName = db = "rag_chatbot"

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
from typing import Iterator
from mlflow import MlflowClient

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1/ Deploy our Model with Inference tables
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-1.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's start by deploying our model endpoint.
# MAGIC
# MAGIC Simply define the `auto_capture_config` parameter during the deployment (or through the UI) to define the table where the endpoint request payload will automatically be saved.
# MAGIC
# MAGIC Databricks will fill the table for you in the background, as a fully managed service.

# COMMAND ----------

class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models, auto_capture_config = None):
        data = {"name": endpoint_name, "config": {"served_models": served_models, "auto_capture_config": auto_capture_config}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}", allow_error=True)
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"enpoint exists ? {ep}")
      return True

    def create_endpoint_if_not_exists(self, endpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True, auto_capture_config = None, environment_vars = {}):
      models = [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
            "environment_vars": environment_vars
      }]
      if not self.inference_endpoint_exists(endpoint_name):
        r = self.create_inference_endpoint(endpoint_name, models, auto_capture_config)
      #Make sure we have the proper version deployed
      else:
        ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            self.wait_endpoint_start(endpoint_name)
            ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            model_deployed = ep['pending_config']['served_models'][0]
            print(f"Error with the model deployed: {model_deployed} - state {ep['state']}")
        else:
            model_deployed = ep['config']['served_models'][0]
        if model_deployed['model_version'] != model_version:
          print(f"Current model is version {model_deployed['model_version']}. Updating to {model_version}...")
          u = self.update_model_endpoint(endpoint_name, {"served_models": models})
      if wait_start:
        self.wait_endpoint_start(endpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        if i % 10 == 0:
          print("waiting for endpoint to build model image and start...")
        time.sleep(10)
        i += 1
      ep = self.get_inference_endpoint(endpoint_name)
      if ep['state'].get("ready", None) != "READY":
        print(f"Error creating the endpoint: {ep}")
        
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        r.raise_for_status()
      return r.json()

# COMMAND ----------

import urllib
import json
import mlflow
import requests
import time

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_name = f"{catalog}.{db}.dbdemos_advanced_chatbot_model"
serving_endpoint_name = f"dbdemos_endpoint_advanced_{catalog}_{db}"[:63]
latest_model = client.get_model_version_by_alias(model_name, "prod")

#TODO: use the sdk once model serving is available.
serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": db,
    "table_name_prefix": serving_endpoint_name
    }
environment_vars={"DATABRICKS_TOKEN": "{{secrets/pcl-scope/pcl-token}}"}
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)

# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# DBTITLE 1,Let's send a query to our chatbot
serving_client.query_inference_endpoint(
    serving_endpoint_name,
    {
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"},
            {
                "role": "assistant",
                "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics.",
            },
            {"role": "user", "content": "Does it support streaming?"},
        ]
    },
)

# COMMAND ----------

# DBTITLE 1,function: display_gradio_app
def display_gradio_app(space_name = "databricks-demos-chatbot"):
    displayHTML(f'''<div style="margin: auto; width: 1000px"><iframe src="https://{space_name}.hf.space" frameborder="0" width="1000" height="950" style="margin: auto"></iframe></div>''')

# COMMAND ----------

# DBTITLE 1,Let's try using Gradio as UI
display_gradio_app("databricks-demos-chatbot")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 2/ Online LLM evaluation with Lakehouse Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's analyse and monitor our model.
# MAGIC
# MAGIC Here are the required steps:
# MAGIC * Make sure the Inference Table is enabeld (it was automatically set up in the previous cell)
# MAGIC * Consume all the Inference Table payload, and measure the model answer metrics (perplexity, complexity, etc.)
# MAGIC * Save the results in your metric table. This can first be used to plot the metrics over time
# MAGIC * Leverage Lakehouse Monitoring to analylse the metric evolution over time

# COMMAND ----------

# DBTITLE 1,function: table_exists
#Temp workaround to test if a table exists in shared cluster mode in DBR 14.2 (see SASP-2467)
def table_exists(table_name):
    try:
        spark.table(table_name).isEmpty()
    except:
        return False
    return True

# COMMAND ----------

# DBTITLE 1,function: send_requests_to_endpoint_and_wait_for_payload_to_be_available
from concurrent.futures import ThreadPoolExecutor

def send_requests_to_endpoint_and_wait_for_payload_to_be_available(endpoint_name, question_df, limit=50):
  print(f'Sending {limit} requests to the endpoint {endpoint_name}, this will takes a few seconds...')
  #send some requests
  serving_client = EndpointApiClient()
  def answer_question(question):
    data = {"messages": [{"role": "user", "content": question}]}
    answer = serving_client.query_inference_endpoint(endpoint_name, data)
    return answer[0]

  df_questions = question_df.limit(limit).toPandas()['question']
  with ThreadPoolExecutor(max_workers=5) as executor:
      results = list(executor.map(answer_question, df_questions))
  print(results)

  #Wait for the inference table to be populated
  print('Waiting for the inference to be in the Inference table, this can take a few seconds...')
  from time import sleep
  for i in range(10):
    if table_exists(f'{endpoint_name}_payload') and not spark.table(f'{endpoint_name}_payload').count() < len(df_questions):
      break
    sleep(10)

# COMMAND ----------

#Let's generate some traffic to our endpoint. We send 50 questions and wait for them to be in our inference table
send_requests_to_endpoint_and_wait_for_payload_to_be_available(serving_endpoint_name, spark.table("evaluation_dataset").select('question'), limit=42)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-inference-table.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's analyse what's inside our Inference Table. The Inference Table name can be fetched from the model serving endpoint configuration.
# MAGIC
# MAGIC We will first get the table name and simply run a query to view its content.

# COMMAND ----------

# Set widgets for required parameters for this notebook.
dbutils.widgets.text("endpoint", f"dbdemos_endpoint_advanced_{catalog}_{db}"[:63], label = "Name of Model Serving Endpoint")
endpoint_name = dbutils.widgets.get("endpoint")
if len(endpoint_name) == 0:
    raise Exception("Please fill in the required information for endpoint name.")


# Location to store streaming checkpoint
dbutils.widgets.text("checkpoint_location", f'dbfs:/Volumes/{catalog}/{db}/volume_databricks_documentation/checkpoints/payload_metrics', label = "Checkpoint Location")
checkpoint_location = dbutils.widgets.get("checkpoint_location")

# COMMAND ----------

import requests
from typing import Dict


def get_endpoint_status(endpoint_name: str) -> Dict:
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with text evaluation metrics will be saved to: {processed_table_name}")

payloads = spark.table(payload_table_name).where('status_code == 200').limit(10)
display(payloads)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Unpacking the inference table requests and responses, and computing the LLM metrics
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-1.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC The request and response columns contains your model input and output as a `string`. Note that the format depends on your model definition and can be custom. Inputs are usually represented as JSON with TF format, and the output depends on your model definition.
# MAGIC
# MAGIC Because our model is designed to potentially batch multiple entries, we need to unpack the avlue from the request and response.
# MAGIC
# MAGIC We will use Spark JSON Path annotation to directly access the query and response as string, concatenate the input / ouput together with an `array_zip` and ultimately `explode` the content to have one input / output per line (i.e. unpacking the batches).
# MAGIC
# MAGIC **Make sure you change the following selectors based on your model definition.**
# MAGIC
# MAGIC *Note: This will be easier within the product directly, we provide this notebook to simplify this task for now.*

# COMMAND ----------

# DBTITLE 1,Define the JSON path to extract the input and output values
# The format of the input payloads, following the TF "inputs" serving format with a "query" field.
# Single query input format: {"inputs": [{"query": "User question?"}]}
# INPUT_REQUEST_JSON_PATH = "inputs[*].query"
# Matches the schema returned by the JSON selector (inputs[*].query is an array of string)
# INPUT_JSON_PATH_TYPE = "array<string>"
# KEEP_LAST_QUESTION_ONLY = False

# Example for format: {"dataframe_split": {"columns": ["messages"], "data": [[{"messages": [{"role": "user", "content": "What is Apache Spark?"}, {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, {"role": "user", "content": "Does it support streaming?"}]}]]}}
INPUT_REQUEST_JSON_PATH = "messages[*].content"
INPUT_JSON_PATH_TYPE = "array<string>"
# As we send in history, we only want to evaluate the last history input which is the new question.
KEEP_LAST_QUESTION_ONLY = True

# Answer format: {"predictions": ["answer"]}
#OUTPUT_REQUEST_JSON_PATH = "predictions"
# Matches the schema returned by the JSON selector (predictions is an array of string)
#OUPUT_JSON_PATH_TYPE = "array<string>"

# Answer format: {"predictions": [{"sources": ["https://docs"], "result": "  Yes."}]}
OUTPUT_REQUEST_JSON_PATH = "[*].result"
# Matches the schema returned by the JSON selector (predictions is an array of string)
OUPUT_JSON_PATH_TYPE = "array<string>"

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, pandas_udf, transform, size, element_at

def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str,
                    keep_last_question_only: False) -> DataFrame:
    # Rename the date column and convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumnRenamed("date", "__db_date")
        .withColumn("__db_timestamp", (col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "__db_model_id",
        F.concat(
            col("request_metadata").getItem("model_name"),
            F.lit("_"),
            col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(col("status_code") == "200")

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type))
        .withColumn("response", F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type)))
    
    if keep_last_question_only:
        requests_unpacked = requests_unpacked.withColumn("request", F.array(F.element_at(F.col("request"), -1)))

    # Explode batched requests into individual rows.
    requests_exploded = (requests_unpacked
        .withColumn("__db_request_response", F.explode(F.arrays_zip(col("request").alias("input"), col("response").alias("output"))))
        .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
        )

    return requests_exploded

# Let's try our unpacking function. Make sure input & output columns are not null
display(unpack_requests(payloads, INPUT_REQUEST_JSON_PATH, INPUT_JSON_PATH_TYPE, OUTPUT_REQUEST_JSON_PATH, OUPUT_JSON_PATH_TYPE, KEEP_LAST_QUESTION_ONLY))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute the Input / Output text evaluation metrics (e.g. toxicity, perplexity, readability)
# MAGIC
# MAGIC Now that our input and output are unpacked and available as string, we can compute their metrics. These will be analysed by Lakehouse Monitoring so taht we can understand how these metrics change over time.
# MAGIC
# MAGIC Feel free to add your own custom evaluation metrics here.

# COMMAND ----------

import tiktoken, textstat, evaluate
import pandas as pd


@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))

@pandas_udf("double")
def flesch_kincaid_grade(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.flesch_kincaid_grade(text) for text in texts])
 
@pandas_udf("double")
def automated_readability_index(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.automated_readability_index(text) for text in texts])

@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = evaluate.load("toxicity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = evaluate.load("perplexity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(perplexity.compute(data=texts.fillna(""), model_id="gpt2")["perplexities"]).where(texts.notna(), None)

# COMMAND ----------

def compute_metrics(requests_df: DataFrame, column_to_measure = ["input", "output"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = (
      requests_df.withColumn(f"toxicity({column_name})", compute_toxicity(F.col(column_name)))
                 .withColumn(f"perplexity({column_name})", compute_perplexity(F.col(column_name)))
                 .withColumn(f"token_count({column_name})", compute_num_tokens(F.col(column_name)))
                 .withColumn(f"flesch_kincaid_grade({column_name})", flesch_kincaid_grade(F.col(column_name)))
                 .withColumn(f"automated_readability_index({column_name})", automated_readability_index(F.col(column_name)))
    )
  return requests_df

# Initialize the processed requests table. Turn on CDF (for monitoring) and enable special characters in column names. 
def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    (DeltaTable.createIfNotExists(spark)
        .tableName(table_name)
        .addColumns(requests_with_metrics.schema)
        .property("delta.enableChangeDataFeed", "true")
        .property("delta.columnMapping.mode", "name")
        .execute())

# COMMAND ----------

# MAGIC %md
# MAGIC We can now incrementally consume new payload from the inference table, unpack them, compute metrics and save them to our final processed table.

# COMMAND ----------

from delta.tables import DeltaTable

# Check whether the table exists before proceeding.
DeltaTable.forName(spark, payload_table_name)

# Unpack the requests as a stream.
requests_raw = spark.readStream.table(payload_table_name)
requests_processed = unpack_requests(requests_raw, INPUT_REQUEST_JSON_PATH, INPUT_JSON_PATH_TYPE, OUTPUT_REQUEST_JSON_PATH, OUPUT_JSON_PATH_TYPE, KEEP_LAST_QUESTION_ONLY)

# Drop columns that we don't need for monitoring analysis.
requests_processed = requests_processed.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id")

# Compute text evaluation metrics.
requests_with_metrics = compute_metrics(requests_processed)

# Persist the requests stream, with a defined checkpoint path for this table.
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics)
(requests_with_metrics.writeStream
                      .trigger(availableNow=True)
                      .format("delta")
                      .outputMode("append")
                      .option("checkpointLocation", checkpoint_location)
                      .toTable(processed_table_name).awaitTermination())

# Display the table (with requests and text evaluation metrics) that will be monitored.
display(spark.table(processed_table_name))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Monitor the Inference Table
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Here we create a monitor on our Inference Table by using the `create_monitor` API. If the monitor already exists, we pass the same parameters to `update_monitor`. In steady state, this should result in no change to the monitor.
# MAGIC
# MAGIC Afterwards, we queue a metric refresh so that the monitor analyses the latest processed requests.
# MAGIC

# COMMAND ----------

"""
Optional parameters to control monitoring analysis. For help, use the command help(lm.create_monitor).
"""
GRANULARITIES = ["1 day"]              # Window sizes to analyze data over
SLICING_EXPRS = None                   # Expressions to slice data with

CUSTOM_METRICS = None                  # A list of custom metrics to compute
BASELINE_TABLE = None                  # Baseline table name, if any, for computing baseline drift

# COMMAND ----------

import databricks.lakehouse_monitoring as lm


monitor_params = {
    "profile_type": lm.TimeSeries(
        timestamp_col="__db_timestamp",
        granularities=GRANULARITIES,
    ),
    "output_schema_name": f"{catalog}.{schema}",
    "schedule": None,  # We will refresh the metrics on-demand in this notebook
    "baseline_table_name": BASELINE_TABLE,
    "slicing_exprs": SLICING_EXPRS,
    "custom_metrics": CUSTOM_METRICS,
}

try:
    info = lm.create_monitor(table_name=processed_table_name, **monitor_params)
    print(info)
except Exception as e:
    # Ensure the exception was expected
    assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    lm.update_monitor(table_name=processed_table_name, updated_params=monitor_params)
    # Refresh metrics calculated on the requests table.
    refresh_info = lm.run_refresh(table_name=processed_table_name)
    print(refresh_info)

# COMMAND ----------

monitor = lm.get_monitor(table_name=processed_table_name)
url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/sql/dashboards/{monitor.dashboard_id}'
print(f"You can monitor the performance of your chatbot at {url}")

# COMMAND ----------

dbutils.notebook.exit(monitor)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Congratulations! You have learned how to autoamte GenAI application with Databricks! 
# MAGIC
# MAGIC We have investigated the use of customer LLM metrics to track our Databricks Q&A chatbot model performance over time.
# MAGIC
# MAGIC Note that for a real use case, you would likely want to add a human feedback loop, reviewing where your model doesn't perform well (e.g. by providing your customer a simple way to flag incorrect answers).
# MAGIC
# MAGIC This is also a good opportunity to either improve your documentation or adjust your prompt, and ultimately add the corecct ansewr to your evaluation dataset.

# COMMAND ----------


