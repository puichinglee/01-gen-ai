# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC ## Section 4: Evalute the RAG chatbot with LLM-as-a-Judge for automated evaluation
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC We have seen how we can improve our chatbot, adding more advanced capabilities to handle a chat history. 
# MAGIC
# MAGIC As you add capabilities to your model and tune the prompt, it will get harder to evaluate your model performance in a repeatable way. Your new prompt might work well for what you tried to fix, but could impact other questions.
# MAGIC
# MAGIC To solve this issue, we need a repeatable way of testing our model as part of our LLMOps deployment. Evaluating LLMs can be challening as existing benchmarks and metrics cannot measure them comprehensively. Humans are often involved in these tasks (see [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback), but it doesn't scale well: humans are slow and expensive!).
# MAGIC
# MAGIC ### Introducing LLM-as-a-Judge
# MAGIC
# MAGIC We will automate the evaluation process with a trending approach in the LLM community: LLM-as-a-Judge. Faster and cheaper than human evaluation, LLM-as-a-Judge leverages an external agent who judges the generative model predictions given what is expected from it.
# MAGIC
# MAGIC Superior models are typicallly used for such evaluations (e.g. `GPT4` judges `databricks-dbrx-instruct`, or `llama2-70B` judges `llama2-7B`).
# MAGIC
# MAGIC We will explore LLM-as-a-Judge evaluation methods introduced in MLflow 2.9, with its powerful `mlflow.evaluate()` API.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0/ Install required external libraries, set parameters, define functions

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC %pip install --upgrade --force-reinstall flask-sqlalchemy sqlalchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="pcl_vs_endpoint"
catalog = "pcl_catalog"
dbName = db = "rag_chatbot"
volume_folder =  f"/Volumes/{catalog}/{db}/volume_databricks_documentation/evaluation_dataset"

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
# MAGIC ## 1/ Create an external model endpoint with Azure OpenAI as a judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/create-external-endpoint.png?raw=true" style="float:right" width="500px" />
# MAGIC
# MAGIC Databricks Serving Endpoint can be of 3 types:
# MAGIC
# MAGIC * Your own models, deployed as an endpoint (a chatbot model, your custom fine tuned LLM)
# MAGIC * Fully managed, serverless Foundation Models (e.g. DBRX Instruct, llama2, MPT...)
# MAGIC * An external Foundation Model (e.g. Azure OpenAI)
# MAGIC
# MAGIC Let's create a external model endpoint using Azure OpenAI.
# MAGIC
# MAGIC Note that you'll need to change the values with your own Azure OpenAI configuration. Alternatively, you can setup a connection to another provider like OpenAI.
# MAGIC
# MAGIC *Note: If you don't have an Azure OpenAI deployment, this demo will fallback to a Databricks managed llama 2 model. Evaluation won't be as good.* 

# COMMAND ----------

from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

try:
    endpoint_name  = "dbdemos-azure-openai"
    deploy_client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "name": endpoint_name,
                    "external_model": {
                        "name": "gpt-35-turbo",
                        "provider": "openai",
                        "task": "llm/v1/chat",
                        "openai_config": {
                            "openai_api_type": "azure",
                            "openai_api_key": "{{secrets/dbdemos/azure-openai}}", #Replace with your own azure open ai key
                            "openai_deployment_name": "dbdemo-gpt35",
                            "openai_api_base": "https://dbdemos-open-ai.openai.azure.com/",
                            "openai_api_version": "2023-05-15"
                        }
                    }
                }
            ]
        }
    )
except Exception as e:
    if 'RESOURCE_ALREADY_EXISTS' in str(e):
        print('Endpoint already exists')
    else:
        print(f"Couldn't create the external endpoint with Azure OpenAI: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
        endpoint_name = "databricks-llama-2-70b-chat"

#Let's query our external model endpoint
answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is Apache Spark?"}]})
answer_test['choices'][0]['message']['content']

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Offline LLM evaluation
# MAGIC
# MAGIC We will start with offline evaluation, scoring our model before its deployment. This requires a set of questions we want to ask to our model.
# MAGIC
# MAGIC In our case, we are fortuante enough to have a labelled training set (questions + answers) with state-of-the-art technical answers from our Databricks support team. Let's leverage it so we can compre our RAG predicts and ground-truth answers in MLflow.
# MAGIC
# MAGIC *Note: This is optional. We can benefit from the LLM-as-a-Judge appraoch without ground-truth labels. This is typically the case if you want to evalute "live" models answering any customer questions.*

# COMMAND ----------

import requests
import collections
import os
 
def download_file_from_git(dest, owner, repo, path):
    def download_file(url, destination):
      local_filename = url.split('/')[-1]
      # NOTE the stream=True parameter below
      with requests.get(url, stream=True) as r:
          r.raise_for_status()
          print('saving '+destination+'/'+local_filename)
          with open(destination+'/'+local_filename, 'wb') as f:
              for chunk in r.iter_content(chunk_size=8192): 
                  # If you have chunk encoded response uncomment if
                  # and set chunk_size parameter to None.
                  #if chunk: 
                  f.write(chunk)
      return local_filename

    if not os.path.exists(dest):
      os.makedirs(dest)
    from concurrent.futures import ThreadPoolExecutor
    files = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents{path}').json()
    files = [f['download_url'] for f in files if 'NOTICE' not in f['name']]
    def download_to_dest(url):
         download_file(url, dest)
    with ThreadPoolExecutor(max_workers=10) as executor:
        collections.deque(executor.map(download_to_dest, files))

# COMMAND ----------

def upload_pdfs_to_volume(volume_path):
  download_file_from_git(volume_path, "databricks-demos", "dbdemos-dataset", "/llm/databricks-pdf-documentation")

def upload_dataset_to_volume(volume_path):
  download_file_from_git(volume_path, "databricks-demos", "dbdemos-dataset", "/llm/databricks-documentation")

# COMMAND ----------

#Load the eval dataset from the repository to our volume
upload_dataset_to_volume(volume_folder)

# COMMAND ----------

# DBTITLE 1,Prepare our evaluation dataset
spark.sql(f'''
CREATE OR REPLACE TABLE evaluation_dataset AS
  SELECT q.id, q.question, a.answer FROM parquet.`{volume_folder}/training_dataset_question.parquet` AS q
    LEFT JOIN parquet.`{volume_folder}/training_dataset_answer.parquet` AS a
      ON q.id = a.question_id ;''')

display(spark.table('evaluation_dataset'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automated evaluation of our chatbot model registered in Unity Catalog
# MAGIC
# MAGIC Let's retrieve the chatbot model we registered in Unity Catalog and predict answers for each questions in the evaluation set.

# COMMAND ----------

# DBTITLE 1,function: get_latest_model_version
# Helper function
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

import mlflow
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
model_name = f"{catalog}.{db}.dbdemos_advanced_chatbot_model"
model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

@pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        dialog = {"messages": [{"role": "user", "content": question}]}
        return rag_model.invoke(dialog)['result']
    return questions.apply(answer_question)

# COMMAND ----------

df_qa = (spark.read.table('evaluation_dataset')
                  .selectExpr('question as inputs', 'answer as targets')
                  .where("targets is not null")
                  .sample(fraction=0.005, seed=40)) #small sample for interactive demo

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()

display(df_qa_with_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM-as-a-Judge: Automated LLM evaluation with out of the box and custom GenAI metrics
# MAGIC
# MAGIC MLflow 2.8 provides out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC * MLflow will automatically compute relevant task-related metrics. In our case, `model_type = 'question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC * Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric.
# MAGIC * Finally, we can define customer metrics. Here, creativity is the only limit. We will evaluate the `professionalism` of our Q&A chatbot.

# COMMAND ----------

# DBTITLE 1,Customer correctness answer
from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Because we have our labels (answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. Again, this is optional.
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(answer_correctness_metrics)

# COMMAND ----------

# DBTITLE 1,Adding custom professionalim metric
professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

# DBTITLE 1,Start the evaluation run
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name="chatbot_rag") as run:
    eval_results = mlflow.evaluate(data = df_qa_with_preds.toPandas(), # evaluation data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="preds", # prediction column_name from eval_df
                                   targets = "targets",
                                   extra_metrics=[answer_correctness_metrics, professionalism])
    
eval_results.metrics

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3/ Visualisation of our GenAI metrics produced by our GPT4 judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge-mlflow.png?raw=true" style="float: right; margin-left:10px" width="600px">
# MAGIC
# MAGIC You can open your MLflow experiment runs from the Experiments menu on the right.
# MAGIC
# MAGIC From here, you can compare multiple model versions, and filter by correctness to spot where your model doesn't answer well.
# MAGIC
# MAGIC Based on that and depending on the issue, you can either fine tune your prompt, your model fine runing instruction with RLHF, or improve your documentation.
# MAGIC
# MAGIC You can equally plot the evaluation metrics directly from the run, or pulling the data from MLflow.

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")

# COMMAND ----------

# Counting the occurrences of each answer correctness score
px.bar(df_genai_metrics['answer_correctness/v1/score'].value_counts(), title='Answer Correctness Score Distribution')

# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")

# COMMAND ----------

# MAGIC %md
# MAGIC ### This looks good. Let's tag out model as Production ready.
# MAGIC
# MAGIC After reviewing the model correctness and potentially comparing its behaviour to your other previous versions, we can flag our model as ready to be deployed.
# MAGIC
# MAGIC *Note: Evaluation can be automated as part of a MLOps step. Once you deploy a new chatbot version with a new prompt, run the evaluation job and benchmark your model behaviour vs. the previous version.*

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)

# COMMAND ----------


