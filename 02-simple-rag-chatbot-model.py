# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Section 2: Deploying a RAG chatbot endpoint with Databricks DBRX Instruct Foundation Endpoint
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-0.png?raw=true" style="float: right; width: 500px; margin-left: 10px">
# MAGIC
# MAGIC Our data is ready and we will use the **Databricks-Managed Vector Search Index (pages from Databricks documentation website)** to answer similarity queries, finding documentation related to our user question.
# MAGIC
# MAGIC We will now create a langchain model with an augmented prompt, accessing the Databricks DBRX Instruct model to answer advanced Databricks questions:
# MAGIC 1. Build Langchain retriever
# MAGIC 2. Build chat model using the Databricks DBRX Instruct foundation model
# MAGIC 3. Assemble the RAG chain and deploy chat model as serverless model endpoint, then use Gradio as the UI
# MAGIC
# MAGIC The flow will be the following:
# MAGIC * A user asks a question
# MAGIC * The question is sent to our serverless Chatbot RAG endpoint
# MAGIC * The endpoint computes the embeddings and searches for docs similar to the question, leveraging the Vector Search Index
# MAGIC * The endpoint creates a prompt enriched with the doc
# MAGIC * The prompt is sent to the DBRX Instruct Foundation Model Serving Endpoint
# MAGIC * We display the output to our users

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

# MAGIC %md
# MAGIC ### This demo requires a secret to work:
# MAGIC
# MAGIC Your Model Serving Endpoint needs a secret to authenticate against your Vector Search Index. 
# MAGIC
# MAGIC **Note: If you are using a shared demo workspace and you see that the secret is set up, please do not run these steps and do not override its value**
# MAGIC
# MAGIC * You will need to set up the Databricks CLI on your laptop or using this cluster terminal: `pip install databricks-cli`
# MAGIC * Configure the CLI. You will need your workspace URL and a PAT token from your profile page: `databricks configure`
# MAGIC * Create the dbdemos scope: `databricks secrets create-scope dbdemos`
# MAGIC * Save your service principal secret. It wil be used by the Model Endpoint to authenticate: `databricks secrets put-secret dbdemos rag_sp_token`
# MAGIC
# MAGIC Note: Make sure your service principal has access to the Vector Store Index:
# MAGIC ```
# MAGIC spark.sql('GRANT USAGE ON CATALOG <catalog> TO `<YOUR_SP>`);
# MAGIC spark.sql('GRANT USAGE ON DATABASE <catalog>.<db> TO `<YOUR_SP>`);
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import databricks.sdk.service.catalog as c
# MAGIC WorkspaceClient().grants.update(c.SecurableType.TABLE, <index_name>, 
# MAGIC                                 changes[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="<YOUR_SP>")])
# MAGIC ```

# COMMAND ----------

spark.sql('GRANT USAGE ON CATALOG pcl_catalog TO `puiching.lee@databricks.com`');
spark.sql('GRANT USAGE ON DATABASE pcl_catalog.rag_chatbot TO `puiching.lee@databricks.com`');
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
WorkspaceClient().grants.update(c.SecurableType.TABLE, "pcl_catalog.rag_chatbot.databricks_documentation_vs_index",
changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="puiching.lee@databricks.com")])

# COMMAND ----------

# DBTITLE 1,function: test_demo_permissions
def test_demo_permissions(host, secret_scope, secret_key, vs_endpoint_name, index_name, embedding_endpoint_name = None, managed_embeddings = True):
  error = False
  CSS_REPORT = """
  <style>
  .dbdemos_install{
                      font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji,FontAwesome;
  color: #3b3b3b;
  box-shadow: 0 .15rem 1.15rem 0 rgba(58,59,69,.15)!important;
  padding: 10px 20px 20px 20px;
  margin: 10px;
  font-size: 14px !important;
  }
  .dbdemos_block{
      display: block !important;
      width: 900px;
  }
  .code {
      padding: 5px;
      border: 1px solid #e4e4e4;
      font-family: monospace;
      background-color: #f5f5f5;
      margin: 5px 0px 0px 0px;
      display: inline;
  }
  </style>"""

  def display_error(title, error, color=""):
    displayHTML(f"""{CSS_REPORT}
      <div class="dbdemos_install">
                          <h1 style="color: #eb0707">Configuration error: {title}</h1> 
                            {error}
                        </div>""")
  
  def get_email():
    try:
      return spark.sql('select current_user() as user').collect()[0]['user']
    except:
      return 'Uknown'

  def get_token_error(msg, e):
    return f"""
    {msg}<br/><br/>
    Your model will be served using Databrick Serverless endpoint and needs a Pat Token to authenticate.<br/>
    <strong> This must be saved as a secret to be accessible when the model is deployed.</strong><br/><br/>
    Here is how you can add the Pat Token as a secret available within your notebook and for the model:
    <ul>
    <li>
      first, setup the Databricks CLI on your laptop or using this cluster terminal:
      <div class="code dbdemos_block">pip install databricks-cli</div>
    </li>
    <li> 
      Configure the CLI. You'll need your workspace URL and a PAT token from your profile page
      <div class="code dbdemos_block">databricks configure</div>
    </li>  
    <li>
      Create the dbdemos scope:
      <div class="code dbdemos_block">databricks secrets create-scope dbdemos</div>
    <li>
      Save your service principal secret. It will be used by the Model Endpoint to autenticate. <br/>
      If this is a demo/test, you can use one of your PAT token.
      <div class="code dbdemos_block">databricks secrets put-secret dbdemos rag_sp_token</div>
    </li>
    <li>
      Optional - if someone else created the scope, make sure they give you read access to the secret:
      <div class="code dbdemos_block">databricks secrets put-acl dbdemos '{get_email()}' READ</div>

    </li>  
    </ul>  
    <br/>
    Detailed error trying to access the secret:
      <div class="code dbdemos_block">{e}</div>"""

  try:
    secret = dbutils.secrets.get(secret_scope, secret_key)
    secret_principal = "__UNKNOWN__"
    try:
      from databricks.sdk import WorkspaceClient
      w = WorkspaceClient(token=dbutils.secrets.get(secret_scope, secret_key), host=host)
      secret_principal = w.current_user.me().emails[0].value
    except Exception as e_sp:
      error = True
      display_error(f"Couldn't get the SP identity using the Pat Token saved in your secret", 
                    get_token_error(f"<strong>This likely means that the Pat Token saved in your secret {secret_scope}/{secret_key} is incorrect or expired. Consider replacing it.</strong>", e_sp))
      return
  except Exception as e:
    error = True
    display_error(f"We couldn't access the Pat Token saved in the secret {secret_scope}/{secret_key}", 
                  get_token_error("<strong>This likely means your secret isn't set or not accessible for your user</strong>.", e))
    return
  
  try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=secret, disable_notice=True)
    vs_index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name)
    if embedding_endpoint_name:
      if managed_embeddings:
        from langchain_community.embeddings import DatabricksEmbeddings
        results = vs_index.similarity_search(query_text='What is Apache Spark?', columns=["content"], num_results=1)
      else:
        from langchain_community.embeddings import DatabricksEmbeddings
        embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)
        embeddings = embedding_model.embed_query('What is Apache Spark?')
        results = vs_index.similarity_search(query_vector=embeddings, columns=["content"], num_results=1)

  except Exception as e:
    error = True
    vs_error = f"""
    Why are we getting this error?<br/>
    The model is using the Pat Token saved with the secret {secret_scope}/{secret_key} to access your vector search index '{index_name}' (host:{host}).<br/><br/>
    To do so, the principal owning the Pat Token must have USAGE permission on your schema and READ permission on the index.<br/>
    The principal is the one who generated the token you saved as secret: `{secret_principal}`. <br/>
    <i>Note: Production-grade deployement should to use a Service Principal ID instead.</i><br/>
    <br/>
    Here is how you can fix it:<br/><br/>
    <strong>Make sure your Service Principal has USE privileve on the schema</strong>:
    <div class="code dbdemos_block">
    spark.sql('GRANT USAGE ON CATALOG `{catalog}` TO `{secret_principal}`');<br/>
    spark.sql('GRANT USAGE ON DATABASE `{catalog}`.`{db}` TO `{secret_principal}`');<br/>
    </div>
    <br/>
    <strong>Grant SELECT access to your SP to your index:</strong>
    <div class="code dbdemos_block">
    from databricks.sdk import WorkspaceClient<br/>
    import databricks.sdk.service.catalog as c<br/>
    WorkspaceClient().grants.update(c.SecurableType.TABLE, "{index_name}",<br/>
                                            changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="{secret_principal}")])
    </div>
    <br/>
    <strong>If this is still not working, make sure the value saved in your {secret_scope}/{secret_key} secret is your SP pat token </strong>.<br/>
    <i>Note: if you're using a shared demo workspace, please do not change the secret value if was set to a valid SP value by your admins.</i>

    <br/>
    <br/>
    Detailed error trying to access the endpoint:
    <div class="code dbdemos_block">{str(e)}</div>
    </div>
    """
    if "403" in str(e):
      display_error(f"Permission error on Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
    else:
      display_error(f"Unkown error accessing the Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
  def get_wid():
    try:
      return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('orgId')
    except:
      return None
  if get_wid() in ["5206439413157315", "984752964297111", "1444828305810485", "2556758628403379"]:
    print(f"----------------------------\nYou are in a Shared FE workspace. Please don't override the secret value (it's set to the SP `{secret_principal}`).\n---------------------------")

  if not error:
    print('Secret and permissions seems to be properly setup, you can continue the demo!')

# COMMAND ----------

index_name=f"{catalog}.{db}.databricks_documentation_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

test_demo_permissions(host, secret_scope="pclscope", secret_key="pcltoken", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 1/ Build Langchain retriever
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-1.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's start by building our Langchain retriever.
# MAGIC
# MAGIC It will be in charge of:
# MAGIC * Creating the input question (our Managed Vector Search Index will compute the embeddings for us)
# MAGIC * Calling the Vector Search Index to find similar documents to augment the prompt with
# MAGIC
# MAGIC Databricks Langchain wrapper makes it easy to do it in one step, handling all the underlying logic and API calls for you.

# COMMAND ----------

# DBTITLE 1,Setup authentication for our model
# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("pclscope", "pcltoken")

# COMMAND ----------

# DBTITLE 1,Databricks Embedding Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# Test embedding Langchain model
# NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever()


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("How do I track my Databricks Billing?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2/ Build our chat model to query Databricks DBRX Instruct Foundation Model
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-3.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Our chatbot will be using Databricks DBRX Instruct foundation model to provide answers. DBRX Instruct is a general-purpose LLM, built to develop enterprise grade GenAI applciations, unlocking your use cases with capabilities that were previously limited to closed model APIs.
# MAGIC
# MAGIC According to our measurements, DBRX surpasses GPT-3.5, and it is competitive with Gemini 1.0 Pro. It is an especially capable code model, rivaling specialised models like CodeLLaMA-70B on programming in addition to its strength as a general-purpose LLM.
# MAGIC
# MAGIC Note: multiple type of endpoints or langchain models can be used:
# MAGIC * Databricks Foundation models (what we will use)
# MAGIC * Your fine-tuned model
# MAGIC * An external model provider (e.g. Azure OpenAI)

# COMMAND ----------

# Test Databricks Foundation LLM model
from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)
print(f"Test chat model: {chat_model.predict('What is Apache Spark')}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3/ Assemble the complete RAG Chain
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's now merge the retriever and the model in a single Langchain chain.
# MAGIC
# MAGIC We will use a custom langchain template for our assistant to give a proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.

# COMMAND ----------

# DBTITLE 1,Databricks Assistant Chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """You are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# DBTITLE 1,Let's try our chatbot in the notebook directly
# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"query": "How can I track billing usage on my workspaces?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save our model to Unity Catalog registry
# MAGIC Now that our model is ready, we can register it within out Unity Catalog schema

# COMMAND ----------

# DBTITLE 1,Register our chain to MLFlow
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.dbdemos_chatbot_model"

with mlflow.start_run(run_name="dbdemos_chatbot_rag") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy our Chat Model as a Serverless Model Endpoint
# MAGIC
# MAGIC Our model is saved in Unity Catalog. The last step is to deploy it as a Model Serving Endpoint. We will then be able to send requests from our frontend UI.
# MAGIC
# MAGIC You can search endpoint name on the [Serving Endpoint UI](#/mlflow/endpoints) and visualise its performance.
# MAGIC
# MAGIC We can then run a REST query to try it in Python. As you can see, we send the `test sentence` doc and it returns an embedding representing our document.

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

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = f"{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/pclscope/pcltoken}}" # <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
question = "How can I track billing usage on my workspaces?"

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])

# COMMAND ----------


