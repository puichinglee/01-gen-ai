# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Section 3: Advanced chatbot with message history and filter using Langchain and DBRX Instruct
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-0.png?raw=true" style="float: right; width: 500px; margin-left: 10px">
# MAGIC
# MAGIC Our data is ready and we will use the **Self-Managed Vector Search Index (PDF documents)** to create a more advanced langchain model to perform RAG.
# MAGIC
# MAGIC We will improve our langchain model with the following:
# MAGIC * Build a complete chain supporting a chat history, using Databricks DBRX Instruct input style
# MAGIC * Add a filter to only answer Databricks-related questions
# MAGIC * Compute the embeddings with Databricks BGE models within our chain to query the Self-Managed Vector Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0/ Install required external libraries, set parameters, define functions

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2
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
# MAGIC ## 1/ Build a chain supporting a chat history, using Databricks DBRX Instruct input style
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain

# COMMAND ----------

# DBTITLE 1,Spark Chat Model Prompt
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is Spark?"}))

# COMMAND ----------

# DBTITLE 1,Adding conversation history to the prompt
prompt_with_history_str = """
Your are a Big Data chatbot. Please answer Big Data question only. If you don't know or not related to Big Data, don't answer.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md 
# MAGIC When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"}, 
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.

# COMMAND ----------

# DBTITLE 1,Chat History Extractor Chain
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Let's add a filter on top to only answer Databricks-related questions
# MAGIC
# MAGIC We want our chatbot to be professional and only answer questions related to Databricks. Let's create a small chain and add a first classification step.
# MAGIC
# MAGIC *Note: This is a fairly naive implementation, another solution could be adding a small classification model based on the question embedding, providing faster classification*

# COMMAND ----------

# DBTITLE 1,Databricks Inquiry Classifier
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

is_question_about_databricks_str = """
You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Workspaces, Databricks account and cloud infrastructure setup, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or something from a very different field. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is Databricks?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is Databricks?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Test with a random question
#Return "no" as this isn't about Databricks
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3/ Use LangChain to retrieve documents from the Vector Store
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-1.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's add our LangChain retriever. 
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question embeddings (with Databricks `bge-large-en`)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with
# MAGIC
# MAGIC Databricks LangChain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

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

index_name=f"{catalog}.{db}.databricks_pdf_documentation_self_managed_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(host, secret_scope="dbdemos", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en", managed_embeddings = False)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

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
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is Apache Spark?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history
# MAGIC
# MAGIC We need to retrieve documents related the the last question but also the history.
# MAGIC
# MAGIC One solution is to add a step for our LLM to summarise the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain.

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's put it together
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC
# MAGIC Let's now merge the retriever and the full LangChain chain.
# MAGIC
# MAGIC We will use a custom LangChain template for our assistant to give a proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about Databricks.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# DBTITLE 1,function: display_chat
def display_chat(chat_history, response):
  def user_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #c2efff; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px;">
        {message}
      </div>"""
  def assistant_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #e3f6fc; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
        <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>
        {message}
      </div>"""
  chat_history_html = "".join([user_message_html(m["content"]) if m["role"] == "user" else assistant_message_html(m["content"]) for m in chat_history])
  answer = response["result"].replace('\n', '<br/>')
  sources_html = ("<br/><br/><br/><strong>Sources:</strong><br/> <ul>" + '\n'.join([f"""<li><a href="{s}">{s}</a></li>""" for s in response["sources"]]) + "</ul>") if response["sources"] else ""
  response_html = f"""{answer}{sources_html}"""

  displayHTML(chat_history_html + assistant_message_html(response_html))

# COMMAND ----------

# DBTITLE 1,Asking an out-of-scope question
import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Asking a relevant question
dialog = {
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.dbdemos_advanced_chatbot_model"

with mlflow.start_run(run_name="dbdemos_chatbot_rag") as run:
    #Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
    )

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

# COMMAND ----------


