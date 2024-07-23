# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Section 1: Ingest and prep data
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-0.png?raw=true" style="float: right; width: 500px; margin-left: 10px">
# MAGIC
# MAGIC The first step is to ingest data is to ingest and prepare the data before we can make use of our Vector Search index.
# MAGIC
# MAGIC In this notebook, we will ingest our documentation pages and index them with a Vector Search index to help our chatbot provide better answers. 
# MAGIC
# MAGIC Preparing high quality data is key for your chatbot performance. We recommend taking time to implement these next steps with your own dataset.
# MAGIC
# MAGIC For this example, we will use Databricks documentation from [docs.databricks.com](docs.databricks.com):
# MAGIC
# MAGIC **0/ Install required external libraries, set parameters, define functions**
# MAGIC * pip install mlflow, lxml, transformers, unstructured, langchain, llama-index, vectorsearch, pydantic
# MAGIC * import libraries pyspark.sql.functions, pandas, os, mlflow, typing
# MAGIC * define vse, databricks url, catalog and schema names
# MAGIC * define functions:
# MAGIC   * `use_and_create_db`: create catalog, schema
# MAGIC * apply `use_and_create_db`
# MAGIC
# MAGIC **1/ Download the Databricks documentation pages**
# MAGIC * define functions:
# MAGIC   * `table_exists`: check if table exists
# MAGIC   * `download_databricks_documentation_articles`: 
# MAGIC     * fetch xml content from sitemap
# MAGIC     * find all URLs in the xml
# MAGIC     * create dataframe from URLs
# MAGIC     * define pandas udf to fetch HTML content for a batch of URLs
# MAGIC     * define pandas udf to process HTML content and extract text using BeautifulSoup
# MAGIC     * apply both udfs to the URLs dataframe
# MAGIC     * select and filter non-null results
# MAGIC * apply `table_exists` on `raw_documentation` table, where if empty, apply `download_databricks_documentation_articles` and save as delta table `raw_documentation`
# MAGIC
# MAGIC **2/ Split documentation pages into small chunks**
# MAGIC * define max_chunk_size, tokenizer (`openai-gpt`), text_splitter (`RecursiveCharacterTextSplitter` class from langchain), html_splitter (`HTMLHeaderTextSplitter` class from langchain)
# MAGIC * define functions:
# MAGIC   * `split_html_on_h2`: split on H2, but merge small H2 chunks together to avoid too small chunks
# MAGIC * create empty `databricks_documentation` delta table
# MAGIC * create pandas udf to chunk documents using `split_html_on_h2` with spark
# MAGIC * apply udf on `raw_documentation` table, creating a new column `content` (which are the chunks), then save as delta table `databricks_documentation`
# MAGIC
# MAGIC **3/ Create our Vector Search Index with Managed Embeddings and BGE**
# MAGIC * define functions:
# MAGIC   * `endpoint_exists`: check if vse exists
# MAGIC   * `wait_for_vs_endpoint_to_be_ready`: check status of vse
# MAGIC   * `index_exists`: check if embedding index within vse exists
# MAGIC   * `wait_for_index_to_be_ready`: check status of index within vse
# MAGIC * using VectorSearchClient, create vse
# MAGIC * create managed vector search index `databricks_documentation_vs_index` for `databricks_documentation` using vse
# MAGIC * test our index with similarity_search

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-0.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC We will also add Databricks ebook PDFs from the [Databricks resources page](https://www.databricks.com/resources) to our knowledge database:
# MAGIC
# MAGIC **4/ Use AutoLoader to ingest Databricks ebook PDFs and extract their pages**
# MAGIC * define functions:
# MAGIC   * `download_file_from_git`
# MAGIC   * `upload_pdfs_to_volume`
# MAGIC   * `upload_dataset_to_volume`
# MAGIC * apply `upload_pdfs_to_volume`
# MAGIC * use AutoLoader to ingest data from volume path, then save as delta table `pdf_raw`
# MAGIC
# MAGIC **5/ Extract our PDF content as text chunks**
# MAGIC * define functions:
# MAGIC   * `install_ocr_on_nodes`: install ocr libraries
# MAGIC   * `extract_doc_text`: uses `unstructured` and ocr libraries to transform pdf as text
# MAGIC * test `extract_doc_text` on a single PDF file
# MAGIC * create pandas udf `read_as_chunk` to apply `extract_doc_text` across multiple nodes, also wrapping it with `SentenceSplitter` from llama_index (parses text with a preference for complete sentences in a node chunk)
# MAGIC
# MAGIC **6/ Create our Vector Search Index with Self-Managed Embeddings and BGE**
# MAGIC * create empty delta table `databricks_pdf_documentation`
# MAGIC * create pandas udf `get embedding` to compute emebddings using the foundation model endpoint (bge)
# MAGIC * save data to delta table `databricks_pdf_documentation`
# MAGIC   * apply `read_as_chunk` on `pdf_raw`, creating a new column called `content`
# MAGIC   * apply `get_embedding` on `content` column from `pdf_raw`, creating a new column called `embedding`
# MAGIC   * apply `get_embedding` on `content` column from `databricks_documentation` (from databricks documentation web pages above)
# MAGIC * using VectorSearchClient, create vse if not already existing
# MAGIC * create self-managed vector search index `databricks_pdf_documentation_self_managed_vs_index` for `databricks_pdf_documentation` using vse
# MAGIC * test our index with similarity_search

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0/ Install required external libraries, set parameters, define functions

# COMMAND ----------

# DBTITLE 1,pip install
# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,import libraries
from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
from typing import Iterator
from mlflow import MlflowClient

# COMMAND ----------

# DBTITLE 1,parameters
VECTOR_SEARCH_ENDPOINT_NAME="pcl_vs_endpoint"
DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"
catalog = "pcl_catalog"
dbName = db = "rag_chatbot"

# COMMAND ----------

# DBTITLE 1,use_and_create_db
def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{dbName}` """)

# COMMAND ----------

assert catalog not in ['hive_metastore', 'spark_catalog']
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
      if catalog == 'dbdemos':
        spark.sql(f"ALTER CATALOG {catalog} OWNER TO `account users`")
  use_and_create_db(catalog, dbName)

if catalog == 'dbdemos':
  try:
    spark.sql(f"GRANT CREATE, USAGE on DATABASE {catalog}.{dbName} TO `account users`")
    spark.sql(f"ALTER SCHEMA {catalog}.{dbName} OWNER TO `account users`")
  except Exception as e:
    print("Couldn't grant access to the schema to all users:"+str(e))    

print(f"using catalog.database `{catalog}`.`{dbName}`")
spark.sql(f"""USE `{catalog}`.`{dbName}`""") 

spark.sql(f"""CREATE VOLUME IF NOT EXISTS volume_databricks_documentation""")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1/ Download the Databricks documentation pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC First, let's create our raw dataset as a Delta Lake table. 
# MAGIC
# MAGIC We will directly download a few documentation pages from `docs.databricks.com` and save the HTML content.

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

# DBTITLE 1,function: download_databricks_documentation_articles
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.types import StringType
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
#Add retries with backoff to avoid 429 while fetching the doc
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429],
)

def download_databricks_documentation_articles(max_documents=None):
    # Fetch the XML content from sitemap
    response = requests.get(DATABRICKS_SITEMAP_URL)
    root = ET.fromstring(response.content)

    # Find all 'loc' elements (URLs) in the XML
    urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if max_documents:
        urls = urls[:max_documents]

    # Create DataFrame from URLs
    df_urls = spark.createDataFrame(urls, StringType()).toDF("url").repartition(10)

    # Pandas UDF to fetch HTML content for a batch of URLs
    @pandas_udf("string")
    def fetch_html_udf(urls: pd.Series) -> pd.Series:
        adapter = HTTPAdapter(max_retries=retries)
        http = requests.Session()
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        def fetch_html(url):
            try:
                response = http.get(url)
                if response.status_code == 200:
                    return response.content
            except requests.RequestException:
                return None
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = list(executor.map(fetch_html, urls))
        return pd.Series(results)

    # Pandas UDF to process HTML content and extract text
    @pandas_udf("string")
    def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
        def extract_text(html_content):
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                article_div = soup.find("div", itemprop="articleBody")
                if article_div:
                    return str(article_div).strip()
            return None

        return html_contents.apply(extract_text)

    # Apply UDFs to DataFrame
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
    final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))

    # Select and filter non-null results
    final_df = final_df.select("url", "text").filter("text IS NOT NULL").cache()
    if final_df.isEmpty():
      raise Exception("Dataframe is empty, couldn't download Databricks documentation, please check sitemap status.")

    return final_df

# COMMAND ----------

if not table_exists("raw_documentation") or spark.table("raw_documentation").isEmpty():
    doc_articles = download_databricks_documentation_articles() # Download Databricks documentation to a DataFrame 
    doc_articles.write.mode('overwrite').saveAsTable("raw_documentation") #Save them as a raw_documentation table

display(spark.table("raw_documentation").limit(2))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2/ Split documentation pages into small chunks
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-2.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Large language models typically have a maximum input context length, and you would not be able to compute embeddings for very long texts. In addition, the longer your context is, the longer it will take for the model to provide a response.
# MAGIC
# MAGIC Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:
# MAGIC * Split document into small chunks (paragraph, headings, etc.)
# MAGIC * Truncate documents to a fixed length
# MAGIC * The chunk size depends on your content and how you will be using it to craft your prompt. Adding multiple small document chunks in your prompt might give different results than sending one big one
# MAGIC * Split into big chunks and ask a separate LLM to summarise each chunk as a one-off job, for faster live inference
# MAGIC * Create multiple agents to evaluate each bigger document in parallel, and ask a final agent to craft your answer

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC
# MAGIC In this demo, we have big documentation articles, which are too long for the prompt model.
# MAGIC
# MAGIC We will not be able to use multiple documents as RAG context as they would exceed our max input size. Some recent studies also suggest that bigger chunk sizes aren't always better, as the LLMs seem to focus on the beginning and end of your prompt.
# MAGIC
# MAGIC In our case, we will split these articles between HTML `h2` tags, remove HTML and ensure that each chunk is less than 500 tokens using LangChain

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM Window size and Tokenizer
# MAGIC Different models might return different tokens for the same sentence. LLMs are shipped with a `Tokenizer` that you can use to count tokens for a given sentence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer you wil be using here matches your model. Databricks DBRX Instruct uses the same tokenizer as GPT4. We will be using the `transformers` library to count DBRX Instruct tokens with its tokenizer. This will also keep our document token size below our embedding max size (1024).

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
help(RecursiveCharacterTextSplitter)

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2, but merge small h2 chunks together to avoid too small. 
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=500):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # Merge chunks together to add text before h2 and avoid too small docs.
  for c in h2_chunks:
    # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # Discard too small chunks
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]
 
# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the chunks and save them to our Delta Table
# MAGIC
# MAGIC The last step is to apply our UDF to all our documentation text and save them to our `databricks_documentation` table.
# MAGIC
# MAGIC *Note that this part would typically be set up as a production-grade job, running as soon as a new documentation page is updated. This could be set up as a Delta Live Tables pipeline to incrementally consume updates.*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the final databricks_documentation table containing chunks
# MAGIC -- Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# Let's create a user-defined function (UDF) to chunk all our documents with spark
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    
(spark.table("raw_documentation")
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable("databricks_documentation"))

display(spark.table("databricks_documentation"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3/ Create our Vector Search Index with Managed Embeddings and BGE
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-managed-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC Databricks provides multiple types of vector search indexes:
# MAGIC * **Managed embeddings**: You provide a text column and endpoint name, and Databricks will automatically compute the embeddings for us. This is the easier mode to get started with Databricks. 
# MAGIC * **Self Managed embeddings**: You compute the embeddings and save them as a field of your Delta Table. Databricks will then synchronise the index.
# MAGIC * **Direct index**: When you want to use and update the index without having a Delta Table.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC A vector search index uses a **Vector search endpoint** to serve the embeddings (you can think about it as your Vector Search API endpoint). Multiple indexes can use the same endpoint. Let's start by creating one.
# MAGIC
# MAGIC You can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation Models are provided by Databricks and can be used out-of-the-box. 
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC * A **foundation model endpoint**, provided by Databricks (e.g. llama2-70B, MPT, BGE)
# MAGIC * An **external endpoint**, acting as a gateway to an external model (e.g. Azure OpenAI)
# MAGIC * A **custom**, fine-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this demo, we will use the foundation model `BGE` (embeddings) and `llama-70B` (chat).
# MAGIC

# COMMAND ----------

# DBTITLE 1,functions: endpoint
import time

def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,functions: index
def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,Create the Vector Search Endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# DBTITLE 1,Create the managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.databricks_documentation"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.databricks_documentation_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='content', #The column containing our text
    embedding_model_endpoint_name='databricks-bge-large-en' #The embedding endpoint used to create the embeddings
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Test our Vector Search Index by searching for similar content
# MAGIC *Note: `similarity_search` also supports a filters parameter. This is useful to add a security layer to your RAG system, i.e. you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "How can I track billing usage on my workspaces?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 4/ Use AutoLoader to ingest Databricks ebook PDFs and extract their pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Now, let's ingest our PDFs as a Delta Lake table with path urls and content in binary format.
# MAGIC
# MAGIC We will use [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingest new files, making it easy to incrementally consume billions of files from the data lake in various data formats. AutoLaoder easily ingests our unstructured PDF data in binary format.

# COMMAND ----------

# DBTITLE 1,function: download_file_from_git
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

# DBTITLE 1,function: upload to volume
def upload_pdfs_to_volume(volume_path):
  download_file_from_git(volume_path, "databricks-demos", "dbdemos-dataset", "/llm/databricks-pdf-documentation")

def upload_dataset_to_volume(volume_path):
  download_file_from_git(volume_path, "databricks-demos", "dbdemos-dataset", "/llm/databricks-documentation")

# COMMAND ----------

# DBTITLE 1,Our pdf or docx files are available in our Volume (or DBFS)
# List our raw PDF docs
volume_folder =  f"/Volumes/{catalog}/{db}/volume_databricks_documentation"
# Let's upload some pdf files to our volume as example. Change this with your own PDFs / docs.
upload_pdfs_to_volume(volume_folder+"/databricks-pdf")

display(dbutils.fs.ls(volume_folder+"/databricks-pdf"))

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Databricks cloudFiles (Autoloader)
df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load('dbfs:'+volume_folder+"/databricks-pdf"))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_docs')
  .table('pdf_raw').awaitTermination())

display(spark.sql("SELECT * FROM pdf_raw LIMIT 2"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 5/ Extract our PDF content as text chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-2.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC We need to convert the PDF document bytes to text, and extract chunks from their content.
# MAGIC
# MAGIC This part can be tricky as PDFs are hard to work with and can be saved as images, for which we'll used an OCR to extract the text.
# MAGIC Using the `Unstructured` library within a Spark UDF makes it easy to extract text.
# MAGIC
# MAGIC *Note: Your cluster will need a few extra libraries that you would typically install with a cluster init script*
# MAGIC
# MAGIC In this scenario, some PDFs are very large, with a lot of text. We will extract the content and then use llama_index `SentenceSplitter` and ensure that each chunk is not bigger than 500 tokens. 
# MAGIC
# MAGIC Remember that your prompt + answer should stay below the model max window size (4096 for llama2).

# COMMAND ----------

# DBTITLE 1,To extract our PDF,  we'll need to setup libraries in our nodes
#install poppler on the cluster (should be done by init scripts)
def install_ocr_on_nodes():
    """
    install poppler on the cluster (should be done by init scripts)
    """
    # from pyspark.sql import SparkSession
    import subprocess
    num_workers = max(1,int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))
    command = "sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get purge && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y" 
    def run_subprocess(command):
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            return output.decode()
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred installing OCR libs:"+ e.output.decode())
    #install on the driver
    run_subprocess(command)
    def run_command(iterator):
        for x in iterator:
            yield run_subprocess(command)
    # spark = SparkSession.builder.getOrCreate()
    data = spark.sparkContext.parallelize(range(num_workers), num_workers) 
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect();
        print("OCR libraries installed")
    except Exception as e:
        print(f"Couldn't install on all node: {e}")
        raise e

# For production use-case, install the libraries at your cluster level with an init script instead. 
install_ocr_on_nodes()

# COMMAND ----------

# DBTITLE 1,Transform PDF as text
from unstructured.partition.auto import partition
import re

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

# DBTITLE 1,Try our text extraction function with a single PDF file
import io
import re
import requests
with requests.get('https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true') as pdf:
  doc = extract_doc_text(pdf.content)  
  print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks great. We will now wrap it with a `text_splitter` to avoid having too big pages, and create a Pandas UDF function to easily scale that across multiple nodes.
# MAGIC
# MAGIC *Note that our PDF text is not clean. To make it nicer, we could use a few extra LLM-based pre-processing steps, asking to remove irrelevant content like the list of chapters, and to only keep the core text.*

# COMMAND ----------

from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 6/ Create our Vector Search Index with Self-Managed Embeddings and BGE
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC We will now show you how to set up a Self-Managed Embeddings index.  
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array<float>`.
# MAGIC
# MAGIC Similar to before, we will be using the foundation model `BGE` (embeddings) and `llama2-70B` (chat).

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model BGE as embedding endpoint
from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
print(embeddings)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_pdf_documentation table containing chunks and embeddings
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Computing the chunk embeddings and saving them to our Delta Table
# MAGIC
# MAGIC Now we need to compute an embedding for all our documentation chunks. Let's create a UDF to compute the embeddings using the foundation model endpoint.

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

volume_folder =  f"/Volumes/{catalog}/{db}/volume_databricks_documentation"

(spark.readStream.table('pdf_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_chunk')
    .table('databricks_pdf_documentation').awaitTermination())

#Let's also add our documentation web page from the simple demo (make sure you run the quickstart demo first)
if table_exists(f'{catalog}.{db}.databricks_documentation'):
  (spark.readStream.option("skipChangeCommits", "true").table('databricks_documentation') #skip changes for more stable demo
      .withColumn('embedding', get_embedding("content"))
      .select('url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/docs_chunks')
    .table('databricks_pdf_documentation').awaitTermination())
  
  display(spark.sql("""SELECT * FROM databricks_pdf_documentation WHERE url like '%.pdf' limit 10"""))

# COMMAND ----------

# DBTITLE 1,Creating the Vector Search endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# DBTITLE 1,Create the Self-Managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.databricks_pdf_documentation"
# Where we want to store our index
vs_index_selfmanaged_fullname = f"{catalog}.{db}.databricks_pdf_documentation_self_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_selfmanaged_fullname):
  print(f"Creating index {vs_index_selfmanaged_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_selfmanaged_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_selfmanaged_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_selfmanaged_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_selfmanaged_fullname).sync()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Test our Vector Search Index by searching for similar content
# MAGIC *Note: `similarity_search` also supports a filters parameter. This is useful to add a security layer to your RAG system, i.e. you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

question = "How can I track billing usage on my workspaces?"

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_selfmanaged_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
print(docs)

# COMMAND ----------


