# Databricks notebook source
# MAGIC %sh
# MAGIC # we need to install into the OS tier in some MLR / Gradio version combos
# MAGIC /databricks/python/bin/pip install fastapi gradio uvicorn pypdf

# COMMAND ----------

# MLR 14.3 and gradio 4.24.0 we can do this
# this is preferable as install to root env as per above requires resetting cluster if we wannt try different versions
# rather than just detach / reattach
%pip install gradio langchainhub==0.1.15 langchain==0.1.13 databricks-vectorsearch==0.23 databricks-sql-connector==3.1.1 flashrank
dbutils.library.restartPython()

# COMMAND ----------

import os

# First we setup the starting location and find the uri
server_port = 8501
os.environ['DB_APP_PORT'] = f'{server_port}'

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
org_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")

# AWS
aws_real_url = f"https://dbc-dp-{org_id}.cloud.databricks.com/driver-proxy/o/{org_id}/{cluster_id}/{server_port}/"
print(f"try this uri if AWS: {aws_real_url}")

# Azure
azure_real_uri = f"https://adb-dp-{org_id}.11.azuredatabricks.net/driver-proxy/o/{org_id}/{cluster_id}/{server_port}"
print(f"try this uri if Azure: {azure_real_uri}")

# COMMAND ----------

workspace_url

# COMMAND ----------

# we can start gradio this way
os.environ['GRADIO_SERVER_NAME'] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = f'{server_port}'

# Create a secret first with the utils notebook then use that here
os.environ['DATABRICKS_HOST'] = f'https://{workspace_url}'
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope='pclscope', key='pcltoken')

# choose the right path_format
os.environ['GRADIO_ROOT_PATH'] = f"https://dbc-dp-{org_id}.cloud.databricks.com/driver-proxy/o/{org_id}/{cluster_id}/{server_port}/"

# COMMAND ----------

!python3 app/main.py

# COMMAND ----------

!python3 app/main2.py

# COMMAND ----------


