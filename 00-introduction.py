# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy LLM Chatbot with the Databricks Data Intelligence Platform and DBRX Instruct
# MAGIC
# MAGIC In this tutorial, you will learn how to build your own Chatbot Assistant to help your end users answer questions about your product (in this case, we'll use Databricks as an example), using Retrieval Augmented Generation (RAG), Databricks State of the Art LLM DBRX Instruct Foundation Model and Vector Search.
# MAGIC
# MAGIC - The **Databricks Data Intelligence Platform** is a revolutionary system that employs AI models to deeply understand the semantics of enterprise data, enabling democratization of data and AI across an organization. It is rooted in a lakehouse, unifying data and governance, and uses a generative AI model to understand the semantics of your data, simplifying the end-to-end experience of using data and AI. This platform allows you to securely get insights in natural language, optimize data automatically, and deploy and manage models all the way through production, thereby simplifying your data architecture for all your workloads.
# MAGIC
# MAGIC - **DBRX Instruct** is a state-of-the-art mixture of experts (MoE) language model developed by Databricks, which outperforms other open-source models on industry benchmarks. It excels at a broad set of natural language tasks such as text summarization, question-answering, extraction, and coding. The model is highly efficient for inference due to its MoE architecture, and it's available to customers via Foundation Model APIs, the Databricks Marketplace, and the Hugging Face Hub.
# MAGIC
# MAGIC - **Retrieval Augmented Generation (RAG)** is a generative AI design pattern that combines a large language model (LLM) with external knowledge retrieval, improving the accuracy and quality of applications by providing your data as context to the LLM at inference time. RAG is commonly used in applications like chatbots, where it can provide answers from a variety of sources such as a Confluence wiki, documentation, or other sources. 
# MAGIC
# MAGIC - **Vector Search** is Databricks' native Vector Database offering, designed to handle a variety of search index use cases. It allows users to create vector indexes from their data and query them in real-time, which is particularly useful for applications like Retrieval Augmented Generation (RAG), document QA, and semantic search. The service is serverless and scalable, and it integrates with other Databricks tools like Delta Sync for automatic data synchronization and Model Serving for embedding generation.
# MAGIC
# MAGIC ## Expected Learning Outcomes
# MAGIC You will learn how to:
# MAGIC 1. Prepare your document dataset, creating **text chunks** from documentation pages
# MAGIC 2. Leverage Databricks Embedding Foundation Model to compute the **chunk embeddings**
# MAGIC 3. Create your **Vector Seach index** against which you would send queries to find relevant documents
# MAGIC 4. Build your **langchain model** leveraging Databricks Foundation Model (DBRX Instruct)
# MAGIC 5. **Evaluate** your model chatbot model correctness with MLflow
# MAGIC 6. **Deploy** the chatbot as a Model Serving Endpoint
# MAGIC 7. **Track** your metrics with Lakehouse Monitoring
# MAGIC
# MAGIC ## Implementing RAG with Databricks AI Foundation models
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-0.png?raw=true" style="margin-left: 10px"  width="1100px;">

# COMMAND ----------


