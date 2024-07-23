import gradio as gr
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain.schema import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

chat_model = 'databricks-dbrx-instruct' # 'pcl_catalog_rag_chatbot'
embedding_model_name = 'databricks-bge-large-en'

endpoint_name = 'pcl_vs_endpoint'
vs_index_fullname = 'pcl_catalog.rag_chatbot.databricks_documentation_vs_index'

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model
)
embedding_model = DatabricksEmbeddings(endpoint=embedding_model_name)

#vector search configuration
vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=endpoint_name,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
        index, text_column="content", embedding=embedding_model
    ).as_retriever()

#prompt
TEMPLATE = """You are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

def predict(message, history):
    #history_langchain_format = []
    #for human, ai in history:
    #    history_langchain_format.append(HumanMessage(content=human))
    #    history_langchain_format.append(AIMessage(content=ai))
    #history_langchain_format.append(HumanMessage(content=message))
    gpt_response = chain.run(message)#llm(history_langchain_format)
    return gpt_response#.content

app = gr.ChatInterface(predict)

if __name__ == '__main__':

    app.launch()