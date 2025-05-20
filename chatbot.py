import numpy as np
import pandas as pd
import os

import langchain
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq

from typing import List, Tuple
import time
import requests
from bs4 import BeautifulSoup
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]="proj7"
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

graph = Neo4jGraph()

# url = "https://www.buildfastwithai.com/" 
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')
# text = soup.get_text()

# doc = Document(page_content=text)

# splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = splitter.split_documents([doc])

# llm_transformer = LLMGraphTransformer(llm=ChatGroq(model_name="llama-3.3-70b-versatile"))   
# graph_documents = llm_transformer.convert_to_graph_documents(docs)
# graph.add_graph_documents(
#     graph_documents,
#     baseEntityLabel=True,
#     include_source=True
# )

# graph.query(""" 
#   CREATE VECTOR INDEX embeds IF NOT EXISTS
#   FOR (d:Document) ON (d.embedding) 
#   OPTIONS { indexConfig: {
#     `vector.dimensions`: 1536,
#     `vector.similarity_function`: 'cosine'
#   }}"""
# )

# graph.query("""
#     MATCH (doc:Document) 
#     WHERE doc.text IS NOT NULL
#     WITH doc, genai.vector.encode(
#       doc.text, 
#       "OpenAI", 
#         {
#           token: $openAiApiKey,
#           endpoint: $openAiEndpoint
#         }) AS vector
#     CALL db.create.setNodeVectorProperty(doc, "embedding", vector)

#     """,
#     params={"openAiApiKey":os.getenv("OPENAI_API_KEY"),"openAiEndpoint": "https://api.openai.com/v1/embeddings"})

index = Neo4jVector.from_existing_graph( 
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

class Entities(BaseModel):   
    names: List[str] = Field(
        ...
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You will extract all entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {question}"
        )
    ]
)

chain = prompt | llm.with_structured_output(Entities) 

def full_text(input: str) -> str:   # Takes the input query and converts it into full-text query, i.e. query to search the DB to retrieve docs
    ft_query = ""
    words = [ch for ch in remove_lucene_chars(input).split() if ch]    # Removes special characters
    for word in words[:-1]:
        ft_query += f" {word}~2 AND"
    ft_query += f" {words[-1]}~1"    # Allows for misspelling
    return ft_query.strip()  # Converted query

def g_retriever(input: str) -> str: 
    result = ""
    entities = chain.invoke({"question": input})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,  # Retrieves the entity queried as well as its neighbors to the LLM
            {"query": full_text(entity)}
            )
        result += "\n".join([el['output'] for el in response])
    return result

def final_retriever(question: str):
    print(f"Search query: {question}")
    structured_data = g_retriever(question)
    unstructured_data = [ch.page_content for ch in index.similarity_search(question)]
    data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
            """
    return data

template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

temp = PromptTemplate.from_template(template)

def chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer   # Chat history stored

def _format_chat_history(history: List[Tuple[str, str]]) -> str:
    return "\n".join([f"Human: {h}\nAI: {a}" for h, a in history])

search_query = RunnableBranch(
    # Integrate chat history in follow-up question so that the LLM remembers
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | temp | ChatOpenAI(temperature=0) | StrOutputParser(),
    ),
    # If no chat history then treat it as normal question
    RunnableLambda(lambda x : x["question"]),
)

template_final = """
You are an intelligent assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know" and do not make up an answer. In all instances, instead of answer "in this context", answer "in the website", or "in Build Fast with AI's website", as appropriate.
If user asks an irrelevant question, politely redirect the conversation back to the context topic.
Do not overuse "In the website", use it sparingly. The conversation should feel human. Answer like you are giving an answer by yourself instead of referring to a context. Mention referring to a website only if the user prompts so.
Introduce yourself as an assistant affiliated with Build Fast with AI in introductory message and if prompted elsewhere.
Provide user with URL link if required.

Context:
{context}

Question:
{question}

Answer:
"""

prompt_final = ChatPromptTemplate.from_template(template_final)

final_chain = (
    RunnableParallel(
            {
            "context": search_query | final_retriever,   # Context is the follow-up Q + chat history + retrieved documents
            "question": RunnablePassthrough(),    # Current Q
        }
    )
    | prompt_final | llm | StrOutputParser()
)


