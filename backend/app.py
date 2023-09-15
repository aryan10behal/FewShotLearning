from flask import Flask, flash, redirect, url_for, render_template, request
import os
from flask_cors import CORS, cross_origin
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
import chromadb
import os
import pathlib
from glob import glob
from typing import List
import openai
import os
import requests
import sys
from num2words import num2words
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv('KEY')


app = Flask(__name__)

UPLOAD_FOLDER = "C:\\Users\\aryanbehal\\Desktop\\hackathon_Bhavesh\\backend\\source_documents"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Model ------------------------------
embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"
model_type = "GPT4All"
model_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
chroma_client = chromadb.PersistentClient(path=persist_directory)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client=chroma_client)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = GPT4All(model=model_path, max_tokens=1000, backend='gptj', n_batch=8, callbacks=[], verbose=False)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# -----------------

# Train Module
persist_directory = "db"
source_directory = "source_documents"
embeddings_model_name = "all-MiniLM-L6-v2"
chunk_size = 2000
chunk_overlap = 250

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#  -------

# ---helper functions

def load_document(file_path: str, file_name) -> List[Document]:
    data = []
    for filename in os.scandir(file_path):
        print(file_name==filename.path)
        print(type(file_name), type(filename.path))
        if filename.is_file() and file_name == filename.path:
            texts = []
            print("Reading ", filename.path)
            if pathlib.Path(filename.path).suffix == ".txt":
                loader = TextLoader(filename.path, encoding="windows-1252")
            elif pathlib.Path(filename.path).suffix == ".pdf":
                loader = PyPDFLoader(filename.path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = text_splitter.split_documents(loader.load())
            data.extend(texts)
    return data

def gpt35(data):
    os.environ['OPENAI_API_KEY'] = 'efe2348f99e14caa94bc64f5a782adb5'
    # os.environ['OPENAI_ENDPOINT'] = 'https://ausopenai.azure-api.net'
    API_KEY = os.getenv("OPENAI_API_KEY") 
    RESOURCE_ENDPOINT = os.getenv("OPENAI_ENDPOINT") 
    # Define the URL to which you want to send the request
    url = 'https://ausopenai.azure-api.net/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview'


    # Send a POST request with the JSON data in the request body
    response = requests.post(url, headers={"api-key": API_KEY, "content-type": "application/json"}, json=data)
    return response

def createJson(content):
    data = {"messages": [
    {"role": "user", "content": "Answer the question: What is the context about? Give a brief summary. using the context: Off the field, Sachin continued to make a difference. He was involved in various philanthropic activities, supporting causes related to education, healthcare, and underprivileged children. His contributions to society were as significant as his contributions to cricket.Chapter 9: The AutobiographyDespite the challenges, Sachin led the team with integrity and determination. He often had to make tough decisions, dropping senior players and nurturing young talent. His leadership may not have resulted in World Cup glory, but it left an indelible mark on the team's ethos.Sachin eventually stepped down from captaincy, allowing others to take up the mantle. He could now concentrate on what he did best—scoring runs and entertaining cricket fans around the world.Throughout the tournament, Sachin played like a man possessed, scoring heavily in every game. His rivalry with the likes of Glenn McGrath, Brett Lee, and Shoaib Akhtar became legendary. But as India reached the final against Australia, Sachin's dream seemed within reach yet tantalizingly out of grasp.Chapter 2: Rise of the Little MasterSachin's journey from the dusty maidans of Mumbai to the hallowed grounds of international cricket was nothing short of a fairy tale. His prowess with the bat was apparent to anyone who watched him play. At the tender age of 16, he made his debut for the Indian national team, facing off against arch-rivals Pakistan."}
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 500
    }
    data["messages"][0]["content"] = content
    return data
# --------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])  
def upload():  
    if request.method == 'POST':  
        f = request.files['file']
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filePath)  
        data = load_document(UPLOAD_FOLDER,filePath)
        if len(data) == 0:
            return "Failure"
        db.add_documents(data)
        db.persist()
        return "Successfully Uploaded File and Trained"
    
@app.route('/query',methods = ['GET'])
def query():
    if request.method == 'GET':
        query = request.args['inputQuery']
        res = qa(query)  
        print("\n\n> Question:", query)
        print("Answer:", res['result'])
        return {"result":res['result']}

@app.route('/queryGpt',methods = ['GET'])
def queryGpt():
    if request.method == 'GET':
        query = request.args['inputQuery']
        print("\n\n> Question:", query)

        x = retriever.get_relevant_documents(query)
        final_context = ""
        for content in x:
            final_context = final_context + "\n\n" + content.page_content

        content = "Answer the question " + query + " using the context: " + final_context
        data = createJson(content)
        result = gpt35(data)
        if result.status_code != 200:
            return{"result":"hagg diya"}
        text = json.loads(result.text)
        text = text['choices'][0]['message']['content']
        return {"result":text}

if __name__ == "__main__":
    app.run(debug=True, threaded=True)



