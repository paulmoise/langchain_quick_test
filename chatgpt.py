from dotenv import load_dotenv, find_dotenv
import os
import sys
import openai
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI

_ = load_dotenv(find_dotenv())  # to read .env file
openai.api_key = os.environ['OPENAI_API_KEY']
query = sys.argv[1]

print(query)
# loader = TextLoader('data.txt')
loader = DirectoryLoader('.', glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI()))
