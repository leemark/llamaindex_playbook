#NOTE: your .env file should contain OPENAI_API_KEY='sk-your-key-here'
#get environment variables
from dotenv import load_dotenv
load_dotenv()
#add logging
import logging
import sys
#DEBUG is verbose, INFO is less verbose
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
#load data from directory
documents = SimpleDirectoryReader("data").load_data()
#load data into vector store
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)