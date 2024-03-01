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

# load data from web
from llama_index.readers.web import SimpleWebPageReader
url = 'https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6'
documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
 
#load data into vector store
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Agentic RAG is an example of:")
print(response)