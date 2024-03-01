#get environment variables
from dotenv import load_dotenv
load_dotenv()
import os

#add logging
import logging
import sys

#DEBUG is verbose, INFO is less verbose
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#the query
my_query = "What is the latest news about llamaindex?"


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools.brave_search import BraveSearchToolSpec

brave_key = os.getenv('BRAVE_API_KEY')
tool_spec = BraveSearchToolSpec(api_key=brave_key)

response = tool_spec.brave_search(query=my_query)

import json

# Extract the text attribute from each Document object
documents = [doc.text for doc in response]

# Parse the JSON string from each document and extract the search results
for document in documents:
    response_data = json.loads(document)  # Parse the JSON string into a Python dictionary

    # Navigate through the dictionary to the relevant part containing search results
    search_results = response_data.get('web', {}).get('results', [])

    # Loop through the search results and print them out
    for result in search_results:
        title = result.get('title')
        url = result.get('url')
        description = result.get('description').replace('<strong>', '').replace('</strong>', '')  # Removing HTML tags
        print(f"Title: {title}\nURL: {url}\nDescription: {description}\n")

# Scrape the content from each URL and add to the vector store
web_page_reader = SimpleWebPageReader(html_to_text=True)
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

# Initialize the list to store documents
all_documents = []

# Set up the session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# Define headers to mimic a web browser
headers = {
    'User-Agent': 'Mozilla/5.0 (compatible; YourBot/1.0; +http://yourwebsite.com/bot.html)'
}

# Loop through the search results to scrape content from each URL
for result in search_results:
    url = result.get('url')
    
    try:
        # Make the request
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status
        
        # Here, we assume that Document can take 'text' as input, you might need to adjust based on the actual API
        doc = Document(text=response.text, url=url)
        all_documents.append(doc)
        
    except requests.exceptions.HTTPError as errh:
        logging.error(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        logging.error(f"OOps: Something Else: {err}")

# Load all the scraped documents into the vector store
index = VectorStoreIndex.from_documents(all_documents)

# Use the index to query with the language model
query_engine = index.as_query_engine()
response = query_engine.query(my_query)
print(response)