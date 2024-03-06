import logging
import sys
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader

def setup_logging():
    """
    Initialize logging configuration to output logs to stdout.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def load_environment_variables():
    """
    Load environment variables from the .env file.
    """
    load_dotenv()

def load_web_data(url):
    """
    Load data from a web page using SimpleWebPageReader.
    :param url: The URL of the web page to load.
    :return: A list of loaded documents.
    """
    return SimpleWebPageReader(html_to_text=True).load_data(urls=[url])

def create_vector_store_index(documents):
    """
    Create a VectorStoreIndex from the loaded documents.
    :param documents: The list of loaded documents.
    :return: The created VectorStoreIndex.
    """
    return VectorStoreIndex.from_documents(documents)

def query_index(index, query):
    """
    Query the VectorStoreIndex using the provided query.
    :param index: The VectorStoreIndex to query.
    :param query: The query string.
    :return: The response from the query engine.
    """
    query_engine = index.as_query_engine()
    return query_engine.query(query)

def main():
    """
    Main function to orchestrate the data loading, indexing, and querying process.
    """
    setup_logging()
    load_environment_variables()

    url = 'https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6'
    documents = load_web_data(url)

    index = create_vector_store_index(documents)

    query = "Agentic RAG is an example of:"
    response = query_index(index, query)
    print(response)

if __name__ == "__main__":
    main()