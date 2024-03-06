"""
This script demonstrates how to use the llama_index library to create and query a vector store index.
It loads documents from a directory, creates an index, and allows querying the index.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def main(data_directory, query):
    try:
        # Load environment variables
        load_dotenv()

        # Configure logging
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        # Load data from directory
        logging.info(f"Loading data from directory: {data_directory}")
        documents = SimpleDirectoryReader(data_directory).load_data()

        # Load data into vector store
        logging.info("Creating vector store index")
        index = VectorStoreIndex.from_documents(documents)

        # Query the index
        query_engine = index.as_query_engine()
        logging.info(f"Querying the index with: {query}")
        response = query_engine.query(query)
        print(response)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and query a vector store index.")
    parser.add_argument("--data_dir", default="data", help="Directory containing the data files.")
    parser.add_argument("--query", default="What did the author do growing up?", help="The query to ask the index.")
    args = parser.parse_args()

    main(args.data_dir, args.query)