"""
This script demonstrates how to use the llama_index library to create and query a vector store index.
It loads documents from a directory, creates an index, and allows querying the index.

usage: python hello_persist.py "What is the author's name and job now?"
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding

def main(query):
    try:
        # Load environment variables
        load_dotenv()

        # Configure logging
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        # Configure embedding model
        Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

        # Set up storage directory
        storage_directory = "./storage"

        if not os.path.exists(storage_directory):
            logging.info("Creating new index...")
            documents = SimpleDirectoryReader("data").load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=storage_directory)
        else:
            logging.info("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_directory)
            index = load_index_from_storage(storage_context)

        # Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        print(response)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a vector store index.")
    parser.add_argument("--query", default="What is the author's name and job now?", help="The query to ask the index.")
    args = parser.parse_args()

    main(args.query)