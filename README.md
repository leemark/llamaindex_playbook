# LlamaIndex RAG Playbook

## Introduction
This repository contains a collection of basic Python examples utilizing LlamaIndex to showcase various chat interfaces and Retrieval-Augmented Generation (RAG) strategies. Each example is designed to be self-contained and demonstrates a unique aspect of working with RAG and chatbot interfaces.

## Requirements
- Python 3.8+
- [LlamaIndex](https://github.com/run-llama/llama_index) library
- Additional requirements are listed in the `requirements.txt` file.

## Installation
To set up your environment to run these examples, follow these steps:
```bash
git clone https://github.com/leemark/llamaindex_ex.git
cd llamaindex_ex
pip install -r requirements.txt
```

## Usage
To run an example, navigate to its directory and execute the Python script:
```bash
cd path/to/examples
python hello.py
```

## Examples
- `hello.py`: Demonstrates a basic "Hello, World!" example with an index.
- `hello_persist.py`: Showcases how to create a persistent RAG index.
- `local_models_ollama.py`: Provides an example of using local models with LlamaIndex and ollama.
- `streamlit_interface.py`: A simple Streamlit app interface for interacting with the model.
- `rag_web_page.py`: An example of augmenting a single web page with RAG.
- `rag_web_search_brave.py`: An example of augmenting brave api web search results with RAG.

## Contributing
We welcome contributions! If you'd like to improve an example or add a new one, please open a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
If you have any questions or want to discuss the examples further, feel free to reach out.
