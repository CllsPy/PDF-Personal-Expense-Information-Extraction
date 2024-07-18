# 📄 Project Description

The goal of this project is to create an interface using **Gradio** where users can upload PDF files containing personal expense information. Utilizing techniques like Text Embeddings and Generative AI, users will be able to extract relevant information from the uploaded data.

## 🛠️ Steps

![Project Workflow](https://github.com/CllsPy/Generative_AI/assets/96326019/920681bf-d869-4db2-aea6-a14b94ba0c8b)
*source: author*

1. **Get user data in the proper format (PDF)**
2. **Apply Text Embedding to the PDF so that the model (LLaMA) can understand it**
3. **Send the data to a Vector Database (Pinecone)**
4. **Adjust the prompt and other aspects related to the desired response**
5. **Provide the data to the model**
6. **Evaluate the response**

## 📋 Requirements

The chosen dynamic language is **Python**, and the preferred IDE is **Google Colab**. For local implementation, additional steps are needed, which can be found in the documentation of the libraries mentioned in this file. Additionally, access to a **GPU** is required.

### **Packages**
Install the following packages:
```python
import os
import torch
import pinecone
import transformers
import gradio as gr
from pinecone import Pinecone
from torch import cuda, bfloat16
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.llms import HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
```

## 🚀 Launch

Download the Python file `app.ipynb` and open it in the IDE.

## 👥 Author

Carlos L. - [GitHub](https://github.com/CllsPy)

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- [Pinecone Docs](https://docs.pinecone.io/integrations/langchain)
- [Langchain](https://www.langchain.com/)
