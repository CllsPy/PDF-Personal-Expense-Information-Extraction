# Descrição do projeto

O objetivo do projeto é construir uma interface via Gradio na qual seja possível o envio de arquivos em PDF contento informações a respeito de gastos pessoas e que através de técnica como Text Embeddings e com Generative AI o usuário possa obter informações relevantes a respeitos dos dados usados como input.

## Etapas
![image](https://github.com/CllsPy/Generative_AI/assets/96326019/920681bf-d869-4db2-aea6-a14b94ba0c8b)
*fonte: autor*
<br>
<br>

1. Obter dados do usuário em formato adequado (PDF)
2. Aplicar Text Embedding ao PDF de modo que o modelo (LLaMA) possa diferir)
3. Enviar os dados para um VectorDataBase(Pinecode)
4. Ajustar prompt e aspectos relacionados a resposta desejada
5. Entregar dados ao modelo
6. Avaliar resposta

## Requirements
A linguagem dinâmica escolhida foi Python, para IDE eu optei pelo Google Colab. Para implementar as etapas localmente passos adicionais são necessários e podem ser encontrados nas documentações das bibliotecas mencionadas neste arquivo. Além disso acesso a GPU.

### Packages
Instale os seguintes pacotes

``` Python
mport os
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

## Launch
Download the Python File `app.ipynb` and open it in the IDE.

## Autor
Carlos L. - [Github](https://github.com/CllsPy)

## License
This project is licensed under the MIT License.

## Acknowledgements

- [Pinecone Docs](https://docs.pinecone.io/integrations/langchain)
- [Langchain](https://www.langchain.com/)
