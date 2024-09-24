import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_BASE = "https://api.galadriel.com/v1"
OPENAI_API_KEY = os.getenv("GALADRIEL_RAG_APP_API_KEY")
OPENAI_MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"

os.environ['USER_AGENT'] = 'myagent'

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

litepaper_loader = WebBaseLoader("https://docs.galadriel.com/litepaper")
litepaper_data = litepaper_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(litepaper_data)

vectorstore = Chroma.from_documents(documents=documents, embedding=embed_model)

llm = ChatOpenAI(temperature=0,     
             openai_api_base=OPENAI_API_BASE, 
             openai_api_key=OPENAI_API_KEY,
             model_name=OPENAI_MODEL_NAME)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever()
)

query = "Explain Galadriel's technology stack, strategy and vision"
print(f"Query: {query}")
response = qa_chain.run(query)
print(response)

