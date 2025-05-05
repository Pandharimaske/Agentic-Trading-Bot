import os
import tempfile
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader , Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from pinecone import ServerlessSpec
from pinecone import Pinecone
from uuid import uuid4
from langchain_pinecone import PineconeVectorStore



class DataIngestion:
    """
    Class to handle document loading, transformation and ingestion into AstraDB vector store.
    """
    
    def __init__(self):
        """
        Initialize envoronment varables , embedding model and config.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.config = load_config()

    def _load_env_variables(self):
        """
        Load and validate required environment variables.
        """
        load_dotenv()

        required_vars = [
            "GOOGLE_API_KEY" , 
            "PINECONE_API_KEY"
        ]

        missing_vars = [var for var in vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")

    def load_documents(self , uploaded_files)-> List[Document]:
        """
        Load documents from uploaded PDF and DOCX files.
        """

        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name_endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False , suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    loader = PyPDFLoader(temp_file.name)
                    documents.extend(loader.load())
            
            elif uploaded_file.name.endswith(".docx"):
                with tempfile.NamedTemporaryFile(delete=False , suffix=".docx") as temp_file:
                    temp_file.write(uploaded_file.read())
                    loader = Docx2txtLoader(temp_file.name)
                    documents.extend(loader.load())
            
            else:
                print(f"Unsupported file type {uploaded_file.name}")
            
            return documents

    def store_in_vector_db(self , documents: List[Document]):
        """
        Split documents and create vector store with embeddings.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000 , 
            chunk_overlap = 200 , 
            length_function = len
        )
        documents = text_splitter.split_documents(documents)

        pc = Pinecone(api_key = self.pinecone_api_key)

        if not pc.has_index(self.config["vector_db"]["index_name"]):
            pc.create_index(
                name = self.config["vector_db"]["index_name"] , 
                dimension = 768 , 
                metric = "cosine" , 
                spec = ServerlessSpec(cloud = "aws" , region = "us-east-1") ,
            )

        index = pc.Index(self.config["vector_db"]["index_name"])


        vector_store = PineconeVectorStore(index = index , embedding = self.model_loader.load_embeddings())

        uuids = [str(uuid4()) for _ in range(len(documents))]

        vector_store.add_documents(documents= documents , ids = uuids)



    def run_pipeline(self , uploaded_files):
        """
        Run full data ingestion: load files , split, embed and store.
        """
        documents = self.load_documents(uploaded_files=uploaded_files)
        if not documents:
            print("No valid documents found.")
            return
        
        self.store_in_vector_db(documents)


if __name__ == "__main__":
    pass



