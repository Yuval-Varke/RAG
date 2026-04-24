from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path("nodejs.pdf")

#Now Load this file in the python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the doc into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents=docs)

#Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="nodejs_docs"
)

print("Indexing of documents is done...")
