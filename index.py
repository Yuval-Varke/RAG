from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = Path("Linux.pdf")

#Now Load this file in the python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the doc into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=400
)
chunks = text_splitter.split_documents(documents=docs)
