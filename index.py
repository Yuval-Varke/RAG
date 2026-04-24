from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path("Linux.pdf")

#Now Load this file in the python program

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print(docs[2].page_content)