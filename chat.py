from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

#Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nodejs_docs",
    embedding=embedding_model,
)

#Take user input
user_query = input("Enter your query: ")

#Relevent chunks from the vector database
searched_results = vector_db.similarity_search(query=user_query)


context = "\n\n\n".join([
    f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
    for result in searched_results
])


SYSTEM_PROMPT = f"""
You are a helpful AI assistant.

Rules:
1. Answer ONLY from the retrieved context.
2. If the answer is not clearly present, say: "I could not find this in the indexed notes."
3. Keep the answer concise and include page numbers when available.

"""

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


res = client.models.generate_content(
        model="gemini-2.5-flash",
    contents=f"""
User question:
{user_query}

Retrieved context:
{context}

""",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
)

print(f"🤖: {res.text}")