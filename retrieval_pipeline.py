from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space" : "cosine"}

)


query = input()

retriever = db.as_retriever(search_kwargs={"k":5}) 
# retrieve top 3 chunks with higher similarity.

""" Differnt approach to do this - only return chunks with consine similarity = 0.3"""

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k":5, "score_threshold":0.3} #better to use smaller_threshold so we will get chunks, else we may don't get
# )

relevant_docs = retriever.invoke(query)

print(f"Users query - {query}")

print("____Context____\n\n")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n{doc.page_content}\n")

