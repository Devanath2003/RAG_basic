from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
load_dotenv()



persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space" : "cosine"}

)

print("=" * 17)
query = input("Enter the query - :")


retriever = db.as_retriever(search_kwargs={"k":5}) 
# retrieve top 3 chunks with higher similarity.

""" Differnt approach to do this - only return chunks with consine similarity = 0.3"""

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k":5, "score_threshold":0.3} #better to use smaller_threshold so we will get chunks, else we may don't get
# )

relevant_docs = retriever.invoke(query)

print(f"Users query - {query}")

# print("____Context____\n\n")
# for i,doc in enumerate(relevant_docs,1):
#     print(f"Document {i}: \n{doc.page_content}\n")

combined_input = f""" Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents,just reply with "not enough information"

"""

model = ChatGroq(model="openai/gpt-oss-120b",temperature=0) #ceating the chatopenai model

#Define messages to model
messages = [SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)]


result = model.invoke(messages)

print("Content only:")
print(result.content)