from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #384 dimensions
db = Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0) 

# Store our conversations as messages.
chat_history = []

def ask_question(user_question):
    print(f"User-question: {user_question}")

    # Make the question clear with chat history

    if chat_history:

        #ask the AI to formulate the question.

        messages = [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable, Just return the rewritten question. "),
                    ] + chat_history + [HumanMessage(content=f"New question: {user_question}")]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"\n-> Searching for : {search_question} ...")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)

    print(f"Found len(docs) relevant documents.")
    for i,doc in enumerate(docs,1):
        #show first two lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc {i}: {preview}...")

    # Create the final prompt
    combined_input = f""" Based on the following documents,please answer this question: {user_question}

Documents: {"\n".join([f"={doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the informatio from these documents. If you can't find the anwer from the given documents- please reply with "Not enough information !"

"""
    messages = [SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation")] + chat_history + [HumanMessage(content=combined_input)]

    result = model.invoke(messages)
    answer = result.content

    # Remember this conversation

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("=^" * 30)
    print("Session started - Ask me questions: Type 'quit' to exit. !")

    while True:
        question = input("Your question : ")

        if question.lower() == "quit":
            print("Session Ended - Good Bye")
            break
        ask_question(question)


if __name__ == "__main__":
    start_chat()
