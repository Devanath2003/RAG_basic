import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader 
# These classes help to read text/ppt/docx files from the folders
from langchain_text_splitters import CharacterTextSplitter
# For chunking
from langchain_openai import OpenAIEmbeddings
# The embedding model to convert chunks into vector embeedings.
from langchain_chroma import Chroma
# The vector database to store the vector embeddings.
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all files from the docs director"""

    print(f"Loading documents from {docs_path} ...")

    # check if the path exist.
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt", #only check for .txt files
        loader_cls=TextLoader,
         loader_kwargs={"encoding":"utf-8"} #for pdfs or csv, we need seperate classes for it.
    )

    documents = loader.load() #invoke the load method - this will give a list of langchain documents

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path} path, Please add necessary documents")


    for i,doc in enumerate(documents[:2]): #show first two documents
        print(f"\nDocument-{i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0): #chunk_size here is 800 characters not tokens
    """ split the doucments into smaller chunks with overlap"""

    text_splitter = CharacterTextSplitter( #most basic textsplitter in langchain
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i,chunk in enumerate(chunks[:5]):
            print(f"\n ____Chunk {i+1}____")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: \n{chunk.page_content} ")

        if len(chunks) > 5:
            print(f"\n.... and {len(chunks)-5} more chunks. ")

    return chunks

def create_vector_store(chunks, persist_directory="db/chrome_db"): # just saving the db locally
    """ create and persist ChromaDB vector store"""

    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create ChromaDB vector store

    print("___Creating vector store____")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"} #algorithm is cosine similarity which will be used to compare the chunks
    )

    print("____Finished creating vector store____")

    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


    
def main():
    print()
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
