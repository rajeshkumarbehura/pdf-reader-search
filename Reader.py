from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from secret_key import openai_key  # only for personal secret key

import os

os.environ['OPENAI_API_KEY'] = openai_key  # add your own OpenAI key

# Postgress Database Connection details
HOST = "localhost"
DATABASE = "vectordb"
USER = "testuser"
PWD = "testpwd"
PORT = 6432
DRIVER = "psycopg2"

CONNECTION_STRING = PGVector.connection_string_from_db_params(DRIVER, HOST, PORT, DATABASE, USER, PWD, )

FILE_NAME = "data/LearnJava.pdf"
COLLECTION_NAME = "Book: Teach Yourself Java in 21days"


def pdf_content_into_documents():
    loader = PyPDFLoader(FILE_NAME)
    documents = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs


def embeddings_from_documents(docs):
    embeddings = OpenAIEmbeddings()
    # # create the store
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=False,
    )


def query_result():
    embeddings = OpenAIEmbeddings()
    query = "What is Incrementing and Decrementing ?"
    # load the store
    pgvector_docsearch = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    searched_docs = pgvector_docsearch.search(query, 'mmr', k=1)
    result = searched_docs[0].page_content
    print(result)


if __name__ == '__main__':
    print("Hello")
    # split_texts = pdf_content_into_documents()
    # embeddings_from_documents(split_texts)

    query_result()
    print("Done")
