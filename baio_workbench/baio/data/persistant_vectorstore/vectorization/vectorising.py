import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from typing import List
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import VectorStore
####REQ FOR NCBI DOC
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

embedding = OpenAIEmbeddings()



#####
#####       MYGENE DOC
#####

#we load the api instruction txt
loader = TextLoader("/home/data/persistant_files/user_manuals/api_documentation/mygene/raw_doc_mygene.txt")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
mygene_all_splits = text_splitter.split_documents(data)



####
####         NCBI & BIOPYTHON
####

loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)
loader_pdf = DirectoryLoader('/home/data/persistant_files/user_manuals/general_bioinformatics', 
                             glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader_pdf.load()

#next up is making the index
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

#now we splitted the documents into chunk sizes of 500 with 100 overlapping
general_docs_all_split = text_splitter.split_documents(documents)



####
####        ANISEED
####

#we load the api data
loader = TextLoader("/home/data/persistant_files/user_manuals/api_documentation/aniseed/aniseed_api_doc.txt")
anisseed_data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
anisseed_data_all_splits = text_splitter.split_documents(anisseed_data)

persist_directory_aniseed = '/home/data/persistant_files/vectorstores/aniseed_datastore/'
vectordb_anissed = Chroma.from_documents(documents=anisseed_data_all_splits,
                                            embedding=embedding,
                                            persist_directory=persist_directory_aniseed)

###
# ##
# ###
# ####aniseed updated
loader = TextLoader("/usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/aniseed/aniseed_api_modified.txt")
anisseed_data = loader.load()
docs = text_splitter.split_documents(anisseed_data)
embeddings = OpenAIEmbeddings()

aniseed_faiss_db = FAISS.from_documents(docs, embeddings)
aniseed_faiss_db.save_local("/usr/src/app/baio/data/persistant_files/vectorstores/aniseed_db_faiss_index")

aniseed_faiss_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/aniseed_db_faiss_index", embeddings)


####
####        STORING ALL SPLITS
####
# Store in one db
persist_directory = '/home/data/persistant_files/vectorstores/general_datastore/'

vectordb = Chroma.from_documents(documents= mygene_all_splits 
                                            + general_docs_all_split
                                            + anisseed_data_all_splits,

                                            embedding=embedding,
                                            persist_directory=persist_directory)


####NCBI doc 
loader = TextLoader("/usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ncbi/jin_et_al.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

ncbi_jin_db = FAISS.from_documents(docs, embeddings)
ncbi_jin_db.save_local("/usr/src/app/baio/data/persistant_files/vectorstores/ncbi_jin_db_faiss_index")

ncbi_jin_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/ncbi_jin_db_faiss_index", embeddings)