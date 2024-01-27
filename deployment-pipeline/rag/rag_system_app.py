from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.palm import PaLM
from llama_index import ServiceContext
from llama_index.llms import PaLM, OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from fastapi import FastAPI, BackgroundTasks
import boto3
import chromadb
import os

embed_model = None #HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
model = None #PaLM(api_key="AIzaSyD-gUGR1747OmPBrTEBk2dJBo2yBLzlBQ8")

app = FastAPI()

def download_s3_dir(key_id, secret_key, bucket, folder):
  # A function to download all the files in its original directory format from provided s3 bucket (with a defined user having access_keys)
  s3 = boto3.resource('s3', aws_access_key_id=key_id, aws_secret_access_key=secret_key)
  bucket = s3.Bucket(bucket)

  data_directory = './data/'
  if not os.path.exists(data_directory):
    os.makedirs(data_directory)
  for obj in bucket.objects.filter(Prefix = folder):
    if not obj.key.endswith('/'): 
      if not os.path.exists(data_directory + '/'.join(obj.key.split('/')[:-1])):
        os.makedirs(data_directory + '/'.join(obj.key.split('/')[:-1]))
      if not os.path.exists(data_directory + f'{obj.key}'):
        print(f'Downloading {obj.key}')
        bucket.download_file(obj.key, data_directory + f'{obj.key}')

def create_new_vdb(folder_name):
  folder_name = os.path.join('./data/', folder_name)
  # Function to create a vector database of files taken from the provided local directory
  documents = SimpleDirectoryReader(folder_name).load_data()
  db = chromadb.PersistentClient(path="./chroma_db")
  chroma_collection = db.get_or_create_collection("test_vector_db")

  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  
  service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=model, chunk_size=512)
  index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context, show_progress=True)

def download_s3_and_new_vector_db(aws_access_key_id, aws_secret_access_key, bucket_name, folder):
  download_s3_dir(aws_access_key_id, aws_secret_access_key, bucket_name, folder)
  create_new_vdb(folder_name=folder)

async def make_query(query):
  # Function to answer a query using the defined vector db and palm llm model
  db = chromadb.PersistentClient(path="./chroma_db")
  chroma_collection = db.get_or_create_collection("test_vector_db")
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)

  service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=model, chunk_size=512)

  # load your index from stored vectors
  index = VectorStoreIndex.from_vector_store(
      vector_store, service_context=service_context, storage_context=storage_context, show_progress=True
  )
  query_engine = index.as_query_engine()
  response = query_engine.query(query)
  return {'response': response}

@app.post("/settings/")
def setting(data: dict):
  model_name = data.get('model_name', 'palm')
  embed_model_name = data.get('embed_model', "BAAI/bge-small-en-v1.5")
  openai_api_key = data.get("openai_api_key")
  palm_api_key = data.get("palm_api_key")

  global model
  global embed_model

  # Set the language model and embedding model to OpenAI
  if openai_api_key:
    model = OpenAI(api_key=openai_api_key)
    embed_model = OpenAIEmbedding(api_key=openai_api_key)
    return {"Message": "Model and embedding model set to OpenAI"}
  
  # Set the language model and embedding models
  if model_name == 'palm':
    model = PaLM(api_key=palm_api_key)
  elif model_name == 'vllm':
    pass
  embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
  return {"Message": f"Language model set to {model_name} and Embedding model set to {embed_model_name}"}  

@app.post("/connects3/")
async def s3_sync(data: dict, background_tasks: BackgroundTasks):
  # An API endpoint to let the user download data from the provided s3 bucket and create a vector database locally to be able to answer queries
  aws_access_key_id = data.get('access_key_id')
  aws_secret_access_key = data.get('secret_access_key')
  bucket_name = data.get('bucket_name')
  folder = data.get('folder_name', '')

  download_s3_and_new_vector_db(aws_access_key_id = aws_access_key_id, aws_secret_access_key=aws_secret_access_key, bucket_name=bucket_name, folder=folder)
  return {"Message": "Syncing with your s3 bucket done"}

@app.get("/makequery/")
async def query(data: dict):
  # API endpoint to answer user queries using the vector database
  question = data.get('question')
  return await make_query(question)
