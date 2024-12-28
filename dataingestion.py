import logging
import json
import os
import pymongo
import numpy as np
import torch
import redis
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
import fitz  # PyMuPDF for PDF processing
import cv2
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

logging.basicConfig(
    filename="data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "multimodal_db"
COLLECTION_NAME = "data_collection"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
chat_cache = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    logging.info("Successfully connected to MongoDB.")
except PyMongoError as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise


def load_data_in_batches(batch_size=1000):
    try:
        cursor = collection.find({}, no_cursor_timeout=True).batch_size(batch_size)
        for record in cursor:
            yield record
    except PyMongoError as e:
        logging.error(f"Error while fetching data from MongoDB: {e}")
    finally:
        cursor.close()
def process_files_in_parallel(file_paths):
    try:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for file_path in file_paths:
                file_type = file_path.split('.')[-1]
                executor.submit(process_file, file_path, file_type)
    except Exception as e:
        logging.error(f"Error processing files in parallel: {e}")
