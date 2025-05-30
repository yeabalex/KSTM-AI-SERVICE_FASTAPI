import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
import uuid
import time
import os
import pickle
import redis
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain

# Custom imports from your libs
from libs.load_and_process_documents import load_and_process_documents
from libs.load_an_process_pdf import load_and_process_pdf
from libs.load_and_process_csv import load_and_process_csv
from libs.load_and_process_txt import load_and_process_txt
from libs.load_and_process_json import load_and_process_json
from config.db import create_vectordb
from static.resources import create_chains
from config.db import load_vectordb

logger = logging.getLogger("uvicorn")

# Load environment variables
load_dotenv()
os.environ.update({
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "LANGSMITH_TRACING": "true",
    "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY")
})

# Redis connection
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=False)
    redis_client.ping()
    print("Redis connection established.")
    logger.info("Successfully connected to Redis.")
except redis.ConnectionError as e:
    print(f"Failed to connect to Redis: {e}")
    logger.error(f"Failed to connect to Redis: {e}")

# FastAPI app
app = FastAPI()

# Enable CORS for specific domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.yeabsiraa.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CreateBotRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    bot_id: str = Field(..., min_length=1)
    kb_id: str = Field(..., min_length=1)
    csv: Optional[List[str]] = []
    pdf: Optional[List[str]] = []
    txt: Optional[List[str]] = []
    json: Optional[List[str]] = []
    web_url: Optional[List[HttpUrl]] = []
    prompt_template: Optional[str] = None
    temperature: Optional[float] = Field(default=0.7, ge=0, le=1)
    theme: Optional[str] = None
    bot_name: str = Field(..., min_length=1)
    profile_image: Optional[HttpUrl] = None
    model: Optional[str] = None

class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    bot_id: str = Field(..., min_length=1)
    kb_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    input_text: str = Field(..., min_length=1)

# POST /create-bot
@app.post("/create-bot")
def create_bot(request: CreateBotRequest):
    user_id, bot_id, kb_id = request.user_id, request.bot_id, request.kb_id
    documents = []

    try:
        # Load and process web URLs
        if request.web_url:
            for url in request.web_url:
                documents.extend(load_and_process_documents(str(url), refresh=True))

        # Load and process files
        loaders = [load_and_process_pdf, load_and_process_csv, load_and_process_txt, load_and_process_json]
        for file_list, loader in zip([request.pdf, request.csv, request.txt, request.json], loaders):
            if file_list:
                for path in file_list:
                    documents.extend(loader(path, refresh=True))

        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents provided")

        # Create vector DB and save it
        create_vectordb(documents, kb_id)
        print("Here fine")
        redis_client.set(f"last_refresh:{user_id}:{bot_id}:{kb_id}", str(time.time()))
        print("Here fine 2")

        if request.prompt_template:
            redis_client.set(f"prompt_template:{user_id}:{bot_id}", request.prompt_template)

        return {"status": "success", "last_refresh": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST /query
@app.post("/query")
def query_bot(request: QueryRequest):
    user_id, bot_id, kb_id, session_id, input_text = (
        request.user_id, request.bot_id, request.kb_id, request.session_id, request.input_text
    )

    try:
        db = load_vectordb()

        memory_key = f"memory:{user_id}:{bot_id}:{kb_id}:{session_id}"
        memory_data = redis_client.get(memory_key)
        memory = pickle.loads(memory_data) if memory_data else ConversationBufferMemory(return_messages=True)

        memory.chat_memory.add_user_message(input_text)

        # Load prompt template
        prompt_template = redis_client.get(f"prompt_template:{user_id}:{bot_id}")
        if prompt_template:
            prompt_template = prompt_template.decode("utf-8")

        # Create and run retrieval chain
        base_chain = create_chains(prompt_template)
        retriever = db.as_retriever(search_kwargs={
            "filter": {
                "source_url": kb_id  # or your specific URL/id
            }
        })

        retrieval_chain = create_retrieval_chain(retriever, base_chain)
        response = retrieval_chain.invoke({
            "input": input_text,
            "chat_history": memory.chat_memory.messages[:-1]
        })

        memory.chat_memory.add_ai_message(response["answer"])

        # Store updated memory
        redis_client.set(memory_key, pickle.dumps(memory))

        return {
            "status": "success",
            "answer": response["answer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
