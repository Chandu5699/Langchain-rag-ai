import logging
import json
import redis
from langchain.memory import ChatMessageHistory

# Setup logging
logging.basicConfig(
    filename="chat_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize chat message history
chat_history = ChatMessageHistory()

# Function to add a message to the chat buffer
def add_message(role, content):
    try:
        chat_history.add_message(role=role, content=content)
        logging.info(f"Added message to chat buffer: {role} - {content}")
    except Exception as e:
        logging.error(f"Error adding message to chat buffer: {e}")

# Function to get the chat history
def get_chat_history():
    try:
        messages = chat_history.messages
        logging.info("Retrieved chat history from buffer")
        return messages
    except Exception as e:
        logging.error(f"Error retrieving chat history from buffer: {e}")
        return []

# Example usage
add_message('user', 'Hello, how are you?')
add_message('assistant', 'I am fine, thank you! How can I assist you today?')

# Retrieve and print chat history
for message in get_chat_history():
    print(f"{message['role']}: {message['content']}")

# Initialize Redis client
try:
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    logging.info("Connected to Redis")
except redis.ConnectionError as e:
    logging.error(f"Error connecting to Redis: {e}")
    redis_client = None

# Function to add a message to the Redis chat buffer
def add_message_to_redis(role, content):
    if redis_client is None:
        logging.error("Redis client is not initialized")
        return
    try:
        message = {"role": role, "content": content}
        redis_client.rpush('chat_history', json.dumps(message))
        logging.info(f"Added message to Redis: {role} - {content}")
    except Exception as e:
        logging.error(f"Error adding message to Redis: {e}")

# Function to get the chat history from Redis
def get_chat_history_from_redis():
    if redis_client is None:
        logging.error("Redis client is not initialized")
        return []
    try:
        messages = redis_client.lrange('chat_history', 0, -1)
        logging.info("Retrieved chat history from Redis")
        return [json.loads(message) for message in messages]
    except Exception as e:
        logging.error(f"Error retrieving chat history from Redis: {e}")
        return []

# Example usage
add_message_to_redis('user', 'Hello, how are you?')
add_message_to_redis('assistant', 'I am fine, thank you! How can I assist you today?')

# Retrieve and print chat history from Redis
for message in get_chat_history_from_redis():
    print(f"{message['role']}: {message['content']}")