import logging
import os
import requests
import fitz  # PyMuPDF
import cv2
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Setup logging
logging.basicConfig(filename='process_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the language model (LLM)
llm = OpenAI(model="text-davinci-003", temperature=0.7)

# Function to extract text and metadata from a PDF file
def extract_text_and_metadata_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        text = ""
        for page in doc:
            text += page.get_text("text")
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the main points from this PDF content: {text}"
        )
        result = llm(prompt.format(text=text))
        return {"content": result, "metadata": metadata}
    except Exception as e:
        logging.error(f"Error extracting text and metadata from PDF {pdf_path}: {e}")
        return {"content": "", "metadata": {}}

# Function to extract metadata and transcribed text from an audio file
def extract_text_and_metadata_from_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        metadata = {
            "duration_seconds": len(audio) / 1000,
            "file_size": os.path.getsize(audio_path),
            "format": audio.format_description,
        }
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Transcribe and refine the text from this audio: {text}"
        )
        result = llm(prompt.format(text=text))
        return {"content": result, "metadata": metadata}
    except Exception as e:
        logging.error(f"Error extracting text and metadata from audio {audio_path}: {e}")
        return {"content": "", "metadata": {}}

# Function to extract metadata and transcribed text from a video file
def extract_text_and_metadata_from_video(video_path):
    try:
        video = VideoFileClip(video_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        metadata = {
            "duration_seconds": video.duration,
            "resolution": video.size,
            "fps": video.fps,
            "file_size": os.path.getsize(video_path)
        }
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the meaningful content extracted from this video: {text}"
        )
        result = llm(prompt.format(text=text))
        return {"content": result, "metadata": metadata}
    except Exception as e:
        logging.error(f"Error extracting text and metadata from video {video_path}: {e}")
        return {"content": "", "metadata": {}}

# Function to process JSON content with metadata
def extract_text_and_metadata_from_json(json_content):
    try:
        metadata = {
            "record_count": len(json_content) if isinstance(json_content, list) else 1,
        }
        prompt = PromptTemplate(
            input_variables=["json_content"],
            template="Summarize the key information from this JSON data: {json_content}"
        )
        result = llm(prompt.format(json_content=json_content))
        return {"content": result, "metadata": metadata}
    except Exception as e:
        logging.error(f"Error processing JSON content: {e}")
        return {"content": "", "metadata": {}}

# Function to extract content and metadata from Jira
def extract_text_and_metadata_from_jira(jira_url, auth_token):
    try:
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = requests.get(jira_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            metadata = {
                "issue_key": data.get("key"),
                "project": data.get("fields", {}).get("project", {}).get("name"),
                "status": data.get("fields", {}).get("status", {}).get("name"),
            }
            prompt = PromptTemplate(
                input_variables=["jira_data"],
                template="Summarize the Jira issues and comments: {jira_data}"
            )
            result = llm(prompt.format(jira_data=data))
            return {"content": result, "metadata": metadata}
        else:
            logging.error(f"Failed to fetch Jira content: {response.status_code}")
            return {"content": "", "metadata": {}}
    except Exception as e:
        logging.error(f"Error extracting text and metadata from Jira: {e}")
        return {"content": "", "metadata": {}}

# Function to extract content and metadata from Confluence
def extract_text_and_metadata_from_confluence(confluence_url, auth_token):
    try:
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = requests.get(confluence_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            metadata = {
                "title": data.get("title"),
                "author": data.get("history", {}).get("createdBy", {}).get("displayName"),
                "created_date": data.get("history", {}).get("createdDate"),
            }
            prompt = PromptTemplate(
                input_variables=["confluence_data"],
                template="Summarize the Confluence page content: {confluence_data}"
            )
            result = llm.prompt.format(confluence_data=data)
            return {"content": result, "metadata": metadata}
        else:
            logging.error(f"Failed to fetch Confluence content: {response.status_code}")
            return {"content": "", "metadata": {}}
    except Exception as e:
        logging.error(f"Error extracting text and metadata from Confluence: {e}")
        return {"content": "", "metadata": {}}

# Main function to process records based on type
def process_record(record):
    content_with_metadata = {}
    try:
        file_type = record.get("type")
        if file_type == "pdf":
            content_with_metadata = extract_text_and_metadata_from_pdf(record.get("file_path"))
        elif file_type == "audio":
            content_with_metadata = extract_text_and_metadata_from_audio(record.get("file_path"))
        elif file_type == "video":
            content_with_metadata = extract_text_and_metadata_from_video(record.get("file_path"))
        elif file_type == "json":
            content_with_metadata = extract_text_and_metadata_from_json(record.get("content"))
        elif file_type == "jira":
            content_with_metadata = extract_text_and_metadata_from_jira(record.get("url"), record.get("auth_token"))
        elif file_type == "confluence":
            content_with_metadata = extract_text_and_metadata_from_confluence(record.get("url"), record.get("auth_token"))
        else:
            logging.warning(f"Unsupported file type: {file_type}")

        return content_with_metadata
    except Exception as e:
        logging.error(f"Error processing record {record}: {e}")
        return {"content": "", "metadata": {}}

# Function to chunk data
def process_and_chunk_data(records, chunk_size=100, chunk_overlap=10):
    try:
        data_batch = []
        for record in records:
            content_with_metadata = process_record(record)
            data_batch.append(content_with_metadata)
            
            if len(data_batch) >= chunk_size:
                yield data_batch[:chunk_size]
                data_batch = data_batch[chunk_size - chunk_overlap:]
        
        if data_batch:
            yield data_batch
    except Exception as e:
        logging.error(f"Error in processing and chunking data: {e}")
        raise

# Example usage
if __name__ == "__main__":
    records = [
        {"type": "pdf", "file_path": "example.pdf"},
        {"type": "audio", "file_path": "example.mp3"},
        {"type": "video", "file_path": "example.mp4"},
        {"type": "json", "content": '{"key": "value"}'},
        {"type": "jira", "url": "https://jira.example.com/rest/api/2/issue/EX-1", "auth_token": "your_jira_token"},
        {"type": "confluence", "url": "https://confluence.example.com/rest/api/content/12345", "auth_token": "your_confluence_token"},
    ]

    for chunk in process_and_chunk_data(records, chunk_size=2, chunk_overlap=1):
        print(f"Processed chunk:\n{chunk}\n")