import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import google.generativeai as genai
import textwrap
from dataclasses import dataclass
from PIL import Image
import time
from ratelimit import limits, sleep_and_retry
import fitz
import io
from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
import pyttsx3

@dataclass
class Config:
    """Configuration class for the application"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"
    TEXT_EMBEDDING_MODEL_ID: str = "models/embedding-001"
    DPI: int = 300

class PDFProcessor:
    """Handles PDF processing using PyMuPDF"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = Config.DPI) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
        return images

class GeminiClient:
    """Handles interactions with the Gemini API"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
    
    def analyze_page(self, image: Image.Image) -> str:
        """Analyze a PDF page using Gemini's vision capabilities"""
        prompt = """Extract text from the given document image."""
        try:
            response = self.model.generate_content([prompt, image])
            return response.text if response.text else ""
        except Exception as e:
            print(f"Error analyzing page: {e}")
            return ""

    @sleep_and_retry
    @limits(calls=30, period=60)
    def create_embeddings(self, data: str):
        """Create embeddings with rate limiting"""
        try:
            response = genai.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_ID,
                content=data,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return response["embedding"] if isinstance(response, dict) and "embedding" in response else []
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

class RAGApplication:
    """Main RAG application class"""
    
    def __init__(self, api_key: str):
        self.gemini_client = GeminiClient(api_key)
        self.data_df = None
    
    def process_pdf(self, pdf_path: str):
        """Process PDF using Gemini's vision capabilities"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        images = PDFProcessor.pdf_to_images(pdf_path)
        page_analyses = [self.gemini_client.analyze_page(image) for image in tqdm(images)]
        
        self.data_df = pd.DataFrame({
            'Page': list(range(1, len(page_analyses) + 1)),
            'Content': page_analyses,
            'Embeddings': [self.gemini_client.create_embeddings(text) for text in tqdm(page_analyses)]
        })
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the processed data"""
        if self.data_df is None or self.data_df.empty:
            raise ValueError("Process a PDF first to extract content.")
    
        extracted_text = "\n".join(self.data_df["Content"])
    
        prompt = textwrap.dedent(f""" Below is some extracted text from a document. Use it to answer the given question:
                                 DOCUMENT TEXT:
                                 {extracted_text}
                                 QUESTION: {question}
                                 Provide a concise and accurate response based on the document text.
                                 """)

        try:
            response = self.gemini_client.model.generate_content(prompt)
            return response.text if response.text else "No relevant answer found."
        except Exception as e:
            print(f"Error answering question: {e}")
            return "Error generating answer."

recognizer = sr.Recognizer()
engine = pyttsx3.init()
speech_thread = None

import threading
def speak(text):

    """Stop previous speech thread and run text-to-speech safely"""
    global speech_thread

    if speech_thread and speech_thread.is_alive():
        return  

    def run_tts():
        engine.say(text)
        engine.runAndWait()

    speech_thread = threading.Thread(target=run_tts, daemon=True)
    speech_thread.start()


def get_voice_input():
    """Capture voice input and convert it to text"""
    with sr.Microphone() as source:
        st.write("Listening...")
        try:
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError:
            return "Could not request results, please check your internet connection."
        


def delete_temp_file(temp_pdf_path):
    """Delete the temporary PDF file if it exists."""
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)
        print(f"Deleted old temp file: {temp_pdf_path}")



def main():
    load_dotenv()
    st.set_page_config(page_title='Ask the Doc App')
    st.title('Ask the Doc App')
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    
    app = RAGApplication(api_key)
    
    # File uploader & text input 
    with st.form(key="form"):
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
        question = st.text_input("Enter your question")
        submit = st.form_submit_button("Submit")

    voice_query = st.button("Ask with Voice")

    if not pdf_file:
        st.warning("Please upload a PDF before asking a question.")
        return  

    temp_pdf_path = f"temp_{pdf_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    st.write("Processing PDF...")
    app.process_pdf(temp_pdf_path)
    st.success("PDF processed successfully! You can now ask questions.")

    # Handling written query
    if submit and question:
        st.write("Generating answer...")
        answer = app.answer_question(question)
        st.write("Answer:", answer)
        speak(answer)

    # Handling voice query
    if voice_query:
        st.write("Listening for voice query...")
        question = get_voice_input()
        st.write(f"Recognized question: {question}")

        if question:
            st.write("Generating answer...")
            answer = app.answer_question(question)
            st.write("Answer:", answer)
            speak(answer)





if __name__ == "__main__":
    main()