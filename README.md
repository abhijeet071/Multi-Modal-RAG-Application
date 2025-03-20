# Multi-Modal RAG Application for Document Querying

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents, extract text and images, and query the content using natural language. The application supports both **text and voice inputs** and provides **spoken or written responses**.


## Features

- **Multi-Modal Processing**: Extracts text and images from PDF documents using **Google Gemini API**.
- **Natural Language Querying**: Answers user questions based on the document content.
- **Voice and Text Input**: Supports both **text and voice queries**.
- **Text-to-Speech**: Provides spoken responses using **pyttsx3** .
- **Streamlit Web Interface**: User-friendly web app for document uploads and querying.

---

## Technologies Used

- **AI/ML**:
  - Google Gemini API (for text and image analysis)
  - PyMuPDF (for PDF processing)
  - Text embeddings (for retrieval)
- **Web Development**:
  - Streamlit (for the web interface)
  - SpeechRecognition (for voice input)
  - pyttsx3 (for text-to-speech)
- **Languages**:
  - Python

---

## How It Works

1. **Document Upload**:
   - Users upload a PDF document through the Streamlit interface.
2. **Document Processing**:
   - The PDF is converted into images using **PyMuPDF**.
   - Each page is analyzed using **Google Gemini API** to extract text and generate embeddings.
3. **Querying**:
   - Users can ask questions using **text or voice input**.
   - The system retrieves relevant information from the document and generates a response.
4. **Response**:
   - The answer is displayed as text and can also be spoken aloud using **text-to-speech**.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Google API key (for Gemini API)

### Steps

Clone the repository:
   ```bash
   git clone https://github.com/abhijeet071/Multi-Modal-RAG-Application.git
   cd Multi-Modal-RAG-Application
   ```
### Install Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Set up environment variables:

Create a .env file in the root directory and add your Google Gemini API key:

  ```bash
  GOOGLE_API_KEY=your_api_key_here
  ```

### Run the Streamlit app:

```bash
streamlit run app.py
```

