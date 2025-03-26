# CSV RAG (Retrieval-Augmented Generation) API

A FastAPI-based application that enables users to upload and query CSV files using Retrieval-Augmented Generation (RAG). The system stores CSV data along with metadata in MongoDB and allows users to interact with it via chat using OpenAI's GPT models.

## Features

- 📤 Multiple CSV upload methods:
  - Direct file upload
  - Load from disk location
  - Load from project directory
- 📊 Advanced CSV processing and metadata extraction
- 💬 Interactive querying using OpenAI's GPT-3.5
- 🔄 Real-time streaming responses
- 📱 Built-in web interface
- 🗄️ MongoDB integration for efficient storage
- ⚡ Asynchronous processing

## Prerequisites

- Python 3.8+
- MongoDB
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mugesh-rao/Task-RAG.git
```

2. Install required packages:
```bash
pip install fastapi uvicorn motor pandas openai python-multipart
```

## Configuration

Update the following variables in `main.py`:

```python
OPENAI_API_KEY = "your-openai-api-key"
MONGODB_URL = "your-mongodb-connection-string"
```

## Running the Application

1. Start the server:
```bash
python main.py
```

2. Access the web interface at `http://localhost:8000`
