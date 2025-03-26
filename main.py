from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import motor.motor_asyncio
from bson import ObjectId
import openai
from io import StringIO
import logging
from pathlib import Path
import os
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="CSV RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENAI_API_KEY = ""
MONGODB_URL = ""


openai.api_key = OPENAI_API_KEY


try:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = client.csv_database
    files_collection = db.files
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

class QueryRequest(BaseModel):
    file_id: str
    query: str

class FileSourceRequest(BaseModel):
    source_type: str  # "upload", "disk", "project"
    file_path: Optional[str] = None

class StreamingResponse(BaseModel):
    file_id: str
    query: str
    stream: bool = False

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>CSV RAG API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; max-width: 800px; margin: 0 auto; }
        .section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        textarea { width: 100%; min-height: 100px; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>CSV RAG API</h1>
    
    <div class="section">
        <h2>Upload CSV</h2>
        <input type="file" id="csvFile" accept=".csv">
        <button onclick="uploadFile()">Upload</button>
        <div id="uploadResult"></div>
    </div>

    <div class="section">
        <h2>Files</h2>
        <button onclick="getFiles()">Refresh File List</button>
        <ul id="fileList"></ul>
    </div>

    <div class="section">
        <h2>Query CSV</h2>
        <select id="fileSelect"><option value="">Select a file</option></select>
        <textarea id="queryInput" placeholder="Enter your query"></textarea>
        <button onclick="queryFile()">Query</button>
        <div id="queryResult"></div>
    </div>

    <script>
        async function uploadFile() {
            const fileInput = document.getElementById('csvFile');
            if (!fileInput.files[0]) {
                document.getElementById('uploadResult').innerHTML = '<span class="error">Please select a file</span>';
                return;
            }
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                document.getElementById('uploadResult').innerHTML = 
                    response.ok ? `Success: ${result.message} (ID: ${result.file_id})` : 
                    `<span class="error">Error: ${result.detail}</span>`;
                if (response.ok) getFiles();
            } catch (error) {
                document.getElementById('uploadResult').innerHTML = `<span class="error">Error: ${error.message}</span>`;
            }
        }

        async function getFiles() {
            try {
                const response = await fetch('/files');
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                const fileList = document.getElementById('fileList');
                const fileSelect = document.getElementById('fileSelect');
                
                fileList.innerHTML = '';
                fileSelect.innerHTML = '<option value="">Select a file</option>';
                
                if (data.files && Array.isArray(data.files)) {
                    data.files.forEach(file => {
                        const li = document.createElement('li');
                        li.innerHTML = `${file.file_name} (ID: ${file.file_id}) 
                            <button onclick="deleteFile('${file.file_id}')">Delete</button>`;
                        fileList.appendChild(li);
                        
                        const option = document.createElement('option');
                        option.value = file.file_id;
                        option.text = file.file_name;
                        fileSelect.appendChild(option);
                    });
                } else {
                    fileList.innerHTML = 'No files found';
                }
            } catch (error) {
                console.error('Error fetching files:', error);
                document.getElementById('fileList').innerHTML = 
                    `<span class="error">Error loading files: ${error.message}</span>`;
            }
        }

        async function queryFile() {
            const fileId = document.getElementById('fileSelect').value;
            const query = document.getElementById('queryInput').value;
            if (!fileId || !query) {
                document.getElementById('queryResult').innerHTML = 
                    '<span class="error">Please select a file and enter a query</span>';
                return;
            }
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_id: fileId, query: query })
                });
                const result = await response.json();
                document.getElementById('queryResult').innerHTML = 
                    response.ok ? result.response : `<span class="error">Error: ${result.detail}</span>`;
            } catch (error) {
                document.getElementById('queryResult').innerHTML = `<span class="error">Error: ${error.message}</span>`;
            }
        }

        async function deleteFile(fileId) {
            try {
                const response = await fetch(`/file/${fileId}`, { method: 'DELETE' });
                if (response.ok) {
                    await getFiles();
                } else {
                    const result = await response.json();
                    console.error('Delete failed:', result.detail);
                }
            } catch (error) {
                console.error('Error deleting file:', error);
            }
        }

        // Initial load
        getFiles();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_file(
    file: Optional[UploadFile] = None,
    source: Optional[FileSourceRequest] = None
):
    try:
        if file:
            # Handle direct file upload
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="Only CSV files are allowed")
            content = await file.read()
            return await process_csv(content.decode(), file.filename)
        
        elif source:
            # Handle disk/project directory files
            if source.source_type not in ["disk", "project"]:
                raise HTTPException(status_code=400, detail="Invalid source type")
            
            if not source.file_path:
                raise HTTPException(status_code=400, detail="File path is required")
                
            try:
                file_path = Path(source.file_path)
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                if not file_path.suffix == '.csv':
                    raise HTTPException(status_code=400, detail="Only CSV files are allowed")
                
                with open(file_path, 'r') as f:
                    content = f.read()
                return await process_csv(content, file_path.name)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail="No file or source provided")
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def get_files():
    try:
        files = await get_all_files()
        return {"files": files}
    except Exception as e:
        logger.error(f"Get files error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_file(request: QueryRequest):
    try:
        response = await query_csv(request.file_id, request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        if "File not found" in str(e):
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_file_stream(request: StreamingResponse):
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(request.file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Prepare context with metadata
        context = (
            f"CSV Summary: {file_doc['summary']}\n"
            f"Metadata: {file_doc['metadata']}\n"
            f"CSV Sample Content (first few rows): "
            f"{str(file_doc['metadata']['sample_rows'])}"
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant that analyzes CSV data. "
                    "Provide clear, concise answers and include relevant statistics "
                    "when available."
                )},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.query}"}
            ],
            stream=request.stream
        )
        
        if request.stream:
            return StreamingResponse(response)
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        if "File not found" in str(e):
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/file/{file_id}")
async def delete_file(file_id: str):
    try:
        await delete_file(file_id)
        return {"message": "File deleted successfully"}
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        if "File not found" in str(e):
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=500, detail=str(e))

async def process_csv(file_content: str, filename: str) -> Dict:
    try:
        df = pd.read_csv(StringIO(file_content))
        csv_str = df.to_csv(index=False)
        
        # Generate enhanced metadata
        metadata = {
            "file_name": filename,
            "upload_date": datetime.utcnow(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "sample_rows": df.head(5).to_dict('records'),
            "numeric_columns_stats": {
                col: {
                    "mean": float(df[col].mean()) if not np.isnan(df[col].mean()) else None,
                    "min": float(df[col].min()) if not np.isnan(df[col].min()) else None,
                    "max": float(df[col].max()) if not np.isnan(df[col].max()) else None
                }
                for col in df.select_dtypes(include=[np.number]).columns
            }
        }
        
        # Store both content and metadata
        document = {
            "file_name": filename,
            "content": csv_str,
            "metadata": metadata,
            "summary": generate_csv_summary(df, metadata)
        }
        
        result = await files_collection.insert_one(document)
        return {"file_id": str(result.inserted_id), "message": "Upload successful"}
    except Exception as e:
        raise Exception(f"Error processing CSV: {str(e)}")

def generate_csv_summary(df: pd.DataFrame, metadata: Dict) -> str:
    summary_parts = [
        f"This CSV file contains {metadata['row_count']} rows and {metadata['column_count']} columns.",
        f"Columns: {', '.join(metadata['columns'])}.",
    ]
    
    # Add numeric column statistics
    if metadata['numeric_columns_stats']:
        summary_parts.append("\nNumeric column statistics:")
        for col, stats in metadata['numeric_columns_stats'].items():
            if all(v is not None for v in stats.values()):
                summary_parts.append(
                    f"- {col}: mean={stats['mean']:.2f}, "
                    f"range: {stats['min']:.2f} to {stats['max']:.2f}"
                )
    
    return " ".join(summary_parts)

async def query_csv(file_id: str, query: str) -> str:
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise Exception("File not found")
        context = f"CSV Summary: {file_doc['summary']}\nCSV Content: {file_doc['content'][:1000]}..."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about CSV data."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error querying CSV: {str(e)}")

async def get_all_files() -> List[Dict]:
    try:
        files = await files_collection.find({}).to_list(length=None)
        return [{"file_id": str(file["_id"]), "file_name": file["file_name"]} for file in files]
    except Exception as e:
        raise Exception(f"Error retrieving files: {str(e)}")

async def delete_file(file_id: str) -> bool:
    try:
        result = await files_collection.delete_one({"_id": ObjectId(file_id)})
        if result.deleted_count == 0:
            raise Exception("File not found")
        return True
    except Exception as e:
        raise Exception(f"Error deleting file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)