import pandas as pd
from typing import Dict, List
import motor.motor_asyncio
from bson import ObjectId
import openai
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = ""

# Configure MongoDB
MONGODB_URL = ""
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client.csv_database
files_collection = db.files

async def process_csv(file_content: str, filename: str) -> Dict:
    """Process CSV file and store in MongoDB"""
    try:
        # Read CSV
        df = pd.read_csv(StringIO(file_content))
        
        # Convert DataFrame to string representation for storage
        csv_str = df.to_csv(index=False)
        
        # Create document for MongoDB
        document = {
            "file_name": filename,
            "content": csv_str,
            "summary": generate_csv_summary(df)
        }
        
        # Insert into MongoDB
        result = await files_collection.insert_one(document)
        
        return {"file_id": str(result.inserted_id), "message": "Upload successful"}
    except Exception as e:
        raise Exception(f"Error processing CSV: {str(e)}")

def generate_csv_summary(df: pd.DataFrame) -> str:
    """Generate a summary of the CSV file for context"""
    columns = df.columns.tolist()
    row_count = len(df)
    summary = f"This CSV has {row_count} rows and the following columns: {', '.join(columns)}."
    return summary

async def query_csv(file_id: str, query: str) -> str:
    """Query CSV data using OpenAI"""
    try:
        # Get file from MongoDB
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise Exception("File not found")

        # Create context from CSV summary and content
        context = f"CSV Summary: {file_doc['summary']}\nCSV Content: {file_doc['content'][:1000]}..."

        # Query OpenAI
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
    """Get list of all files"""
    try:
        files = await files_collection.find({}).to_list(length=None)
        return [{"file_id": str(file["_id"]), "file_name": file["file_name"]} for file in files]
    except Exception as e:
        raise Exception(f"Error retrieving files: {str(e)}")

async def delete_file(file_id: str) -> bool:
    """Delete file from MongoDB"""
    try:
        result = await files_collection.delete_one({"_id": ObjectId(file_id)})
        if result.deleted_count == 0:
            raise Exception("File not found")
        return True
    except Exception as e:
        raise Exception(f"Error deleting file: {str(e)}") 