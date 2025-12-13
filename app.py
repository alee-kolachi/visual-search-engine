from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import uuid
import numpy as np
from typing import List
from pydantic import BaseModel
import chromadb
import zipfile
import shutil
from datetime import datetime


app = FastAPI()

os.makedirs("images", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
MAX_SIZE = 5 * 1024 * 1024
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP Model")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("Model Loaded!")

#Initialize ChromaDB
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="image_embeddings",
    metadata={"hnsw:space": "cosine"}
)
print(f"ChromaDB ready. Images indexed: {collection.count()}")

batch_status = {}

#Request model for text search
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class BatchUploadResponse(BaseModel):
    job_id: str
    message: str
    status: str

@app.get("/")
def home():
    return {"message": "visual search engine"}

def get_image_embedding(image):
    """
    Convert image to 512-dimensional vector.
    """
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    # Convert to numpy array
    return embedding.cpu().numpy().flatten()

def get_text_embedding(text: str):
    """
    Convert text to 512-dimensional vector
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

def cosine_similarity(vec1, vec2):
    """
    Calculate similarity between two vectors.
    Returns 0 to 1
    """
    return np.dot(vec1, vec2)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {ALLOWED_EXTENSIONS} allowed")
    
    content = await file.read()

    if len(content) > MAX_SIZE:
        raise HTTPException(400, f"File too large, Max allowed size is {MAX_SIZE}")
    
    image_id = str(uuid.uuid4())
    new_filename = f"{image_id}.{extension}"
    file_path = f"images/{new_filename}"

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        image = Image.open(file_path)
        image.verify()
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(400, f"Invalid image: {str(e)}")
    
    img = Image.open(file_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(file_path, quality=95)

    #Extract embedding
    embedding = get_image_embedding(img)

    collection.add(
        ids=[image_id],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "filename": file.filename,
            "filepath": new_filename,
            "size": len(content)
        }]
    )


    return {
        "image_id": image_id,
        "original_filename": file.filename,
        "saved_as": new_filename,
        "size": f"{len(content)} bytes",
        "total_indexed": collection.count()
    }
        
@app.post("/upload/batch", response_model=BatchUploadResponse)
async def batch_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip files allowed.")
    
    #Generate job id
    job_id = str(uuid.uuid4())

    #Save zip file
    zip_path = f"images/batch_{job_id}.zip"
    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    #initialize job status
    batch_status[job_id] = {
        "status": "processing",
        "total": 0,
        "processed": 0,
        "failed": 0,
        "started_at": datetime.now().isoformat()
    }

    #Process in background (don't block response)
    background_tasks.add_task(process_batch, job_id, zip_path)

    return BatchUploadResponse(
        job_id=job_id,
        message="Batch processing started",
        status="processing"
    )

def process_batch(job_id: str, zip_path: str):
    extract_path = f"images/batch_{job_id}"
    os.makedirs(extract_path, exist_ok=True)

    try:
        #Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        image_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                ext = file.split(".")[-1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    image_files.append(os.path.join(root, file))
        
        batch_status[job_id]["total"] = len(image_files)

        #process each image
        batch_embeddings = []
        batch_ids = []
        batch_metadatas = []

        for img_path in image_files:
            try:
                #Load and process
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((224, 224), Image.Resampling.LANCZOS)

                #Generate ID and save
                image_id = str(uuid.uuid4())
                ext = img_path.split(".")[-1].lower()
                new_filename = f"{image_id}.{ext}"
                new_path = f"images/{new_filename}"
                img.save(new_path, quality=95)

                #Extract embedding
                embedding = get_image_embedding(img)

                #Collect for batch insert
                batch_ids.append(image_id)
                batch_embeddings.append(embedding.tolist())
                batch_metadatas.append({
                    "filename": os.path.basename(img_path),
                    "filepath": new_filename,
                    "size": os.path.getsize(new_path),
                    "batch_job": job_id
                })

                batch_status[job_id]["processed"] += 1

                #Batch insert every 50 images
                if len(batch_ids) >= 50:
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )

                    batch_ids = []
                    batch_embeddings = []
                    batch_metadatas = []
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                batch_status[job_id]["failed"] += 1

        #Insert remaining images
        if batch_ids:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

        #Cleanup
        shutil.rmtree(extract_path)
        os.remove(zip_path)

        batch_status[job_id]["status"] = "completed"
        batch_status[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        batch_status[job_id]["status"] = "failed"
        batch_status[job_id]["error"] = str(e)
                
@app.get("/batch/status/{job_id}")
def get_batch_status(job_id: str):
    """Check batch processing status"""
    if job_id not in batch_status:
        raise HTTPException(404, "Job not found")
    
    return batch_status[job_id]

@app.post("/index/folder")
async def index_folder(
    background_tasks: BackgroundTasks,
    folder_path: str
):
    if not os.path.exists(folder_path):
        raise HTTPException(404, "Folder not found")
    
    job_id = str(uuid.uuid4())

    batch_status[job_id] = {
        "status": "processing",
        "total": 0,
        "processed": 0,
        "failed": 0,
        "started_at": datetime.now().isoformat()
    }

    background_tasks.add_task(process_folder, job_id, folder_path)

    return {
        "job_id": job_id,
        "message": "folder indexing started",
        "status": "processing"
    }

def process_folder(job_id: str, folder_path: str):
    """Process all image in a  folder"""
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = file.split(".")[-1].lower()
            if ext in ALLOWED_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    
    batch_status[job_id]["total"] = len(image_files)

    batch_embeddings = []
    batch_ids = []
    batch_metadatas = []

    for img_path in image_files:
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            image_id = str(uuid.uuid4())
            ext = img_path.split(".")[-1].lower()
            new_filename = f"{image_id}.{ext}"
            new_path = f"images/{new_filename}"
            img.save(new_path, quality=95)

            embedding = get_image_embedding(img)

            batch_ids.append(image_id)
            batch_embeddings.append(embedding.tolist())
            batch_metadatas.append({
                "filename": os.path.basename(img_path),
                "filepath": new_filename,
                "size": os.path.getsize(new_path),
                "batch_job": job_id
            })

            batch_status[job_id]["processed"] += 1

            if len(batch_ids) >= 50:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                batch_ids = []
                batch_embeddings = []
                batch_metadatas = []

        except Exception as e:
            print(f"Failed: {img_path}: {e}")
            batch_status[job_id]["failed"] += 1

    if batch_ids:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

    batch_status[job_id]["status"] = "completed"
    batch_status[job_id]["completed_at"] = datetime.now().isoformat()


@app.get("/search/image/{image_id}")
def search_by_image(image_id: str, top_k: int = 5):
    """
    Find similar images to the given image_id
    """
    try:
        result = collection.get(ids=[image_id], include=["embeddings"])
        if not result["embeddings"]:
            raise HTTPException(404, "Image not found")
        query_embedding = result["embeddings"][0]
    except:
        raise HTTPException(404, "Image not found")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k+1,
        include=["metadatas", "distances"]
    )
    #format response
    search_results = []
    for i, (img_id, distance, metadata) in enumerate(zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0]
    )):
        if img_id == image_id:
            continue

        similarity = 1 - distance

        search_results.append({
            "image_id": img_id,
            "filename": metadata["filepath"],
            "similarity": float(similarity),
            "original_name": metadata["filename"]
        })

    return {
        "query_image_id": image_id,
        "results": search_results[:top_k]
    }

@app.post("/search/text")
def search_by_text(request: TextSearchRequest):
    query_embedding = get_text_embedding(request.query)

    #search in chromadb
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=request.top_k,
        include=["metadatas", "distances"]
    )

    #Format response
    search_results = []
    for img_id, distance, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        similarity = 1 - distance
        search_results.append({
            "image_id": img_id,
            "filename": metadata["filepath"],
            "similarity": float(similarity),
            "original_name": metadata["filename"]
        })

    return {
        "query_text": request.query,
        "results": search_results
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    """Serve image file"""
    filepath = f"images/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(404, "image not found")
    return FileResponse(filepath)

@app.get("/stats")
def get_stats():
    """Get database statistics"""

    return {
        "total_images": collection.count(),
        "storage_path": "chroma_db/",
        "collection_name": collection.name,
        "status": "active"
    }

@app.delete("/image/{image_id}")
def delete_image(image_id: str):
    """Delete image from database and disk"""
    
    #get metadata to find file
    result = collection.get(ids=[image_id], include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(404, "Image not found")
    
    filepath = f"image/{result["metadatas"][0]["filepath"]}"
    collection.delete(ids=[image_id])

    if os.path.exists(filepath):
        os.remove(filepath)
    
    return {
        "message": "image deleted",
        "image_id": image_id,
        "remaining_images": collection.count()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)