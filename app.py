from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import uuid
import numpy as np
from typing import List

app = FastAPI()

os.makedirs("images", exist_ok=True)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
MAX_SIZE = 5 * 1024 * 1024
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP Model")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("Model Loaded!")

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

    embedding_file = f"images/{image_id}.npy"
    np.save(embedding_file, embedding)


    return {
        "image_id": image_id,
        "original_filename": file.filename,
        "saved_as": new_filename,
        "size": f"{len(content)} bytes",
        "embedding_shape": embedding.shape,
        "embedding_file": embedding_file
    }
        
@app.get("/search/image/{image_id}")
def search_by_image(image_id: str, top_k: int = 5):
    """
    Find similar images to the given image_id
    """
    query_file = f"images/{image_id}.npy"
    if not os.path.exists(query_file):
        raise HTTPException(404, "Image not found")
    
    query_embedding = np.load(query_file)

    #Get all embeddings
    results = []
    for filename in os.listdir("images"):
        if not filename.endswith(".npy"):
            continue

        #Skip query image itself
        current_id = filename.replace(".npy", "")
        if current_id == image_id:
            continue

        #Load embedidng and calculate similarity
        embedding = np.load(f"images/{filename}")
        similarity = cosine_similarity(query_embedding, embedding)

        #Find the image file
        image_file = None
        for ext in ALLOWED_EXTENSIONS:
            img_path = f"images/{current_id}.{ext}"
            if os.path.exists(img_path):
                image_file = f"{current_id}.{ext}"
                break

        results.append({
            "image_id": current_id,
            "filename": image_file,
            "similarity": float(similarity)
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "query_image_id": image_id,
        "results": results[:top_k]
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    """Serve image file"""
    filepath = f"images/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(404, "image not found")
    return FileResponse(filepath)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)