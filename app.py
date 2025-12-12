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
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)