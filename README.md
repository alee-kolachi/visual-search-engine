# 🔍 Visual Search Engine

A FastAPI + Streamlit-based **visual search engine** that allows you to upload images, index them, and perform searches by image or text. This project leverages **CLIP embeddings** for semantic image search and **ChromaDB** for fast vector storage.

---

## Features

- Upload single or multiple images (batch upload via ZIP)
- Index images and generate embeddings for similarity search
- Search for images by:
  - Uploading an image
  - Providing an image ID
  - Text-based queries
- Monitor batch processing status
- Delete images from database and storage
- Display database statistics (total images, collection info)
- Streamlit UI for easy interaction

---

## Screenshots

[![Home Page Screenshot](ss-1.png)](ss-1.png)

---

## Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Image Processing:** PIL (Pillow)  
- **Embeddings:** HuggingFace CLIP (`openai/clip-vit-base-patch32`)  
- **Vector Database:** ChromaDB  
- **Asynchronous Processing:** FastAPI `BackgroundTasks`  
- **File Handling:** Python `os`, `shutil`, `zipfile`

---
