import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Visual Search Engine", layout="wide")

# Header
st.title("🔍 Visual Search Engine")
st.markdown("Upload images and search using text or similar images")

# Sidebar - Stats
with st.sidebar:
    st.header("📊 Statistics")
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        
        st.metric("Total Images", stats["total_images"])
        st.metric("Collection", stats["collection_name"])
        
        # Delete image section
        st.markdown("---")
        st.subheader("🗑️ Delete Image")
        delete_id = st.text_input("Image ID to delete", key="delete_input")
        if st.button("Delete", key="delete_btn"):
            if delete_id:
                response = requests.delete(f"{API_URL}/image/{delete_id}")
                if response.status_code == 200:
                    st.success("Image deleted!")
                else:
                    st.error("Image not found")
    except:
        st.error("API not running")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload", "🔤 Text Search", "🖼️ Image Search", "📁 Batch Upload", "📊 Batch Status"])

# Tab 1: Upload
with tab1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    category = st.text_input("Category (optional)", placeholder="e.g., animals, vehicles")
    
    if uploaded_file and st.button("Upload & Index"):
        with st.spinner("Processing..."):
            files = {"file": uploaded_file.getvalue()}
            url = f"{API_URL}/upload" + (f"?category={category}" if category else "")
            
            response = requests.post(url, files={"file": uploaded_file})
            result = response.json()
            
            st.success(f"✅ Uploaded successfully!")
            st.code(f"Image ID: {result['image_id']}")
            st.info(f"Total indexed: {result['total_indexed']}")

# Tab 2: Text Search
with tab2:
    st.header("Search by Text Description")
    text_query = st.text_input("Describe what you're looking for", placeholder="e.g., red car, cute cat")
    text_top_k = st.slider("Number of results", 1, 20, 5)
    
    if st.button("Search", key="text_search"):
        if text_query:
            with st.spinner("Searching..."):
                response = requests.post(
                    f"{API_URL}/search/text",
                    json={"query": text_query, "top_k": text_top_k}
                )
                results = response.json()
                
                st.success(f"Found {len(results['results'])} results")
                
                # Display results in grid
                cols = st.columns(3)
                for idx, result in enumerate(results['results']):
                    with cols[idx % 3]:
                        img_url = f"{API_URL}/image/{result['filename']}"
                        st.image(img_url, width=250)
                        st.caption(f"**{result['similarity']*100:.1f}% match**")
                        st.caption(f"ID: {result['image_id'][:8]}...")
        else:
            st.warning("Please enter search text")

# Tab 3: Image Search - TWO OPTIONS
with tab3:
    st.header("Find Similar Images")
    
    search_method = st.radio("Search method:", ["Upload Image", "Use Image ID"])
    
    if search_method == "Upload Image":
        st.subheader("Upload Image to Search")
        search_file = st.file_uploader("Upload image to find similar", type=["jpg", "jpeg", "png", "webp"], key="search_upload")
        search_top_k = st.slider("Number of results", 1, 20, 5, key="upload_slider")
        
        if search_file and st.button("Find Similar Images", key="upload_search"):
            with st.spinner("Searching..."):
                # First upload the image (temporary)
                files = {"file": search_file.getvalue()}
                upload_response = requests.post(f"{API_URL}/upload", files={"file": search_file})
                temp_result = upload_response.json()
                temp_id = temp_result['image_id']
                
                # Now search using that ID
                response = requests.get(f"{API_URL}/search/image/{temp_id}?top_k={search_top_k}")
                
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Found {len(results['results'])} similar images")
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(results['results']):
                        with cols[idx % 3]:
                            img_url = f"{API_URL}/image/{result['filename']}"
                            st.image(img_url, width=250)
                            st.caption(f"**{result['similarity']*100:.1f}% match**")
                            st.caption(f"ID: {result['image_id'][:8]}...")
                else:
                    st.error("Search failed")
    
    else:  # Use Image ID
        st.subheader("Search by Image ID")
        image_id = st.text_input("Enter Image ID", placeholder="Paste image ID from upload")
        id_top_k = st.slider("Number of results", 1, 20, 5, key="id_slider")
        
        if st.button("Find Similar", key="id_search"):
            if image_id:
                with st.spinner("Searching..."):
                    response = requests.get(f"{API_URL}/search/image/{image_id}?top_k={id_top_k}")
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.success(f"Found {len(results['results'])} similar images")
                        
                        cols = st.columns(3)
                        for idx, result in enumerate(results['results']):
                            with cols[idx % 3]:
                                img_url = f"{API_URL}/image/{result['filename']}"
                                st.image(img_url, width=250)
                                st.caption(f"**{result['similarity']*100:.1f}% match**")
                    else:
                        st.error("Image ID not found")
            else:
                st.warning("Please enter image ID")

# Tab 4: Batch Upload
with tab4:
    st.header("📦 Batch Upload")
    st.markdown("Upload a ZIP file containing multiple images")
    
    zip_file = st.file_uploader("Choose ZIP file", type=["zip"])
    
    if zip_file and st.button("Upload ZIP"):
        with st.spinner("Uploading ZIP file..."):
            files = {"file": zip_file.getvalue()}
            response = requests.post(f"{API_URL}/upload/batch", files={"file": zip_file})
            result = response.json()
            
            st.success(f"✅ Batch upload started!")
            st.code(f"Job ID: {result['job_id']}")
            st.info("Go to 'Batch Status' tab to check progress")

# Tab 5: Batch Status
with tab5:
    st.header("📊 Batch Processing Status")
    
    job_id_input = st.text_input("Enter Job ID", placeholder="Paste job ID from batch upload")
    
    if st.button("Check Status") or job_id_input:
        if job_id_input:
            try:
                response = requests.get(f"{API_URL}/batch/status/{job_id_input}")
                status = response.json()
                
                # Status display
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Status", status['status'].upper())
                col2.metric("Total", status['total'])
                col3.metric("Processed", status['processed'])
                col4.metric("Failed", status['failed'])
                
                # Progress bar
                if status['total'] > 0:
                    progress = status['processed'] / status['total']
                    st.progress(progress)
                
                # Timestamps
                st.markdown(f"**Started:** {status['started_at']}")
                if 'completed_at' in status:
                    st.markdown(f"**Completed:** {status['completed_at']}")
                
                # Auto-refresh if processing
                if status['status'] == 'processing':
                    st.info("🔄 Processing... Refresh to update")
                    
            except:
                st.error("Job ID not found")
        else:
            st.warning("Please enter Job ID")