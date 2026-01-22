import streamlit as st
import numpy as np
import torch
from PIL import Image
import timm
from pathlib import Path
import pickle
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Face Similarity Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .match-card {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f8ff;
        margin: 10px 0;
    }
    .percentage {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .camera-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


class FastEmbeddingModel:
    """Fast embedding model using MobileNetV3"""
    
    def __init__(self, model_name: str = 'mobilenetv3_small_100', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model-specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
    
    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a single image"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(img_tensor)
        
        # Normalize embedding
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0]


@st.cache_resource
def load_model(model_name: str = 'mobilenetv3_small_100'):
    """Load and cache the embedding model"""
    with st.spinner('🔄 Loading model...'):
        model = FastEmbeddingModel(model_name)
    return model


@st.cache_data
def load_vector_store(path: str):
    """Load and cache the vector store"""
    with st.spinner('📂 Loading vector store...'):
        if path.endswith('.npz'):
            data = np.load(path, allow_pickle=True)
            return {
                'embeddings': data['embeddings'],
                'image_paths': data['image_paths'].tolist(),
                'model_name': str(data.get('model_name', 'mobilenetv3_small_100'))
            }
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)


def cosine_similarity_to_percentage(similarity: float) -> float:
    """Convert cosine similarity to percentage (0 to 100)"""
    percentage = max(0, min(100, similarity * 100))
    return percentage


def get_match_color(percentage: float) -> str:
    """Get color based on match percentage"""
    if percentage >= 80:
        return "#2ecc71"  # Green
    elif percentage >= 60:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red


def search_similar_images(query_embedding: np.ndarray, vector_store: dict, top_k: int = 5):
    """Search for similar images in vector store"""
    embeddings = vector_store['embeddings']
    similarities = np.dot(embeddings, query_embedding)
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [
        (
            vector_store['image_paths'][idx],
            float(similarities[idx]),
            cosine_similarity_to_percentage(similarities[idx])
        )
        for idx in top_indices
    ]
    
    return results


def save_captured_image(image: Image.Image, save_folder: str = "captured_images") -> str:
    """Save captured image to folder"""
    Path(save_folder).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = Path(save_folder) / filename
    image.save(filepath)
    return str(filepath)


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Face Similarity Search</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    if 'saved_image_path' not in st.session_state:
        st.session_state.saved_image_path = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        vector_store_path = st.text_input(
            "Vector Store Path",
            value="vector_store.npz",
            help="Path to your vector store file (.pkl or .npz)"
        )
        
        model_name = st.selectbox(
            "Model",
            options=[
                'mobilenetv3_small_100',
                'mobilenetv3_large_100',
                'efficientnet_lite0',
                'efficientnet_b0'
            ],
            index=0,
            help="Select the embedding model"
        )
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of similar faces to show"
        )
        
        save_captures = st.checkbox(
            "💾 Save Captured Photos",
            value=True,
            help="Save captured photos to 'captured_images' folder"
        )
        
        st.markdown("---")
        st.info("📌 Use camera to capture a photo and search for similar faces")
        
        # Display stats if vector store is loaded
        try:
            vector_store = load_vector_store(vector_store_path)
            st.success(f"✅ Vector store loaded")
            st.metric("Total Images", f"{len(vector_store['embeddings']):,}")
            st.metric("Embedding Dim", vector_store['embeddings'].shape[1])
        except Exception as e:
            st.error(f"❌ Error loading vector store: {e}")
            vector_store = None
    
    # Main content area
    if vector_store is None:
        st.error("⚠️ Please provide a valid vector store path in the sidebar")
        st.info("Run the vector store creation script first to generate the embeddings")
        return
    
    # Load model
    try:
        model = load_model(model_name)
        st.success(f"✅ Model loaded on: {model.device}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Input method selection
    st.subheader("📸 Choose Input Method")
    input_method = st.radio(
        "How would you like to provide the image?",
        options=["📷 Use Camera", "📤 Upload Image"],
        horizontal=True
    )
    
    query_image = None
    image_source_label = ""
    
    if input_method == "📷 Use Camera":
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📷 Camera Capture Instructions:
        1. **Browser will ask for camera permission** - Click "Allow"
        2. Look for the **camera icon/button below** 
        3. Click it to open your device camera
        4. Take a photo and click "Capture" or checkmark ✓
        5. Photo will appear below automatically
        """)
        
        col_cam1, col_cam2 = st.columns([2, 1])
        
        with col_cam1:
            st.markdown("#### 🎥 Click Camera Icon Below:")
            st.markdown("*(If you don't see camera, check browser permissions)*")
            
            # The camera input - this will show a camera icon
            camera_image = st.camera_input("📸 Open Camera and Take Picture")
            
            if camera_image is not None:
                # Convert to PIL Image
                query_image = Image.open(camera_image).convert('RGB')
                st.session_state.captured_image = query_image
                
                # Save if option is enabled
                if save_captures:
                    saved_path = save_captured_image(query_image)
                    st.session_state.saved_image_path = saved_path
                    st.success(f"✅ Photo saved: {saved_path}")
                
                image_source_label = "📷 Camera capture"
        
        with col_cam2:
            if st.session_state.captured_image is not None:
                st.markdown("#### ✅ Captured Photo:")
                st.image(st.session_state.captured_image, use_container_width=True)
                
                # Option to retake
                if st.button("🔄 Retake Photo", use_container_width=True):
                    st.session_state.captured_image = None
                    st.session_state.saved_image_path = None
                    st.rerun()
            else:
                st.info("📸 No photo captured yet. Click the camera icon on the left.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Use the captured image
        if st.session_state.captured_image is not None:
            query_image = st.session_state.captured_image
    
    else:  # Upload Image
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload a photo of the person you want to search for"
        )
        
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert('RGB')
            image_source_label = "📤 Uploaded image"
    
    # Display results section
    if query_image is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Query Image")
            st.image(query_image, use_container_width=True)
            st.caption(image_source_label)
            
            if st.session_state.saved_image_path:
                st.caption(f"💾 Saved as: {Path(st.session_state.saved_image_path).name}")
            
            # Search button
            search_button = st.button("🔎 Search Similar Faces", type="primary", use_container_width=True)
            
            if search_button:
                with st.spinner('🔍 Searching for similar faces...'):
                    start_time = time.time()
                    
                    # Get query embedding
                    query_embedding = model.get_embedding(query_image)
                    
                    # Search for similar images
                    results = search_similar_images(query_embedding, vector_store, top_k)
                    
                    search_time = time.time() - start_time
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['search_time'] = search_time
        
        with col2:
            if 'results' in st.session_state:
                st.subheader(f"🎯 Top {len(st.session_state['results'])} Matches")
                st.caption(f"⚡ Search completed in {st.session_state['search_time']:.3f} seconds")
                
                results = st.session_state['results']
                
                # Display best match prominently
                if results:
                    best_match_path, best_similarity, best_percentage = results[0]
                    
                    st.markdown("### 🏆 Best Match")
                    
                    # Create a card for best match
                    match_color = get_match_color(best_percentage)
                    st.markdown(f"""
                        <div class="match-card">
                            <div class="percentage" style="color: {match_color};">
                                {best_percentage:.1f}% Match
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        best_image = Image.open(best_match_path)
                        st.image(best_image, use_container_width=True)
                        st.caption(f"📁 {Path(best_match_path).name}")
                        st.caption(f"🔢 Cosine Similarity: {best_similarity:.4f}")
                    except Exception as e:
                        st.error(f"Could not load image: {best_match_path}")
                    
                    # Display other matches
                    if len(results) > 1:
                        st.markdown("---")
                        st.markdown("### 📊 Other Matches")
                        
                        # Create columns for other matches
                        num_cols = min(4, len(results) - 1)
                        cols = st.columns(num_cols)
                        
                        for idx, (img_path, similarity, percentage) in enumerate(results[1:], start=1):
                            col_idx = (idx - 1) % num_cols
                            
                            with cols[col_idx]:
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, use_container_width=True)
                                    
                                    match_color = get_match_color(percentage)
                                    st.markdown(f"""
                                        <div style="text-align: center;">
                                            <span style="font-size: 1.5rem; font-weight: bold; color: {match_color};">
                                                {percentage:.1f}%
                                            </span>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.caption(f"#{idx + 1} - {Path(img_path).name}")
                                except Exception as e:
                                    st.error(f"Error loading image {idx + 1}")
                
                # Download results option
                st.markdown("---")
                if st.button("📥 Download Results as Text"):
                    results_text = "Face Similarity Search Results\n"
                    results_text += "=" * 50 + "\n\n"
                    results_text += f"Query Image: {image_source_label}\n"
                    if st.session_state.saved_image_path:
                        results_text += f"Saved Path: {st.session_state.saved_image_path}\n"
                    results_text += f"Search Time: {st.session_state['search_time']:.3f}s\n\n"
                    
                    for idx, (img_path, similarity, percentage) in enumerate(results, start=1):
                        results_text += f"Rank {idx}:\n"
                        results_text += f"  Path: {img_path}\n"
                        results_text += f"  Match: {percentage:.1f}%\n"
                        results_text += f"  Similarity: {similarity:.4f}\n\n"
                    
                    st.download_button(
                        label="💾 Download Results",
                        data=results_text,
                        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    else:
        # Show instructions when no image
        st.markdown("---")
        st.info("👆 Capture or upload an image above to start searching for similar faces")
        
        st.markdown("### 📖 How to Use:")
        st.markdown("""
        1. **Camera Mode**: 
           - Click the camera button to capture a photo
           - The photo will be automatically saved (if enabled)
           - Click "Search Similar Faces" to find matches
           - Use "Retake Photo" if you want to capture again
        
        2. **Upload Mode**: 
           - Upload an image from your device
           - Click "Search Similar Faces" to find matches
        
        3. View results with match percentages
        """)
        
        st.markdown("### 🎯 Match Percentage Guide:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **80-100%**: Excellent match")
        with col2:
            st.markdown("🟡 **60-80%**: Good match")
        with col3:
            st.markdown("🔴 **<60%**: Low match")


if __name__ == "__main__":
    main()