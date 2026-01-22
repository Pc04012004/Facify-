#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:21:51 2026

@author: prasadchede
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Custom dataset for loading images efficiently"""
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path, True
        except Exception as e:
            logger.warning(f"Failed to load image: {img_path} - {e}")
            return None, img_path, False


class FastEmbeddingModel:
    """Fast embedding model using MobileNetV3 or EfficientNet"""
    
    def __init__(self, model_name: str = 'mobilenetv3_small_100', device: str = None):
        """
        Initialize fast embedding model
        
        Args:
            model_name: Options:
                - 'mobilenetv3_small_100' (fastest, ~2.5M params)
                - 'mobilenetv3_large_100' (faster, ~5.4M params)
                - 'efficientnet_b0' (balanced, ~5.3M params)
                - 'efficientnet_lite0' (very fast, ~4.7M params)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pretrained model
        logger.info(f"Loading model: {model_name}")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model-specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Embedding dimension: {self.model.num_features}")
    
    @torch.no_grad()
    def get_embeddings_batch(self, images: torch.Tensor) -> np.ndarray:
        """Extract embeddings for a batch of images"""
        images = images.to(self.device)
        embeddings = self.model(images)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def get_transform(self):
        """Get the preprocessing transform"""
        return self.transform


def collect_image_paths(image_folder: str, extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')) -> List[str]:
    """Collect all image paths from folder recursively"""
    image_paths = []
    image_folder = Path(image_folder)
    
    logger.info(f"Scanning {image_folder} for images...")
    
    for ext in extensions:
        image_paths.extend(list(image_folder.rglob(f'*{ext}')))
        image_paths.extend(list(image_folder.rglob(f'*{ext.upper()}')))
    
    image_paths = [str(p) for p in image_paths]
    logger.info(f"Found {len(image_paths)} images")
    
    return image_paths


def collate_fn(batch):
    """Custom collate function to handle failed images"""
    images = []
    paths = []
    success_flags = []
    
    for img, path, success in batch:
        if success:
            images.append(img)
        else:
            images.append(torch.zeros(3, 224, 224))  # Placeholder
        paths.append(path)
        success_flags.append(success)
    
    return torch.stack(images), paths, success_flags


def create_vector_store(
    image_folder: str,
    output_path: str = "vector_store.pkl",
    model_name: str = 'mobilenetv3_small_100',
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = None,
    use_fp16: bool = True
):
    """
    Create vector store from images using fast embedding model
    
    Args:
        image_folder: Path to folder containing images
        output_path: Path to save the vector store
        model_name: Model to use (mobilenetv3_small_100, efficientnet_lite0, etc.)
        batch_size: Batch size for processing (higher = faster on GPU)
        num_workers: Number of workers for data loading
        device: 'cuda', 'cpu', or None (auto-detect)
        use_fp16: Use mixed precision for faster inference (GPU only)
    """
    
    # Collect image paths
    image_paths = collect_image_paths(image_folder)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    # Initialize model
    logger.info("Initializing embedding model...")
    model = FastEmbeddingModel(model_name, device)
    
    # Enable mixed precision if requested and on GPU
    use_amp = use_fp16 and model.device == 'cuda'
    if use_amp:
        logger.info("Using mixed precision (FP16) for faster inference")
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform=model.get_transform())
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if model.device == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Process images and create embeddings
    vector_store = {
        'embeddings': [],
        'image_paths': [],
        'failed_images': [],
        'model_name': model_name,
        'embedding_dim': model.model.num_features
    }
    
    logger.info("Processing images and creating embeddings...")
    logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    # Estimate processing speed
    import time
    start_time = time.time()
    processed_count = 0
    
    for batch_idx, (images, paths, success_flags) in enumerate(tqdm(dataloader, desc="Creating embeddings")):
        # Get embeddings with automatic mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                embeddings = model.get_embeddings_batch(images)
        else:
            embeddings = model.get_embeddings_batch(images)
        
        # Store results
        for emb, path, success in zip(embeddings, paths, success_flags):
            if success:
                vector_store['embeddings'].append(emb)
                vector_store['image_paths'].append(path)
                processed_count += 1
            else:
                vector_store['failed_images'].append(path)
        
        # Log speed every 100 batches
        if batch_idx > 0 and batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            speed = processed_count / elapsed
            remaining = len(image_paths) - processed_count
            eta = remaining / speed if speed > 0 else 0
            logger.info(f"Speed: {speed:.0f} images/sec | ETA: {eta/60:.1f} minutes")
    
    # Convert embeddings to numpy array
    vector_store['embeddings'] = np.array(vector_store['embeddings'], dtype=np.float32)
    
    total_time = time.time() - start_time
    logger.info(f"Successfully processed {len(vector_store['embeddings'])} images in {total_time/60:.1f} minutes")
    logger.info(f"Average speed: {len(vector_store['embeddings'])/total_time:.0f} images/sec")
    logger.info(f"Failed to process {len(vector_store['failed_images'])} images")
    logger.info(f"Embedding shape: {vector_store['embeddings'].shape}")
    
    # Save vector store
    logger.info(f"Saving vector store to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(vector_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save as npz for efficient loading
    npz_path = output_path.replace('.pkl', '.npz')
    np.savez_compressed(
        npz_path,
        embeddings=vector_store['embeddings'],
        image_paths=np.array(vector_store['image_paths'], dtype=object),
        model_name=model_name
    )
    logger.info(f"Also saved as compressed npz: {npz_path}")
    
    return vector_store


def load_vector_store(path: str) -> Dict:
    """Load vector store from disk"""
    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        return {
            'embeddings': data['embeddings'],
            'image_paths': data['image_paths'].tolist(),
            'model_name': str(data['model_name'])
        }
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def search_similar_images(
    query_image_path: str,
    vector_store: Dict,
    model: FastEmbeddingModel = None,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Search for similar images in vector store
    
    Args:
        query_image_path: Path to query image
        vector_store: Dictionary containing embeddings and paths
        model: FastEmbeddingModel instance (will create if None)
        top_k: Number of top results to return
    
    Returns:
        List of (image_path, similarity_score) tuples
    """
    # Initialize model if not provided
    if model is None:
        model_name = vector_store.get('model_name', 'mobilenetv3_small_100')
        model = FastEmbeddingModel(model_name)
    
    # Get query embedding
    img = Image.open(query_image_path).convert('RGB')
    img_tensor = model.get_transform()(img).unsqueeze(0)
    query_emb = model.get_embeddings_batch(img_tensor)[0]
    
    # Compute similarities (cosine similarity)
    embeddings = vector_store['embeddings']
    similarities = np.dot(embeddings, query_emb)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return results
    results = [
        (vector_store['image_paths'][idx], float(similarities[idx]))
        for idx in top_indices
    ]
    
    return results


def build_faiss_index(vector_store: Dict, use_gpu: bool = False):
    """
    Build FAISS index for even faster similarity search
    
    Args:
        vector_store: Vector store dictionary
        use_gpu: Use GPU for FAISS (requires faiss-gpu)
    
    Returns:
        FAISS index
    """
    try:
        import faiss
    except ImportError:
        logger.error("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
        return None
    
    embeddings = vector_store['embeddings'].astype('float32')
    dimension = embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        logger.info("Using GPU for FAISS index")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add vectors to index
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors")
    
    return index


if __name__ == "__main__":
    # Configuration
    IMAGE_FOLDER = "images2"  # Change this to your image folder path
    OUTPUT_PATH = "vector_store.pkl"
    
    # Model options (faster to slower, but generally better quality):
    # 'mobilenetv3_small_100' - Fastest (2.5M params) - ~5000 img/sec on GPU
    # 'mobilenetv3_large_100' - Fast (5.4M params) - ~3000 img/sec on GPU
    # 'efficientnet_lite0' - Very fast (4.7M params) - ~3500 img/sec on GPU
    # 'efficientnet_b0' - Balanced (5.3M params) - ~2500 img/sec on GPU
    
    MODEL_NAME = 'mobilenetv3_small_100'
    BATCH_SIZE = 256  # Increase if you have more GPU memory
    NUM_WORKERS = 8   # Increase based on your CPU cores
    USE_FP16 = True   # Use mixed precision (GPU only)
    
    # Create vector store
    try:
        vector_store = create_vector_store(
            image_folder=IMAGE_FOLDER,
            output_path=OUTPUT_PATH,
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_fp16=USE_FP16
        )
        
        logger.info("Vector store creation completed successfully!")
        logger.info(f"Total embeddings: {len(vector_store['embeddings'])}")
        
        # Optional: Build FAISS index for fast search
        # faiss_index = build_faiss_index(vector_store, use_gpu=True)
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise
    
    # Example: Search for similar images
    # vector_store = load_vector_store(OUTPUT_PATH)
    # model = FastEmbeddingModel(MODEL_NAME)
    # results = search_similar_images("query_image.jpg", vector_store, model, top_k=5)
    # for img_path, score in results:
    #     print(f"{img_path}: {score:.4f}")