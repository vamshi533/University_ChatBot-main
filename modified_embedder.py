#!/usr/bin/env python3
"""
SRM University-AP Modified Embedder for Deep Scraping Data

This module loads scraped data from deep scraping, chunks the text, and generates embeddings.
It automatically detects and processes deep scraping data if available.
"""

import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Any, Optional


os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/modified_embedder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SRMAPModifiedEmbedder")


required_packages = ["numpy", "tqdm", "sentence-transformers", "faiss-cpu"]
missing_packages = []

for package in required_packages:
    try:
        __import__(package.replace("-", "_").split(">=")[0].split("==")[0])
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        logger.info("Package installation completed")
    except Exception as e:
        logger.error(f"Failed to install packages: {str(e)}")
        logger.error("Please install the following packages manually: " + ", ".join(missing_packages))
        sys.exit(1)


import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("Failed to import SentenceTransformer. Please install it manually with: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
except ImportError:
    logger.error("Failed to import FAISS. Please install it manually with: pip install faiss-cpu")
    sys.exit(1)

class ModifiedTextEmbedder:
    """Text chunking and embedding for RAG system with deep scraping support."""
    
    def __init__(
        self,
        input_file: str = None,  # Will auto-detect
        output_index_file: str = None,  # Will auto-set based on input
        output_metadata_file: str = None,  # Will auto-set based on input
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_gpu: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize the embedder with configuration parameters.
        
        Args:
            input_file: Path to the scraped data pickle file (auto-detects if None)
            output_index_file: Path to save the FAISS index (auto-sets if None)
            output_metadata_file: Path to save the chunk metadata (auto-sets if None)
            model_name: Name of the SentenceTransformer model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            use_gpu: Whether to use GPU for embedding generation
            batch_size: Batch size for embedding generation
        """
        
        if input_file is None:
            if os.path.exists("srmap_data_deep.pkl"):
                input_file = "srmap_data_deep.pkl"
                logger.info("Detected deep scraping data file")
            else:
                input_file = "srmap_data_async.pkl"
                logger.info("Using standard data file")
        
       
        if "deep" in input_file:
            if output_index_file is None:
                output_index_file = "srmap_faiss_deep.index"
            if output_metadata_file is None:
                output_metadata_file = "srmap_metadata_deep.pkl"
        else:
            if output_index_file is None:
                output_index_file = "srmap_faiss_async.index"
            if output_metadata_file is None:
                output_metadata_file = "srmap_metadata_async.pkl"
        
        self.input_file = input_file
        self.output_index_file = output_index_file
        self.output_metadata_file = output_metadata_file
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output index file: {self.output_index_file}")
        logger.info(f"Output metadata file: {self.output_metadata_file}")
        
        
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model {model_name} with embedding dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            logger.info("Falling back to a different model...")
            try:
                fallback_model = "paraphrase-MiniLM-L6-v2"
                self.model = SentenceTransformer(fallback_model, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_name = fallback_model
                logger.info(f"Loaded fallback model {fallback_model} with embedding dimension {self.embedding_dim}")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {str(e2)}")
                raise RuntimeError("Failed to load any embedding model")
        
        
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Initialized FAISS index")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise
        
        
        self.chunks = []
        self.metadata = []
    
    def _get_device(self) -> str:
        """
        Determine the best available device for the model.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if not self.use_gpu:
            return 'cpu'
        
        try:
            import torch
            if torch.cuda.is_available():
                # Check CUDA memory
                try:
                    free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3) 
                    if free_memory < 1.0:  
                        logger.warning(f"Low CUDA memory ({free_memory:.2f} GB free). Using CPU instead.")
                        return 'cpu'
                except:
                    pass  
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'
    
    def load_data(self) -> Dict[str, str]:
        """
        Load the scraped data from the pickle file.
        
        Returns:
            Dictionary of scraped data {url: text_content}
        """
        try:
            if not os.path.exists(self.input_file):
                logger.error(f"Input file {self.input_file} does not exist")
                
                
                error_file = f"{self.input_file}.error"
                partial_file = f"{self.input_file}.partial"
                interrupted_file = f"{self.input_file}.interrupted"
                
                if os.path.exists(interrupted_file):
                    logger.info(f"Found interrupted file {interrupted_file}, using it instead")
                    self.input_file = interrupted_file
                elif os.path.exists(error_file):
                    logger.info(f"Found error file {error_file}, using it instead")
                    self.input_file = error_file
                elif os.path.exists(partial_file):
                    logger.info(f"Found partial file {partial_file}, using it instead")
                    self.input_file = partial_file
                else:
                   
                    try:
                        from sample_data import generate_sample_data
                        logger.info("Input file not found. Generating sample data...")
                        generate_sample_data()
                        if os.path.exists("srmap_data_async.pkl"):
                            logger.info(f"Successfully generated sample data")
                            self.input_file = "srmap_data_async.pkl"
                        else:
                            raise FileNotFoundError(f"Failed to generate sample data")
                    except ImportError:
                        raise FileNotFoundError(f"Input file {self.input_file} not found and sample_data.py not available")
            
            with open(self.input_file, 'rb') as f:
                data = pickle.load(f)
            
            if not data:
                logger.warning(f"Loaded data from {self.input_file} is empty")
                return {}
                
            logger.info(f"Loaded {len(data)} documents from {self.input_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {self.input_file}: {str(e)}")
            
            
            try:
                from sample_data import generate_sample_data
                logger.info("Generating sample data as fallback...")
                generate_sample_data()
                
                with open("srmap_data_async.pkl", 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded {len(data)} documents from generated sample data")
                return data
            except:
                logger.error("Failed to generate sample data as fallback")
                return {}
    
    def chunk_text(self, url: str, text: str) -> List[Tuple[str, Dict]]:
        """
        Split text into overlapping chunks.
        
        Args:
            url: Source URL of the text
            text: Text to chunk
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        
        if not text or len(text) < 50:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            
            
            if len(chunk) < 50:
                break
            
            
            chunk_metadata = {
                'url': url,
                'start': start,
                'end': end,
                'chunk_id': len(self.chunks)
            }
            
            
            chunks.append((chunk, chunk_metadata))
            
            
            start = start + self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_data(self, data: Dict[str, str]) -> None:
        """
        Process the data: chunk text and generate embeddings.
        
        Args:
            data: Dictionary of scraped data {url: text_content}
        """
        
        logger.info("Chunking text...")
        try:
            for url, text in tqdm(data.items(), desc="Chunking"):
                chunks = self.chunk_text(url, text)
                for chunk_text, chunk_metadata in chunks:
                    self.chunks.append(chunk_text)
                    self.metadata.append(chunk_metadata)
            
            logger.info(f"Created {len(self.chunks)} chunks from {len(data)} documents")
            
            
            self._save_intermediate_metadata()
            
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            if self.chunks:
                logger.info(f"Proceeding with {len(self.chunks)} chunks that were successfully created")
            else:
                raise
        
        
        logger.info("Generating embeddings...")
        try:
            all_embeddings = []
            
            
            for i in tqdm(range(0, len(self.chunks), self.batch_size), desc="Embedding"):
                batch = self.chunks[i:i+self.batch_size]
                try:
                    embeddings = self.model.encode(batch, show_progress_bar=False)
                    all_embeddings.append(embeddings)
                    
                    
                    if (i // self.batch_size) % 10 == 0 and i > 0:
                        self._save_intermediate_embeddings(all_embeddings)
                        
                except Exception as e:
                    logger.error(f"Error embedding batch {i//self.batch_size}: {str(e)}")
                    
                    smaller_batch_size = max(1, self.batch_size // 2)
                    logger.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                    
                    for j in range(i, min(i+self.batch_size, len(self.chunks)), smaller_batch_size):
                        sub_batch = self.chunks[j:j+smaller_batch_size]
                        try:
                            sub_embeddings = self.model.encode(sub_batch, show_progress_bar=False)
                            all_embeddings.append(sub_embeddings)
                        except Exception as e2:
                            logger.error(f"Error embedding sub-batch: {str(e2)}")
                           
                            continue
            
            
            if all_embeddings:
                embeddings = np.vstack(all_embeddings)
                
                
                logger.info("Adding embeddings to FAISS index...")
                self.index.add(embeddings.astype(np.float32))
                
                logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
                
                
                if len(embeddings) < len(self.chunks):
                    logger.warning(f"Only {len(embeddings)} embeddings were created for {len(self.chunks)} chunks")
                    self.chunks = self.chunks[:len(embeddings)]
                    self.metadata = self.metadata[:len(embeddings)]
            else:
                raise RuntimeError("No embeddings were successfully created")
                
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise
    
    def _save_intermediate_metadata(self) -> None:
        """Save intermediate metadata during processing."""
        try:
            metadata_dict = {
                'chunks': self.chunks,
                'metadata': self.metadata,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }
            
            with open(f"{self.output_metadata_file}.partial", 'wb') as f:
                pickle.dump(metadata_dict, f)
            
            logger.info(f"Saved intermediate metadata to {self.output_metadata_file}.partial")
        except Exception as e:
            logger.error(f"Error saving intermediate metadata: {str(e)}")
    
    def _save_intermediate_embeddings(self, all_embeddings: List[np.ndarray]) -> None:
        """
        Save intermediate embeddings during processing.
        
        Args:
            all_embeddings: List of embedding arrays
        """
        try:
            
            embeddings = np.vstack(all_embeddings)
            temp_index = faiss.IndexFlatL2(self.embedding_dim)
            temp_index.add(embeddings.astype(np.float32))
            
            
            faiss.write_index(temp_index, f"{self.output_index_file}.partial")
            
            logger.info(f"Saved intermediate embeddings to {self.output_index_file}.partial")
        except Exception as e:
            logger.error(f"Error saving intermediate embeddings: {str(e)}")
    
    def save_index_and_metadata(self) -> None:
        """Save the FAISS index and chunk metadata."""
        
        try:
            faiss.write_index(self.index, self.output_index_file)
            logger.info(f"Saved FAISS index to {self.output_index_file}")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {self.output_index_file}: {str(e)}")
            
           
            try:
                backup_file = f"backup_index_{int(time.time())}.index"
                faiss.write_index(self.index, backup_file)
                logger.info(f"Saved backup FAISS index to {backup_file}")
            except Exception as e2:
                logger.error(f"Error saving backup FAISS index: {str(e2)}")
        
        
        try:
            metadata_dict = {
                'chunks': self.chunks,
                'metadata': self.metadata,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }
            
            with open(self.output_metadata_file, 'wb') as f:
                pickle.dump(metadata_dict, f)
            
            logger.info(f"Saved metadata to {self.output_metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata to {self.output_metadata_file}: {str(e)}")
            
            
            try:
                backup_file = f"backup_metadata_{int(time.time())}.pkl"
                with open(backup_file, 'wb') as f:
                    pickle.dump(metadata_dict, f)
                logger.info(f"Saved backup metadata to {backup_file}")
            except Exception as e2:
                logger.error(f"Error saving backup metadata: {str(e2)}")
    
    def run(self) -> bool:
        """
        Run the embedder pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            
            data = self.load_data()
            if not data:
                logger.error("No data to process. Exiting.")
                return False
            
            
            self.process_data(data)
            
            
            self.save_index_and_metadata()
            
            logger.info("Embedding process completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in embedder pipeline: {str(e)}")
            return False

def main():
    """Main function to run the embedder."""
    try:
        
        embedder = ModifiedTextEmbedder()
        
        
        success = embedder.run()
        return success
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return False

if __name__ == "__main__":
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Embedder stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)
