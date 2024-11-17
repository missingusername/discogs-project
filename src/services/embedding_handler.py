import datetime
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Set project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from src.utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


class EmbeddingHandler:
    """
    Manages the generation and storage of CLIP embeddings for images.

    Attributes:
        model_name (str): Name of the CLIP model being used
        device (str): Device to run the model on ('cuda' or 'cpu')
        batch_size (int): Size of batches for processing
        db_client (DatabaseManager): Database client for storing embeddings
    """

    def __init__(
        self,
        db_client: DatabaseManager,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the EmbeddingManager.

        Args:
            db_client: DatabaseManager instance for storing embeddings
            model_name: Name of the CLIP model to use
            device: Device to run the model on. If None, automatically selects GPU if available
            batch_size: Size of batches for processing
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.db_client = db_client

        # Initialize model and processor
        self.model, self.processor = self._setup_model()
        logger.info(f"Initialized EmbeddingManager with device: {self.device}")

    def _setup_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Initialize and setup CLIP model and processor."""
        logger.debug("Setting up CLIP model and processor...")
        model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(self.model_name)
        model.eval()  # Set model to evaluation mode
        return model, processor

    @staticmethod
    def load_image(image_data: bytes) -> Optional[Image.Image]:
        """Safely load an image from bytes."""
        try:
            return Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def process_image_batch(self, images: List[Image.Image]) -> Optional[torch.Tensor]:
        """Process a batch of images and return embeddings."""
        logger.info(f"Processing batch of {len(images)} images...")
        try:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
                size=224,
            )
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings.cpu()

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return None

    def process_document_batch(
        self,
        documents: List[Dict],
        image_field: str = "album_cover.image_data",
        embedding_field: str = "album_cover",
    ) -> bool:
        """
        Process and store embeddings for a batch of documents.

        Args:
            documents: Batch of documents to process
            image_field: Dot-notation path to the image data field
            embedding_field: Name of the field to store embeddings in

        Returns:
            bool: True if successful, False otherwise
        """
        current_batch = []
        current_batch_ids = []

        # Process documents
        for doc in documents:
            # Directly access the image data field
            image_data = doc.get("album_cover", {}).get("image_data", {})

            if not image_data:
                logger.warning(f"No image data found for document {doc.get('_id')}")
                continue

            image = self.load_image(image_data)

            if image is not None:
                current_batch.append(image)
                current_batch_ids.append(doc["_id"])
            else:
                logger.warning(f"Failed to load image for document {doc.get('_id')}")

        if not current_batch:
            return True  # No images to process in this batch

        # Generate embeddings
        batch_embeddings = self.process_image_batch(current_batch)

        if batch_embeddings is None:
            return False

        # Prepare documents for update
        update_documents = []
        for idx, doc_id in enumerate(current_batch_ids):
            embedding_list = batch_embeddings[idx].numpy().tolist()

            update_doc = {
                "_id": doc_id,
                embedding_field: {
                    "embedding": embedding_list,
                    "fetching_status": "embeddings_generated",
                    "embedding_model": self.model_name,
                    "embedding_dimension": len(embedding_list),
                    "last_updated": datetime.datetime.utcnow(),
                },
            }
            update_documents.append(update_doc)

        # Update database
        success = self.db_client.update_documents_batch(
            documents=update_documents, embedded_fields=[embedding_field]
        )

        if not success:
            logger.error("Failed to update batch")
            return False

        logger.debug(
            f"Successfully processed and stored batch of {len(update_documents)} documents"
        )
        return True

    def process_all_documents(
        self,
        query_path: str,
        image_field: str = "album_cover.image_data",
        embedding_field: str = "album_cover",
        total_docs: Optional[int] = None,
    ) -> bool:
        """
        Process all documents matching the query, generating and storing embeddings in batches.

        Args:
            query_path: Path to the query in the queries file
            image_field: Dot-notation path to the image data field
            embedding_field: Name of the field to store embeddings in
            total_docs: Optional total number of documents to process

        Returns:
            bool: True if all batches were processed successfully
        """
        logger.info("Starting document processing")

        try:
            processed_docs = 0

            with tqdm(total=total_docs, desc="Processing documents") as pbar:
                while True:
                    # Retrieve a batch of documents
                    documents = self.db_client.retrieve_documents_by_query(
                        query_path=query_path,
                        limit=self.batch_size,
                        as_list=True,
                        log_execution=False,
                    )

                    if not documents:
                        break  # No more documents to process

                    success = self.process_document_batch(
                        documents, image_field, embedding_field
                    )

                    if not success:
                        logger.error("Failed to process batch, stopping")
                        return False

                    processed_docs += len(documents)
                    pbar.update(len(documents))

            logger.info(f"Successfully processed {processed_docs} documents")
            return True

        except Exception as e:
            logger.error(f"Error during document processing: {str(e)}")
            return False
