import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from umap import UMAP

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from src.utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 42
    n_components: int = 2


class DimensionalityReducer:
    """
    Handles dimensionality reduction of image embeddings and their visualization.
    Uses UMAP for reduction and manages database storage of reduced embeddings.

    Attributes:
        config (UMAPConfig): Configuration parameters for UMAP
        db_client (DatabaseManager): Database client for storing reduced embeddings
        reducer (Optional[UMAP]): UMAP reducer instance after fitting
    """

    def __init__(
        self,
        db_client: DatabaseManager,
        config: Optional[UMAPConfig] = None,
    ):
        """
        Initialize the DimensionalityReducer.

        Args:
            db_client: DatabaseManager instance for storing reduced embeddings
            config: UMAP configuration parameters (optional)
        """
        if db_client is None:
            raise ValueError("DatabaseManager instance must be provided")

        self.config = config or UMAPConfig()
        self.db_client = db_client
        self.reducer: Optional[UMAP] = None

        logger.info(
            f"Initialized DimensionalityReducer with config: "
            f"n_neighbors={self.config.n_neighbors}, "
            f"min_dist={self.config.min_dist}, "
            f"n_components={self.config.n_components}"
        )

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Validate input embeddings array.

        Args:
            embeddings: Input embeddings array

        Raises:
            ValueError: If embeddings are invalid
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array")
        if embeddings.size == 0:
            raise ValueError("Embeddings array is empty")
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

    def fit_transform(
        self,
        embeddings: np.ndarray,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fit UMAP to the embeddings and transform them to lower dimensionality.

        Args:
            embeddings: Array of high-dimensional embeddings
            n_components: Optional override for number of components

        Returns:
            np.ndarray: Reduced embeddings

        Raises:
            ValueError: If embeddings are invalid
            RuntimeError: If UMAP reduction fails
        """
        try:
            self._validate_embeddings(embeddings)

            n_components = n_components or self.config.n_components
            logger.info(f"Fitting UMAP with {n_components} components...")

            self.reducer = UMAP(
                n_neighbors=self.config.n_neighbors,
                min_dist=self.config.min_dist,
                n_components=n_components,
                random_state=self.config.random_state,
            )

            reduced_embeddings = self.reducer.fit_transform(embeddings)
            logger.info("UMAP reduction completed successfully")
            return reduced_embeddings

        except Exception as e:
            logger.error(f"Error during UMAP reduction: {str(e)}")
            raise RuntimeError(f"Failed to perform UMAP reduction: {str(e)}") from e

    def transform_new_points(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using the fitted UMAP model.

        Args:
            embeddings: Array of new embeddings to transform

        Returns:
            np.ndarray: Reduced embeddings

        Raises:
            ValueError: If reducer isn't fitted or embeddings are invalid
            RuntimeError: If transformation fails
        """
        if self.reducer is None:
            raise ValueError("Reducer must be fitted before transforming new points")

        try:
            self._validate_embeddings(embeddings)
            return self.reducer.transform(embeddings)

        except Exception as e:
            logger.error(f"Error transforming new points: {str(e)}")
            raise RuntimeError(f"Failed to transform new points: {str(e)}") from e

    def _collect_embeddings(
        self, documents: List[Dict[str, Any]], embedding_field: str
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Collect embeddings and document IDs from database documents.

        Args:
            documents: List of database documents
            embedding_field: Field name containing embeddings

        Returns:
            Tuple of (embeddings list, document IDs list)

        Raises:
            ValueError: If no valid embeddings found
        """
        logger.info(f"Collecting embeddings from {len(documents)} documents...")

        all_embeddings = []
        all_doc_ids = []

        for doc in documents:
            embedding_data = doc.get(embedding_field, {}).get("embedding")
            if embedding_data:
                try:
                    # Convert embedding to numpy array if it isn't already
                    embedding = np.array(embedding_data, dtype=np.float32)
                    if not np.isnan(embedding).any():
                        all_embeddings.append(embedding)
                        all_doc_ids.append(doc["_id"])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid embedding for document {doc['_id']}: {e}")
                    continue

        if not all_embeddings:
            raise ValueError("No valid embeddings found to process")

        return all_embeddings, all_doc_ids

    def process_and_store_reduced_embeddings(
        self,
        query_path: str,
        embedding_field: str = "album_cover",
    ) -> bool:
        try:
            logger.info(
                f"Retrieving embeddings from database using query: {query_path}"
            )
            documents = self.db_client.retrieve_documents_by_query(
                query_path=query_path,
                limit=None,
                as_list=True,
            )
            logger.info(f"Retrieved {len(documents)} documents")

            if not documents:
                logger.warning("No documents found to process")
                return False

            logger.info("Extracting embeddings from documents...")
            # Collect and validate embeddings
            all_embeddings, all_doc_ids = self._collect_embeddings(
                documents, embedding_field
            )

            logger.info(f"Processing {len(all_doc_ids)} embeddings...")
            # Convert to numpy array and reduce dimensionality
            embeddings_array = np.stack(all_embeddings)
            reduced_embeddings = self.fit_transform(embeddings_array)

            # Prepare updates
            logger.info("Storing reduced embeddings in database...")
            update_documents = []
            for idx, doc_id in enumerate(all_doc_ids):
                reduced_coords = reduced_embeddings[idx].tolist()
                update_doc = {
                    "_id": doc_id,
                    embedding_field: {
                        "reduced_embedding_all": reduced_coords,
                        "last_updated": datetime.datetime.now(datetime.timezone.utc),
                    },
                }
                update_documents.append(update_doc)

            # Pass all updates to client method
            success = self.db_client.update_documents_batch(
                documents=update_documents,
                embedded_fields=[embedding_field],
            )

            if success:
                logger.info(
                    f"Successfully processed and stored {len(all_doc_ids)} reduced embeddings"
                )
            return success

        except Exception as e:
            error_msg = f"Error processing and storing reduced embeddings: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
