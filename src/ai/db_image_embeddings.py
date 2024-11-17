import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Set project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from src.services.embedding_handler import EmbeddingHandler
from src.utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


# Example usage:
def main():
    # Setup
    env_file_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file_path)

    mongo_uri = os.environ.get("MONGODB_URI_ALL")
    queries_file_path = (
        Path(__file__).resolve().parents[1] / "config/mongodb_queries.json"
    )
    queries_index_path = "covers.get_albums_has_image_data_no_embeddings.query"

    # Initialize database connection
    db_client = DatabaseManager(
        uri=mongo_uri,
        db_name="discogs_data",
        collection_name="albums",
        queries_file=queries_file_path,
    )

    # Initialize embedding manager
    embedding_manager = EmbeddingHandler(db_client=db_client)

    total_docs = db_client.count_documents_by_filter(queries_index_path)
    logger.info(f"Total documents to process: {total_docs}")

    # Process all documents
    success = embedding_manager.process_all_documents(
        queries_index_path, total_docs=total_docs
    )


if __name__ == "__main__":
    main()
