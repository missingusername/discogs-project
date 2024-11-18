import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Set project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from src.services.dimensionality_reducer import DimensionalityReducer
from src.utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


def main():
    # Setup
    env_file_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file_path)

    mongo_uri = os.environ.get("MONGODB_URI_LOCAL_INTERNAL")
    queries_file_path = (
        Path(__file__).resolve().parents[1] / "config/mongodb_queries.json"
    )
    queries_index_path = "covers.get_albums_has_embeddings.query"

    # Initialize database connection
    db_client = DatabaseManager(
        uri=mongo_uri,
        db_name="discogs_data",
        collection_name="albums",
        queries_file=queries_file_path,
    )

    # Initialize dimensionality reducer with base config from dataclass
    reducer = DimensionalityReducer(db_client=db_client)

    reducer.process_and_store_reduced_embeddings(query_path=queries_index_path)


if __name__ == "__main__":
    main()
