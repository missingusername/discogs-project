import os
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


def main():
    # Load environment variables from .env file from root directory
    load_dotenv()

    # Setup variables for accessing database
    uri = os.getenv("MONGODB_URI_ALL")
    db_name = "discogs_data"
    collection_name = "albums"

    queries_file_path = (
        Path(__file__).resolve().parents[0] / "config/mongodb_queries.json"
    )

    # Initialize the DatabaseManager
    client = DatabaseManager(uri, db_name, collection_name, queries_file_path)

    # client.convert_string_arrays_to_arrays("artist_names")
    merge_test = client.merge_duplicate_documents(
        field="master_id", merge_strategy={"master_id": "first"}, dry_run=False
    )
    counts = client.analyze_array_field_types("artist_names")

    # client.identify_duplicate_documents_by_field("master_id")
    """ num_query_docs = client.count_documents_by_filter(
        "album_metadata.get_artist_names_field_string.query"
    )
    logger.info(f"Number of documents matching query: {num_query_docs}")

    num_docs = client.count_documents_agg_pipeline(
        "album_metadata.get_artist_names_field_string.pipeline"
    )
    logger.info(f"Number of documents matching pipeline: {num_docs}")

    num_unique_masters = client.count_documents_agg_pipeline(
        "album_metadata.get_unique_master_ids.pipeline"
    )
    logger.info(f"Number of unique master ids: {num_unique_masters}")
 """
    # client.create_dump()


if __name__ == "__main__":
    main()
