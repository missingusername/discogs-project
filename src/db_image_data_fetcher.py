import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from utils.logger_utils import get_logger

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from src.services.discogs_interface import DiscogsFetcher

# og så tilføje typer cli implementation
def main():
    env_file_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_file_path)

    mongo_uri = os.environ.get("MONGODB_URI_ALL")
    queries_file_path = (
        Path(__file__).resolve().parents[0] / "config/mongodb_queries.json"
    )
    queries_index_path = "covers.get_albums_no_image_data.query"

    client = DatabaseManager(
        uri=mongo_uri,
        db_name="discogs_data",
        collection_name="albums",
        queries_file=queries_file_path,
    )

    discogs_user_token = os.getenv("DISCOGS_API_KEY")
    jl_agent = "jl-prototyping/0.1"
    cloudflare_rate_limit = {
        "max_reqs": 30,
        "interval": 60,
    }
    fetcher = DiscogsFetcher(
        user_token=discogs_user_token,
        mongodb_client=client,
        user_agent=jl_agent,
        rate_limit_stats=cloudflare_rate_limit,
        mode = "img_download",
    )
    fetcher.process_batch(batch_size=30, db_query_index=queries_index_path, embedded_fields_to_update=["album_cover"])

if __name__ == "__main__":
    main()