import os
from pathlib import Path
import sys
from typing import Optional

from dotenv import load_dotenv
import typer

from utils.logger_utils import get_logger

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.database_manager import DatabaseManager
from src.services.discogs_interface import DiscogsFetcher

logger = get_logger(__name__, "INFO")
app = typer.Typer()


@app.command()
def fetch_images(
    batch_size: int = typer.Option(
        6, help="Number of documents to process in each batch"
    ),
    env_file: Optional[Path] = typer.Option(None, help="Path to the .env file"),
    mongo_uri: Optional[str] = typer.Option(None, help="MongoDB URI"),
    db_name: str = typer.Option("discogs_data", help="MongoDB database name"),
    collection_name: str = typer.Option("albums", help="MongoDB collection name"),
    user_agent: str = typer.Option(
        "jl-prototyping/0.1", help="User-Agent for Discogs API requests"
    ),
    timestamp: bool = typer.Option(
        True, help="Add timestamp to documents when updating"
    ),
    queries_file: str = typer.Option(
        "./src/config/mongodb_queries.json",
        help="Path to the MongoDB queries JSON file",
    ),
    queries_index_path: str = typer.Option(
        "covers.get_albums_no_uri.query", help="Path to the query in the JSON file"
    ),
):
    """
    Fetch image URIs for Discogs albums and update the MongoDB database.
    """
    try:
        if env_file:
            load_dotenv(dotenv_path=env_file)
        else:
            env_file_path = Path(__file__).resolve().parents[1] / ".env"
            logger.debug(env_file_path)
            load_dotenv(dotenv_path=env_file_path)

        if not mongo_uri:
            mongo_uri = os.environ.get("MONGODB_URI_ALL")
        if not mongo_uri:
            typer.secho(
                "MONGO_URI_ALL environment variable not set or empty",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        logger.debug(f"Mongo URI: {mongo_uri}")

        # Resolve the relative path to an absolute path
        queries_file_path = Path(queries_file).resolve()

        mongodb_client = DatabaseManager(
            uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
            queries_file=queries_file_path,  # Pass the resolved path here
        )

        discogs_user_token = os.getenv("DISCOGS_API_KEY")
        discogs_rate_limit = {
            "max_reqs": 60,
            "interval": 60,
        }
        if not discogs_user_token:
            typer.secho(
                "DISCOGS_API_KEY environment variable not set or empty",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        fetcher = DiscogsFetcher(
            discogs_user_token, mongodb_client, user_agent, discogs_rate_limit, "tracklist", 
        )
        fetcher.process_batch(batch_size=batch_size, db_query_index=queries_index_path)

    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
