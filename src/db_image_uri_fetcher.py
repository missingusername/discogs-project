from collections import deque
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import socket
import time
from typing import List, Optional

from dotenv import load_dotenv
import netifaces
import pymongo
import requests
from tqdm import tqdm
import typer

from utils.logger_utils import get_logger

logger = get_logger(__name__, "DEBUG")
app = typer.Typer()

class NetworkUtils:
    @staticmethod
    def get_machine_ip():
        ip_addresses = []
        
        # Method 1: Using socket
        try:
            socket_ip = socket.gethostbyname(socket.gethostname())
            if socket_ip != "127.0.0.1":
                ip_addresses.append(("socket", socket_ip))
            logger.info(f"[SOCKET] IP address: {socket_ip}")
        except Exception as e:
            logger.error(f"Failed to fetch IP address using socket: {e}")
        
        # Method 2: Using netifaces (cross-platform)
        try:
            for interface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr['addr']
                        if ip != "127.0.0.1":
                            ip_addresses.append(("netifaces", ip))
                            logger.info(f"[NETIFACES] IP address for {interface}: {ip}")
        except Exception as e:
            logger.error(f"Failed to fetch IP addresses using netifaces: {e}")
        
        # Analyze results
        if not ip_addresses:
            logger.warning("No non-loopback IP addresses found.")
            return None
        elif len(ip_addresses) == 1:
            logger.info(f"Found one IP address: {ip_addresses[0][1]}")
            return ip_addresses[0][1]
        else:
            logger.info(f"Found multiple IP addresses: {[ip for _, ip in ip_addresses]}")
            return ip_addresses[0][1]  # Returning the first non-loopback IP found

class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        logger.debug(f"Initializing DatabaseManager with URI: {uri}, DB: {db_name}, Collection: {collection_name}")
        try:
            self.client = pymongo.MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            # Test the connection
            self.client.server_info()
        except pymongo.errors.ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def retrieve_documents_without_image_uri(
        self,
        limit: int,
        as_list=False,
    ) -> Optional[List[dict]]:
        logger.debug(f"Retrieving documents without image URI, limit: {limit}, as_list: {as_list}")
        try:
            query = {
                "$and": [
                    {"$or": [
                        {"image_uri": {"$exists": False}},
                        {"image_uri": ""}
                    ]},
                    {"fetching_status": {"$ne": "fetching"}}  # Add this condition
                ]
            }

            documents = []
            for _ in range(limit):
                document = self.collection.find_one_and_update(
                    query,
                    {"$set": {"fetching_status": "fetching"}},
                    return_document=pymongo.ReturnDocument.AFTER
                )
                if document:
                    documents.append(document)
                else:
                    break

            logger.debug(f"Retrieved {len(documents)} documents")
            return documents if as_list else iter(documents)
        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return None

    def update_documents_batch(self, documents: List[dict], timestamp: Optional[bool] = False) -> bool:
        logger.debug(f"Updating batch of {len(documents)} documents with timestamp: {timestamp}")
        try:
            bulk_operations = []
            for document in documents:
                update_fields = {
                    "image_uri": document["image_uri"],
                    "fetching_status": "processed"
                }
                if timestamp:
                    update_fields["timestamp"] = time.time()

                bulk_operations.append(
                    pymongo.UpdateOne(
                        {"_id": document["_id"]},
                        {"$set": update_fields},
                    )
                )

            if bulk_operations:
                result = self.collection.bulk_write(bulk_operations)
                logger.info(f"Bulk write result: {result.bulk_api_result}")
            return True
        except pymongo.errors.BulkWriteError as e:
            logger.error(f"Bulk write operation failed: {e}")
            return False
        
    def reset_fetching_status(self, document_ids: List):
        logger.debug(f"Resetting fetching status for {len(document_ids)} documents")
        try:
            self.collection.update_many(
                {"_id": {"$in": document_ids}},
                {"$set": {"fetching_status": "unprocessed"}}
            )
            logger.info(f"Reset fetching status for {len(document_ids)} documents")
        except pymongo.errors.BulkWriteError as e:
            logger.error(f"Failed to reset fetching status: {e}")


class RateLimiter:
    def __init__(self, max_requests, period):
        self.max_requests = max_requests
        self.period = period
        self.request_times = deque()

    def get_token(self):
        now = time.time()
        while self.request_times and now - self.request_times[0] > self.period:
            self.request_times.popleft()

        if len(self.request_times) < self.max_requests:
            self.request_times.append(now)
            return True
        return False

def rate_limited(func):
    limiter = RateLimiter(max_requests=60, period=60)  # Adjust these values based on Discogs API limits
    max_backoff = 64  # Maximum backoff time in seconds

    @wraps(func)
    def wrapper(*args, **kwargs):
        backoff = 1
        while not limiter.get_token():
            logger.info(f"Rate limit reached, sleeping for {backoff} seconds.")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)  # Exponential backoff with a cap
        logger.debug("Proceeding with the request.")
        return func(*args, **kwargs)

    return wrapper

class DiscogsFetcher:
    def __init__(self, user_token, mongodb_client, user_agent):
        self.user_token = user_token
        self.mongodb_client = mongodb_client
        self.user_agent = user_agent
        self.session = requests.Session()

    @rate_limited
    def fetch_image_uri(self, master_id):
        try:
            url = f"https://api.discogs.com/masters/{master_id}"
            headers = {
                'User-Agent': 'jl-prototyping/0.1',
                'Authorization': f'Discogs token={self.user_token}'
            }
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            master_data = response.json()
            image_uri = master_data['images'][0]['uri'] if master_data.get('images') else "Image not available"
            return image_uri
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Master ID {master_id} not found. Setting image URI to 'Image not available'.")
                return "Image not available"
            elif e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 30))
                logger.error(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                return self.fetch_image_uri(master_id)
            elif e.response.status_code == 500:
                logger.error(f"Server error (500) when fetching image URI for master_id {master_id}. Setting image URI to 'Image not available'.")
                return "Image not available"
            else:
                logger.error(f"HTTP error when fetching image URI for master_id {master_id}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when fetching image URI for master_id {master_id}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for master_id {master_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error when fetching image URI for master_id {master_id}: {e}")
        return "Image not available"
    
    
    @contextmanager
    def reset_fetching_status_on_exit(self, document_ids):
        try:
            yield
        except Exception:
            # Only reset the fetching status if an exception occurs
            self.mongodb_client.reset_fetching_status(document_ids)
            raise

    def process_batch(self, batch_size=100, timestamp=False):
        while True:
            documents = self.mongodb_client.retrieve_documents_without_image_uri(limit=batch_size, as_list=True)
            if not documents:
                logger.info("No more documents to process")
                break

            document_ids = [document["_id"] for document in documents]
            master_ids = [document.get("master_id") for document in documents if document.get("master_id")]

            logger.debug(f"Processing batch with master_ids: {master_ids}")

            updated_documents = []

            try:
                with self.reset_fetching_status_on_exit(document_ids):
                    for document in tqdm(documents, desc="Processing documents"):
                        master_id = document.get("master_id")
                        if master_id:
                            image_uri = self.fetch_image_uri(master_id)
                            if image_uri:
                                document["image_uri"] = image_uri
                                updated_documents.append(document)

                    if updated_documents:
                        if self.mongodb_client.update_documents_batch(updated_documents, timestamp=timestamp):
                            logger.info(f"Processed and updated {len(updated_documents)} documents.")
                        else:
                            logger.warning("Failed to update documents in the database.")
                    else:
                        logger.warning("No documents were updated in this batch.")
            except Exception as e:
                logger.error(f"An error occurred during batch processing: {e}")
                raise



@app.command()
def fetch_images(
    batch_size: int = typer.Option(60, help="Number of documents to process in each batch"),
    env_file: Optional[Path] = typer.Option(None, help="Path to the .env file"),
    mongo_uri: Optional[str] = typer.Option(None, help="MongoDB URI"),
    db_name: str = typer.Option("discogs_data", help="MongoDB database name"),
    collection_name: str = typer.Option("albums", help="MongoDB collection name"),
    user_agent: str = typer.Option("jl-prototyping/0.1", help="User-Agent for Discogs API requests"),
    timestamp: bool = typer.Option(False, help="Add timestamp to documents when updating")
):
    """
    Fetch image URIs for Discogs albums and update the MongoDB database.
    """
    try:
        if env_file:
            load_dotenv(dotenv_path=env_file)
        else:
            env_file_path = Path(__file__).resolve().parents[2] / "Discogs" / ".env"
            load_dotenv(dotenv_path=env_file_path)
        
        if not mongo_uri:
            mongo_uri = os.environ.get("MONGODB_URI_ALL")
        if not mongo_uri:
            typer.secho("MONGO_URI_ALL environment variable not set or empty", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        logger.debug(f"Mongo URI: {mongo_uri}")
        
        mongodb_client = DatabaseManager(
            uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
        )

        discogs_user_token = os.getenv("DISCOGS_API_KEY")
        if not discogs_user_token:
            typer.secho("DISCOGS_API_KEY environment variable not set or empty", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        fetcher = DiscogsFetcher(discogs_user_token, mongodb_client, user_agent)
        fetcher.process_batch(batch_size=batch_size, timestamp=timestamp)

    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()