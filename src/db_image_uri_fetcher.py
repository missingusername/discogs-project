from collections import deque
from functools import wraps
import json
import os
from pathlib import Path
import time
from typing import List, Optional

from dotenv import load_dotenv
import pymongo
from tqdm import tqdm
import requests

from utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


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
    ) -> Optional[pymongo.cursor.Cursor | List[dict]]:
        logger.debug(f"Retrieving documents without image URI, limit: {limit}, as_list: {as_list}")
        try:
            query = {
                "$or": [
                    {"image_uri": {"$exists": False}},
                    {"image_uri": {"$not": {"$regex": "^http"}}},
                    {"image_uri": "Image not available"},
                ]
            }

            cursor = self.collection.find(query).limit(limit)
            documents = list(cursor) if as_list else cursor
            logger.debug(f"Retrieved {len(documents) if as_list else 'a cursor'} documents")
            return documents
        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return None

    def update_documents_batch(self, documents: List[dict]) -> bool:
        logger.debug(f"Updating batch of {len(documents)} documents")
        try:
            bulk_operations = []
            for document in documents:
                bulk_operations.append(
                    pymongo.UpdateOne(
                        {"_id": document["_id"]},
                        {"$set": {"image_uri": document["image_uri"]}},
                    )
                )

            if bulk_operations:
                result = self.collection.bulk_write(bulk_operations)
                logger.info(f"Bulk write result: {result.bulk_api_result}")
            return True
        except pymongo.errors.BulkWriteError as e:
            logger.error(f"Bulk write operation failed: {e}")
            return False


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
    def __init__(self, user_token, mongodb_client):
        self.user_token = user_token
        self.mongodb_client = mongodb_client
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
            else:
                logger.error(f"HTTP error when fetching image URI for master_id {master_id}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when fetching image URI for master_id {master_id}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for master_id {master_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error when fetching image URI for master_id {master_id}: {e}")
        return None

    def process_batch(self, batch_size=100):
        while True:
            documents = self.mongodb_client.retrieve_documents_without_image_uri(limit=batch_size, as_list=True)
            if not documents:
                logger.info("No more documents to process")
                break

            updated_documents = []
            for document in tqdm(documents, desc="Processing documents"):
                master_id = document.get("master_id")
                if master_id:
                    image_uri = self.fetch_image_uri(master_id)
                    if image_uri:
                        document["image_uri"] = image_uri
                        updated_documents.append(document)

            if updated_documents:
                if self.mongodb_client.update_documents_batch(updated_documents):
                    logger.info(f"Processed and updated {len(updated_documents)} documents.")
                else:
                    logger.warning("Failed to update documents in the database.")
            else:
                logger.warning("No documents were updated in this batch.")


def main():
    try:
        env_file_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_file_path)
        
        mongo_uri = os.environ.get("MONGODB_URI_ALL")
        if not mongo_uri:
            logger.error("MONGO_URI_ALL environment variable not set or empty")
            return
        logger.debug(f"Mongo URI: {mongo_uri}")
        
        mongodb_client = DatabaseManager(
            uri=mongo_uri,
            db_name="discogs_data",
            collection_name="albums",
        )

        discogs_user_token = os.getenv("DISCOGS_API_KEY")
        
        fetcher = DiscogsFetcher(discogs_user_token, mongodb_client)
        fetcher.process_batch(batch_size=16)

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()