# Skal refactor DiscogsFetchers fetching metoder til at returne en tuple med værdier

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import re
import socket
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import netifaces
import pymongo
import requests
from tqdm import tqdm
import typer

from utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")
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
                        {"album_cover.image_uri": {"$exists": False}},
                        {"album_cover.image_uri": ""},
                        
                    ]},
                    {"album_cover.fetching_status": {"$ne": "fetching"}},
                    {"image_uri": {"$exists": False}},
                ]
            }

            documents = []
            for _ in range(limit):
                document = self.collection.find_one_and_update(
                    query,
                    {"$set": {"album_cover.fetching_status": "fetching"}},
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
                update_fields = {}
                album_cover_fields = document.get("album_cover", {})

                # Handle all other fields dynamically
                for key, value in document.items():
                    if key not in ["_id", "image_uri", "album_cover"]:
                        update_fields[key] = value

                if timestamp:
                    album_cover_fields["uri_time_fetched"] = time.time()

                if album_cover_fields:
                    album_cover_fields["fetching_status"] = "processed"
                    update_fields["album_cover"] = album_cover_fields

                bulk_operations.append(
                    pymongo.UpdateOne(
                        {"_id": document["_id"]},
                        {"$set": update_fields},
                    )
                )

            if bulk_operations:
                result = self.collection.bulk_write(bulk_operations)
                logger.debug(f"Bulk write result: {result.bulk_api_result}")
            return True
        except pymongo.errors.BulkWriteError as e:
            logger.error(f"Bulk write operation failed: {e}")
        return False
        
    def reset_fetching_status(self, document_ids: List):
        logger.debug(f"Resetting fetching status for {len(document_ids)} documents")
        try:
            self.collection.update_many(
                {"_id": {"$in": document_ids}},
                {"$set": {"album_cover.fetching_status": "unprocessed"}}
            )
            logger.info(f"Reset fetching status for {len(document_ids)} documents")
        except pymongo.errors.BulkWriteError as e:
            logger.error(f"Failed to reset fetching status: {e}")

class RateLimiter:
    def __init__(self, max_requests, period):
        self.max_requests = max_requests
        self.period = period
        self.remaining = max_requests
        self.reset_time = time.time() + period

    def update_limits(self, remaining: int, reset_time: float):
        self.remaining = remaining
        self.reset_time = reset_time

    def get_token(self):
        now = time.time()
        if now >= self.reset_time:
            # If we've passed the reset time, reset the remaining requests
            self.remaining = self.max_requests
            self.reset_time = now + self.period
        
        if self.remaining > 0:
            self.remaining -= 1
            return True
        return False

def rate_limited(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        while not self.rate_limiter.get_token():
            sleep_time = max(0, self.rate_limiter.reset_time - time.time())
            logger.info(f"Rate limit reached. Remaining: {self.rate_limiter.remaining}, Reset Time: {self.rate_limiter.reset_time}, Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        return func(self, *args, **kwargs)
    return wrapper

class DiscogsFetcher:
    def __init__(self, user_token, mongodb_client, user_agent, mode):
        self.user_token = user_token
        self.mongodb_client = mongodb_client
        self.user_agent = user_agent
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(max_requests=60, period=60)
        self.fetch_mode = mode

    def _find_result_by_master_id(self, data, target_master_id):
        for result in data['results']:
            if result['master_id'] == target_master_id:
                return result
        return None

    @rate_limited
    def make_request(self, url, headers, params):
        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        self.update_rate_limiter(response)
        return response

    def update_rate_limiter(self, response):
        remaining = int(response.headers.get('X-Discogs-Ratelimit-Remaining', self.rate_limiter.remaining))
        reset_time = float(response.headers.get('X-Discogs-Ratelimit-Reset', time.time() + 60))
        logger.debug(f"API Response - Rate limit remaining: {remaining}, reset time: {reset_time}")
        logger.debug(f"Before update - Rate limiter state: remaining={self.rate_limiter.remaining}, reset_time={self.rate_limiter.reset_time}")
        self.rate_limiter.update_limits(remaining, reset_time)
        logger.warning(response.headers)
        logger.debug(f"After update - Rate limiter state: remaining={self.rate_limiter.remaining}, reset_time={self.rate_limiter.reset_time}")

    @rate_limited
    def fetch_master_popularity(self, master_id: int, document: dict):
        try:
            url = "https://api.discogs.com/database/search"
            headers = {
                'User-Agent': self.user_agent,
                'Authorization': f"Discogs token={self.user_token}",
                'Accept': 'application/vnd.discogs.v2.discogs+json'
            }
            cleaned_artists = [re.sub(r'\s*\(\d\)\s*', '', artist) for artist in document.get("artist_names", [])]
            
            params = {
                'release_title': document.get("title"),
                'type': 'master',
                'artist': ','.join(cleaned_artists)
                
            }
            start_time = time.time()
            
            response = self.make_request(url, headers, params)
            search_data = response.json()
            num_items = search_data.get('pagination', {}).get('items', 0)
            master_data = self._find_result_by_master_id(search_data, master_id)
            
            logger.debug(f"Initial search for master_id {master_id} returned {num_items} results. Master data found: {master_data is not None}")
           
            # If cant find a match due to pagination, retry with modified parameters to narrow search
            if master_data is None and num_items >= 50:
                logger.warning(f"More than 50 results found for {master_id}, modifying parameters and retrying.")
                genres = document.get("genres")
                artists = document.get("artist_names")
                if genres:
                    params['genre'] = ','.join(genres)
                if artists:
                    # Clean artist names by removing parentheses containing a single number
                    cleaned_artists = [re.sub(r'\s*\(\d\)\s*', '', artist) for artist in artists]
                    params['artist'] = ','.join(cleaned_artists)
                logger.debug(f"Modified search parameters: {params}")
                response = self.make_request(url, headers, params)
                search_data = response.json()
                num_items = search_data.get('pagination', {}).get('items', 0)
                master_data = self._find_result_by_master_id(search_data, master_id)
                if master_data:
                    logger.info(f"Found master ID {master_id} after retry.")
                else:
                    logger.warning(f"{num_items} results found for {master_id} after retry. Exiting.")
                    return None, response.status_code, time.time() - start_time

            # Handle edge cases
            if num_items == 0 or master_data is None:
                logger.warning(f"{num_items} results found for Master ID: {master_id} (Release_title: {document.get('title')}), modifying parameters and retrying.")
                params['query'] = params.pop('release_title')
                response = self.make_request(url, headers, params)
                search_data = response.json()
                num_items = search_data.get('pagination', {}).get('items', 0)
                logger.debug(f"Retry search returned {num_items} results. Master data found: {master_data is not None}")
                if num_items == 0 or master_data is None:
                    logger.warning("No results found after retry. Exiting.")
                    return None, response.status_code, time.time() - start_time
            
            if master_data:
                # Construct dictionary here
                new_document_fields = {
                    'image_uri': master_data.get('cover_image', "Image not available"),
                    'popularity': master_data.get('community', {}),
                    'year': master_data.get('year', 0),
                    'country': master_data.get('country', ""),
                    'format': master_data.get('format', [])
                }
                
                return new_document_fields, response.status_code, time.time() - start_time
            else:
                logger.warning(f"Master ID {master_id} not found in search results. Setting image URI to 'Image not available'.")
                return "Image not available", response.status_code, time.time() - start_time
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Master ID {master_id} not found. Setting image URI to 'Image not available'.")
                return "Image not available", e.response.status_code, time.time() - start_time
            elif e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 30))
                logger.warning(f"Rate limit exceeded. Updating rate limiter.")
                self.rate_limiter.update_limits(0, time.time() + retry_after)
                return None, e.response.status_code, time.time() - start_time
            else:
                logger.error(f"HTTP error when fetching image URI for master_id {master_id}: {e}")
                return None, e.response.status_code, time.time() - start_time
        except Exception as e:
            logger.error(f"Unexpected error when fetching image URI for master_id {master_id}: {e}")
            return None, 0, time.time() - start_time

    @rate_limited
    def fetch_image_uri_and_tracklist(self, master_id: int):
        try:
            url = f"https://api.discogs.com/masters/{master_id}"
            headers = {
                'User-Agent': self.user_agent,
                'Authorization': f'Discogs token={self.user_token}',
                'Accept': 'application/vnd.discogs.v2.discogs+json'
            }
            start_time = time.time()
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            # Update rate limiter based on response headers
            remaining = int(response.headers.get('X-Discogs-Ratelimit-Remaining', self.rate_limiter.remaining))
            reset_time = float(response.headers.get('X-Discogs-Ratelimit-Reset', time.time() + 60))
            logger.debug(f"API Response - Rate limit remaining: {remaining}, reset time: {reset_time}")
            logger.debug(f"Before update - Rate limiter state: remaining={self.rate_limiter.remaining}, reset_time={self.rate_limiter.reset_time}")
            self.rate_limiter.update_limits(remaining, reset_time)
            logger.debug(f"After update - Rate limiter state: remaining={self.rate_limiter.remaining}, reset_time={self.rate_limiter.reset_time}")
            logger.warning(response.headers.get("X-Discogs-Ratelimit-Reset"))
            
            master_data = response.json()
            new_document_fields = {
            'image_uri': master_data['images'][0]['uri'] if master_data.get('images') else "Image not available",
            'track_list': master_data['tracklist'] if master_data.get('tracklist') else []
            }
            return new_document_fields, response.status_code, time.time() - start_time
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Master ID {master_id} not found. Setting image URI to 'Image not available'.")
                return "Image not available", e.response.status_code, time.time() - start_time
            elif e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 30))
                logger.error(f"Rate limit exceeded. Updating rate limiter.")
                self.rate_limiter.update_limits(0, time.time() + retry_after)
                return None, e.response.status_code, time.time() - start_time
            else:
                logger.error(f"HTTP error when fetching image URI for master_id {master_id}: {e}")
                return None, e.response.status_code, time.time() - start_time
        except Exception as e:
            logger.error(f"Unexpected error when fetching image URI for master_id {master_id}: {e}")
            return None, 0, time.time() - start_time
    
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
            start_time = time.time()
            documents = self.mongodb_client.retrieve_documents_without_image_uri(limit=batch_size, as_list=True)
            if not documents:
                logger.info("No more documents to process")
                break

            document_ids = [document["_id"] for document in documents]
            master_ids = [document.get("master_id") for document in documents if document.get("master_id")]

            logger.debug(f"Processing batch with master_ids: {master_ids}")

            updated_documents = []
            request_times = []

            try:
                with self.reset_fetching_status_on_exit(document_ids):
                    for document in tqdm(documents, desc="Processing documents"):
                        master_id = document.get("master_id")
                        if master_id:
                            while True:
                                if self.fetch_mode == "tracklist":
                                    document_fields, status_code, request_time = self.fetch_image_uri_and_tracklist(master_id)
                                elif self.fetch_mode == "popularity":
                                    document_fields, status_code, request_time = self.fetch_master_popularity(master_id, document)
                                    time.sleep(0.6)
                                else:
                                    logger.error(f"Unknown fetch mode: {self.fetch_mode}")
                                    raise ValueError(f"Unknown fetch mode: {self.fetch_mode}")

                                request_times.append(request_time)

                                if status_code == 429:
                                    logger.info("Rate limit reached. Waiting before retrying.")
                                    sleep_time = max(0, self.rate_limiter.reset_time - time.time())
                                    time.sleep(sleep_time)
                                    continue

                                if document_fields:
                                    # Update document with new fields
                                    if 'image_uri' in document_fields:
                                        document['album_cover'] = document.get('album_cover', {})
                                        document['album_cover']['image_uri'] = document_fields.pop('image_uri')
                                    document.update(document_fields)
                                    updated_documents.append(document)
                                else:
                                    logger.warning(f"Failed to fetch data for master_id {master_id}. Status code: {status_code}")

                                break  # Exit the while loop after successful fetch or non-429 error

                    if updated_documents:
                        if self.mongodb_client.update_documents_batch(updated_documents, timestamp=timestamp):
                            logger.debug(f"Processed and updated {len(updated_documents)} documents.")
                        else:
                            logger.warning("Failed to update documents in the database.")
                    else:
                        logger.warning("No documents were updated in this batch.")

                    # Log request time statistics
                    if request_times:
                        sum_time = sum(request_times)
                        avg_time = sum_time / len(request_times)
                        max_time = max(request_times)
                        min_time = min(request_times)
                        logger.info(f"Request time stats - Total: {sum_time:.2f}s, Avg: {avg_time:.2f}s, Max: {max_time:.2f}s, Min: {min_time:.2f}s")

            except KeyboardInterrupt:
                logger.error("Keyboard interrupt detected. Resetting fetching status for current batch.")
                self.mongodb_client.reset_fetching_status(document_ids)
                raise  # Re-raise the exception to halt the script
            except Exception as e:
                logger.error(f"An error occurred during batch processing: {e}")
                raise

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Batch processing time: {elapsed_time:.2f} seconds")


@app.command()
def fetch_images(
    batch_size: int = typer.Option(60, help="Number of documents to process in each batch"),
    env_file: Optional[Path] = typer.Option(None, help="Path to the .env file"),
    mongo_uri: Optional[str] = typer.Option(None, help="MongoDB URI"),
    db_name: str = typer.Option("discogs_data", help="MongoDB database name"),
    collection_name: str = typer.Option("albums", help="MongoDB collection name"),
    user_agent: str = typer.Option("jl-prototyping/0.1", help="User-Agent for Discogs API requests"),
    timestamp: bool = typer.Option(True, help="Add timestamp to documents when updating")
):
    """
    Fetch image URIs for Discogs albums and update the MongoDB database.
    """
    try:
        if env_file:
            load_dotenv(dotenv_path=env_file)
        else:
            env_file_path = Path(__file__).resolve().parents[2] / "Discogs/" ".env"
            logger.info(env_file_path)
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

        discogs_user_token = os.getenv("DISCOGS_API_KEY_GABRIEL")
        if not discogs_user_token:
            typer.secho("DISCOGS_API_KEY environment variable not set or empty", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        fetcher = DiscogsFetcher(discogs_user_token, mongodb_client, user_agent, 'tracklist')
        fetcher.process_batch(batch_size=batch_size, timestamp=timestamp)

    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()