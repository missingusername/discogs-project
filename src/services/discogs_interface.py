from contextlib import contextmanager
from io import BytesIO
import requests
import re
import time

from PIL import Image
from pymongo import MongoClient
from tqdm import tqdm

from .rate_limiter import SlidingWindowRateLimiter
from ..utils.logger_utils import get_logger

logger = get_logger(__name__, "Info")


class DiscogsFetcher:
    def __init__(
        self,
        user_token: str,
        mongodb_client: MongoClient,
        user_agent: str,
        rate_limit_stats: dict,
        mode: str = None,
    ):
        self.user_token = user_token
        self.mongodb_client = mongodb_client
        self.user_agent = user_agent
        self.session = requests.Session()
        self.rate_limiter = SlidingWindowRateLimiter(
            max_requests=rate_limit_stats.get("max_reqs", 60),
            period=rate_limit_stats.get("interval", 60),
        )
        self.fetch_mode = mode

    @staticmethod
    def is_valid_image_url(image_url: str) -> bool:
        """
        Check if the given URL is valid.
        """
        return image_url.startswith("http://") or image_url.startswith("https://")

    def _find_result_by_master_id(self, data, target_master_id):
        for result in data["results"]:
            if result["master_id"] == target_master_id:
                return result
        return None

    def _make_request(self, url, headers, params=None):
        while True:
            self.rate_limiter.wait_for_token()
            response = self.session.get(url, headers=headers, params=params)
            if response.status_code == 429:
                logger.warning("Received 429 Too Many Requests. Backing off...")
                self.rate_limiter.backoff()
            else:
                response.raise_for_status()
                return response

    def fetch_image_data(self, image_uri):
        start_time = time.time()
        if not image_uri or not self.is_valid_image_url(image_uri):
            logger.error(f"Invalid image URI: {image_uri}. Skipping download.")
            return {}, 400, time.time() - start_time

        try:
            headers = {
                "User-Agent": self.user_agent,
                "Authorization": f"Discogs token={self.user_token}",
                "Accept": "application/vnd.discogs.v2.discogs+json",
            }
            response = self._make_request(image_uri, headers)

            with BytesIO() as buffer:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                image_format = img.format if img.format else "JPEG"
                img.save(buffer, format=image_format)
                buffer.seek(0)
                image_data_dict = {
                    "image_data": buffer.getvalue(),
                    "data_time_fetched": time.time(),
                    "fetching_status": "processed_and_downloaded",
                }
            document_fields = self._reshape_document_fields_to_embedded(
                embedded_doc_key="album_cover", fields_to_add=image_data_dict
            )
            return document_fields, response.status_code, time.time() - start_time
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            logger.error(f"HTTP error when fetching image data: {e}")
            return {}, status_code, time.time() - start_time

    def fetch_master_popularity(self, master_id: int, document: dict):
        try:
            url = "https://api.discogs.com/database/search"
            headers = {
                "User-Agent": self.user_agent,
                "Authorization": f"Discogs token={self.user_token}",
                "Accept": "application/vnd.discogs.v2.discogs+json",
            }
            cleaned_artists = [
                re.sub(r"\s*\(\d\)\s*", "", artist)
                for artist in document.get("artist_names", [])
            ]

            params = {
                "release_title": document.get("title"),
                "type": "master",
                "artist": ",".join(cleaned_artists),
            }
            start_time = time.time()

            response = self._make_request(url, headers, params)
            search_data = response.json()
            num_items = search_data.get("pagination", {}).get("items", 0)
            master_data = self._find_result_by_master_id(search_data, master_id)

            logger.debug(
                f"Initial search for master_id {master_id} returned {num_items} results. Master data found: {master_data is not None}"
            )

            # If cant find a match due to pagination, retry with modified parameters to narrow search
            if master_data is None and num_items >= 50:
                logger.warning(
                    f"More than 50 results found for {master_id}, modifying parameters and retrying."
                )
                genres = document.get("genres")
                artists = document.get("artist_names")
                if genres:
                    params["genre"] = ",".join(genres)
                if artists:
                    # Clean artist names by removing parentheses containing a single number
                    cleaned_artists = [
                        re.sub(r"\s*\(\d\)\s*", "", artist) for artist in artists
                    ]
                    params["artist"] = ",".join(cleaned_artists)
                logger.debug(f"Modified search parameters: {params}")
                response = self._make_request(url, headers, params)
                search_data = response.json()
                num_items = search_data.get("pagination", {}).get("items", 0)
                master_data = self._find_result_by_master_id(search_data, master_id)
                if master_data:
                    logger.info(f"Found master ID {master_id} after retry.")
                else:
                    logger.warning(
                        f"{num_items} results found for {master_id} after retry. Exiting."
                    )
                    return None, response.status_code, time.time() - start_time

            # Handle edge cases
            if num_items == 0 or master_data is None:
                logger.warning(
                    f"{num_items} results found for Master ID: {master_id} (Release_title: {document.get('title')}), modifying parameters and retrying."
                )
                params["query"] = params.pop("release_title")
                response = self._make_request(url, headers, params)
                search_data = response.json()
                num_items = search_data.get("pagination", {}).get("items", 0)
                logger.debug(
                    f"Retry search returned {num_items} results. Master data found: {master_data is not None}"
                )
                if num_items == 0 or master_data is None:
                    logger.warning("No results found after retry. Exiting.")
                    return None, response.status_code, time.time() - start_time

            if master_data:
                # Construct dictionary here
                new_document_fields = {
                    "image_uri": master_data.get("cover_image", "Image not available"),
                    "popularity": master_data.get("community", {}),
                    "year": master_data.get("year", 0),
                    "country": master_data.get("country", ""),
                    "format": master_data.get("format", []),
                }

                return (
                    new_document_fields,
                    response.status_code,
                    time.time() - start_time,
                )
            else:
                logger.warning(
                    f"Master ID {master_id} not found in search results. Setting image URI to 'Image not available'."
                )
                return (
                    "Image not available",
                    response.status_code,
                    time.time() - start_time,
                )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(
                    f"Master ID {master_id} not found. Setting image URI to 'Image not available'."
                )
                return (
                    "Image not available",
                    e.response.status_code,
                    time.time() - start_time,
                )
            else:
                logger.error(
                    f"HTTP error when fetching image URI for master_id {master_id}: {e}"
                )
                return None, e.response.status_code, time.time() - start_time
        except Exception as e:
            logger.error(
                f"Unexpected error when fetching image URI for master_id {master_id}: {e}"
            )
            return None, 0, time.time() - start_time

    def _reshape_document_fields_to_embedded(
        self, embedded_doc_key: str, fields_to_add: dict
    ):
        reshaped_document_fields = {}
        reshaped_document_fields[embedded_doc_key] = {}
        for key, value in fields_to_add.items():
            reshaped_document_fields[embedded_doc_key][key] = value
        return reshaped_document_fields

    def fetch_image_uri_and_tracklist(self, master_id: int):
        start_time = time.time()
        try:
            url = f"https://api.discogs.com/masters/{master_id}"
            headers = {
                "User-Agent": self.user_agent,
                "Authorization": f"Discogs token={self.user_token}",
                "Accept": "application/vnd.discogs.v2.discogs+json",
            }
            response = self._make_request(url, headers)

            master_data = response.json()
            album_document_fields = {
                "image_uri": (
                    master_data["images"][0]["uri"]
                    if master_data.get("images")
                    else "Image not available"
                ),
                "uri_time_fetched": time.time(),
                "fetching_status": "processed",
            }
            tracklist_document_fields = {"track_list": master_data.get("tracklist", [])}
            album_document_fields = self._reshape_document_fields_to_embedded(
                embedded_doc_key="album_cover", fields_to_add=album_document_fields
            )
            all_document_fields = {**album_document_fields, **tracklist_document_fields}
            return all_document_fields, response.status_code, time.time() - start_time

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(
                    f"Master ID {master_id} not found. Setting image URI to 'Image not available'."
                )
                return (
                    {"image_uri": "Image not available", "track_list": []},
                    e.response.status_code,
                    time.time() - start_time,
                )
            else:
                logger.error(
                    f"HTTP error when fetching image URI for master_id {master_id}: {e}"
                )
                return (
                    {"error": f"HTTP error: {e}"},
                    e.response.status_code,
                    time.time() - start_time,
                )

        except Exception as e:
            logger.error(
                f"Unexpected error when fetching image URI for master_id {master_id}: {e}"
            )
            return {"error": f"Unexpected error: {e}"}, 0, time.time() - start_time

    @contextmanager
    def reset_fetching_status_on_exit(self, document_ids):
        try:
            yield
        except Exception:
            # Only reset the fetching status if an exception occurs
            self.mongodb_client.reset_fetching_status(document_ids)
            raise

    def process_batch(
        self,
        batch_size=100,
        db_query_index="",
        embedded_fields_to_update=[],
        total_num_images_to_download=50000,
    ):
        if self.fetch_mode == "img_download":
            number_of_images = self.mongodb_client.check_num_documents_by_query(
                "covers.get_num_albums_has_image_data.query"
            )
            logger.info(f"Number of images already downloaded: {number_of_images}")

        while True:
            start_time = time.time()
            documents = self.mongodb_client.retrieve_documents_by_query(
                limit=batch_size, as_list=True, query_path=db_query_index
            )
            if not documents:
                logger.info("No more documents to process")
                break

            document_ids = [document["_id"] for document in documents]
            master_ids = [
                document.get("master_id")
                for document in documents
                if document.get("master_id")
            ]

            logger.debug(f"Processing batch with master_ids: {master_ids}")

            updated_documents = []
            request_times = []

            try:
                with self.reset_fetching_status_on_exit(document_ids):
                    for document in tqdm(documents, desc="Processing documents"):
                        master_id = document.get("master_id")
                        if master_id:
                            if self.fetch_mode == "tracklist":
                                document_fields, status_code, request_time = (
                                    self.fetch_image_uri_and_tracklist(master_id)
                                )
                            elif self.fetch_mode == "popularity":
                                document_fields, status_code, request_time = (
                                    self.fetch_master_popularity(master_id, document)
                                )
                            elif self.fetch_mode == "img_download":
                                if total_num_images_to_download is None:
                                    total_num_images_to_download = float("inf")

                                logger.debug(
                                    f"Number of images currently processed: {number_of_images}"
                                )
                                if number_of_images >= total_num_images_to_download:
                                    logger.warning(
                                        f"Reached the limit of {total_num_images_to_download} images to download. Exiting."
                                    )
                                    raise SystemExit(
                                        "Download limit reached. Exiting script."
                                    )
                                image_uri = document.get("album_cover", {}).get(
                                    "image_uri"
                                )
                                document_fields, status_code, request_time = (
                                    self.fetch_image_data(image_uri)
                                )
                            else:
                                logger.error(f"Unknown fetch mode: {self.fetch_mode}")
                                raise ValueError(
                                    f"Unknown fetch mode: {self.fetch_mode}"
                                )

                            request_times.append(request_time)

                            if document_fields:
                                document.update(document_fields)
                                updated_documents.append(document)
                            else:
                                logger.warning(
                                    f"Failed to fetch data for master_id {master_id}. Status code: {status_code}"
                                )

                    if updated_documents:
                        if self.mongodb_client.update_documents_batch(
                            updated_documents, embedded_fields=embedded_fields_to_update
                        ):
                            logger.info(
                                f"Processed and updated {len(updated_documents)} documents."
                            )
                        else:
                            logger.warning(
                                "Failed to update documents in the database."
                            )
                    else:
                        logger.warning("No documents were updated in this batch.")

                    if request_times:
                        sum_time = sum(request_times)
                        avg_time = sum_time / len(request_times)
                        max_time = max(request_times)
                        min_time = min(request_times)
                        logger.info(
                            f"Request time stats - Total: {sum_time:.2f}s, Avg: {avg_time:.2f}s, Max: {max_time:.2f}s, Min: {min_time:.2f}s"
                        )

            except KeyboardInterrupt:
                logger.error(
                    "Keyboard interrupt detected. Resetting fetching status for current batch."
                )
                self.mongodb_client.reset_fetching_status(document_ids)
                raise
            except Exception as e:
                logger.error(f"An error occurred during batch processing: {e}")
                raise

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Batch processing time: {elapsed_time:.2f} seconds")
