import json
import time
from typing import List, Optional

import pymongo

from ..utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str, queries_file: str):
        logger.debug(
            f"Initializing DatabaseManager with URI: {uri}, DB: {db_name}, Collection: {collection_name}"
        )
        try:
            self.client = pymongo.MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            # Test the connection
            self.client.server_info()

            # Load queries from JSON file
            with open(queries_file, "r") as file:
                self.queries = json.load(file)
        except (pymongo.errors.ConnectionFailure, FileNotFoundError) as e:
            logger.error(f"Failed to initialize DatabaseManager: {e}")
            raise

    def retrieve_documents_by_query(
        self,
        query_path: str,
        limit: int,
        as_list=False,
        log_execution=False,
    ) -> Optional[List[dict]]:
        logger.debug(
            f"Retrieving documents for query path: {query_path}, limit: {limit}, as_list: {as_list}"
        )
        start_time = time.time()
        try:
            # Retrieve the query using the query_path
            query = self._get_query_by_path(query_path)
            if query is None:
                logger.error(f"Query not found for path: {query_path}")
                return None

            if log_execution:
                execution_plan = self.collection.find(query).explain()
                logger.info(f"Query execution plan: {execution_plan}")

            documents = []
            for _ in range(limit):
                document = self.collection.find_one_and_update(
                    query,
                    {"$set": {"album_cover.fetching_status": "fetching"}},
                    return_document=pymongo.ReturnDocument.AFTER,
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
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Time taken to retrieve documents: {duration:.2f} seconds")

    def _get_query_by_path(self, query_path: str) -> Optional[dict]:
        """
        Retrieve a query from self.queries using a dot-notated path.
        """
        logger.debug(f"Getting query for path: {query_path}")
        parts = query_path.split('.')
        current = self.queries
        for part in parts:
            logger.debug(f"Current part: {part}")
            logger.debug(f"Current structure: {current}")
            if part in current:
                current = current[part]
            else:
                logger.error(f"Invalid query path: {query_path}")
                return None
        if isinstance(current, dict) and "query" in current:
            logger.debug(f"Found query: {current['query']}")
            return current["query"]
        elif isinstance(current, dict):
            logger.debug(f"Found query: {current}")
            return current

    def update_documents_batch(
        self, documents: List[dict], timestamp: Optional[bool] = True
    ) -> bool:
        logger.info(
            f"Updating batch of {len(documents)} documents with timestamp: {timestamp}"
        )
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
                {"$set": {"album_cover.fetching_status": "unprocessed"}},
            )
            logger.info(f"Reset fetching status for {len(document_ids)} documents")
        except pymongo.errors.BulkWriteError as e:
            logger.error(f"Failed to reset fetching status: {e}")
