import json
import time
from typing import List, Optional, Union, Iterator

import pymongo
from pymongo.cursor import Cursor

from ..utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str, queries_file: str):
        logger.info(f"Initializing DatabaseManager with db_name: {db_name}, collection_name: {collection_name}")
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
    
    def check_num_documents_by_query(self, query_path: str) -> int:
        logger.debug(f"Checking number of documents for query path: {query_path}")
        try:
            query = self._get_query_by_path(query_path)
            if query is None:
                logger.error(f"Query not found for path: {query_path}")
                return 0
            num_documents = self.collection.count_documents(query)
            logger.debug(f"Found {num_documents} documents")
            return num_documents
        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to check number of documents: {e}")
            return 0

    def retrieve_documents_by_query(
        self,
        query_path: str,
        limit: int,
        as_list: bool = False,
        log_execution: bool = False,
    ) -> Optional[Union[List[dict], Iterator[dict]]]:
        logger.debug(
            f"Retrieving documents for query path: {query_path}, limit: {limit}, as_list: {as_list}"
        )
        start_time = time.time()

        try:
            query = self._get_query_by_path(query_path)
            if query is None:
                logger.error(f"Query not found for path: {query_path}")
                return None

            if log_execution:
                execution_plan = self.collection.find(query).explain()
                logger.info(f"Query execution plan: {execution_plan}")

            update = {
                "$set": {"album_cover.fetching_status": "fetching"},
                "$currentDate": {"lastModified": True}
            }

            cursor: Cursor = self.collection.find(query).sort("_id", pymongo.ASCENDING).limit(limit)
            
            documents = []
            for doc in cursor:
                updated_doc = self.collection.find_one_and_update(
                    {"_id": doc["_id"]},
                    update,
                    return_document=pymongo.ReturnDocument.AFTER
                )
                if updated_doc:
                    documents.append(updated_doc)

            result = documents if as_list else iter(documents)
            logger.debug(f"Retrieved {len(documents)} documents")
            return result

        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return None

        finally:
            duration = time.time() - start_time
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

    def update_documents_batch(self, documents: List[dict], embedded_fields: List[str] = None) -> bool:
        logger.info(f"Updating batch of {len(documents)} documents")
        try:
            bulk_operations = []
            for document in documents:
                update_fields = {key: value for key, value in document.items() if key != "_id"}
                
                # Handle embedded documents
                embedded_updates = {}
                if embedded_fields:
                    for field in embedded_fields:
                        if field in update_fields:
                            embedded_updates.update({f"{field}.{k}": v for k, v in update_fields[field].items()})
                            del update_fields[field]
                
                # Combine regular and embedded updates
                update_operation = {"$set": update_fields}
                if embedded_updates:
                    update_operation["$set"].update(embedded_updates)
                
                bulk_operations.append(
                    pymongo.UpdateOne(
                        {"_id": document["_id"]},
                        update_operation,
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
