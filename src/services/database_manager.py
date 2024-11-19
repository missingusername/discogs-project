import io
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
import pymongo
import pymongo.errors
from PIL import Image
from pymongo.cursor import Cursor
from pymongo.operations import UpdateOne
from tqdm import tqdm

from ..utils.logger_utils import get_logger

logger = get_logger(__name__, "INFO")


class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str, queries_file: str):
        logger.info(
            f"Initializing DatabaseManager with db_name: {db_name}, collection_name: {collection_name}"
        )
        try:
            self.uri = uri
            self.client = pymongo.MongoClient(self.uri)
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

    @staticmethod
    def split_string_return_last_element(string: str, delimiter: str) -> str:
        return string.split(delimiter)[-1]

    def export_documents_to_csv(self, output_dir, output_file_name, pipeline_path):
        try:
            pipeline = self._get_query_by_path(pipeline_path)
            if pipeline is None:
                logger.error(f"Pipeline not found for path: {pipeline_path}")
                return False

            logger.info(f"Exporting documents to CSV using pipeline: {pipeline}")

            cursor = self.collection.aggregate(pipeline, allowDiskUse=True)
            documents: List[dict] = []
            for doc in cursor:
                documents.append(doc)

            logger.info(len(documents))

            if not documents:
                logger.warning("No documents found with reduced embeddings")
                return False

            data = []
            for doc in documents:
                embedding = doc["reduced_embedding_all"]
                row = {
                    "document_id": str(doc.get("_id", "unknown")),
                    "master_id": doc.get("master_id", "unknown"),
                    "title": doc.get("title", "untitled"),
                    "year": doc.get("year", 0),
                    "artists": doc.get("artist_names", []),
                    "genres": doc.get("genres", []),
                    "x": embedding[0] if embedding and len(embedding) > 0 else 0.0,
                    "y": embedding[1] if embedding and len(embedding) > 1 else 0.0,
                }
                data.append(row)

            df = pd.DataFrame(data)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            output_path = output_dir / output_file_name
            df.to_csv(output_path, index=False)
            logger.info(
                f"Successfully exported {len(documents)} embeddings to {output_file_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to export reduced embeddings: {e}")
            return False

    def merge_duplicate_documents(
        self,
        field: str,
        merge_strategy: Dict[str, str] = None,
        query_filter: dict = None,
        min_occurrences: int = 2,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Merge documents that have the same value for a specified field.

        Args:
            field: The field to check for duplicates
            merge_strategy: Dictionary specifying how to merge specific fields
                        e.g. {"tags": "union", "views": "sum", "description": "longest"}
                        Supported strategies: "first", "last", "union", "sum", "max", "min", "longest"
            query_filter: Optional filter to apply before looking for duplicates
            min_occurrences: Minimum number of occurrences to consider as duplicate
            dry_run: If True, only shows what would be merged without making changes

        Returns:
            Dictionary containing merge statistics and results
        """
        logger.info(f"Starting duplicate document merge for field: {field}")

        try:
            # Default merge strategy if none provided
            if merge_strategy is None:
                merge_strategy = {}

            # Find duplicate groups
            pipeline = [
                {"$match": query_filter} if query_filter else {"$match": {}},
                {
                    "$group": {
                        "_id": f"${field}",
                        "docs": {"$push": "$$ROOT"},
                        "count": {"$sum": 1},
                    }
                },
                {"$match": {"count": {"$gte": min_occurrences}}},
                {"$sort": {"count": -1}},
            ]

            cursor = self.collection.aggregate(pipeline, allowDiskUse=True)
            merge_stats = {"groups_processed": 0, "documents_merged": 0, "errors": 0}

            def merge_field_values(
                docs: List[dict], field_name: str, strategy: str
            ) -> Any:
                try:
                    values = [doc.get(field_name) for doc in docs if field_name in doc]
                    if not values:
                        return None

                    if strategy == "first":
                        return values[0]
                    elif strategy == "last":
                        return values[-1]
                    elif strategy == "union" and all(
                        isinstance(v, (list, set)) for v in values
                    ):
                        return list(set().union(*[set(v) for v in values]))
                    elif strategy == "sum" and all(
                        isinstance(v, (int, float)) for v in values
                    ):
                        return sum(values)
                    elif strategy == "max" and all(
                        isinstance(v, (int, float, str)) for v in values
                    ):
                        return max(values)
                    elif strategy == "min" and all(
                        isinstance(v, (int, float, str)) for v in values
                    ):
                        return min(values)
                    elif strategy == "longest" and all(
                        isinstance(v, str) for v in values
                    ):
                        return max(values, key=len)
                    else:
                        # Default to first value if strategy doesn't match
                        return values[0]
                except Exception as e:
                    logger.error(f"Error merging field {field_name}: {e}")
                    return None

            for group in cursor:
                try:
                    docs = group["docs"]
                    merged_doc = {}

                    # Get all possible fields from all documents
                    all_fields = set()
                    for doc in docs:
                        all_fields.update(doc.keys())

                    # Keep the _id from the first document
                    merged_doc["_id"] = docs[0]["_id"]

                    # Merge each field according to strategy
                    for field_name in all_fields:
                        if field_name == "_id":
                            continue

                        strategy = merge_strategy.get(field_name, "first")
                        merged_value = merge_field_values(docs, field_name, strategy)
                        if merged_value is not None:
                            merged_doc[field_name] = merged_value

                    if not dry_run:
                        # Delete all documents in the group
                        doc_ids = [doc["_id"] for doc in docs[1:]]  # Exclude first doc
                        if doc_ids:
                            self.collection.delete_many({"_id": {"$in": doc_ids}})

                        # Update the first document with merged data
                        self.collection.update_one(
                            {"_id": merged_doc["_id"]}, {"$set": merged_doc}
                        )

                    merge_stats["groups_processed"] += 1
                    merge_stats["documents_merged"] += len(docs) - 1

                except Exception as e:
                    logger.error(f"Error processing duplicate group: {e}")
                    merge_stats["errors"] += 1
                    continue

            summary = {
                "status": "completed",
                "dry_run": dry_run,
                "stats": merge_stats,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Merge operation completed: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Failed to merge duplicate documents: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def convert_string_arrays_to_arrays(
        self, field_name: str, batch_size: int = 50000
    ) -> Tuple[int, int]:
        """
        Convert string arrays to actual arrays for a specific field.
        Args:
            field_name: The field to convert
            batch_size: Number of documents to process in each batch
        Returns:
            Tuple of (processed_count, error_count)
        """
        logger.info(f"Converting string arrays to arrays for field: {field_name}")
        processed_count = 0
        error_count = 0
        try:
            # Find documents where the field is a string and starts with '['
            query = {
                field_name: {
                    "$type": "string",  # Ensure it's a string
                    "$regex": r"^\[.*\]$",  # Matches strings that look like arrays
                    "$not": {"$type": "array"},  # Exclude actual arrays
                }
            }
            total_docs = self.collection.count_documents(query)
            logger.info(f"Found {total_docs} documents to process")

            def string_to_array(s: str) -> List[Any]:
                try:
                    # Normalize the string by converting single quotes to double quotes carefully
                    normalized = re.sub(r"(?<=\[)\s*'|'(?=\s*,|\s*\])", '"', s)
                    # Convert the normalized string into JSON
                    return json.loads(normalized)
                except json.JSONDecodeError:
                    logger.error(f"Failed to convert string to array: {s}")
                    # Final fallback: return the original string wrapped in a list
                    return [s]

            with tqdm(total=total_docs, desc="Converting arrays") as pbar:
                cursor = self.collection.find(query, batch_size=batch_size)
                bulk_updates = []
                for doc in cursor:
                    try:
                        string_value = doc[field_name]
                        array_value = string_to_array(string_value)
                        bulk_updates.append(
                            UpdateOne(
                                {"_id": doc["_id"]}, {"$set": {field_name: array_value}}
                            )
                        )
                        processed_count += 1

                        # Execute bulk write when batch size is reached
                        if len(bulk_updates) >= batch_size:
                            self.collection.bulk_write(bulk_updates, ordered=False)
                            bulk_updates = []  # Clear the batch after execution
                    except Exception as e:
                        logger.error(f"Error processing document {doc['_id']}: {e}")
                        error_count += 1
                    finally:
                        pbar.update(1)

                # Process any remaining updates in the final batch
                if bulk_updates:
                    self.collection.bulk_write(bulk_updates, ordered=False)

            return processed_count, error_count
        except Exception as e:
            logger.error(f"Error in convert_string_arrays_to_arrays: {e}")
            return processed_count, error_count

    def analyze_array_field_types(self, field_name: str) -> Dict[str, int]:
        logger.info(f"Analyzing field types for {field_name}")
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": {"$type": f"${field_name}"},
                        "count": {"$sum": 1},
                        "sample": {"$first": f"${field_name}"},
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            type_counts = {"array": 0, "string_array": 0, "other": 0}

            for result in results:
                field_type = result["_id"]
                sample = result["sample"]
                count = result["count"]

                if field_type == "array":
                    type_counts["array"] = count
                elif (
                    field_type == "string"
                    and isinstance(sample, str)
                    and sample.startswith("[")
                ):
                    type_counts["string_array"] = count
                else:
                    type_counts["other"] = count

            logger.info(f"Field type analysis for {field_name}: {type_counts}")
            return type_counts
        except Exception as e:
            logger.error(f"Error analyzing field types: {e}")
            return None

    def count_documents_agg_pipeline(self, pipeline_path: List[dict]) -> int:
        if not pipeline_path or not isinstance(pipeline_path, str):
            raise ValueError("Pipeline path must be a non-empty string")

        logger.debug(f"Checking number of documents for query path: {pipeline_path}")
        try:
            pipeline = self._get_query_by_path(pipeline_path)
            if pipeline is None:
                logger.error(f"Pipeline not found for path: {pipeline_path}")
                return None
            logger.info(f"Using aggregation pipeline: {pipeline}")
            result = list(self.collection.aggregate(pipeline))

            # If no documents match, the count stage returns empty result
            if not result:
                document_count = 0
            else:
                # Use the count field name from your pipeline ("matching_docs")
                document_count = result[0]["matching_docs"]
            return document_count

        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to check number of documents: {e}")
            return None
        except pymongo.errors.InvalidOperation as e:
            logger.error("Invalid query operation: %s", str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error counting documents: %s", str(e))
            return None

    def count_documents_by_filter(self, query_path: str) -> int:
        if not query_path or not isinstance(query_path, str):
            raise ValueError("Query path must be a non-empty string")

        logger.debug(f"Checking number of documents for query path: {query_path}")
        try:
            query_or_pipeline = self._get_query_by_path(query_path)
            if query_or_pipeline is None:
                logger.error(f"Query not found for path: {query_path}")
                return None
            filter_type = self.split_string_return_last_element(query_path, ".")

            if isinstance(query_or_pipeline, list) and filter_type == "pipeline":
                logger.info(f"Using aggregation pipeline: {query_or_pipeline}")
                num_documents = len(list(self.collection.aggregate(query_or_pipeline)))
            elif isinstance(query_or_pipeline, dict) and filter_type == "query":
                logger.info(f"Using query: {query_or_pipeline}")
                num_documents = self.collection.count_documents(
                    filter=query_or_pipeline
                )
            else:
                logger.error(f"Invalid query or pipeline format for path: {query_path}")
                return None

            logger.info(f"Found {num_documents} documents")
            return num_documents

        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to check number of documents: {e}")
            return None
        except pymongo.errors.InvalidOperation as e:
            logger.error("Invalid query operation: %s", str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error counting documents: %s", str(e))
            return None

    def retrieve_documents_by_query(
        self,
        query_path: str,
        limit: Optional[int] = None,
        as_list: bool = False,
        log_execution: bool = False,
    ) -> Optional[Union[List[dict], Iterator[dict]]]:
        logger.debug(
            f"Retrieving documents for query path: {query_path}, limit: {'unlimited' if limit is None else limit}, as_list: {as_list}"
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
                "$currentDate": {"lastModified": True},
            }

            # Create base cursor with sorting
            cursor: Cursor = self.collection.find(query).sort("_id", pymongo.ASCENDING)

            # Apply limit only if specified
            if limit is not None:
                cursor = cursor.limit(limit)

            documents = []
            for doc in cursor:
                updated_doc = self.collection.find_one_and_update(
                    {"_id": doc["_id"]},
                    update,
                    return_document=pymongo.ReturnDocument.AFTER,
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
        retrieve_type = self.split_string_return_last_element(query_path, ".")
        logger.info(f"Retrieving {retrieve_type} for query path: {query_path}")

        logger.debug(f"Getting query for path: {query_path}")
        parts = query_path.split(".")
        current = self.queries
        for part in parts:
            logger.debug(f"Current part: {part}")
            if part in current:
                current = current[part]
            else:
                logger.error(f"Invalid query path: {query_path}")
                return None
        if isinstance(current, dict) and retrieve_type == "query":
            logger.debug(f"Found query: {current}")
            return current
        elif isinstance(current, list) and retrieve_type == "pipeline":
            logger.debug(f"Found pipeline: {current}")
            return current

    def update_documents_batch(
        self, documents: List[dict], embedded_fields: List[str] = None
    ) -> bool:
        logger.info(f"Updating batch of {len(documents)} documents")
        try:
            bulk_operations = []
            for document in documents:
                update_fields = {
                    key: value for key, value in document.items() if key != "_id"
                }

                # Handle embedded documents
                embedded_updates = {}
                if embedded_fields:
                    for field in embedded_fields:
                        if field in update_fields:
                            embedded_updates.update(
                                {
                                    f"{field}.{k}": v
                                    for k, v in update_fields[field].items()
                                }
                            )
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
        finally:
            self.close_connection()

    def identify_duplicate_documents_by_field(
        self,
        field: str,
        query_filter: dict = None,
        min_occurrences: int = 2,
        include_docs: bool = False,
        export_results_to_csv: bool = True,
    ) -> List[dict]:
        logger.info(f"Identifying duplicate documents by field: {field}")
        try:
            pipeline = [
                {"$match": query_filter} if query_filter else {"$match": {}},
                {
                    "$group": {
                        "_id": f"${field}",
                        "count": {"$sum": 1},
                        "doc_ids": {"$push": "$_id"},
                    }
                },
                {"$match": {"count": {"$gte": min_occurrences}}},
                {"$sort": {"count": -1}},
            ]

            cursor = self.collection.aggregate(pipeline, allowDiskUse=True)
            duplicate_groups = []
            total_documents = 0

            for group in cursor:
                total_documents += group["count"]

                duplicate_group = {"value": group["_id"], "count": group["count"]}

                if include_docs:
                    documents = list(
                        self.collection.find({"_id": {"$in": group["doc_ids"]}})
                    )
                    duplicate_group["documents"] = documents

                duplicate_groups.append(duplicate_group)

            result = {
                "total_duplicates": len(duplicate_groups),
                "total_duplicate_documents": total_documents,
                "duplicate_groups": duplicate_groups,
            }

            logger.info(
                f"Found {result['total_duplicates']} duplicate groups "
                f"containing {total_documents} documents for field: {field}"
            )

            if export_results_to_csv:
                df = pd.DataFrame(duplicate_groups)
                df.to_csv(f"duplicate_documents_{field}.csv", index=False)
                logger.info(f"Results exported to duplicate_documents_{field}.csv")

            return result
        except pymongo.errors.OperationFailure as e:
            logger.error(f"Failed to find duplicates: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while finding duplicates: {e}")
            return None

    def download_image_data(
        self, query_path, output_dir: Path = None, limit: int = 100000
    ) -> None:
        logger.info(f"Downloading image data for {limit} documents")
        start_time = time.time()

        if not output_dir:
            output_dir = Path(__file__).resolve().parents[2] / "images"
        try:
            documents = self.retrieve_documents_by_query(
                query_path, limit, as_list=True
            )
            if not documents:
                logger.warning("No documents found for the given query path")
                return

            # Ensure the output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            with tqdm(total=len(documents), desc="Downloading images") as pbar:
                for doc in documents:
                    try:
                        image_data = doc.get("album_cover", {}).get("image_data")
                        master_id = doc.get("master_id")
                        if not image_data or not master_id:
                            logger.warning(
                                f"Missing image data or master_id for document ID: {doc['_id']}"
                            )
                            continue

                        # Open the image using Pillow
                        image = Image.open(io.BytesIO(image_data))

                        # Save the image to a file named after the master_id
                        image_path = output_dir / f"{master_id}.jpg"
                        image.save(image_path)

                        logger.debug(
                            f"Saved image for document ID: {doc['_id']} to {image_path}"
                        )

                    except Exception as e:
                        logger.error(f"Error processing document ID: {doc['_id']}: {e}")
                    finally:
                        pbar.update(1)

        except Exception as e:
            logger.error(f"Error downloading image data: {e}")
        finally:
            duration = time.time() - start_time
            logger.info(f"Time taken to download images: {duration:.2f} seconds")

    def close_connection(self):
        logger.info("Closing MongoDB connection")
        self.client.close()

    def create_dump(
        self,
        output_dir: str = "mongodb_dumps",
        collections: Optional[List[str]] = None,
        compress: bool = True,
    ) -> Optional[str]:

        logger.info(f"Creating MongoDB dump for database: {self.db}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_dir = Path(output_dir) / f"dump_{timestamp}"

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Build mongodump command
            cmd = ["mongodump", "--uri", self.uri, "--out", str(dump_dir)]

            if collections:
                cmd.extend(["--db", self.db_name])
                for collection in collections:
                    cmd.extend(["--collection", collection])

            if compress:
                cmd.append("--gzip")

            logger.info(f"Starting MongoDB dump to {dump_dir}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("MongoDB dump completed successfully")
                return str(dump_dir)
            else:
                logger.error(f"MongoDB dump failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error during MongoDB dump: {str(e)}")
            return None
