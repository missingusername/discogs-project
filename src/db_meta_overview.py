import json
import os
from typing import List, Tuple

from dotenv import load_dotenv
import pymongo
from tqdm import tqdm

from utils.logger_utils import get_logger

logger = get_logger(__name__, 'INFO')

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

    def get_num_documents_in_collection(self) -> int:
        return self.collection.count_documents({})
    
    def get_num_documents_with_field(self, field: str) -> int:
        return self.collection.count_documents({field: {"$exists": True}})

    def get_documents_sorted_by_field(self, sort_options: Tuple[str, str]) -> list:
        sort_direction = pymongo.ASCENDING if sort_options[1].lower() == 'asc' else pymongo.DESCENDING
        return list(self.collection.find().sort([(sort_options[0], sort_direction)]))
    
    def count_unique_values(self, field: str) -> int:
        """
        Count the number of unique values for a given field in the collection.
        """
        aggregation_pipeline = [
            {"$group": {"_id": f"${field}"}},
            {"$count": "uniqueCount"}
        ]
        
        result = list(self.collection.aggregate(aggregation_pipeline))
        
        if result:
            return result[0]["uniqueCount"]
        else:
            return 0
    
    def analyze_document_structure(self, sample_size=100):
        sample = list(self.collection.aggregate([
            { "$sample": { "size": sample_size } },
            { "$project": { "_id": 0 } }
        ]))
        return sample
        
    
    def count_embedded_documents(self, sample_size=1000):
        sample = self.collection.aggregate([
            { "$sample": { "size": sample_size } },
            { "$project": { "_id": 0 } }
        ])
        
        total_embedded = 0
        for doc in sample:
            total_embedded += self._count_nested(doc) - 1  # Subtract 1 to not count the top-level document
        
        avg_embedded = total_embedded / sample_size
        estimated_total = int(avg_embedded * self.get_num_documents_in_collection())
        
        return {
            "sample_size": sample_size,
            "total_embedded_in_sample": total_embedded,
            "avg_embedded_per_document": avg_embedded,
            "estimated_total_embedded": estimated_total
        }

    def _count_nested(self, obj):
        count = 1  # Count this object
        if isinstance(obj, dict):
            for value in obj.values():
                count += self._count_nested(value)
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_nested(item)
        return count
    
    def count_documents_without_field(self, field: str) -> int:
        """Count documents that don't have the specified field."""
        return self.collection.count_documents({field: {"$exists": False}})

    def count_documents_with_empty_field(self, field: str) -> int:
        """Count documents where the specified field is an empty string."""
        return self.collection.count_documents({field: ""})

    def count_documents_with_null_field(self, field: str) -> int:
        """Count documents where the specified field is null."""
        return self.collection.count_documents({field: None})
    
    def check_for_duplicate_documents(self, fields: List[str], chunk_size: int = 100000) -> List[dict]:
        group_fields = {field: f"${field}" for field in fields}
        total_docs = self.collection.count_documents({})
        duplicate_groups = []

        logger.info(f"Total documents to process: {total_docs}. Processing in chunks of {chunk_size} documents")

        for start in tqdm(range(0, total_docs, chunk_size), desc="Checking for duplicates"):
            logger.debug(f"Processing chunk starting at document {start}")
            aggregation_pipeline = [
                {"$skip": start},
                {"$limit": chunk_size},
                # Group documents by the fields you want to check for duplicates
                {"$group": {
                    "_id": group_fields,
                    "count": {"$sum": 1},
                    "docs": {"$push": "$_id"}
                }},
                # Filter to only include groups with more than one document
                {"$match": {
                    "count": {"$gt": 1}
                }},
                # Sort by count in descending order
                {"$sort": {"count": -1}}
            ]
            # Execute the aggregation with allowDiskUse set to True
            result = self.collection.aggregate(aggregation_pipeline, allowDiskUse=True)
            
            chunk_duplicate_groups = list(result)
            duplicate_groups.extend(chunk_duplicate_groups)
            
            for group in chunk_duplicate_groups:
                logger.info(f"Found {group['count']} duplicate documents:")
                for doc_id in group['docs']:
                    logger.debug(f"Document ID: {doc_id}")
    
        logger.info(f"Total duplicate groups found: {len(duplicate_groups)}")
        return duplicate_groups
    
def main():
    # Load environment variables from .env file from root directory
    load_dotenv()

    # Setup variables for accessing database
    uri = os.getenv("MONGODB_URI_ALL")
    db_name = "discogs_data"
    collection_name = "albums"

    # Initialize the DatabaseManager
    db_manager = DatabaseManager(uri, db_name, collection_name)

    # Count the number of documents in the collection
    num_documents = db_manager.get_num_documents_in_collection()
    logger.info(f"Number of documents in collection: {num_documents}")

    # Count the number of documents with a specific field
    num_documents_with_field = db_manager.get_num_documents_with_field("image_uri")
    logger.info(f"Number of documents with 'image_url' field: {num_documents_with_field}")
    logger.info(f"Percentage of documents with 'image_url' field: {num_documents_with_field / num_documents * 100:.2f}%")

    # Get unique values for a specific field
    unique_masters = db_manager.count_unique_values("master_id")
    logger.info(f"Number of unique master IDs: {unique_masters}")

    # Get documents without a specific field
    docs_without_master_id = db_manager.count_documents_without_field("master_id")
    logger.info(f"Number of documents without a master ID: {docs_without_master_id}")

    # Get documents with empty or null values for a specific field
    docs_with_empty_master_id = db_manager.count_documents_with_empty_field("master_id")
    logger.info(f"Number of documents with empty master ID: {docs_with_empty_master_id}")

    # Get documents with null values for a specific field
    docs_with_null_master_id = db_manager.count_documents_with_null_field("master_id")
    logger.info(f"Number of documents with null master ID: {docs_with_null_master_id}")

    # Analyze document structure
    logger.debug("Analyzing document structure...")
    db_manager.analyze_document_structure(sample_size=5)  # Adjust sample size as needed

    # Check for duplicate documents grouped by field
    duplicates = db_manager.check_for_duplicate_documents(["master_id"])
    logger.info(f"Number of duplicate groups: {len(duplicates)}")

    # Count embedded documents
    embedded_stats = db_manager.count_embedded_documents()
    logger.info(f"Embedded document statistics: {json.dumps(embedded_stats, indent=2)}")


if __name__ == "__main__":
    main()
