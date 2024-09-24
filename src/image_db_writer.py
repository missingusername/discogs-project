from io import BytesIO
import os
import pymongo.collection
import requests
import time
from tqdm import tqdm

from dotenv import load_dotenv
from PIL import Image
import pymongo

from utils.logger_utils import get_logger

logger = get_logger(__name__)

DELAY_BETWEEN_REQUESTS = 1  # Delay between requests in seconds

def is_valid_image_url(image_url: str) -> bool:
    """
    Check if the given URL is valid.
    """
    return image_url.startswith("http://") or image_url.startswith("https://")

def retrieve_unprocessed_document(collection: pymongo.collection.Collection) -> dict:
    document = collection.find_one({
        '$and': [
            {
                '$or': [
                    {'image_data': {'$exists': False}},
                    {'image_data': None},
                ]
            },
            {'Image_uri': {'$ne': 'Image not available'}},
            {'Image_uri': {'$ne': ''}}
        ]
    })
    return document


def download_album_buffer_by_uri(document: dict, user_agent: str = "Mozilla/5.0", max_retries: int = 5) -> bytes:
    headers = {"User-Agent": user_agent}
    try:
        image_url = document["Image_uri"]

        if not is_valid_image_url(image_url):
            logger.error(f"Not a valid image url fire. Invalid image URL: {image_url}")
            logger.error(f"Document: {document}")
            return None
        
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.thumbnail((512, 512))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            return buffer
        else:
            logger.error(f"HTTP error occurred: {e}")
    except Exception as e:
        logger.error(f"Error downloading image from {document['Image_uri']}. Error: {e}")

    logger.error(f"Failed to download image from {document['Image_uri']} after {max_retries} attempts.")
    return None


def update_document_with_image_data(collection: pymongo.collection.Collection, document: dict, image_data: bytes) -> None:
    result = collection.update_one(
        {"_id": document["_id"]},
        {"$set": {"image_data": image_data}}
    )
    logger.debug(f"Document {document['_id']} updated with image data")


def main():
    load_dotenv()

    mongo_uri = os.getenv("MONGODB_URI")

    client = pymongo.MongoClient(mongo_uri, serverApi=pymongo.ServerApi('1'))
    db = client["album_covers"]
    sample_collection = db["5K-albums-sample"]

    # Get the total number of documents to process
    total_documents = sample_collection.count_documents({
        '$and': [
            {
                '$or': [
                    {'image_data': {'$exists': False}},
                    {'image_data': None},
                    {'image_data': ''}
                ]
            },
            {'image_data': {'$ne': 'Image not available'}}
        ]
    })

    # Use tqdm to create a progress bar
    with tqdm(total=total_documents, desc="Processing documents") as pbar:
        while True:
            document = retrieve_unprocessed_document(sample_collection)
            if document:
                buffer = download_album_buffer_by_uri(document)
                if buffer:
                    update_document_with_image_data(sample_collection, document, buffer.getvalue())
                pbar.update(1)  # Update the progress bar
                time.sleep(DELAY_BETWEEN_REQUESTS)  # Add a delay between processing each document
            else:
                logger.info("No more unprocessed documents.")
                break


if __name__ == "__main__":
    main()