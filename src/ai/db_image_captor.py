import os
from tqdm import tqdm
import pymongo
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

class Album:
    def __init__(self, master_id, artist, album_title, image_uri):
        self.master_id = master_id
        self.artist = artist
        self.album_title = album_title
        self.image_uri = image_uri
        self.cover_desc = None

def initialize_client(api_key):
    print("Initializing client...")
    return InferenceClient(api_key=api_key)

def prepare_messages(image, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

def get_chat_completion(client, model, messages, max_tokens):
    response = ""
    print("Getting chat completion...")
    for message in client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    ):
        response += message.choices[0].delta.content
    return response

def image_prompt(client, model, image, prompt, max_tokens=300):
    messages = prepare_messages(image, prompt)
    return get_chat_completion(client, model, messages, max_tokens)

def connect_to_db(uri, db_name, collection_name):
    print("Connecting to database...")
    mongo_uri = os.getenv(uri)
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Create index on master_id for faster queries
    collection.create_index("master_id")
    
    return collection

def fetch_albums(collection, batch_size, max_batches=None):
    print("Fetching albums from database...")
    albums = []
    total_documents = collection.count_documents({"image_uri": {"$exists": True}, "cover_desc": {"$exists": False}})
    if max_batches:
        total_documents = min(total_documents, batch_size * max_batches)
    for i in tqdm(range(0, total_documents, batch_size), ascii=True, desc="Fetching albums"):
        documents = list(collection.find(
            {"image_uri": {"$exists": True}, "cover_desc": {"$exists": False}}
        ).sort("master_id", pymongo.ASCENDING).limit(batch_size).allow_disk_use(True))
        
        for doc in documents:
            album = Album(
                master_id=doc["master_id"],
                artist=doc["artist_names"],
                album_title=doc["title"],
                image_uri=doc["image_uri"]
            )
            albums.append(album)
    return albums

def update_documents_bulk(collection, albums):
    print("Updating documents in bulk...")
    bulk_operations = []
    for album in albums:
        bulk_operations.append(
            pymongo.UpdateOne(
                {"master_id": album.master_id},
                {"$set": {"cover_desc": album.cover_desc}}
            )
        )
    
    if bulk_operations:
        result = collection.bulk_write(bulk_operations)
        print(f"Matched {result.matched_count} documents and modified {result.modified_count} documents.")

def process_albums(client, model, albums):
    print("Processing albums...")
    for album in albums:
        print(f"Processing album: {album.album_title} by {album.artist}")
        prompt = f"""
        Visually describe this album cover. 
        The album is {album.album_title} by {album.artist}. 
        only write 4-6 sentences. 
        don't explain or comment on the emotions it evokes. ONLY describe the visuals. 
        be structured.

        Follow this format:
        **Central Image:**
            describe the main image or object

        **Text Elements:**
            describe any text elements on the image, inclu

        **Background Context:**
            describe the image background
        """
        album.cover_desc = image_prompt(client, model, album.image_uri, prompt)

def main():
    print("Starting main process...")
    load_dotenv()
    api_key = "HF_API_KEY"
    client = initialize_client(api_key)
    model = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    mongo_uri = "MONGODB_URI"
    db_name = "album_covers"
    collection_name = "fiveK-albums-sample-copy"
    collection = connect_to_db(mongo_uri, db_name, collection_name)

    batch_size = 16
    max_batches = 1  # Optional: Set to None to process entire db

    albums = fetch_albums(collection, batch_size, max_batches)
    process_albums(client, model, albums)
    update_documents_bulk(collection, albums)
    print("Process completed.")

if __name__ == "__main__":
    main()
