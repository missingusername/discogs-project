from io import BytesIO
import os
import platform
import queue
import socket
import threading

import customtkinter as ctk
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import pymongo
import pymongo.cursor

from utils.logger_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

BATCH_SIZE = 16


class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def retrieve_unprocessed_documents(
        self, limit: int = 16, update_status: bool = False, as_list: bool = True
    ) -> pymongo.cursor.Cursor | list:
        # Find documents that match the criteria
        cursor = self.collection.find(
            {
                "$and": [
                    {"image_data": {"$exists": True}},
                    {"tagging_status": {"$eq": "unprocessed"}},
                ]
            }
        ).limit(limit)
        
        # Convert cursor to list to avoid cursor exhaustion
        documents = list(cursor)
        
        # Optionally update the tagging_status to "in progress"
        if update_status:
            self.update_documents_to_in_progress(documents)

        # Optionally return the documents as a list
        if as_list:
            return documents
        
        return cursor

    def update_documents_to_in_progress(self, cursor: pymongo.cursor.Cursor) -> None:
        # Collect the IDs of the documents to update
        document_ids = [doc["_id"] for doc in cursor]

        # Update the tagging_status to "in progress"
        self.collection.update_many(
            {"_id": {"$in": document_ids}}, {"$set": {"tagging_status": "in progress"}}
        )

    def update_batch_tagging_status(self, documents: list) -> None:
        for doc in documents:
            self.collection.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "tagging_status": doc["tagging_status"],
                        "tag": doc["tag"],
                        "tagged_by": doc["tagged_by"]
                    }
                }
            )
    def test_connection(self) -> bool:
        try:
            # Perform a simple operation to test the connection
            self.collection.estimated_document_count()
            logger.info("Database connection test passed")
            return True
        except pymongo.errors.PyMongoError as e:
            logger.error(f"Database connection test failed: {e}")
            return False
        
    def count_unprocessed_documents(self) -> int:
        # Count documents that match the criteria
        count = self.collection.count_documents(
            {
                "$and": [
                    {"image_data": {"$exists": True}},
                    {"tagging_status": {"$eq": "unprocessed"}},
                ]
            }
        )
        logger.info(f"Found {count} unprocessed documents")
        return count


class DiscogsCoverTagger:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tagger = self.get_tagger_identity()
        self.current_index = 0
        self.processed_indices = set()
        self.documents = []
        self.processed_documents = []
        
        self.batch_size = BATCH_SIZE
        self.batch_processed_count = 0
        self.current_batch_ids = set()
        self.document_queue = queue.Queue()
        self.is_fetching = False

        # initial pictures
        self.retrieve_unprocessed_covers()
        
        self.app = ctk.CTk()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("src/style.json")

        self.app.title("Discogs Cover Tagger")
        self.app.geometry("520x750")

        self.master_frame = ctk.CTkFrame(self.app, fg_color="#202027")
        self.master_frame.pack(expand=True, fill="both")

        self.app.bind("<Left>", self.on_key_press)
        self.app.bind("<Right>", self.on_key_press)

        self.initial_frame = ctk.CTkFrame(self.master_frame, corner_radius=10)
        self.initial_frame.pack(expand=True)

        self.label = ctk.CTkLabel(
            self.initial_frame, text="Select an image folder to process"
        )
        self.label.pack(pady=10, padx=20)

        self.user_input_var = ctk.StringVar()
        self.image_label = ctk.CTkLabel(self.master_frame, text="")
        self.image_label.pack(expand=True)
        self.info_label = ctk.CTkLabel(self.master_frame, text="", font=("Helvetica", 14))
        self.info_label.pack(pady=5)
       
        self.create_widgets()
        self.display_image()

    def process_current_image(self):
        if self.current_index >= len(self.documents):
            return

        document = self.documents[self.current_index]
        tag = self.user_input_var.get()
        document_id = document["_id"]

        if document_id in self.current_batch_ids:
            processed_document = {
                "_id": document_id,
                "tagging_status": "tagged",
                "tag": tag,
                "tagged_by": {
                    "tagger_id": self.tagger,
                    "timestamp": pd.Timestamp.now(),
                    "role": "primary_tagger" if document['tagging_status'] == "unprocessed" else "secondary_tagger"
                }
            }

            self.processed_documents.append(processed_document)
            self.processed_indices.add(self.current_index)  # Mark this index as processed
            self.batch_processed_count += 1

            logger.debug(f"Processed document {self.batch_processed_count}/{self.batch_size}")

            if self.batch_processed_count >= self.batch_size:
                self.write_processed_documents_to_db()
                self.prepare_next_batch()
        
    def write_processed_documents_to_db(self):
        self.db_manager.update_batch_tagging_status(self.processed_documents)
        logger.info(f"Batch of {len(self.processed_documents)} documents written to the database.")
        self.processed_documents = []
        self.batch_processed_count = 0

    def retrieve_unprocessed_covers(self):
        self.documents = self.db_manager.retrieve_unprocessed_documents(limit=self.batch_size, update_status=True, as_list=True)
        self.current_batch_ids = set(doc['_id'] for doc in self.documents)
        self.batch_processed_count = 0
        logger.info(f"Retrieved {len(self.documents)} unprocessed documents.")

    def retrieve_unprocessed_covers_async(self):
        if not self.is_fetching:
            self.is_fetching = True
            threading.Thread(target=self._fetch_documents_async).start()

    def _fetch_documents_async(self):
        new_documents = self.db_manager.retrieve_unprocessed_documents(limit=self.batch_size, update_status=True, as_list=True)
        for doc in new_documents:
            self.document_queue.put(doc)
        logger.info(f"Fetched {len(new_documents)} new documents asynchronously.")
        self.is_fetching = False

    def prepare_next_batch(self):
        # Only prepare a new batch if all documents in the current batch have been processed
        if len(self.processed_indices) == len(self.documents):
            self.documents = []
            self.current_batch_ids = set()
            self.batch_processed_count = 0
            self.processed_indices.clear()
            
            while len(self.documents) < self.batch_size and not self.document_queue.empty():
                doc = self.document_queue.get()
                self.documents.append(doc)
                self.current_batch_ids.add(doc['_id'])

            if len(self.documents) < self.batch_size:
                self.retrieve_unprocessed_covers_async()

            self.current_index = 0
        else:
            # Find the first unprocessed document in the current batch
            self.current_index = next(i for i in range(len(self.documents)) if i not in self.processed_indices)

    def update_gui(self):
        if not self.documents or self.current_index >= len(self.documents):
            self.info_label.configure(text="No more images to display.")
            return

        document = self.documents[self.current_index]
        image_data = document['image_data']
        category = document['tagging_status']

        img = Image.open(BytesIO(image_data))
        self.current_image = ctk.CTkImage(img, size=(500, 500))

        self.image_label.configure(image=self.current_image)
        self.image_label.image = self.current_image

        self.info_label.configure(text=f"Image {self.current_index + 1} out of {len(self.documents)}\nCategory: {'Not tagged' if category == 'unprocessed' else category}")

        self.update_progress_bars()

    def display_image(self):
        if self.current_index < len(self.documents):
            self.update_gui()
        else:
            print("No more images.")

    def create_widgets(self):
        self.create_button_frame()
        self.create_progress_frame()

    def create_button_frame(self):
        button_frame = ctk.CTkFrame(self.master_frame, corner_radius=10)
        button_frame.pack(side="bottom", pady=10)
        button_padding = 5

        ctk.CTkButton(button_frame, text="Vinyl", command=lambda: self.on_button_click('vinyl')).grid(row=0, column=0, padx=button_padding, pady=button_padding)
        ctk.CTkButton(button_frame, text="Other", command=lambda: self.on_button_click('other')).grid(row=0, column=1, padx=button_padding, pady=button_padding)
        ctk.CTkButton(button_frame, text="Cover", command=lambda: self.on_button_click('cover')).grid(row=0, column=2, padx=button_padding, pady=button_padding)
        ctk.CTkButton(button_frame, text="Back", command=self.on_back_button_click).grid(row=1, column=0, padx=button_padding, pady=button_padding)
        ctk.CTkButton(button_frame, text="Next", command=self.on_next_button_click).grid(row=1, column=2, padx=button_padding, pady=button_padding)

    def create_progress_frame(self):
        self.progress_frame = ctk.CTkFrame(self.master_frame, corner_radius=10)
        self.progress_frame.pack(pady=10, padx=10, fill="x")

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="0/0 covers tagged", font=("Helvetica", 14))
        self.progress_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.progressbar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal")
        self.progressbar.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.progressbar.set(0)

        self.cover_album_label = ctk.CTkLabel(self.progress_frame, text="0 covers / 0 vinyls", font=("Helvetica", 14))
        self.cover_album_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.cover_album_progressbar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal")
        self.cover_album_progressbar.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.cover_album_progressbar.set(0)

        self.progress_frame.grid_columnconfigure(1, weight=1)

    def update_progress_bars(self):
        tagged_images_count = len(self.processed_indices)
        progress = tagged_images_count / len(self.documents)
        self.progressbar.set(progress)
        self.progress_label.configure(text=f"{tagged_images_count}/{len(self.documents)} covers tagged")

        covers_count = sum(1 for i, doc in enumerate(self.documents) if i in self.processed_indices and doc['tag'] == 'cover')
        albums_count = sum(1 for i, doc in enumerate(self.documents) if i in self.processed_indices and doc['tag'] == 'vinyl')
        total_tagged = covers_count + albums_count

        cover_album_progress = covers_count / total_tagged if total_tagged > 0 else 0
        self.cover_album_progressbar.set(cover_album_progress)
        self.cover_album_label.configure(text=f"{covers_count} covers / {albums_count} vinyls")

    def on_button_click(self, value):
        logger.info(f"Button clicked: {value}")
        self.user_input_var.set(value)
        self.process_current_image()
        self.on_next_button_click()

    def on_next_button_click(self):
        self.current_index += 1
        if self.current_index < len(self.documents):
            # If we've reached a processed document, skip to the next unprocessed one
            while self.current_index in self.processed_indices and self.current_index < len(self.documents):
                self.current_index += 1
            if self.current_index < len(self.documents):
                self.update_gui()
            else:
                self.prepare_next_batch()
        else:
            self.prepare_next_batch()

        if not self.documents:
            logger.info("No more unprocessed documents available.")
        else:
            self.update_gui()

        # Check if we need to fetch more documents
        if self.document_queue.qsize() < self.batch_size and not self.is_fetching:
            self.retrieve_unprocessed_covers_async()

    def on_back_button_click(self):
        if self.current_index > 0:
            self.current_index -= 1
            # If we've reached a processed document, skip to the previous unprocessed one
            while self.current_index in self.processed_indices and self.current_index > 0:
                self.current_index -= 1
            self.update_gui()

    def run_main_gui_loop(self):
        self.app.mainloop()

    def on_key_press(self, event):
        if event.keysym == "Left":
            self.on_button_click("vinyl")
        elif event.keysym == "Right":
            self.on_button_click("cover")

    def get_tagger_identity(self):
        # Get the computer name through multiple methods for rigour + cross platform
        computer_name_01 = platform.node()
        computer_name_02 = socket.gethostname()
        computer_name_03 = os.getenv("COMPUTER_NAME")
        
        # Evaluate the computer names
        if computer_name_01 == computer_name_02 or computer_name_01 == computer_name_03:
            eval_computer_name = computer_name_01
        elif computer_name_02 == computer_name_03:
            eval_computer_name = computer_name_02
        else:
            raise Exception("Computer names do not match.")
        
        # Map the computer name to a unique identifier, maybe use a hash function
        pc_identity_mapping = {
            "DESKTOP-FMMNKQ2": "Gabriel"
        }

        return pc_identity_mapping.get(eval_computer_name, "Unknown")

def main():
    mongo_uri = os.getenv("MONGODB_URI")
    db_manager = DatabaseManager(
        uri=mongo_uri, db_name="album_covers", collection_name="fiveK-albums-sample"
    )
    app = DiscogsCoverTagger(db_manager=db_manager)
    app.run_main_gui_loop()


if __name__ == "__main__":
    main()