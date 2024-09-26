from io import BytesIO
import json
import os
from pathlib import Path
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

# Load configuration from JSON file
config_path = Path(__file__).resolve().parents[0] / "config" / "gui_config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

BATCH_SIZE = config['BATCH_SIZE']
LOG_LEVEL = config['LOG_LEVEL']
MODE = config['MODE']

logger = get_logger(__name__, level=LOG_LEVEL)

# Load environment variables
load_dotenv()

class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def retrieve_unprocessed_documents(
        self,
        limit: int = 16,
        update_status: bool = False,
        as_list: bool = True,
        mode: str = "TAGGING",
        tagger: str = None,
    ) -> pymongo.cursor.Cursor | list:
        query = {
            "$and": [
                {"image_data": {"$exists": True}},
            ]
        }

        if mode == "TAGGING":
            query["$and"].append({"tagging_status": {"$eq": "unprocessed"}})
        elif mode == "VALIDATION" and tagger:
            query["$and"].append({"tagging_status": {"$ne": "unprocessed"}})
            query["$and"].append({"tagged_by.tagger_id": {"$ne": tagger}})

        cursor = self.collection.find(query).limit(limit)

        documents = list(cursor)

        if update_status and mode == "TAGGING":
            self.update_documents_to_in_progress(documents)

        if as_list:
            return documents

        return cursor

    def reset_current_batch_status(self, batch_ids, mode):
        if mode == "TAGGING" and batch_ids:
            self.collection.update_many(
                {"_id": {"$in": list(batch_ids)}},
                {"$set": {"tagging_status": "unprocessed"}},
            )
            logger.info("Reset status of current batch documents to 'unprocessed'.")

    def update_documents_to_in_progress(self, cursor: pymongo.cursor.Cursor) -> None:
        document_ids = [doc["_id"] for doc in cursor]
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
                        "tagged_by": doc["tagged_by"],
                    }
                },
            )

    def test_connection(self) -> bool:
        try:
            self.collection.estimated_document_count()
            logger.info("Database connection test passed")
            return True
        except pymongo.errors.PyMongoError as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def count_unprocessed_documents(self) -> int:
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


class GuiManager:
    def __init__(self, tagger):
        self.tagger = tagger
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

        self.label = ctk.CTkLabel(self.initial_frame, text="")
        self.label.pack(pady=10, padx=20)

        self.user_input_var = ctk.StringVar()
        self.image_label = ctk.CTkLabel(self.master_frame, text="")
        self.image_label.pack(expand=True)
        self.info_label = ctk.CTkLabel(
            self.master_frame, text="", font=("Helvetica", 14)
        )
        self.info_label.pack(pady=5)

        # Add a label to display the current mode
        self.mode_label = ctk.CTkLabel(
            self.master_frame,
            text=f"Mode: {self.tagger.mode.upper()}",
            font=("Helvetica", 14),
        )
        self.mode_label.pack(pady=5)

        self.create_widgets()
        self.display_image()

        self.app.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.tagger.reset_current_batch_status(self.tagger.mode)
        self.app.destroy()

    def create_widgets(self):
        self.create_button_frame()
        self.create_progress_frame()

    def create_button_frame(self):
        button_frame = ctk.CTkFrame(self.master_frame, corner_radius=10)
        button_frame.pack(side="bottom", pady=10)
        button_padding = 5

        ctk.CTkButton(
            button_frame, text="Vinyl", command=lambda: self.on_button_click("vinyl")
        ).grid(row=0, column=0, padx=button_padding, pady=button_padding)
        ctk.CTkButton(
            button_frame, text="Other", command=lambda: self.on_button_click("other")
        ).grid(row=0, column=1, padx=button_padding, pady=button_padding)
        ctk.CTkButton(
            button_frame, text="Cover", command=lambda: self.on_button_click("cover")
        ).grid(row=0, column=2, padx=button_padding, pady=button_padding)
        ctk.CTkButton(
            button_frame, text="Back", command=self.on_back_button_click
        ).grid(row=1, column=0, padx=button_padding, pady=button_padding)
        ctk.CTkButton(
            button_frame, text="Next", command=self.on_next_button_click
        ).grid(row=1, column=2, padx=button_padding, pady=button_padding)

    def create_progress_frame(self):
        self.progress_frame = ctk.CTkFrame(self.master_frame, corner_radius=10)
        self.progress_frame.pack(pady=10, padx=10, fill="x")

        self.progress_label = ctk.CTkLabel(
            self.progress_frame, text="0/0 covers tagged", font=("Helvetica", 14)
        )
        self.progress_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.progressbar = ctk.CTkProgressBar(
            self.progress_frame, orientation="horizontal"
        )
        self.progressbar.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.progressbar.set(0)

        self.cover_album_label = ctk.CTkLabel(
            self.progress_frame, text="0 covers / 0 vinyls", font=("Helvetica", 14)
        )
        self.cover_album_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.cover_album_progressbar = ctk.CTkProgressBar(
            self.progress_frame, orientation="horizontal"
        )
        self.cover_album_progressbar.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.cover_album_progressbar.set(0)

        self.progress_frame.grid_columnconfigure(1, weight=1)

    def update_progress_bars(self):
        tagged_images_count = self.tagger.batch_processed_count
        progress = tagged_images_count / len(self.tagger.documents)
        self.progressbar.set(progress)
        self.progress_label.configure(
            text=f"{tagged_images_count}/{len(self.tagger.documents)} covers tagged"
        )

        covers_count = sum(
            1 for doc in self.tagger.processed_documents if doc["tag"] == "cover"
        )
        albums_count = sum(
            1 for doc in self.tagger.processed_documents if doc["tag"] == "vinyl"
        )
        total_tagged = covers_count + albums_count

        cover_album_progress = covers_count / total_tagged if total_tagged > 0 else 0
        self.cover_album_progressbar.set(cover_album_progress)
        self.cover_album_label.configure(
            text=f"{covers_count} covers / {albums_count} vinyls"
        )

    def display_image(self):
        if self.tagger.current_index < len(self.tagger.documents):
            self.update_gui()
        else:
            logger.info("No more unprocessed documents available.")

    def update_gui(self):
        if not self.tagger.documents or self.tagger.current_index >= len(
            self.tagger.documents
        ):
            self.info_label.configure(text="No more images to display.")
            return

        document = self.tagger.documents[self.tagger.current_index]
        image_data = document["image_data"]

        img = Image.open(BytesIO(image_data))
        self.current_image = ctk.CTkImage(img, size=(500, 500))

        self.image_label.configure(image=self.current_image)
        self.image_label.image = self.current_image

        artist_names = document.get("artist_names", ["Unknown Artist"])
        artist = ", ".join(artist_names)
        album_title = document.get("title", "Unknown Album")
        release_year = document.get("year", "Unknown Year")
        document_id = document.get("_id", "Unknown ID")
        self.label.configure(
            text=f"ID: {document_id}\nArtist: {artist}, Album: {album_title}, Year: {release_year}"
        )

        # Update the mode label to match the styling of the artist and album title label
        self.mode_label.configure(
            text=f"Mode: {self.tagger.mode}", font=("Helvetica", 14)
        )
        self.mode_label.pack(after=self.label, pady=5)

        self.info_label.configure(
            text=f"Image {self.tagger.current_index + 1} out of {len(self.tagger.documents)}"
        )

        # Display tagging information if mode is VALIDATION
        if self.tagger.mode == "VALIDATION" and "tagged_by" in document:
            tagging_info = "\n".join(
                [
                    f"Tagger: {tag['tagger_id']}, Tag: {tag['tag_value']}, Role: {tag['role']}"
                    for tag in document["tagged_by"]
                ]
            )
            self.info_label.configure(
                text=f"{self.info_label.cget('text')}\n\nTagging Info:\n{tagging_info}"
            )

        self.update_progress_bars()

    def on_button_click(self, value):
        logger.debug(f"Button clicked: {value}")
        self.user_input_var.set(value)
        self.tagger.process_current_image()
        if not self.tagger.batch_prepared:
            self.on_next_button_click()

    def on_next_button_click(self):
        logger.debug(
            f"Current index before on_next_button_click: {self.tagger.current_index}"
        )
        if self.tagger.current_index < len(self.tagger.documents) - 1:
            self.tagger.current_index += 1
            logger.debug(
                f"Current index after increment in on_next_button_click: {self.tagger.current_index}"
            )
            self.update_gui()
        else:
            logger.info("Reached the end of the current batch.")
            return

        if not self.tagger.documents:
            logger.info("No more unprocessed documents available.")
        else:
            self.update_gui()

        if (
            self.tagger.document_queue.qsize() < self.tagger.batch_size
            and not self.tagger.is_fetching
        ):
            self.tagger.retrieve_unprocessed_covers_async()

    def on_back_button_click(self):
        logger.debug(
            f"Current index before on_back_button_click: {self.tagger.current_index}"
        )
        if self.tagger.current_index > 0:
            self.tagger.current_index -= 1
            logger.debug(
                f"Current index after decrement in on_back_button_click: {self.tagger.current_index}"
            )
            self.update_gui()

    def on_key_press(self, event):
        if event.keysym == "Left":
            self.on_button_click("vinyl")
        elif event.keysym == "Right":
            self.on_button_click("cover")

    def run_main_gui_loop(self):
        self.app.mainloop()


class DiscogsCoverTagger:
    def __init__(self, db_manager: DatabaseManager, mode: str = "TAGGING"):
        self.db_manager = db_manager
        self.tagger = self.get_tagger_identity()
        self.current_index = 0
        self.documents = []
        self.processed_documents = []

        self.batch_size = BATCH_SIZE
        self.batch_processed_count = 0
        self.current_batch_ids = set()
        self.document_queue = queue.Queue()
        self.is_fetching = False
        self.batch_prepared = False
        self.mode = mode

        self.retrieve_unprocessed_covers()

        self.gui_manager = GuiManager(self)

    def reset_current_batch_status(self, mode):
        self.db_manager.reset_current_batch_status(self.current_batch_ids, mode)

    def process_current_image(self):
        if self.current_index >= len(self.documents):
            return

        document = self.documents[self.current_index]
        tag = self.gui_manager.user_input_var.get()
        document_id = document["_id"]

        if self.mode == "TAGGING":
            processed_document = {
                "_id": document_id,
                "tagging_status": "tagged",
                "tag": tag,
                "tagged_by": {
                    "tagger_id": self.tagger,
                    "tag_value": tag,
                    "timestamp": pd.Timestamp.now(),
                    "role": (
                        "primary_tagger"
                        if document["tagging_status"] == "unprocessed"
                        else "secondary_tagger"
                    ),
                },
            }
        elif self.mode == "VALIDATION":
            processed_document = document
            if "tagged_by" not in processed_document:
                processed_document["tagged_by"] = []
            processed_document["tagged_by"].append(
                {
                    "tagger_id": self.tagger,
                    "tag_value": tag,
                    "timestamp": pd.Timestamp.now(),
                    "role": "validator",
                }
            )

        self._increment_batch_processed_count(document_id)
        self._add_or_update_processed_document(processed_document)

        logger.debug(
            f"Processed document {self.batch_processed_count}/{self.batch_size}"
        )

        if self.batch_processed_count >= self.batch_size:
            self.write_processed_documents_to_db()
            self.prepare_next_batch()
            self.batch_prepared = True
        else:
            self.batch_prepared = False

    def _increment_batch_processed_count(self, document_id):
        if not any(doc["_id"] == document_id for doc in self.processed_documents):
            self.batch_processed_count += 1
            logger.debug(
                f"Incremented batch_processed_count to {self.batch_processed_count}"
            )
        else:
            logger.debug(f"Document {document_id} already counted in the batch")

    def _add_or_update_processed_document(self, processed_document):
        for i, doc in enumerate(self.processed_documents):
            if doc["_id"] == processed_document["_id"]:
                self.processed_documents[i] = processed_document
                logger.debug(
                    f"Updated processed document: {processed_document['_id']} with tag: {processed_document['tag']}"
                )
                return
        self.processed_documents.append(processed_document)
        logger.debug(
            f"Added new processed document: {processed_document['_id']} with tag: {processed_document['tag']}"
        )

    def write_processed_documents_to_db(self):
        self.db_manager.update_batch_tagging_status(self.processed_documents)
        logger.info(
            f"Batch of {len(self.processed_documents)} documents written to the database."
        )
        self.processed_documents = []

    def retrieve_unprocessed_covers(self):
        self.documents = self.db_manager.retrieve_unprocessed_documents(
            limit=self.batch_size,
            update_status=True,
            as_list=True,
            mode=self.mode,
            tagger=self.tagger,
        )
        self.current_batch_ids = set(doc["_id"] for doc in self.documents)
        self.batch_processed_count = 0
        logger.info(f"Retrieved {len(self.documents)} unprocessed documents.")

    def retrieve_unprocessed_covers_async(self):
        if not self.is_fetching:
            self.is_fetching = True
            threading.Thread(target=self._fetch_documents_async).start()

    def _fetch_documents_async(self):
        new_documents = self.db_manager.retrieve_unprocessed_documents(
            limit=self.batch_size,
            update_status=True,
            as_list=True,
            mode=self.mode,
            tagger=self.tagger,
        )
        for doc in new_documents:
            self.document_queue.put(doc)
        logger.info(f"Fetched {len(new_documents)} new documents asynchronously.")
        logger.debug(
            f"Document queue size: {self.document_queue.qsize()} images to be processed"
        )
        self.is_fetching = False

    def prepare_next_batch(self):
        logger.debug("Preparing next batch of documents.")
        logger.debug(f"Document queue size: {self.document_queue.qsize()}")
        logger.debug(
            f"The index is {self.batch_processed_count} and the length of documents is {len(self.documents)}"
        )

        if self.batch_processed_count >= self.batch_size:
            logger.debug("Batch processed count reached batch size. Resetting state.")
            self.documents = []
            self.current_batch_ids = set()
            self.batch_processed_count = 0

            while (
                len(self.documents) < self.batch_size
                and not self.document_queue.empty()
            ):
                doc = self.document_queue.get()
                self.documents.append(doc)
                self.current_batch_ids.add(doc["_id"])

            if len(self.documents) < self.batch_size:
                logger.debug(
                    "Not enough documents in the queue. Fetching more documents asynchronously."
                )
                self.retrieve_unprocessed_covers_async()

            self.current_index = 0
            logger.debug(f"Reset current_index to {self.current_index}")
            self.gui_manager.update_gui()
        else:
            logger.debug(
                "Batch processed count has not reached batch size. Moving to the next document."
            )
            logger.debug(f"Current index before increment: {self.current_index}")
            self.current_index += 1
            logger.debug(f"Current index after increment: {self.current_index}")
            if self.current_index >= len(self.documents):
                self.current_index = 0
                logger.debug(f"Current index reset to {self.current_index}")
            logger.debug(f"Updated current_index to {self.current_index}")
            self.gui_manager.update_gui()

    def get_tagger_identity(self):
        computer_name_01 = platform.node()
        computer_name_02 = socket.gethostname()
        computer_name_03 = os.getenv("COMPUTER_NAME")

        if computer_name_01 == computer_name_02 or computer_name_01 == computer_name_03:
            eval_computer_name = computer_name_01
        elif computer_name_02 == computer_name_03:
            eval_computer_name = computer_name_02
        else:
            raise Exception("Computer names do not match.")

        pc_identity_mapping = {"DESKTOP-FMMNKQ2": "Gabriel"}

        return pc_identity_mapping.get(eval_computer_name, "Unknown")


def main():
    mongo_uri = os.getenv("MONGODB_URI")
    db_manager = DatabaseManager(
        uri=mongo_uri, db_name="album_covers", collection_name="fiveK-albums-sample"
    )
    app = DiscogsCoverTagger(db_manager=db_manager, mode=MODE)
    app.gui_manager.run_main_gui_loop()


if __name__ == "__main__":
    main()
