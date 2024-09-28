import customtkinter as ctk
from PIL import Image
import os
import pymongo
from dotenv import load_dotenv
from io import BytesIO
import datetime
import json

# Global variables
queue = []  # Queue of images to be tagged
processed_images = []   # List of processed images
current_image = None    # Current image being displayed
image_label = None  # Label to display the image
info_label = None   # Label to display the index, master ID, tag, and tagged by information
progressbar = None  # Progress bar to display the progress of the tagging process
progress_label = None   # Label to display the progress of the tagging process
cover_album_progressbar = None      # Progress bar for the number of covers, vinyls, and others
cover_album_label = None    # Label to display the number of covers, vinyls, and others
current_index = 0   # Index of the current image being displayed
current_document = None  # Current document being displayed
processing_threshold = 5  # If the number of unprocessed objects in the queue is less than this, refill the queue.
# Load configuration from JSON file
with open(os.path.join('src', 'config', 'GUI_CONFIG.JSON'), 'r') as config_file:
    config = json.load(config_file)

mode = config.get("MODE")

class Document:
    def __init__(self, master_id, image_data, tagging_status="unprocessed", tag="not tagged", index=0, tagged_by=None):
        self.master_id = master_id
        self.image_data = image_data
        self.tagging_status = tagging_status
        self.tag = tag
        self.index = index
        self.tagged_by = tagged_by if tagged_by else []

    def update_status(self, status):
        self.tagging_status = status

    def update_tag(self, tag, tagger_id, role="primary_tagger", favorite=False):
        self.tag = tag  # Corrected this line
        self.tagged_by.append({
            "tagger_id": tagger_id,
            "tag": tag,
            "timestamp": datetime.datetime.now(),
            "role": role,
            "favorite": favorite
        })

    def to_dict(self):
        return {
            "master_id": self.master_id,
            "image_data": self.image_data,
            "tagging_status": self.tagging_status,
            "tag": self.tag,
            "index": self.index,
            "tagged_by": self.tagged_by
        }

def get_and_update_objects(sample_collection, n, sort_field):
    if mode == "TAGGING":
        print(f"Retrieving {n} objects sorted by {sort_field} where tagging_status is 'unprocessed'...")
        results = sample_collection.find({"tagging_status": "unprocessed", "image_data": {"$exists": True}}).sort(sort_field, pymongo.ASCENDING).limit(n)
    elif mode == "VALIDATION":
        print(f"Retrieving {n} objects sorted by {sort_field} where tagging_status is 'tagged'...")
        results = sample_collection.find({"tagging_status": "tagged", "image_data": {"$exists": True}}).sort(sort_field, pymongo.ASCENDING).limit(n)

    updated_objects = []
    master_ids_to_update = []

    for result in results:
        master_ids_to_update.append(result["master_id"])
        document = Document(
            master_id=result["master_id"],
            image_data=result["image_data"],
            tagging_status="processing" if mode == "TAGGING" else "tagged",
            tag=result.get("tag", "not tagged"),
            index=sample_collection.count_documents({"master_id": {"$lt": result["master_id"]}}) + 1,
            tagged_by=result.get("tagged_by", [])
        )
        updated_objects.append(document)

    if mode == "TAGGING":
        sample_collection.update_many(
            {"master_id": {"$in": master_ids_to_update}},
            {"$set": {"tagging_status": "processing"}}
        )

    print(f"Updated {len(master_ids_to_update)} objects to 'processing' status.")
    return updated_objects

def display_image():
    global current_document
    if queue:
        current_document = queue[current_index]  # Use current_index
        print(f"Displaying image with master ID: {current_document.master_id}")
        image = Image.open(BytesIO(current_document.image_data))
        current_image = ctk.CTkImage(image, size=(500, 500))
        image_label.configure(image=current_image)
        image_label.image = current_image
        update_info_label()  # Update info label when displaying image
    else:
        print("No images found in the queue.")

def create_widgets():
    create_button_frame()
    create_info_label()
    create_progress_frame()
    create_image_label()
    initialize_user_input_var()

def create_button_frame():
    button_frame = ctk.CTkFrame(master_frame, corner_radius=10)
    button_frame.pack(side="bottom", pady=10)
    button_padding = 5

    ctk.CTkButton(button_frame, text="Vinyl (←)", command=lambda: on_button_click('vinyl')).grid(row=0, column=0, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Other (↑)", command=lambda: on_button_click('other')).grid(row=0, column=1, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Cover (→)", command=lambda: on_button_click('cover')).grid(row=0, column=2, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Back", command=on_back_button_click).grid(row=1, column=0, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Favorite (↓)", command=on_favorite_button_click).grid(row=1, column=1, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Next", command=on_next_button_click).grid(row=1, column=2, padx=button_padding, pady=button_padding)

def create_info_label():
    global info_label
    info_label = ctk.CTkLabel(master_frame, text="Index: 0, Master ID: N/A, Tag: N/A, Tagged By: N/A", font=("Helvetica", 14))
    info_label.pack(pady=5)

def create_progress_frame():
    global progressbar, progress_label, cover_album_label
    progress_frame = ctk.CTkFrame(master_frame, corner_radius=10)
    progress_frame.pack(pady=10, padx=10, fill="x")

    total_elements = get_total_elements()
    processed_elements = get_processed_elements()
    vinyl_count = get_category_count("vinyl")
    other_count = get_category_count("other")
    cover_count = get_category_count("cover")

    vinyl_percentage = (vinyl_count / processed_elements * 100) if processed_elements > 0 else 0
    other_percentage = (other_count / processed_elements * 100) if processed_elements > 0 else 0
    cover_percentage = (cover_count / processed_elements * 100) if processed_elements > 0 else 0

    progress_label = ctk.CTkLabel(progress_frame, text=f"{processed_elements}/{total_elements} covers tagged", font=("Helvetica", 14))
    progress_label.pack(side="top", anchor="w", padx=10, pady=5)

    progressbar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
    progressbar.pack(side="top", fill="x", padx=10, pady=5)
    progressbar.set(processed_elements / total_elements if total_elements > 0 else 0)

    cover_album_label = ctk.CTkLabel(progress_frame, text=f"{cover_count} covers ({cover_percentage:.2f}%), {vinyl_count} vinyls ({vinyl_percentage:.2f}%), {other_count} others ({other_percentage:.2f}%)", font=("Helvetica", 14))
    cover_album_label.pack(side="top", anchor="w", padx=10, pady=5)

def create_image_label():
    global image_label
    image_label = ctk.CTkLabel(master_frame, text="")
    image_label.pack(expand=True)

def initialize_user_input_var():
    global user_input_var
    user_input_var = ctk.StringVar()

def get_username():
    return os.getlogin()

def get_total_elements():
    return sample_collection.count_documents({"image_data": {"$exists": True}})

def get_processed_elements():
    return sample_collection.count_documents({"tag": {"$exists": True, "$ne": None}})

def get_category_count(category):
    return sample_collection.count_documents({"tag": category})

def process_current_image():
    global queue, processed_images, current_document

    category = user_input_var.get()
    if mode == "TAGGING":
        current_document.update_status("tagged")
        current_document.update_tag(category, get_username(), role="primary_tagger")
    elif mode == "VALIDATION":
        if current_document.tag == category:    # If the current tag matches the category the user selected, update the status to "validated", otherwise "disagreement"
            current_document.update_status("validated")
        else:
            current_document.update_status("disagreement")
        current_document.update_tag(category, get_username(), role="secondary_tagger")
        print(f"Master {current_document.master_id} Tagged as {category} by {current_document.tagged_by[-1]['tagger_id']}")

    # Update the document in the database
    sample_collection.update_one(
        {"master_id": current_document.master_id},
        {"$set": current_document.to_dict()}
    )

    # Add to processed images if not already in the list
    if current_document not in processed_images:
        processed_images.append(current_document)

    # Refill the queue if necessary. check how many objects in the queue have "tagging_status" = "processing", if less than the threshold, refill the queue.
    unprocessed_queue_count = sum(1 for doc in queue if doc.tagging_status == "processing")
    print(f"Unprocessed queue: {unprocessed_queue_count}")
    if unprocessed_queue_count <= processing_threshold:
        refill_queue()

    # Update progress and category counts
    update_progress_and_counts()

def update_progress_and_counts():
    total_elements = get_total_elements()
    processed_elements = get_processed_elements()
    vinyl_count = get_category_count("vinyl")
    other_count = get_category_count("other")
    cover_count = get_category_count("cover")

    vinyl_percentage = (vinyl_count / processed_elements * 100) if processed_elements > 0 else 0
    other_percentage = (other_count / processed_elements * 100) if processed_elements > 0 else 0
    cover_percentage = (cover_count / processed_elements * 100) if processed_elements > 0 else 0

    progress_label.configure(text=f"{processed_elements}/{total_elements} covers tagged")
    progressbar.set(processed_elements / total_elements if total_elements > 0 else 0)
    cover_album_label.configure(text=f"{cover_count} covers ({cover_percentage:.2f}%), {vinyl_count} vinyls ({vinyl_percentage:.2f}%), {other_count} others ({other_percentage:.2f}%)")

def update_info_label():
    if current_document:
        primary_tag = None
        secondary_tag = None
        favorite_status = "No"
        for tag_info in current_document.tagged_by:
            if tag_info["role"] == "primary_tagger":
                primary_tag = tag_info["tag"]
            elif tag_info["role"] == "secondary_tagger":
                secondary_tag = tag_info["tag"]
            if tag_info.get("favorite", False):
                favorite_status = "Yes"

        tagged_by_info = ""
        if mode == "VALIDATION" and current_document.tagged_by:
            tagged_by_info = f"\nTagged By: {current_document.tagged_by[0]['tagger_id']}"

        info_label.configure(text=f"Index: {current_document.index}\nMaster ID: {current_document.master_id}\nPrimary Tag: {primary_tag}\nSecondary Tag: {secondary_tag}\nFavorite: {favorite_status}{tagged_by_info}")
    else:
        info_label.configure(text="Index: 0, Master ID: N/A, Primary Tag: N/A, Secondary Tag: N/A, Favorite: No, Tagged By: N/A")

def on_button_click(value):
    global current_index
    print(f"Button clicked: {value}")
    user_input_var.set(value)
    process_current_image()
    if current_index < len(queue) - 1:
        current_index += 1
        display_image()
    else:
        print("No more images.")

def on_next_button_click():
    global current_index
    if current_index < len(queue) - 1:
        current_index += 1
        display_image()
    else:
        print("No more images.")
        refill_queue()

def on_back_button_click():
    global current_index
    if current_index > 0:
        current_index -= 1
        display_image()
    else:
        print("No previous images.")

def on_favorite_button_click():
    global current_document
    if current_document:
        # Toggle the favorite status for the current tagger
        tagger_id = get_username()
        for tag_info in current_document.tagged_by:
            if tag_info["tagger_id"] == tagger_id:
                tag_info["favorite"] = not tag_info.get("favorite", False)
                break
        else:
            # If the tagger_id is not found, add a new entry with favorite set to True
            current_document.update_tag(current_document.tag, tagger_id, role="primary_tagger", favorite=True)

        # Update the document in the database
        sample_collection.update_one(
            {"master_id": current_document.master_id},
            {"$set": current_document.to_dict()}
        )

        # Update the info label to reflect the change
        update_info_label()

def on_key_press(event):
    if event.keysym == 'Left':
        on_button_click('vinyl')
    elif event.keysym == 'Right':
        on_button_click('cover')
    elif event.keysym == 'Up':
        on_button_click('other')
    elif event.keysym == 'Down':
        on_favorite_button_click()
    

def refill_queue():
    global queue
    new_objects = get_and_update_objects(sample_collection, 10, "master_id")
    queue.extend(new_objects)
    print(f"Queue refilled. New queue length: {len(queue)}")

def reset_processing_objects():
    global queue
    master_ids_to_reset = [document.master_id for document in queue if document.tagging_status == "processing"]
    if master_ids_to_reset:
        sample_collection.update_many(
            {"master_id": {"$in": master_ids_to_reset}},
            {"$set": {"tagging_status": "unprocessed"}}
        )
        print(f"Reset {len(master_ids_to_reset)} objects to 'unprocessed' status.")

def on_closing():
    reset_processing_objects()
    app.destroy()

# Initialize the main application window
app = ctk.CTk()

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("src/style.json")

app.title("Discogs Cover Tagger")
app.geometry("520x770")

master_frame = ctk.CTkFrame(app, fg_color="#202027")
master_frame.pack(expand=True, fill="both")

app.bind('<Left>', on_key_press)
app.bind('<Right>', on_key_press)
app.bind('<Up>', on_key_press)
app.bind('<Down>', on_key_press)

# Initialize the database connection and retrieve entries
load_dotenv()
mongo_uri = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(mongo_uri)
db = client["album_covers"]
sample_collection = db["fiveK-albums-sample-copy"]
queue = get_and_update_objects(sample_collection, 15, "master_id")

# Create widgets
create_widgets()
display_image()

# Bind the window close event
app.protocol("WM_DELETE_WINDOW", on_closing)

app.mainloop()
