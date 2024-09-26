import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import pandas as pd
import os
import pymongo
from dotenv import load_dotenv
from io import BytesIO

# Global variables
queue = []
processed_images = []
current_image = None
image_label = None
info_label = None
progressbar = None
progress_label = None
cover_album_progressbar = None
cover_album_label = None
current_index = 0
processed_counter = 0  # Add this line

def get_and_update_objects(sample_collection, n, sort_field):
    print(f"Retrieving {n} objects sorted by {sort_field} where tagging_status is 'unprocessed'...")
    results = sample_collection.find({"tagging_status": "unprocessed"}).sort(sort_field, pymongo.ASCENDING).limit(n)
    updated_objects = []
    master_ids_to_update = []

    for result in results:
        master_ids_to_update.append(result["master_id"])
        updated_objects.append({
            "master_id": result["master_id"],
            "image_data": result["image_data"],
            "tagging_status": "processing",
            "tag": "not tagged",
            "index": sample_collection.count_documents({"master_id": {"$lt": result["master_id"]}})  # Calculate index based on sorted master_id
        })

    sample_collection.update_many(
        {"master_id": {"$in": master_ids_to_update}},
        {"$set": {"tagging_status": "processing"}}
    )

    print(f"Updated {len(master_ids_to_update)} objects to 'processing' status.")
    return updated_objects

def display_image():
    if queue:
        document = queue[current_index]  # Use current_index
        image_data = document["image_data"]
        image = Image.open(BytesIO(image_data))
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

    ctk.CTkButton(button_frame, text="Vinyl", command=lambda: on_button_click('vinyl')).grid(row=0, column=0, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Other", command=lambda: on_button_click('other')).grid(row=0, column=1, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Cover", command=lambda: on_button_click('cover')).grid(row=0, column=2, padx=button_padding, pady=button_padding)
    ctk.CTkButton(button_frame, text="Back", command=on_back_button_click).grid(row=1, column=0, padx=button_padding, pady=button_padding)
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
    progress_label.pack(side="top", anchor="center", padx=10, pady=5)

    progressbar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
    progressbar.pack(side="top", fill="x", padx=10, pady=5)
    progressbar.set(processed_elements / total_elements if total_elements > 0 else 0)

    cover_album_label = ctk.CTkLabel(progress_frame, text=f"{cover_count} covers ({cover_percentage:.2f}%), {vinyl_count} vinyls ({vinyl_percentage:.2f}%), {other_count} others ({other_percentage:.2f}%)", font=("Helvetica", 14))
    cover_album_label.pack(side="top", anchor="center", padx=10, pady=5)

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
    return sample_collection.count_documents({})

def get_processed_elements():
    return sample_collection.count_documents({"tagging_status": "processed"})

def get_category_count(category):
    return sample_collection.count_documents({"tag": category})

def on_button_click(value):
    global current_index, processed_counter
    print(f"Button clicked: {value}")
    user_input_var.set(value)
    process_current_image()
    processed_counter += 1
    if processed_counter % 10 == 0:
        refill_queue()
    if current_index < len(queue) - 1:
        current_index += 1
        display_image()
    else:
        print("No more images.")

def process_current_image():
    global queue, processed_images

    category = user_input_var.get()
    document = queue[current_index]
    document["tagging_status"] = "processed"
    document["tag"] = category
    document["tagged_by"] = get_username()  # Add the username
    print(f"Tagged as {category} by {document['tagged_by']}")

    # Update the document in the database
    sample_collection.update_one(
        {"master_id": document["master_id"]},
        {"$set": {"tagging_status": "processed", "tag": category, "tagged_by": document["tagged_by"]}}
    )

    # Add to processed images if not already in the list
    if document not in processed_images:
        processed_images.append(document)

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
    if queue:
        document = queue[current_index]  # Use current_index
        index = document["index"]
        master_id = document["master_id"]
        tag = document["tag"]
        tagged_by = document.get("tagged_by", "N/A")  # Get the tagged_by field, default to "N/A" if not present
        info_label.configure(text=f"Index: {index}\nMaster ID: {master_id}\nTag: {tag}\nTagged By: {tagged_by}")
    else:
        info_label.configure(text="Index: 0, Master ID: N/A, Tag: N/A, Tagged By: N/A")

def on_next_button_click():
    global current_index
    if current_index < len(queue) - 1:
        current_index += 1
        display_image()
    else:
        print("No more images.")

def on_back_button_click():
    global current_index
    if current_index > 0:
        current_index -= 1
        display_image()
    else:
        print("No previous images.")

def on_key_press(event):
    if event.keysym == 'Left':
        on_button_click('vinyl')
    elif event.keysym == 'Right':
        on_button_click('cover')

def refill_queue():
    global queue
    new_objects = get_and_update_objects(sample_collection, 10, "master_id")
    queue.extend(new_objects)
    print(f"Queue refilled. New queue length: {len(queue)}")

def reset_processing_objects():
    global queue
    master_ids_to_reset = [document["master_id"] for document in queue if document["tagging_status"] == "processing"]
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

#questions
#back and forwards:
# - should we keep it so that you only go backwards and forwards in your own session queue?
# - or should we keep it so that you can go back and forth in the whole database, and see things tagged by others?
