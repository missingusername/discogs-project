import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import pandas as pd
import os

# Global variables
folder_path = ""
image_files = []
df = pd.DataFrame()
csv_file_path = ""
current_index = 0
image_label = None
current_image = None
info_label = None
user_input_var = None
progressbar = None
progress_label = None
cover_album_progressbar = None
cover_album_label = None

def choose_folder():
    global folder_path, image_files, df, csv_file_path, current_index

    folder_path = filedialog.askdirectory()
    if folder_path:
        hide_initial_widgets()
        image_files = get_image_files(folder_path)
        df, csv_file_path = initialize_dataframe(folder_path, image_files)
        current_index = skip_tagged_images(df)
        create_widgets()
        display_image()

def hide_initial_widgets():
    label.pack_forget()
    choose_folder_button.pack_forget()
    initial_frame.pack_forget()

def get_image_files(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return image_files

def initialize_dataframe(folder_path, image_files):
    master_ids = [os.path.splitext(f)[0] for f in image_files]
    csv_file_path = os.path.join(folder_path, 'tagged_images.csv')
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path, dtype={'master_id': str})
    else:
        df = pd.DataFrame({'master_id': master_ids, 'category': [None] * len(image_files)})
    return df, csv_file_path

def skip_tagged_images(df):
    current_index = 0
    while current_index < len(image_files) and pd.notna(df.at[current_index, 'category']):
        print(f"Skipping already tagged image at index: {current_index}")
        current_index += 1
    return current_index

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
    info_label = ctk.CTkLabel(master_frame, text="", font=("Helvetica", 14))
    info_label.pack(pady=5)

def create_progress_frame():
    global progressbar, progress_label, cover_album_progressbar, cover_album_label
    progress_frame = ctk.CTkFrame(master_frame, corner_radius=10)
    progress_frame.pack(pady=10, padx=10, fill="x")

    progress_label = ctk.CTkLabel(progress_frame, text="0/0 covers tagged", font=("Helvetica", 14))
    progress_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    progressbar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
    progressbar.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
    progressbar.set(0)

    cover_album_label = ctk.CTkLabel(progress_frame, text="0 covers / 0 vinyls", font=("Helvetica", 14))
    cover_album_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    cover_album_progressbar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
    cover_album_progressbar.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
    cover_album_progressbar.set(0)

    progress_frame.grid_columnconfigure(1, weight=1)

def create_image_label():
    global image_label
    image_label = ctk.CTkLabel(master_frame, text="")
    image_label.pack(expand=True)

def initialize_user_input_var():
    global user_input_var
    user_input_var = ctk.StringVar()

def display_image():
    if current_index < len(image_files):
        update_gui()
    else:
        print("No untagged images found.")

def on_button_click(value):
    print(f"Button clicked: {value}")
    user_input_var.set(value)
    process_current_image()
    on_next_button_click()

def update_gui():
    global current_image, image_label, df, info_label, image_files, current_index, progressbar, progress_label, cover_album_progressbar, cover_album_label

    print(f"Updating image for index: {current_index}")

    image_name = df.at[current_index, 'master_id'] + '.jpg'
    category = df.at[current_index, 'category']
    print(f"Current category: {category}")

    image_path = construct_image_path(folder_path, image_name, category)
    print(f"Current image path: {image_path}")

    img = Image.open(image_path)
    current_image = ctk.CTkImage(img, size=(500, 500))

    image_label.configure(image=current_image)
    image_label.image = current_image

    info_label.configure(text=f"Image {current_index + 1} out of {len(image_files)}\nCurrent image: {os.path.basename(image_path)}\nCategory: {'Not tagged' if pd.isna(category) else category}")

    update_progress_bars(df, image_files)

def construct_image_path(folder_path, image_name, category):
    if pd.isna(category):
        return os.path.join(folder_path, image_name)
    else:
        return os.path.join(folder_path, category, image_name)

def update_progress_bars(df, image_files):
    tagged_images_count = df['category'].notna().sum()
    progress = tagged_images_count / len(image_files)
    progressbar.set(progress)
    progress_label.configure(text=f"{tagged_images_count}/{len(image_files)} covers tagged")

    covers_count = df['category'].value_counts().get('cover', 0)
    albums_count = df['category'].value_counts().get('vinyl', 0)
    total_tagged = covers_count + albums_count

    cover_album_progress = covers_count / total_tagged if total_tagged > 0 else 0
    cover_album_progressbar.set(cover_album_progress)
    cover_album_label.configure(text=f"{covers_count} covers / {albums_count} vinyls")

def process_current_image():
    global df, current_index

    category = user_input_var.get()
    current_category = df.at[current_index, 'category']
    df.at[current_index, 'category'] = category
    target_folder = os.path.join(folder_path, category)
    print(f"Tagging as {category}")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created folder: {target_folder}")

    image_name = df.at[current_index, 'master_id'] + '.jpg'
    source_path = construct_image_path(folder_path, image_name, current_category)
    new_image_path = os.path.join(target_folder, image_name)
    os.rename(source_path, new_image_path)
    print(f"Moved image to: {new_image_path}")

    df.to_csv(csv_file_path, index=False)
    print("CSV file updated")

def on_next_button_click():
    global current_index
    current_index += 1
    if current_index < len(image_files):
        update_gui()
    else:
        print("No more images.")

def on_back_button_click():
    global current_index
    if current_index > 0:
        current_index -= 1
        update_gui()

def on_key_press(event):
    if event.keysym == 'Left':
        on_button_click('vinyl')
    elif event.keysym == 'Right':
        on_button_click('cover')

# Initialize the main application window
app = ctk.CTk()

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("src/style.json")

app.title("Discogs Cover Tagger")
app.geometry("520x750")

master_frame = ctk.CTkFrame(app, fg_color="#202027")
master_frame.pack(expand=True, fill="both")

app.bind('<Left>', on_key_press)
app.bind('<Right>', on_key_press)

initial_frame = ctk.CTkFrame(master_frame, corner_radius=10)
initial_frame.pack(expand=True)

label = ctk.CTkLabel(initial_frame, text="Select an image folder to process")
label.pack(pady=10, padx=20)

choose_folder_button = ctk.CTkButton(initial_frame, text="Choose Folder", command=choose_folder)
choose_folder_button.pack(pady=10, padx=20)

app.mainloop()
