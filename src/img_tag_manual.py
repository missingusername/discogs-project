import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import os
import sys

# Function to handle button clicks
def on_button_click(value):
    print(f"Button clicked: {value}")
    user_input_var.set(value)
    process_current_image()  # Process the current image based on user input
    on_next_button_click()  # Automatically go to the next image after tagging

# Function to update the image display
def update_image_display():
    global current_image, image_label, df, info_label, image_files, current_index, progressbar, progress_label, cover_album_progressbar, cover_album_label

    print(f"Updating image for index: {current_index}")

    # Get the image path
    image_path = df.at[current_index, 'current_path']
    print(f"Current image path: {image_path}")

    # Load and resize the image
    img = Image.open(image_path)
    img = img.resize((500, 500))
    current_image = ImageTk.PhotoImage(img)

    # Update the image label
    image_label.configure(image=current_image)
    image_label.image = current_image  # Keep a reference to avoid garbage collection

    # Update the info label
    info_label.configure(text=f"Image {current_index + 1} out of {len(image_files)}\nCurrent image: {os.path.basename(image_path)}")

    # Calculate the number of tagged images
    tagged_images_count = df['is_vinyl'].notna().sum()

    # Update the progress bar
    progress = tagged_images_count / len(image_files)
    progressbar.set(progress)

    # Update the progress label
    progress_label.configure(text=f"{tagged_images_count}/{len(image_files)} covers tagged")

    # Calculate the number of covers and albums
    covers_count = df['is_vinyl'].value_counts().get(False, 0)
    albums_count = df['is_vinyl'].value_counts().get(True, 0)
    total_tagged = covers_count + albums_count

    # Update the cover vs album progress bar
    if total_tagged > 0:
        cover_album_progress = covers_count / total_tagged
    else:
        cover_album_progress = 0
    cover_album_progressbar.set(cover_album_progress)

    # Update the cover vs album label
    cover_album_label.configure(text=f"{covers_count} covers / {albums_count} vinyls")

# Function to process the current image based on user input
def process_current_image():
    global df, current_index

    # Update the "is_vinyl" column based on user input
    if user_input_var.get() == 'vinyl':
        df.at[current_index, 'is_vinyl'] = True
        target_folder = os.path.join(folder_path, 'vinyl')
        print("Tagging as vinyl")
    elif user_input_var.get() == 'cover':
        df.at[current_index, 'is_vinyl'] = False
        target_folder = os.path.join(folder_path, 'cover')
        print("Tagging as cover")
    else:
        return  # No valid input, return early

    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created folder: {target_folder}")

    # Move the image to the target folder
    image_path = df.at[current_index, 'current_path']
    new_image_path = os.path.join(target_folder, os.path.basename(image_path))
    os.rename(image_path, new_image_path)
    df.at[current_index, 'current_path'] = new_image_path
    print(f"Moved image to: {new_image_path}")

    # Save the updated CSV file after each tagging
    df.to_csv(csv_file_path, index=False)
    print("CSV file updated")

# Function to handle the "Next" button click
def on_next_button_click():
    global current_index
    current_index += 1
    if current_index < len(image_files):
        update_image_display()
    else:
        print("No more images.")

# Function to handle the "Back" button click
def on_back_button_click():
    global current_index
    if current_index > 0:
        current_index -= 1
        update_image_display()

# Function to handle keyboard input
def on_key_press(event):
    if event.keysym == 'Left':
        on_button_click('vinyl')
    elif event.keysym == 'Right':
        on_button_click('cover')

# Function to handle folder selection
def choose_folder():
    global folder_path, image_files, df, csv_file_path, current_index, image_label, current_image, info_label, user_input_var, progressbar, progress_label, cover_album_progressbar, cover_album_label

    folder_path = filedialog.askdirectory()
    if folder_path:
        # Hide the initial label and button
        label.pack_forget()
        choose_folder_button.pack_forget()
        initial_frame.pack_forget()
        
        # Get a list of image files in the selected folder
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Sort the image files based on their numeric names
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Initialize the DataFrame to keep track of tagged images
        csv_file_path = os.path.join(folder_path, 'tagged_images.csv')
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
        else:
            df = pd.DataFrame({'original_path': image_files, 'current_path': image_files, 'is_vinyl': [None] * len(image_files)})

        # Variable to keep track of the current index
        current_index = 0

        # Skip already tagged images
        while current_index < len(image_files) and pd.notna(df.at[current_index, 'is_vinyl']):
            print(f"Skipping already tagged image at index: {current_index}")
            current_index += 1

        # Create and pack the frame to hold the buttons
        button_frame = ctk.CTkFrame(master_frame)
        button_frame.pack(side="bottom", pady=10)
        
        # Show the buttons in a 2x2 grid
        button_padding = 10
        
        vinyl_button = ctk.CTkButton(button_frame, text="Vinyl", command=lambda: on_button_click('vinyl'))
        vinyl_button.grid(row=0, column=0, padx=button_padding, pady=button_padding)
        
        cover_button = ctk.CTkButton(button_frame, text="Cover", command=lambda: on_button_click('cover'))
        cover_button.grid(row=0, column=1, padx=button_padding, pady=button_padding)
        
        back_button = ctk.CTkButton(button_frame, text="Back", command=on_back_button_click)
        back_button.grid(row=1, column=0, padx=button_padding, pady=button_padding)

        next_button = ctk.CTkButton(button_frame, text="Next", command=on_next_button_click)
        next_button.grid(row=1, column=1, padx=button_padding, pady=button_padding)
        
        # Create a label to display the image info
        info_label = ctk.CTkLabel(master_frame, text="", font=("Helvetica", 14))
        info_label.pack(pady=10)

        # Create a frame to hold the progress labels and progress bars
        progress_frame = ctk.CTkFrame(master_frame)
        progress_frame.pack(pady=10, padx=20, fill="x")

        # Create a label to display the progress text
        progress_label = ctk.CTkLabel(progress_frame, text="0/0 covers tagged", font=("Helvetica", 14))
        progress_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Create a progress bar
        progressbar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
        progressbar.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        progressbar.set(0)  # Initialize progress to 0

        # Create a label to display the cover vs album progress text
        cover_album_label = ctk.CTkLabel(progress_frame, text="0 covers / 0 vinyls", font=("Helvetica", 14))
        cover_album_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        # Create a second progress bar for covers vs albums
        cover_album_progressbar = ctk.CTkProgressBar(progress_frame, orientation="horizontal")
        cover_album_progressbar.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        cover_album_progressbar.set(0)  # Initialize progress to 0

        # Configure grid to make progress bars expand
        progress_frame.grid_columnconfigure(1, weight=1)

        # Create a label to display the image
        image_label = ctk.CTkLabel(master_frame, text="")
        image_label.pack(expand=True)

        # Variable to store user input
        user_input_var = ctk.StringVar()

        # Display the first image
        if current_index < len(image_files):
            update_image_display()
        else:
            print("No untagged images found.")

# Initialize the main application window
app = ctk.CTk()

ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("src/style.json")  # Themes: blue (default), dark-blue, green

# Set the title of the window
app.title("Discogs Cover Tagger")

# Set the size of the window
app.geometry("520x750")

# Create a master frame to hold all other widgets
master_frame = ctk.CTkFrame(app, fg_color="#202027")
master_frame.pack(expand=True, fill="both")

# Bind keyboard events
app.bind('<Left>', on_key_press)
app.bind('<Right>', on_key_press)

# Create a frame to hold the initial label and button
initial_frame = ctk.CTkFrame(master_frame)
initial_frame.pack(expand=True)

# Create and pack the initial label and button inside the frame with margins
label = ctk.CTkLabel(initial_frame, text="Select an image folder to process")
label.pack(pady=10, padx=20)

choose_folder_button = ctk.CTkButton(initial_frame, text="Choose Folder", command=choose_folder)
choose_folder_button.pack(pady=10, padx=20)

# Initialize the image label variable
image_label = None

# Run the main loop to display the window
app.mainloop()
