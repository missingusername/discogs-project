import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

# Load the CSV file
input_file = 'out/random_sample_100_with_image_uri.csv'
df = pd.read_csv(input_file)

# Add "is_vinyl" column if it doesn't exist
if 'is_vinyl' not in df.columns:
    df['is_vinyl'] = None

# Function to update image and get user input
def update_image_and_get_input(index):
    global img_label, df, root, next_button, info_label

    # Get the image path
    master_id = df.at[index, 'master_id']
    image_path = f'out/images/{master_id}.jpg'

    # Load and resize the image
    img = Image.open(image_path)
    img = img.resize((500, 500))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img

    # Update the info label
    info_label.config(text=f"Image {index + 1} out of {len(df)}\nCurrent image: {master_id}")

    # Wait for user input
    next_button.wait_variable(user_input_var)

    # Update the "is_vinyl" column based on user input
    if user_input_var.get() == 'y':
        df.at[index, 'is_vinyl'] = True
    elif user_input_var.get() == 'n':
        df.at[index, 'is_vinyl'] = False

# Function to handle button clicks
def on_button_click(value):
    user_input_var.set(value)

# Function to handle window close event
def on_close():
    root.quit()
    root.destroy()

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.protocol("WM_DELETE_WINDOW", on_close)  # Bind the close event to the on_close function

# Create a label to display the image
info_label = tk.Label(root, text="", font=("Helvetica", 14))
info_label.pack()

img_label = tk.Label(root)
img_label.pack()

# Variable to store user input
user_input_var = tk.StringVar()

# Create buttons for user input
button_frame = tk.Frame(root)
button_frame.pack()

yes_button = tk.Button(button_frame, text="Yes", command=lambda: on_button_click('y'))
yes_button.pack(side=tk.LEFT)

no_button = tk.Button(button_frame, text="No", command=lambda: on_button_click('n'))
no_button.pack(side=tk.LEFT)

# Create a button to proceed to the next image
next_button = tk.Button(button_frame, text="Next", command=lambda: user_input_var.set('next'))
next_button.pack(side=tk.LEFT)

# Iterate through each row
for index in df.index:
    update_image_and_get_input(index)

# Save the updated CSV file
df.to_csv(input_file, index=False)

# Close the tkinter window
root.destroy()
