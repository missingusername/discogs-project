from io import BytesIO
import requests
import time
import pandas as pd
from pathlib import Path
from PIL import Image

from image_uri_fetcher import load_csv_as_df

def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by replacing or removing invalid characters.

    Parameters:
    filename (str): The original filename.

    Returns:
    str: The sanitized filename.
    """
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_image_extension(content_type: str) -> str:
    """
    Get the image file extension based on the content type.

    Parameters:
    content_type (str): The content type of the image.

    Returns:
    str: The image file extension.
    """
    if content_type == "image/jpeg":
        return ".jpg"
    elif content_type == "image/png":
        return ".png"
    elif content_type == "image/gif":
        return ".gif"
    else:
        return ".jpg"  # Default to .jpg if the content type is unknown

def download_album_cover_by_uri(image_url: str, save_directory: Path, image_name: str, user_agent: str = "Mozilla/5.0") -> None:
    """
    Download an image from discogs URL and save it to a specified directory.

    Parameters:
    image_url (str): The URL of the image to download.
    save_directory (Path): The directory where the image will be saved.
    image_name (str): The name of the image file.
    user_agent (str): The User-Agent header to use for the request. Default is "Mozilla/5.0".

    Raises:
    Exception: If there is an error during the download or save process.
    """
    try:
        headers = {"User-Agent": user_agent}
        # Send a GET request to the image URL with the User-Agent header
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Determine the image format from the content type
        content_type = response.headers.get("Content-Type")
        extension = get_image_extension(content_type)

        # Sanitize the image name and append the extension
        sanitized_image_name = sanitize_filename(image_name) + extension

        # Create the full file path
        file_path = save_directory / sanitized_image_name

        # Open the image and save it
        image = Image.open(BytesIO(response.content))
        image_format = image.format if image.format else 'JPEG'
        image.save(file_path, format=image_format)

        print(f"Image successfully downloaded and saved to {file_path}")
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")

def download_covers_from_df(df: pd.DataFrame, output_dir: Path):
    # Ensure the save directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_url = row['Image_uri']
        album_title = row['title']
        download_album_cover_by_uri(image_url, output_dir, album_title)
        time.sleep(1)

def main():
    input_csv_path = Path("out") / "100_masters_with_image_uri.csv"
    image_output_directory = Path("out") / "images"

    df = load_csv_as_df(input_csv_path)
    download_covers_from_df(df, image_output_directory)

if __name__ == "__main__":
    main()
