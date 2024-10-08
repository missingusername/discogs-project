from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def get_master_data(driver, url):
    try:
        driver.get(url)
        
        # Extract image src
        try:
            div_element = driver.find_element(By.CLASS_NAME, "image_3rzgk.bezel_2NSgk")
            img_element = div_element.find_element(By.TAG_NAME, "img")
            img_src = img_element.get_attribute("src")
            print(f"Image src for {url}: {img_src}")
        except Exception as e:
            print(f"Error extracting image src for {url}")
        
        # Extract tracklist
        try:
            tracklist_element = driver.find_element(By.CSS_SELECTOR, "div.main_2FbVC[style='order:7'] section#release-tracklist")
            track_rows = tracklist_element.find_elements(By.TAG_NAME, "tr")
            tracklist = {}
            for row in track_rows:
                track_position = row.get_attribute("data-track-position")
                track_title = row.find_element(By.CLASS_NAME, "trackTitleNoArtist_ANE8Q").text
                track_duration = row.find_element(By.CLASS_NAME, "duration_25zMZ").text
                tracklist[track_position] = {"title": track_title, "duration": track_duration}
            print(f"Tracklist for {url}: {tracklist}")
        except Exception as e:
            print(f"Error extracting tracklist for {url}")
        
        # Extract rating data
        try:
            rating_element = driver.find_element(By.CLASS_NAME, "items_Y-X8L")
            rating_text = rating_element.text.split('\n')
            avg_rating = rating_text[0].split(': ')[1]
            total_ratings = rating_text[1].split(': ')[1]
            print(f"Avg Rating for {url}: {avg_rating}")
            print(f"Total Ratings for {url}: {total_ratings}")
        except Exception as e:
            print(f"Error extracting rating data for {url}")
        
    except Exception as e:
        print(f"Selenium error for {url}: {e}")

base_url = "https://www.discogs.com/master/"
master_ids = [497, 158, 215, 216, 217]

options = Options()
options.headless = True
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

for master_id in master_ids:
    url = f"{base_url}{master_id}"
    get_master_data(driver, url)

driver.quit()
