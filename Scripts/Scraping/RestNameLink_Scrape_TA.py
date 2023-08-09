# Simple scrape of all restuarant names and links for Lisbon TA
import pandas as pd
import os
import pickle
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC

#--- Settings 
url = "https://www.tripadvisor.com/Restaurants-g189158-Lisbon_Lisbon_District_Central_Portugal.html"

output_path = "/home/jon/GitRepos/gRATE/Temp/"
output_name = "LX_AllRestaurantNamesLinks_TA"

run_silent = 1 # 1 = don't show scraping
max_rests = 4920 # 4920 = N restaurants for LX TA; treated as soft limit 

#--- Set up scraper
# Start webdriver, optionally run silently
if run_silent == 1:
    from selenium.webdriver.firefox.options import Options
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options = options)
else:
    driver = webdriver.Firefox()

# Get url; handle cookies pop-up (click accept)
driver.get(url)
wait = WebDriverWait(driver, 10)
wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,"#onetrust-accept-btn-handler"))).click()

#--- Scrape names from each page, then move onto next page
all_names = [] 
all_links = []
scrape_total = 0
while scrape_total < max_rests + 1:
    # Get names and links for restaurants
    names_per_page = driver.find_elements(By.XPATH,".//a[@class = 'Lwqic Cj b']")
    links_per_page = [i.get_attribute("href")
         for i in names_per_page 
         if i.text.split(" ")[0][0].isnumeric()]
    names_per_page = [" ".join(i.text.split(" ")[1:]) 
         for i in names_per_page 
         if i.text.split(" ")[0][0].isnumeric()]
    
    # Append them
    all_names = all_names + names_per_page
    all_links = all_links + links_per_page
    
    # Update total scraped item count
    scrape_total = len(all_names)
    
    try:
        # Click next page
        driver.find_element_by_xpath('.//a[@class="nav next rndBtn ui_button primary taLnk"]').click()
        time.sleep(2)
    except:
        pass
driver.close()

# Format df
df = pd.DataFrame(zip(all_names, all_links), 
                  columns = ["Name", "Link"])
df.drop_duplicates(keep = "first", inplace = True)

# Save
pickle_file = open(os.path.join(output_path,output_name),"wb")
pickle.dump(df,pickle_file)
pickle_file.close()

# Open as df2 to check
df2 = pd.read_pickle(os.path.join(output_path,output_name))

