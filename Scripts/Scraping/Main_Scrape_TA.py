# Scrape LX restaurants
import numpy as np
import pandas as pd
import os
import pickle
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

# Input, output paths 
name_link_data = ("/home/jon/GitRepos/LX_Restaurants/Output/" +
                  "LX_AllRestaurantNamesLinks_TA")
output_path = "/home/jon/GitRepos/LX_Restaurants/Output/FileParts/"
output_name = "LX_RestaurantData_TA"

max_reviews = 50 # Max n most recent reviews per restaurant
min_reviews = 10 # Min n most recent reviews per restaurant
run_silent = 1 # 1 = don't show scraping

start_idx = 250 # Inclusive first restaurant selection index 
stop_idx = 500 # Exclusive last restaurant selection
per_step = 25 # Restaurants to scrape per step, such that:
             # abs(start_idx - stop_idx) % per_step = 0

#--- Load name_link_data, get subset of restaurants, define step indices 
df_nl = pd.read_pickle(name_link_data)

# Fixed seeded shuffle order of all df_nl indices
r_seed = 42 
np.random.seed(r_seed)
shuff_idx = np.random.choice(len(df_nl),len(df_nl),replace = False)

# indices to start for each step
step_start_idx = np.arange(start_idx, stop_idx, per_step)

#--- Loop each step (ultimately saves 1 df per step)
for stp_idx, strt_idx in enumerate(step_start_idx):
    #--- Set up webdriver, optionally run silently
    if run_silent == 1:
        options = Options()
        options.headless = True
        driver = webdriver.Firefox(options = options)
    else:
        driver = webdriver.Firefox()
        
    # Initialize lists
    name_list = [None] * per_step
    address_list = [None] * per_step
    skip_code_list = [0] * per_step
    link_list = [None] * per_step
    total_en_reviews_list = [0] * per_step
    avg_rating_list = [None] * per_step
    sub_rating_food_list = [None] * per_step
    sub_rating_service_list = [None] * per_step
    sub_rating_value_list = [None] * per_step
    sub_rating_atmosph_list = [None] * per_step
    about_list = [None] * per_step
    cuisines_list = [None] * per_step
    special_diets_list = [None] * per_step
    meals_list = [None] * per_step
    features_list = [None] * per_step
    price_range_list = [None] * per_step
    tags_review_list = [None] * per_step
    rev_titles = [None] * per_step
    rev_dates = [None] * per_step
    rev_ratings = [None] * per_step
    rev_texts = [None] * per_step
    
    #--- Loop each restaurant 
    s_time = time.time()
    first_idx = 1
    for list_idx, rst in enumerate(np.arange(strt_idx, strt_idx + per_step)):        
        # Get restautant url, assign
        driver.get(df_nl.Link.iloc[shuff_idx[rst]])   
        link_list[list_idx] = df_nl.Link.iloc[shuff_idx[rst]]
        
        if first_idx == 1:
            # Handle cookies pop-up (click accept)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,"#onetrust-accept-btn-handler"))).click()
            first_idx = 0

        #--- Get restaurant details        
        # Name, address, en_reviews, avg_rating
        name = driver.find_elements(By.XPATH,".//h1[@class = 'HjBfq']")   
        address = driver.find_elements(By.XPATH,".//span[@class = 'yEWoV']") 
        en_review_labels = driver.find_elements(By.XPATH,".//label[@class = 'label container']")       
        avg_rating = driver.find_elements(By.XPATH,".//span[@class = 'ZDEqb']")
                 
        # Ratings by sub-criteria
        sub_rate_names = driver.find_elements(By.XPATH,".//div[@class = 'DzMcu']")
        sub_rate_names = [i.find_element_by_class_name("BPsyj").text for i in sub_rate_names]
        sub_rate_values = driver.find_elements(By.XPATH, ".//div[@class = 'DzMcu']/span/span[contains(@class, 'ui_bubble_rating bubble_')]")
        sub_rate_values = [int(i.get_attribute("class").split("_")[-1]) for i in sub_rate_values]

        # "About" info (open pop up, scrape, close)
        # Try/except because some pages are structured differently...
        time.sleep(2)
        about = []
        try: 
            driver.find_element(By.XPATH,'.//a[@class="OTyAN _S b"]').click()
        except:
            try: 
                driver.find_element(By.XPATH,'.//a[@class="GllrM d S4 H3 _S b"]').click()
            except:                
                try: 
                    driver.find_element(By.XPATH,'.//a[@class="yhUxi bknfM S4 H3 _S b"]').click()
                except: pass
        time.sleep(1)
        about = driver.find_elements(By.XPATH,".//div[@class = 'jmnaM']")
        try: about_list[list_idx] = about[0].text
        except: pass
        try: driver.find_element(By.XPATH,'.//div[@class="zPIck _Q Z1 t _U c _S zXWgK"]').click()
        except: pass
       
        # Descriptive tags (cuisines, special_diets, price_range)
        descriptive_names = driver.find_elements(By.XPATH,".//div[@class = 'tbUiL b']")
        descriptive_details = driver.find_elements(By.XPATH,".//div[@class = 'SrqKb']")
        descriptive_names = [i.text for i in descriptive_names]
        descriptive_details = [i.text for i in descriptive_details]
        
        # Review tags (high frequency words across reviews)
        tags_rev = driver.find_elements(By.XPATH,".//div[@class = 'ui_tagcloud_group']")
        
        #--- List assignments (use try as several values are typically missing) 
        try: name_list[list_idx] = name[0].text
        except: pass 
        try: address_list[list_idx] = address[0].text
        except: pass 
        try: avg_rating_list[list_idx] = float(avg_rating[0].text.replace(" ", ""))            
        except: pass
        try: sub_rating_food_list[list_idx] = sub_rate_values[sub_rate_names.index("Food")]
        except: pass
        try: sub_rating_service_list[list_idx] = sub_rate_values[sub_rate_names.index("Service")]
        except: pass
        try: sub_rating_value_list[list_idx] = sub_rate_values[sub_rate_names.index("Value")]
        except: pass
        try: sub_rating_atmosph_list[list_idx] = sub_rate_values[sub_rate_names.index("Atmosphere")]
        except: pass 
        try: price_range_list[list_idx] = descriptive_details[descriptive_names.index("PRICE RANGE")]
        except: pass 
        try: cuisines_list[list_idx] = descriptive_details[descriptive_names.index("CUISINES")].replace(", " , " | ").lower()
        except: pass
        try: special_diets_list[list_idx] = descriptive_details[descriptive_names.index("SPECIAL DIETS")].replace(", " , " | ").lower()
        except: pass
        try: meals_list[list_idx] = descriptive_details[descriptive_names.index("MEALS")].replace(", " , " | ").lower()
        except: pass
        try: features_list[list_idx] = descriptive_details[descriptive_names.index("FEATURES")].replace(", " , " | ").lower()
        except: pass
        try:
            tags_rev = tags_rev[0].text.replace("\n", " | ").lower()
            tags_review_list[list_idx] = tags_rev.replace("all reviews | ", "")
        except: pass    
        
        # Skip reviews if: a) no avg_rating present (new, no details yet)
        # or b) total english reviews <= min_reviews
        total_en_rev_count = 0
        try: total_en_rev_count = int([i.text for i in en_review_labels if "English" in i.text][0].split("(")[1][:-1])
        except: pass    
        total_en_reviews_list[list_idx] = total_en_rev_count 
        if avg_rating == [] or total_en_rev_count <= min_reviews:
           skip_code_list[list_idx] = 1
           continue       
        
        #--- Get Reviews
        
        # Expand reviews on first page    
        try: driver.find_element(By.XPATH, "//span[@class='taLnk ulBlueLinks']").click()
        except: pass
        
        # Get either max_reviews or total_en_rev_count if it is less than this
        n_get_reviews = np.array([total_en_rev_count, max_reviews]).min()
        rst_rev_titles = [" "] * n_get_reviews
        rst_rev_dates = [" "] * n_get_reviews
        rst_rev_ratings = np.zeros(n_get_reviews)
        rst_rev_ratings[:] = np.nan
        rst_rev_texts = [" "] * n_get_reviews
        
        rev_count = -1
        while rev_count < n_get_reviews:
            time.sleep(2)
            
            # Get all reviews on current page, and their different attributes
            page_reviews = driver.find_elements(By.XPATH,".//div[@class='review-container']")           
            for i in range(len(page_reviews)):
                rev_count += 1     
                try:
                    rst_rev_titles[rev_count] = page_reviews[i].find_element_by_xpath(".//span[@class='noQuotes']").text
                    rst_rev_dates[rev_count] = page_reviews[i].find_element_by_xpath(".//span[contains(@class, 'ratingDate')]").get_attribute("title")
                    rst_rev_ratings[rev_count] = int(page_reviews[i].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3])
                    rst_rev_texts[rev_count] = page_reviews[i].find_element_by_xpath(".//p[@class='partial_entry']").text.replace("\n", " ")
                except:
                    pass
                
            # Change the page (else these are all reviews)
            try:
                driver.find_element(By.XPATH,'.//a[@class="nav next ui_button primary"]').click()
            except: break
        
        # Assign reviews (current restaurant)
        rev_titles[list_idx] = rst_rev_titles
        rev_dates[list_idx] = rst_rev_dates
        rev_ratings[list_idx] = rst_rev_ratings
        rev_texts[list_idx] = rst_rev_texts        

    # Assign step data to df         
    df = pd.DataFrame(zip(name_list, skip_code_list, link_list, address_list,
                          total_en_reviews_list, 
                          avg_rating_list, sub_rating_food_list, 
                          sub_rating_service_list, sub_rating_value_list, 
                          sub_rating_atmosph_list, about_list, 
                          cuisines_list, special_diets_list, meals_list,
                          features_list, price_range_list, 
                          tags_review_list, rev_titles, rev_dates, 
                          rev_ratings, rev_texts), 
                      columns = ["Name", "SkipCode","URL","Address", 
                                 "TotalReviewsEN", "AvgRating",
                                 "FoodRating","ServiceRating", 
                                 "ValueRating", "AtmosphereRating",
                                 "About", "Cuisines", "SpecialDiets",
                                 "Meals", "Features",
                                 "PriceRange", "ReviewTags", 
                                 "ReviewTitles", "ReviewDates", 
                                 "ReviewRatings", "ReviewTexts"])
    # Pickle          
    pickle_file = open(os.path.join(output_path,"LX_RestaurantData_TA_mn" + 
                       str(min_reviews) + "_mx" + str(max_reviews) + "_" +
                       str(strt_idx) + "_" + str(strt_idx + per_step - 1)),
                      "wb")
    pickle.dump(df,pickle_file)
    pickle_file.close()
    
    # Show step time
    e_time = (time.time() - s_time)
    print(f"Step Time (seconds): {e_time: .2f}")
    
    driver.close()
        
 



