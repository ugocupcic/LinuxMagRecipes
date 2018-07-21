#setting a huge recursion limit as we will go quite deep
import sys
sys.setrecursionlimit(1000000)


from selenium import webdriver

driver = webdriver.PhantomJS()
driver.set_window_size(1120, 550)

all_urls = []

def crawl_for_url(url):
    if url not in all_urls:
        print("crawling: " + url)
        all_urls.append(url)
        
        driver.get(url)
        continue_link = driver.find_element_by_tag_name('a')
        elem = driver.find_elements_by_xpath("//*[@href]")
        for el in elem:
            url = el.get_attribute("href")
            # first identify all the recipes on the page and write them down
            if "marmiton.org/recettes/" in url:
                if "recettes/recette_" in url and "#" != url[-1]:
                    print("found a recipe: " + url)
                    with open("/home/ugocupcic/all_urls.txt", "a") as f:
                        f.write(url + "\n")
        #and now crawl the first uncrawled url in our list
        for el in elem:
            url = el.get_attribute("href")
            if "marmiton.org/recettes/" in url:
                if url not in all_urls:
                    crawl_for_url(url)

crawl_for_url("http://www.marmiton.org/recettes/top-internautes.aspx") 

