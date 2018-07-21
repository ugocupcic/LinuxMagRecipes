import pandas
import urllib
from bs4 import BeautifulSoup
import urllib.request

def getPageFromURL(url):
    soupedPage = None
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        page = response.read()
        soupedPage = BeautifulSoup(page, "html5lib")
    
        print("reading: " + soupedPage.title.string)
    return soupedPage
        
def extractRecipe(url):
    recipe = []
    
    soupedPage = getPageFromURL(url)
    title = soupedPage.title.string
    
    for ustensil in soupedPage.findAll("span", {"class": "recipe-utensil__name"}):
        recipe_line = {"title": title, "url": url, "type": "ustensil", "value": ustensil.string.strip().strip("\t").strip("\n")}
        recipe.append(recipe_line)
    
    for ingredient in soupedPage.findAll("span", {"class":"ingredient"}):
        recipe_line = {"title": title, "url": url, "type": "ingredient", "value": ingredient.string.strip().strip("\t").strip("\n")}
        recipe.append(recipe_line)
    
    for step in soupedPage.findAll("li", {"class":"recipe-preparation__list__item"}):
        recipe_line = {"title": title, "url": url, "type": "step", "value": step.getText().strip().strip("\t").strip("\n")}
        recipe.append(recipe_line)
    
    tags = []
    for tag in soupedPage.findAll("li", {"class":"mrtn-tag"}):
        if tag.string not in tags:
            tags.append(tag.string.strip().strip("\t").strip("\n"))
    for tag in tags:
        recipe_line = {"title": title, "url": url, "type": "tag", "value": tag}
        recipe.append(recipe_line)
    
    recipe = pandas.DataFrame(recipe)
    
    return recipe


import sys

sys.setrecursionlimit(10000000)
all_urls = []

with open("/home/ugocupcic/all_urls.txt", 'r') as f:
    for line in f.readlines():
        if line not in all_urls:
            all_urls.append(line)

print(len(all_urls))


for index, url in enumerate(all_urls):
    new_recipe = extractRecipe(url)
    new_recipe.to_csv("/home/ugocupcic/all_recipes/"+str(index) + ".csv")

