import pandas, glob, os
os.chdir("all_recipes")

all_recipes = []
for file in glob.glob("*.csv"):
    recipe = pandas.read_csv(file)
    all_recipes.append(recipe)

all_recipes = pandas.concat(all_recipes)

all_recipes.set_index(["title","url","type"], inplace=True)
all_recipes = all_recipes.loc[:, ~all_recipes.columns.str.contains('^Unnamed')]

tags = all_recipes.xs("tag", level="type")
tag_values = tags.set_index(["value"])
unique_tag_values = tag_values.index.get_level_values(0).unique()
unique_tag_values = unique_tag_values.values.tolist()

def tags_to_boolean_vector(tags):
    tags_vector = [0]*len(unique_tag_values)

    for tag in tags:
        index = unique_tag_values.index(tag)
        tags_vector[index] = 1

    return tags_vector


import numpy

x_train = []
y_train = []
x_test = []
y_test = []

all_values = all_recipes.set_index(["value"])
all_values = all_values.index.get_level_values(0).unique()
all_values = all_values.tolist()
recipe_index = 0
for id_recipe, recipe in all_recipes.groupby(level=["title"]):

    if(recipe_index % 1000) == 0:
        print("Extracting..... " + str(recipe_index) + " recipes." )
    recipe_index += 1

    recipe_for_training = {}
    recipe_for_training["ingredient"] = []
    recipe_for_training["ustensil"] = []
    recipe_for_training["step"] = []
    for id_type, item in recipe.groupby(level=["type"]):
        #print(id_type)

        if "tag" == id_type:
            tags_for_current_recipe = []
            for tag in item.values:
                tags_for_current_recipe.append(tag)
            tags_as_vector = tags_to_boolean_vector(tags_for_current_recipe)
            y_train.append(tags_as_vector)
        else:
            recipe_items = []
            for value in item.values:
                recipe_items.extend(value.tolist()[:])
            #print(recipe_items)
            recipe_for_training[id_type] = recipe_items

    #print(recipe_for_training)
    recipe_training_input_as_vector = [recipe_for_training["ingredient"],
                                       recipe_for_training["ustensil"],
                                       recipe_for_training["step"]]
    x_train.append(recipe_training_input_as_vector)

print("Finished computing y_train")

import pickle


print("Pickling X trains")
with open('x_train.pkl', 'wb') as f:
    pickle.dump(x_train, f, pickle.HIGHEST_PROTOCOL)

print("Pickling y trains")
with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
