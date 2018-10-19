import pandas, glob, os
os.chdir("data/all_recipes")

all_recipes = []
for file in glob.glob("*.csv"):
    recipe = pandas.read_csv(file)
    all_recipes.append(recipe)

all_recipes = pandas.concat(all_recipes)
all_recipes.set_index(["title","url","type","value"], inplace=True)
all_recipes = all_recipes.loc[:, ~all_recipes.columns.str.contains('^Unnamed')]

tags = all_recipes[all_recipes.index.get_level_values('type').isin(['tag'])]

tags_col = tags.reset_index(level=['value'])

common_tags = tags_col.apply(pandas.value_counts)[1:6].index.values

filtered_by_tags = tags[tags.index.get_level_values('value').isin(common_tags)]

filtered_titles = filtered_by_tags.index.get_level_values('title').unique()

filtered_recipes = all_recipes[all_recipes.index.get_level_values('title').isin(filtered_titles)]
filtered_recipes = filtered_recipes[filtered_recipes.index.get_level_values('type').isin(['ingredient', 'tag'])]

filtered_recipes = filtered_recipes.reset_index(level=["value"])

import spacy
nlp = spacy.load('fr')


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            #print("label: " + str(label) + " / score: " + str(score) + " / gold: " + str(gold) )
            if label not in gold["cats"]:
                #print("LABEL NOT IN GOLD")
                continue
            #print("LABEL IN GOLD " + str(gold["cats"][label]) )
            if score >= 0.5 and gold["cats"][label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold["cats"][label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold["cats"][label] < 0.5:
                tn += 1
            elif score < 0.5 and gold["cats"][label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


import pickle

x_train = []
y_train = []

recipe_index = 0

for id_recipe, recipe in filtered_recipes.groupby(level=["title"]):
    if (recipe_index % 50) == 0 and len(x_train) > 0:
        print("Extracted..... " + str(recipe_index) + " recipes.")
        break

    recipe_index += 1

    tags_for_current_recipe = {}
    for id_type, item in recipe.groupby(level=["type"]):

        if "tag" == id_type:
            recipe_tags = [" ".join(tag.tolist()[:]) for tag in item.values]

            for tag in common_tags:
                if tag in recipe_tags:
                    tags_for_current_recipe[tag] = 1.
                else:
                    tags_for_current_recipe[tag] = 0.

        elif "ingredient" == id_type:
            ingredients = ""
            for value in item.values:
                ingredients += " ".join(value.tolist()) + " "
            x_train.append(ingredients)

    y_train.append({"cats": tags_for_current_recipe})

os.chdir("../..")
print("And pickling the data....")
with open("data/x_train.pkl", "wb") as f:
    pickle.dump(x_train, f)

with open("data/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

print(".....done")

for i in range(10):
    print(str(x_train[i]) + "  =>  " + str(y_train[i]))

all_labels = common_tags

max_size = min(len(x_train), len(y_train))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train[:max_size], y_train[:max_size], test_size=0.3)

train_data = list(zip(X_train, Y_train))

if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
else:
    textcat = nlp.get_pipe('textcat')

for label in all_labels:
    print("Adding label: " + label)
    textcat.add_label(label)

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    losses = {}
    optimizer = nlp.begin_training()
    print("Training the model...")
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))

    for i in range(2):
        nlp.update(X_train, Y_train, sgd=optimizer, drop=0.2, losses=losses)

        if i % 100 == 0:
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, X_train, Y_train)
                print('Training: {0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
                      .format(losses['textcat'], scores['textcat_p'],
                              scores['textcat_r'], scores['textcat_f']))
                scores = evaluate(nlp.tokenizer, textcat, X_test, Y_test)
                print('Training: {0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
                      .format(losses['textcat'], scores['textcat_p'],
                              scores['textcat_r'], scores['textcat_f']))

nlp.to_disk("model")


test_recipe = "sucre farine oeufs"
doc = nlp(test_recipe)
print(test_recipe, doc.cats)

test_recipe = "poulet lait citron sel poivre pâtes"
doc = nlp(test_recipe)
print(test_recipe, doc.cats)

test_recipe = "farine de blé noir oeuf eau"
doc = nlp(test_recipe)
print(test_recipe, doc.cats)

test_recipe = "sucre farine oeufs lait beurre"
doc = nlp(test_recipe)
print(test_recipe, doc.cats)