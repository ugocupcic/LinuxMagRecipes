import pandas, spacy, pickle


nlp = spacy.load("fr")


x_train = []
with open('bow/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)

bag_of_words_ingredient = {}
with open('bow/ingredients.pkl', 'rb') as f:
    bag_of_words_ingredient = pickle.load(f)

bag_of_words_ustensil = {}
with open('bow/ustensils.pkl', 'rb') as f:
    bag_of_words_ustensil = pickle.load(f)

bag_of_words_step = {}
with open('bow/steps.pkl', 'rb') as f:
    bag_of_words_step = pickle.load(f)

bag_of_words_ingredient_asvec = list(bag_of_words_ingredient.keys())
bag_of_words_ustensil_asvec = list(bag_of_words_ustensil.keys())
bag_of_words_step_asvec = list(bag_of_words_step.keys())

def to_bag_of_words_vector(items, bag_of_words_model):
    bow = [0]*len(bag_of_words_model)
    for item in items:
        doc = nlp(item)
        for token in doc:
            if not token.is_stop:
                index_item = bag_of_words_model.index(token.text.lower())
                bow[index_item] = 1
    return bow

x_train_as_bows = []
for index,x in enumerate(x_train):
    if(index % 1000) == 0:
        print("Processing.... " + str(index) + " recipes analysed.")

    ingredients = x[0]
    bow_ingredients = to_bag_of_words_vector(ingredients, bag_of_words_ingredient_asvec)

    ustensils = x[1]
    bow_ustensils = to_bag_of_words_vector(ustensils, bag_of_words_ustensil_asvec)

    steps = x[2]
    bow_steps = to_bag_of_words_vector(steps, bag_of_words_step_asvec)

    list_of_bows = [bow_ingredients, bow_ustensils, bow_steps]

    flat_concatenated_bows = [item for sublist in list_of_bows for item in sublist]
    x_train_as_bows.append(flat_concatenated_bows)


print(len(x_train_as_bows), " ", len(x_train_as_bows[0]))
print("Pickling x trains")
with open('x_train_as_bows.pkl', 'wb') as f:
        pickle.dump(x_train_as_bows, f, pickle.HIGHEST_PROTOCOL)
