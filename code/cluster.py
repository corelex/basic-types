#!/usr/bin/env python3

"""Usage Notes:
A prerequisite step to running this code is creating the embeddings-{002, all}.pickle files via the basic_types.py module
for use in the cluster visualization.

$ python3 basic_types.py cluster

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings (see: train-semcor
    and train-test modes). Use the -model test option to use the test-sized model. Default full (all) model.

$ python3 basic_types.py word-cluster -w WORD

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings (see: train-semcor
    and train-test modes) for specific words in the corpus. Use the -w WORD option to specify an input word.
    Use the -model test option to use the test-sized model. Default full (all)
    model.

$ python3 basic_types.py polysem-cluster

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings (see: train-semcor
    and train-test modes) for lemmas associate with multiple basic types in the corpus. Use the -model test option to
    use the test-sized model. Default full (all) model."""

import argparse
from collections import Counter

from basic_types import tokentypevector
from semcor import Semcor, SemcorFile

import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE
from nltk.stem import WordNetLemmatizer

def model_cluster(model_name):
    type2vector = tokentypevector(model_name)
    labels, vectors, surface_form = tsne_all_prep(type2vector)
    x, y = create_tsne_model(vectors)
    cmap, c_handles =  create_color_handles(labels)
    display_full_plot(x, y, cmap, c_handles, labels)

def word_cluster(model_name, word):
    type2vector = tokentypevector(model_name)
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    labels, vectors, surface_form = lemma_prep(type2vector, lemma)
    print("Input word: ", word)
    labeled_cluster_output(labels, vectors, surface_form)

def tsne_all_prep(type2vector):
    labels = []
    token_vectors = []
    surface_form = []

    for (b_type, lemma, surface, vector) in type2vector:
        if type(vector) is not str:
            token_vectors.append(vector)
            labels.append(b_type)
            surface_form.append(surface)

    return labels, token_vectors, surface_form


def lemma_prep(type2vector, lemma):
    labels = []
    token_vectors = []
    surface_form = []

    for (b_type, cur_lemma, surface, vector) in type2vector:
        if type(vector) is not str and lemma == cur_lemma:
            token_vectors.append(vector)
            labels.append(b_type)
            surface_form.append(surface)

    return labels, token_vectors, surface_form


def polysem_prep(type2vector):
    all_types = {}
    for (b_type, cur_lemma, surface, vector) in type2vector:
        if type(vector) is not str:
            if cur_lemma not in all_types:
                all_types.update({cur_lemma:
                                      {'b_type':[b_type],
                                       'instances': [
                                           {'surface': surface,
                                            'bert_vector': vector,
                                            'b_type': b_type}]
                                       }
                                  })
            else:
                all_types[cur_lemma]['b_type'].append(b_type)
                all_types[cur_lemma]['instances'].append({'surface': surface,
                                                          'bert_vector': vector,
                                                          'b_type': b_type})
        polysem_labels = []
        polysem_token_vectors = []
        polysem_surface_form = []
        for (entry, values) in all_types.items():
            uniq_types = list(set(values['b_type']))
            if len(uniq_types) > 1:
                for instance in values['instances']:
                    polysem_labels.append(instance['b_type'])
                    polysem_token_vectors.append(instance['bert_vector'])
                    polysem_surface_form.append(instance['surface'])

    return polysem_labels, polysem_token_vectors, polysem_surface_form


def create_tsne_model(token_vectors):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    print("Fitting TSNE model...")
    new_values = tsne_model.fit_transform(token_vectors)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    return x,y

def create_color_handles(labels):
    colormap = cm.colors.ListedColormap(cm.get_cmap('gist_ncar')(np.linspace(0,1,len(set(labels))+1)))
    cmap = {list(set(labels))[i]: colormap.colors[i] for i in range(len(set(labels)))}
    mpatch_handles = [mpatches.Patch(color=color, label=btype) for (btype, color) in cmap.items()]

    return cmap, mpatch_handles

def display_full_plot(x, y, cmap, mpatch_handles, labels):
    print("Generating plot...")
    plt.figure(figsize=(16, 16))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=[cmap[labels[i]]])

    plt.legend(handles=mpatch_handles)
    plt.show()

def display_word_plot(x, y, cmap, mpatch_handles, labels, surface_form):
    print("Generating plot...")
    plt.figure(figsize=(16, 16))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=[cmap[labels[i]]])
        plt.annotate(surface_form[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.legend(handles=mpatch_handles)
    plt.show()

def polysem_cluster(model_name):
    type2vector = tokentypevector(model_name)
    labels, vectors, surface_form = polysem_prep(type2vector)
    x, y = create_tsne_model(vectors)
    cmap, c_handles = create_color_handles(labels)
    display_full_plot(x, y, cmap, c_handles, labels)
    # labeled_cluster_output(labels, vectors, surface_form)

def labeled_cluster_output(labels, vectors, surface_form):
    associated_forms = Counter(surface_form)
    associated_btypes = Counter(labels)
    print("Associated word forms: ")
    for (form, count) in associated_forms.items():
        print("Form: ", form, " Count: ", count)
    print("Associated basic types: ")
    for (btype, count) in associated_btypes.items():
        print("Basic type: ", btype, " Count: ", count)
    if surface_form:
        x, y = create_tsne_model(vectors)
        cmap, c_handles = create_color_handles(labels)
        display_word_plot(x, y, cmap, c_handles, labels, surface_form)
    else:
        print("Error: this token does not occur in the corpus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic vector space visualization for basic types.")
    parser.add_argument('--mode',
                        choices=['cluster', 'word-cluster', 'polysem-cluster'],
                        help="select a mode, modes indicate which cluster module to run")
    parser.add_argument('--model',
                        default='all',
                        choices=['all', 'test'],
                        help="choose a model to run against")
    parser.add_argument('-w',
                        default=None,
                        type=str,
                        help="word input for word cluster model")
    args = parser.parse_args()
    mode = args.mode

    if args.model == "test":
        model = "002"
    else:
        model = args.model

    if mode == 'cluster':
        model_cluster(model)
    elif mode == 'word-cluster':
        word = args.w
        if word:
            word_cluster(model, word)
        else:
            print("Error: No input word provided.")
    elif mode == 'polysem-cluster':
        polysem_cluster(model)