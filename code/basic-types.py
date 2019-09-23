#!/usr/bin/env/python

"""basic-types.py

Module for creating and running the basic types classifier.

Usage

$ python3 basic-types.py train-semcor

    Train a classifier from all of Semcor and save it in
    ../data/classifier-all.pickle. Use flag -create-embedding to save new word embeddings for this model.
    Default off.

$ python3 basic-types.py train-test

    Train a classifier from a fragment of Semcor (two files) and save it in
    ../data/classifier-002.pickle, for testing and debugging purposes.
    Use flag -create-embedding to save new word embeddings for this model.
    Default off.

$ python3 basic-types.py test

    Test the classifier on a feature set and test the evaluation code.

$ python3 basic-types.py classify-file -input-file FILENAME

    Run the classifier on filename, output will be written to the terminal. Specify the file with the -f option.

$ python3 basic-types.py classify-spv1

    Run the classifier on all SPV1 files, output will be written to the out/
    directory.

$ python3 basic-types.py cluster

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings (see: train-semcor
    and train-test modes). Use the -model test option to use the test-sized model. Default full (all) model.

$ python3 basic-types.py word-cluster

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings (see: train-semcor
    and train-test modes) for specific words in the corpus. Use the -w WORD option to specify an input word.
    Use the -model test option to use the test-sized model. Default full (all)
    model.

$ python3 basic-types.py polysem-cluster

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings (see: train-semcor
    and train-test modes) for lemmas associate with multiple basic types in the corpus. Use the -model test option to
    use the test-sized model. Default full (all) model.


The "classify-file" and "classify-spv1" invocations both require the NLTK CoreNLPDependencyParser which
assumes that the Stanford CoreNLP server is running at port 9000. To use the
server run the following from the corenlp directory:

$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,depparse -status_port 9000 -port 9000 -timeout 15000

Note that this invocation does not allow you browser access to port 9000 because
the homepage uses an annotator that is not loaded by the above command.

"""
import argparse, codecs, csv, glob, json, pickle, os, sys
from collections import Counter

from bert_embedding import BertEmbedding
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.patches as mpatches
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
import numpy as np
from semcor import Semcor, SemcorFile
from sklearn.manifold import TSNE
from tqdm import tqdm


SC_SENT = '../data/semcor.sent.tsv'
SC_TOKEN_FT = '../data/semcor.token.tsv'
SC_TOKEN_FT_SMALL = '../data/semcor.token.tsv.10000'
SC_TOKEN_DEP = '../data/semcor.token.fv'


def extract_token_tsv_data(file):
    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        data = [row for row in reader]
        sentences = {}
        for semcor_token_item in data:
            sentence_id = semcor_token_item[1]
            surface_token = semcor_token_item[3]
            if sentence_id in sentences:
                sentences[sentence_id].append(surface_token)
            else:
                sentences.update({sentence_id:[surface_token]})

        return data, sentences

def extract_types(sc):
    """
    Returns a list of types from each wordform in semcor as well as a mapping
    from integers to tokens.
    """
    types = []
    mapping = {}
    i = 0
    sen_list = [file.get_sentences() for file in sc.files]
    for sentences in sen_list:
        for sentence in sentences:
            wf = sentence.wfs
            for form in wf:
                i += 1
                if form.is_word_form():
                    if form.lemma is not None:
                        mapping.update({str(i): form.lemma})
                    else:
                        mapping.update({str(i): form.text})
                    if form.synset is not None:
                            types.append(form.synset.btypes)
                    else:
                        types.append(None)
                else:
                    mapping.update({str(i): form.text})
                    types.append(None)

    return types, mapping


def feature_set(types, token_features, mapping):
    mapped = zip(token_features, types)
    feature_set = []
    for i in mapped:
        if i[1] is not None and len(i[1])<4:
            features = i[0]
            # indexs: [0] token_id, [1] sent_id, [2] token_no, [3] surface,[4] lemma, [5] pos, [6] sense_no, [7] sense_key, [8] ssid,
            # [9] int_dom_token_no, [10] dom_token_id, [11] rel
            #sentence_embedding = bert_embeddings[features[1]]
            #if features[5] != 'VB'and type(sentence_embedding[int(features[2])-1][1]) is not str:
            if features[5] != 'VB':
                feature_dict = {
                    "surface" : features[3],
                    "lemma" : features[4],
                    "pos" : features[5],
                    # "embedding_vector" : sentence_embedding[int(features[2])-1][1][0].tostring(),
                    # "sense_no" : features[6],
                    # "sense_key" : features[7],
                    # "ssid" : features[8],
                    "rel" : features[11],
                }
                if features[9] != '0':
                    feature_dict.update({
                        "int_dom_token": mapping[features[9]],
                        "dom_token": mapping[features[10]]
                    })
                # else:
                #     feature_dict.update({
                #         "int_dom_token": None,
                #         "dom_token": None
                #     })
                feature_set.append((feature_dict, i[1]))

    return feature_set


def split_data(feature_set):
    index = int(len(feature_set) * .8)
    training_set, test_set = feature_set[:index], feature_set[index:]
    return training_set, test_set


def train_classifier(training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # classifier = nltk.MaxentClassifier.train(training_set)
    return classifier

def create_bert_embeddings(sentences, model_name):
    number_of_sents = sent_count(model_name)
    print("Creating embeddings from BERT...")
    bert_embedding = BertEmbedding()
    for i in tqdm(range(1, number_of_sents)):
        sentences[str(i)] = bert_embedding(sentences[str(i)])
    save_embeddings(sentences, model_name)
    return sentences


def save_classifier(classifier, name):
    filename = '../data/classifier-%s.pickle' % name
    print("Saving %s" % filename)
    with open(filename, 'wb') as fh:
        pickle.dump(classifier, fh)

def save_embeddings(embeddings, name):
    filename = '../data/embeddings-%s.pickle' % name
    print("Saving %s" % filename)
    with open(filename, 'wb') as fh:
        pickle.dump(embeddings, fh)


def load_classifier(name):
    filename = '../data/classifier-%s.pickle' % name
    print("Loading %s" % filename)
    with open(filename, 'rb') as fh:
        classifier = pickle.load(fh)
    return classifier


def save_test_features(features, name):
    filename = '../data/test-features-%s.pickle' % name
    print("Saving %s" % filename)
    with open(filename, 'wb') as fh:
        pickle.dump(features, fh)


def load_test_features(name):
    filename = '../data/test-features-%s.pickle' % name
    print("Loading %s" % filename)
    with open(filename, 'rb') as fh:
        features = pickle.load(fh)
    return features

def load_embeddings(name):
    filename = '../data/embeddings-%s.pickle' % name
    print("Loading %s" % filename)
    with open(filename, 'rb') as fh:
        embeddings = pickle.load(fh)
    return embeddings

def evaluate_classifier(classifier, test_set):
    """
    :param classifier: classifier that has been trained on training set.
    :param test_set: 20% of the featureset which includes features and a label.
    :return: percentage accuracy of the classifier being able to label the data correctly based on features.
    """
    accuracy = nltk.classify.accuracy(classifier, test_set)
    return accuracy


def print_type_count(types):
    count = len([t for t in types if t is not None])
    print("Total number of types: %d" % len(types))
    print("Number of non-nil types: %d" % count)


def train_test(create_embeddings):
    """Train a model on the first 2 files of Semcor, using the partial feature file
    SC_TOKEN_FT_SMALL, this evaluates at 0.8808. Model and test features are
    written to ../data."""
    semcor = Semcor(2)
    _train(semcor, SC_TOKEN_FT_SMALL, '002', create_embeddings)

def train(create_embeddings):
    """Train a model on all of Semcor, using the full feature file SC_TOKEN_FT, this
    evaluates at 0.9334. Model and test features are written to ../data."""
    semcor = Semcor()
    _train(semcor, SC_TOKEN_FT, 'all', create_embeddings)

def sent_count(model_name):
    if model_name == "002":
        return int(179)
    else:
        return int(439)

def _train(semcor, features_in, model_name, create_embeddings):
    token_features, sentences = extract_token_tsv_data(features_in)
    if create_embeddings:
         create_bert_embeddings(sentences, model_name)
    # else:
    #     load_embeddings(model_name)
    types_from_semcor, identifier2token = extract_types(semcor)
    print_type_count(types_from_semcor)
    feature_data = feature_set(types_from_semcor, token_features, identifier2token)
    training_set, test_set = split_data(feature_data)
    classifier = train_classifier(training_set)
    print("Labels: %s" % classifier.labels())
    accuracy = evaluate_classifier(classifier, test_set)
    print("Accuracy on test set is %.4f" % accuracy)
    save_classifier(classifier, model_name)
    save_test_features(test_set, model_name)
    #classifier.show_most_informative_features(20)
    

def test_classifier(classifier_name, test_set):
    # just run one set of features through it
    print("Running classifier on one set of features")
    classifier = load_classifier(classifier_name)
    features = {'pos': 'NN', 'rel': 'nsubj',
                'sense_key': '1:09:00::', 'ssid': '05808619', 'sense_no': '1',
                'dom_token': 'produce', 'int_dom_token': 'produce',
                'lemma': 'investigation', 'surface': 'investigation'}
    print(classifier.classify(features))
    print("Evaluating classifier")
    test_set = load_test_features(test_set)
    print(classifier.labels())
    accuracy = evaluate_classifier(classifier, test_set)
    print("Accuracy on test set is %.4f" % accuracy)
    classifier.show_most_informative_features(20)


def run_classifier_on_file(fname_in, fname_out=None):
    classifier = load_classifier('all')
    lemmatizer = WordNetLemmatizer()
    text = codecs.open(fname_in).read()
    if fname_out is None:
        fh_out = sys.stdout
    else:
        fh_out = codecs.open(fname_out, 'w')
    sentences = nltk.sent_tokenize(text)
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    for sentence in sentences:
        parses = parser.parse(nltk.word_tokenize(sentence))
        for parse in parses:
            for (gov, gov_pos), rel, (dep, dep_pos) in parse.triples():
                if dep_pos in ('NN', 'NNS'):
                    lemma = lemmatizer.lemmatize(dep)
                    features = {'pos': dep_pos, 'rel': rel,
                                'lemma': lemma, 'surface': dep,
                                'dom_token': gov, 'int_dom_token': gov}
                    label = classifier.classify(features)
                    fh_out.write("%s\t%s\n" % (lemma, label))
                    print(lemma, label)
        print('')


def run_classifier_on_string(classifier, lemmatizer, text, fname_out):
    fh_out = codecs.open(fname_out, 'w')
    sentences = nltk.sent_tokenize(text)
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    for sentence in sentences:
        parses = parser.parse(nltk.word_tokenize(sentence))
        for parse in parses:
            for (gov, gov_pos), rel, (dep, dep_pos) in parse.triples():
                if dep_pos in ('NN', 'NNS'):
                    lemma = lemmatizer.lemmatize(dep)
                    features = {'pos': dep_pos, 'rel': rel,
                                'lemma': lemma, 'surface': dep,
                                'dom_token': gov, 'int_dom_token': gov}
                    label = classifier.classify(features)
                    fh_out.write("%s\t%s\n" % (lemma, label))


def run_classifier_on_spv1():
    classifier = load_classifier('all')
    lemmatizer = WordNetLemmatizer()
    fnames = glob.glob('/DATA/dtra/spv1-results-lif-ela/documents/*.json')
    for fname in fnames[:2]:
        try:
            with codecs.open(fname) as fh:
                json_object = json.load(fh)
                text = json_object['text']
                print(fname, len(text))
                outfile = os.path.join('out', os.path.basename(fname))
                run_classifier_on_string(classifier, lemmatizer, text, outfile)
        except:
            print('ERROR')


def tokentypevector(model_name):
    token_features, sentences = extract_token_tsv_data(SC_TOKEN_FT_SMALL)
    bert_embeddings = load_embeddings(model_name)
    if model_name == '002':
        semcor = Semcor(2)
    else:
        semcor = Semcor()
    types_from_semcor, identifier2token = extract_types(semcor)
    mapping = zip(token_features, types_from_semcor)
    type2vector = []
    for i in mapping:
        if i[1] is not None and len(i[1])<4:
            features = i[0]
            sentence_embedding = bert_embeddings[features[1]]
            type2vector.append((i[1], i[0][4], i[0][3], sentence_embedding[int(features[2])-1][1][0]))

    return type2vector

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


def model_cluster(model_name):
    type2vector = tokentypevector(model_name)
    labels, vectors, surface_form = tsne_all_prep(type2vector)
    x, y = create_tsne_model(vectors)
    cmap, c_handles =  create_color_handles(labels)
    display_full_plot(x, y, cmap, c_handles, labels)

def polysem_cluster(model_name):
    type2vector = tokentypevector(model_name)
    labels, vectors, surface_form = polysem_prep(type2vector)
    labeled_cluster_output(labels, vectors, surface_form)


def word_cluster(model_name, word):
    type2vector = tokentypevector(model_name)
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    labels, vectors, surface_form = lemma_prep(type2vector, lemma)
    print("Input word: ", word)
    labeled_cluster_output(labels, vectors, surface_form)


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

def create_color_handles(labels):
    colormap = cm.colors.ListedColormap(cm.get_cmap('gist_ncar')(np.linspace(0,1,len(set(labels))+1)))
    cmap = {}
    for i in range(len(set(labels))):
        cmap.update({list(set(labels))[i]: colormap.colors[i]})
    mpatch_handles = []
    for (btype, color) in cmap.items():
        mpatch_handles.append(mpatches.Patch(color=color, label=btype))

    return cmap, mpatch_handles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train-test',
                                         'train-semcor',
                                         'test',
                                         'classify-file',
                                         'classify-spv',
                                         'cluster',
                                         'word-cluster',
                                         'polysem-cluster'])
    parser.add_argument('-create-embeddings',
                        action='store_true',
                        default=False,
                        help="sets create new embeddings in training to true")
    parser.add_argument('-model',
                        default='all',
                        choices=['all', 'test'],
                        help="choose a model to run against")
    parser.add_argument('-input-file',
                        default=None,
                        help="input file for classify-file mode")
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

    if mode == 'train-test':
        train_test(args.creat_embeddings)
    elif mode == 'train-semcor':
        train(args.create_embeddings)
    elif mode == 'test':
        #     test the classifier with the full model on the 002 test set, gives
        #     unrealistic results because the training data probably includes
        #     the test data, just here to see whether the mechanism works
        test_classifier('all', '002')
    elif mode == 'classify-file':
        filename = args.input_file
        if filename:
            run_classifier_on_file(filename)
        else:
            print("Error: No input file provided.")
    elif mode == 'classify-spv':
        run_classifier_on_spv1()
    elif mode == 'cluster':
        model_cluster(model)
    elif mode == 'word-cluster':
        word = args.w
        if word:
            word_cluster(model, word)
        else:
            print("Error: No input word provided.")
    elif mode == 'polysem-cluster':
        polysem_cluster(model)
