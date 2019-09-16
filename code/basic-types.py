#!/usr/bin/env/python

"""basic-types.py

Module for creating and running the basic types classifier.

Usage

$ python3 basic-types.py --train-semcor

    Train a classifier from all of Semcor and save it in
    ../data/classifier-all.pickle.

$ python3 basic-types.py --train-test

    Train a classifier from a fragment of Semcor (two files) and save it in
    ../data/classifier-002.pickle, for testing and debugging purposes.

$ python3 basic-types.py --test

    Test the classifier on a feature set and test the evaluation code.

$ python3 basic-types.py --cluster

    Creates a t-distributed stochastic neighbor embedding model using the created word embeddings. Currently only
    available for the train-test model.

$ python3 basic-types.py --classify-file FILENAME

    Run the classifier on filename, output will be written to the terminal.

$ python3 basic-types.py --classify-spv1

    Run the classifier on all SPV1 files, output will be written to the out/
    directory.

The last two invocations both require the NLTK CoreNLPDependencyParser which
assumes that the Stanford CoreNLP server is running at port 9000. To use the
server run the following from the corenlp directory:

$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,depparse -status_port 9000 -port 9000 -timeout 15000

Note that this invocation does not allow you browser access to port 9000 because
the homepage uses an annotator that is not loaded by the above command.

"""
import os, sys, csv, getopt, pickle, codecs, json, glob
from tqdm import tqdm
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
from semcor import Semcor, SemcorFile
from bert_embedding import BertEmbedding
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def feature_set(types, token_features, mapping, bert_embeddings):
    mapped = zip(token_features, types)
    feature_set = []
    for i in mapped:
        if i[1] is not None and len(i[1])<4:
            features = i[0]
            # indexs: [0] token_id, [1] sent_id, [2] token_no, [3] surface,[4] lemma, [5] pos, [6] sense_no, [7] sense_key, [8] ssid,
            # [9] int_dom_token_no, [10] dom_token_id, [11] rel
            sentence_embedding = bert_embeddings[features[1]]
            if features[5] != 'VB'and type(sentence_embedding[int(features[2])-1][1]) is not str:
                feature_dict = {
                    "surface" : features[3],
                    "lemma" : features[4],
                    "pos" : features[5],
                    "embedding_vector" : sentence_embedding[int(features[2])-1][1][0].tostring(),
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
        bert_embeddings = create_bert_embeddings(sentences, model_name)
    else:
        bert_embeddings = load_embeddings(model_name)
    types_from_semcor, identifier2token = extract_types(semcor)
    print_type_count(types_from_semcor)
    feature_data = feature_set(types_from_semcor, token_features, identifier2token, bert_embeddings)
    training_set, test_set = split_data(feature_data)
    # maybe add an option to train on the entire set
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

def cluster(model_name):
    token_features, sentences = extract_token_tsv_data(SC_TOKEN_FT_SMALL)
    bert_embeddings = load_embeddings(model_name)
    types_from_semcor, identifier2token = extract_types(Semcor(2))
    mapping = zip(token_features, types_from_semcor)
    type2vector = []
    for i in mapping:
        if i[1] is not None and len(i[1])<4:
            features = i[0]
            sentence_embedding = bert_embeddings[features[1]]
            type2vector.append((i[1], i[0][3], sentence_embedding[int(features[2])-1][1][0]))
    tsne_plot(type2vector)


def tsne_plot(type2vector):
    labels = []
    tokens = []
    surface_form = []

    for (b_type, surface, vector) in type2vector:
        if type(vector) is not str:
            tokens.append(vector)
            labels.append(b_type)
            surface_form.append(surface)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10))

    colors = ['red', 'lightcoral', 'darkgreen', 'limegreen', 'darkorange', 'bisque', 'gold', 'yellow', 'fuchsia',
              'black', 'gray', 'deeppink', 'pink', 'blueviolet', 'mediumblue', 'aquamarine', 'cyan', 'darkslategrey', 'lawngreen', 'lightskyblue', 'sandybrown', 'teal']
    cmap = {}
    for i in range(len(set(labels))):
        cmap.update({list(set(labels))[i]: colors[i]})

    print(cmap)

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=cmap[labels[i]])
        plt.annotate(surface_form[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


if __name__ == '__main__':

    options = ['train-test', 'train-semcor', 'test', 'classify-file=', 'classify-spv1','cluster']
    opts, args = getopt.getopt(sys.argv[1:], '', options)

    for opt, val in opts:
        #TODO create options for creating/not creating embeddings
        if opt == '--train-test':
            train_test(False)

        elif opt == '--train-semcor':
            train(True)

        elif opt == '--test':
            # test the classifier with the full model on the 002 test set, gives
            # unrealistic results because the training data probably includes
            # the test data, just here to see whether the mechanism works
            test_classifier('all', '002')

        elif opt == '--classify-file':
            filename = val
            run_classifier_on_file(filename)

        elif opt == '--classify-spv1':
            run_classifier_on_spv1()

        elif opt == '--cluster':
            cluster('002')

