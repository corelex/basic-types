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

The "classify-file" and "classify-spv1" invocations both require the NLTK CoreNLPDependencyParser which
assumes that the Stanford CoreNLP server is running at port 9000. To use the
server run the following from the corenlp directory:

$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,depparse -status_port 9000 -port 9000 -timeout 15000

Note that this invocation does not allow you browser access to port 9000 because
the homepage uses an annotator that is not loaded by the above command.

"""
import argparse, codecs, csv, glob, json, pickle, os, sys
import itertools

from bert_embedding import BertEmbedding
import nltk
from nltk.corpus import wordnet as wn
from nltk.parse.corenlp import CoreNLPDependencyParser
from semcor import Semcor, SemcorFile
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

def get_synset_features(surface: str):
    synsets = wn.synsets(surface, pos=wn.NOUN)

    synset_names = [synset.name() for synset in synsets]
    name_indexes = ["ssid" + str(i) for i in range(0, len(synsets))]
    hypernyms = [synset.name() for synset in list(itertools.chain.from_iterable([synset.hypernyms() for synset in synsets]))]
    hypernym_indexes = ["hnid" + str(i) for i in range(0, len(hypernyms))]

    path_hypernyms = [list(itertools.chain.from_iterable(synset.hypernym_paths())) for synset in synsets]

    if len(path_hypernyms) > 1:
        path_hyp = [hypernym.name() for hypernym_list in path_hypernyms for hypernym in hypernym_list]
        path_hyp_index = ["hpid" + str(i) for i in range(0, len(path_hypernyms))]
        path_hypernym_feature = zip(path_hyp_index, path_hyp)
    else:
        path_hypernym_feature = None

    return zip(name_indexes, synset_names), zip(hypernym_indexes, hypernyms), path_hypernym_feature

def get_embedding_features(sentence_embedding, token_no):
    # print(sentence_embedding)
    token_vector = sentence_embedding[token_no][1][0]
    if type(token_vector) is not str:
        indexes = ["v" + str(i) for i in range(0, token_vector.size)]
        vector_feature = [token_vector[i] for i in range(0, token_vector.size)]
        return zip(indexes, vector_feature)
    else:
        return None


def feature_set(types, token_features, identifier2token, sentence_embeddings):
    mapped = zip(token_features, types)
    feature_set = []
    for i in mapped:
        if i[1] is not None and len(i[1])<4:
            features = i[0]
            # indexs: [0] token_id, [1] sent_id, [2] token_no, [3] surface,[4] lemma, [5] pos, [6] sense_no, [7] sense_key, [8] ssid,
            # [9] int_dom_token_no, [10] dom_token_id, [11] rel
            # sentence_embedding = bert_embeddings[features[1]]
            #if features[5] != 'VB'and type(sentence_embedding[int(features[2])-1][1]) is not str:
            #TODO: Window Features, +/- 3
            #TODO: Semantic context other nouns in the sentence, their types
            #TODO: Verb in sentences
            #TODO: Move to list features -> Sci-kit learn bayesian (sequences, BoW)
            #TODO: Corelex btypes -> read notes, with details
            #TODO: Run the MaxEnt version
            print(features[11])
            if features[5] != 'VB':
                feature_dict = {
                    "surface" : features[3],
                    "lemma" : features[4],
                    "pos" : features[5],
                    "rel_to" : features[11],
                }
                if features[9] != '0':
                    feature_dict.update({
                        "int_dom_token": identifier2token[features[9]],
                        "dom_token": identifier2token[features[10]],
                    })
                # embedding_features = get_embedding_features(sentence_embeddings[features[1]], int(features[2])-1)
                # if embedding_features:
                #     feature_dict.update(embedding_features)
                synset, hypernym, path_hypernym = get_synset_features(features[3])
                if synset:
                    feature_dict.update(synset)
                if hypernym:
                    feature_dict.update(hypernym)
                if path_hypernym:
                    feature_dict.update(path_hypernym)

                # print(feature_dict)
                feature_set.append((feature_dict, i[1]))

    return feature_set


def split_data(feature_set):
    index = int(len(feature_set) * .8)
    training_set, test_set = feature_set[:index], feature_set[index:]
    return training_set, test_set


def train_classifier(training_set):
    #TODO: Move to SciKit Learn, get rid of NLTK
    #TODO: Look at most important feattures, potential bias in training time
    # classifier = nltk.NaiveBayesClassifier.train(training_set)
    classifier = nltk.MaxentClassifier.train(training_set)
    return classifier

def create_bert_embeddings(sentences, model_name):
    number_of_sents = sent_count(model_name)
    print("Creating embeddings from BERT...")
    bert_embedding = BertEmbedding(model='bert_24_1024_16')
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
        sentence_embeddings = create_bert_embeddings(sentences, model_name)
    else:
       sentence_embeddings =  load_embeddings(model_name)
    types_from_semcor, identifier2token = extract_types(semcor)
    print_type_count(types_from_semcor)
    feature_data = feature_set(types_from_semcor, token_features, identifier2token, sentence_embeddings)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['train-test',
                                 'train-semcor',
                                 'test',
                                 'classify-file',
                                 'classify-spv'],
                        help="select a mode, modes indicate which basic-type module to run")
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
    args = parser.parse_args()
    mode = args.mode
    if args.model == "test":
        model = "002"
    else:
        model = args.model

    if mode == 'train-test':
        train_test(args.create_embeddings)
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
