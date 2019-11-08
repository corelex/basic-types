import argparse
import csv
import itertools
import sys, os

from modules import cltypes
from semcor import Semcor, SemcorFile

from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


SC_SENT = '../data/semcor.sent.tsv'
SC_TOKEN_FT = '../data/semcor.token.tsv'
SC_TOKEN_FT_SMALL = '../data/semcor.token.tsv.10000'
SC_TOKEN_DEP = '../data/semcor.token.fv'
POS = ['NN', 'NNP']
BASIC_TYPES = {key:[type.split('.')[0] for type in value[0][1].split()]
               for (key, value) in cltypes.BASIC_TYPES_3_1.items()}

sys.path.append(os.getcwd())

class SemcorCorpus:
    def __init__(self, file, model_name):
        self.tsv_ids = self.extract_tsv_data(file)
        self.semcor = self.create_semcor(model_name)
        self.sen_list = [file.get_sentences() for file in self.semcor.files]
        self.elements = self.extract_basic_type_objects()

    def extract_tsv_data(self, file):
        with open(file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            data = [row for row in reader]
            tsv_ids = {}
            for semcor_token_item in data:
                token_identifier = (semcor_token_item[1], semcor_token_item[3])
                data = {'token_id': semcor_token_item[0],
                        'sent_id': semcor_token_item[1],
                        'token_no': semcor_token_item[2],
                        'surface': semcor_token_item[3],
                        'lemma': semcor_token_item[4],
                        'pos': semcor_token_item[5],
                        'int_dom_token_no': semcor_token_item[9],
                        'dom_token_id': semcor_token_item[10],
                        'rel': semcor_token_item[11]
                        }

                tsv_ids.update({token_identifier: data})

            return tsv_ids

    def get_elements(self):
        return self.elements

    def create_semcor(self, model_name):
        if model_name == "test":
            semcor = Semcor(2)
        else:
            semcor = Semcor()

        return semcor

    def extract_basic_type_objects(self):
        objects_list = []
        for sentences in self.sen_list:
            for sentence in sentences:
                word_forms = sentence.elements
                new_sentence = []
                for word_form in word_forms:
                    if word_form.is_word_form():
                        basic_type = None
                        sid = word_form.sid
                        if word_form.synset is not None:
                            basic_type = word_form.synset.btypes
                        tsv_id = (sid, word_form.position)
                        b_type = BasicTypeObject(surface=word_form.text,
                                                 sentence=sentence,
                                                 lemma=word_form.lemma,
                                                 basic_type=basic_type,
                                                 sid=word_form.sid,
                                                 pos=word_form.pos,
                                                 sentence_position=word_form.position)
                        tsv_features = self.tsv_ids.get(tsv_id)
                        if tsv_features:
                            b_type.update_linking_rel(tsv_features['rel'])
                            dom_id = tsv_features['int_dom_token_no']
                            dom_features = self.tsv_ids.get((sid, dom_id))
                            if dom_features:
                                b_type.update_dom_lemma(dom_features['lemma'])
                        new_sentence.append(b_type)
                    else:
                        new_sentence.append(word_form)
                objects_list.append(new_sentence)
        return objects_list

class BasicTypeObject:
    def __init__(self, surface, sentence, lemma, basic_type, sid, pos, sentence_position):
        self.sentence = sentence
        self.surface_form = surface
        self.lemma = lemma
        self.basic_type = basic_type
        self.sentence_position = sentence_position
        self.sid = sid
        self.pos = pos
        self.embedding = None
        self.linking_rel = None
        self.dom_lemma = None
        self.corelex = None

    def __str__(self):
        b_type = "None"
        if self.basic_type is not None:
            b_type = str(self.basic_type)
        return "<BasicTypeObj (" + self.surface_form + ", " + b_type +")>"

    def get_basic_type(self):
        return self.basic_type

    def is_basic_type(self):
        return True

    def is_punctuation(self):
        return False

    def update_linking_rel(self, linking_rel):
        self.linking_rel = linking_rel

    def update_dom_lemma(self, dom_lemma):
        self.dom_lemma = dom_lemma

    def update_corelex(self, corelex):
        self.corelex = corelex

    def get_corelex(self):
        return self.corelex

class BasicTyper:
    def __init__(self, feature_extractor, classifier_type, model_name):
        self.feature_extractor = feature_extractor
        self.model_name = model_name
        self.classifier = self.set_classifier(classifier_type)
        self.featurized_data = {}


    def set_classifier(self, classifier_type):
        if classifier_type == "NaiveBayes":
            return MultinomialNB()
        else:
            return LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')


    def featurize_corpus(self, semcor_corpus, train_split):
        #TODO: Add ability to k-fold training per PA feedback
        feature_dictionaries = []
        label_list = []
        instance_list = []
        elements = semcor_corpus.get_elements()
        for sen in tqdm(elements, desc="Featurizing sentences"):
            features = self.feature_extractor.extract(sen)
            labels = self.feature_extractor.label_extract(sen)
            instances = self.feature_extractor.instance_extract(sen)
            feature_dictionaries.extend(features)
            label_list.extend(labels)
            instance_list.extend(instances)

        vectorizer = DictVectorizer(dtype=np.uint8, sparse=True)
        X = vectorizer.fit_transform(feature_dictionaries)
        y = np.asarray(label_list)

        self.X = X
        self.y = y

        idx = int(self.X.shape[0]*train_split)
        self.featurized_data.update({'X_train': X[:idx],
                                   'X_test': X[idx:],
                                    'y_train': y[:idx],
                                    'y_test':  y[idx:],
                                    'train_instances': instance_list[:idx],
                                    'test_instances': instance_list[idx:]})


    def train(self):
        X = self.featurized_data['X_train']
        y = self.featurized_data['y_train']
        print("Training classifier ...")
        self.classifier.fit(X=X, y=y)


    def predict(self):
        X = self.featurized_data['X_test']
        print("Predicting labels ...")
        return self.classifier.predict(X=X)


    def predict_probablities(self):
        predicted_labels = []

        X = self.featurized_data['X_test']
        test_instances = self.featurized_data['test_instances']
        probabilities = pd.DataFrame(self.classifier.predict_proba(X), columns=self.classifier.classes_)
        row_count = probabilities.shape[0]

        for i in range(0, row_count):
            prob_instance = probabilities.iloc[i]
            corelex = test_instances[i].get_corelex()
            if corelex:
                predicted_label = prob_instance[test_instances[i].get_corelex()].idxmax(axis=1)
            else:
                predicted_label = prob_instance.idxmax(axis=1)

            predicted_labels.append(predicted_label)

        return predicted_labels


    def validate(self, predicted_labels):
        y_true = self.featurized_data['y_test']
        print(accuracy_score(y_true, predicted_labels))


class WindowedFeatureExtractor:
    def __init__(self, feature_extractors, window_size: int):
        self.feature_extractors = feature_extractors
        self.window_size = window_size

    def extract(self, basic_types):
        features = []
        for current_idx in range(len(basic_types)):
            instance = basic_types[current_idx]
            if not instance.is_punctuation() and instance.pos in POS and instance.get_basic_type():
                if len(instance.get_basic_type()) < 4:
                    feature_dict = {}
                    for feature_extractor in self.feature_extractors:
                        for window_idx in range(-self.window_size, self.window_size+1):
                            relative_idx = current_idx + window_idx
                            if current_idx + window_idx >= 0 and current_idx + window_idx < len(basic_types):
                                cur_instance = basic_types[relative_idx]
                                feature_extractor.extract(cur_instance, window_idx, feature_dict)
                    features.append(feature_dict)
        return features

    def label_extract(self, basic_types):
        labels = []
        for instance in basic_types:
            if not instance.is_punctuation() and instance.pos in POS and instance.get_basic_type():
                if len(instance.get_basic_type()) < 4:
                    labels.append(instance.get_basic_type())
        return labels

    def instance_extract(self, basic_types):
        instances = []
        for instance in basic_types:
            if not instance.is_punctuation() and instance.pos in POS and instance.get_basic_type():
                if len(instance.get_basic_type()) < 4:
                    instances.append(instance)
        return instances


class POSFeatureExtractor:
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation() and instance.pos:
            features.update({'pos[' + str(window_idx) + ']=' + instance.pos: 1.0})


class RelFeatureExtractor:
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation() and window_idx == 0:
            rel = instance.linking_rel
            if rel:
                features.update({'rel=' + rel: 1.0})


class ContextFeatureExtractor:
    #TODO: maybe allow this feature to behavior differently w/r/t windows
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation() and window_idx != 0:
            features.update({'context[' + str(window_idx) + ']=' + instance.surface_form: 1.0})
        elif instance.is_punctuation() and window_idx != 0:
            features.update({'context[' + str(window_idx) + ']=PUNC' : 1.0})

class SynsetFeatureExtractor:
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation():
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            synset_names = [synset.name() for synset in synsets]
            feature_values = ['ssid' + str(i) + '[' + str(window_idx) + ']=' + synset_names[i] for i in range(0, len(synset_names))]
            features.update(zip(feature_values, itertools.repeat(1.0)))


class HypernymFeatureExtractor:
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation():
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            hypernyms = [synset.name() for synset in
                         list(itertools.chain.from_iterable([synset.hypernyms() for synset in synsets]))]
            feature_values = ["hnid" + str(i) + '[' + str(window_idx) + ']=' + hypernyms[i] for i in range(0, len(hypernyms))]
            # print(feature_values)
            features.update(zip(feature_values, itertools.repeat(1.0)))


class HypernymPathFeatureExtractor:
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation():
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            path_hypernyms = [list(itertools.chain.from_iterable(synset.hypernym_paths())) for synset in synsets]

            if len(path_hypernyms) > 1:
                path_hyp = [hypernym.name() for hypernym_list in path_hypernyms for hypernym in hypernym_list]
                feature_values = ["hpid" + str(i) + '[' + str(window_idx) + ']=' + path_hyp[i] for i in range(0, len(path_hyp))]
                # print(path_hyp, instance.get_basic_type())
                features.update(zip(feature_values, itertools.repeat(1.0)))


class CorelexFeatureExtractor:
    def extract(self, instance, window_idx, features):
        if not instance.is_punctuation():
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            path_hypernyms = [list(itertools.chain.from_iterable(synset.hypernym_paths())) for synset in synsets]
            path_hyp = [hypernym.name().split('.')[0] for hypernym_list in path_hypernyms for hypernym in hypernym_list]
            corelex = [key for (key, value) in BASIC_TYPES.items() for hym_name in value if hym_name in path_hyp]
            instance.update_corelex(corelex)
            feature_values = ["corelex" + str(i) + '[' + str(window_idx) + ']=' + corelex[i] for i in range(0, len(corelex))]
            features.update(zip(feature_values, itertools.repeat(1.0)))


def main(model_name, window, train_split, classifier_type, prediction_type):
    features_in = SC_TOKEN_FT
    sc = SemcorCorpus(file=features_in, model_name=model_name)

    model = BasicTyper(
        WindowedFeatureExtractor(
            [
                POSFeatureExtractor(),
                RelFeatureExtractor(),
                ContextFeatureExtractor(),
                SynsetFeatureExtractor(),
                HypernymFeatureExtractor(),
                HypernymPathFeatureExtractor(),
                CorelexFeatureExtractor()
            ],
            window),
        classifier_type,
        model_name)

    model.featurize_corpus(sc, train_split)
    model.train()

    if prediction_type == "filtered":
        predictions = model.predict_probablities()
    else:
        predictions = model.predict()

    model.validate(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A classifier to predict a noun's Corelex Basic Type.")
    parser.add_argument('--mode',
                        default='train-test',
                        choices=['train-test'])
    parser.add_argument('--model',
                        default='all',
                        choices=['all', 'test'],
                        help="Choose the the model size you'd like to train and test against (default=all).")
    parser.add_argument('--architecture',
                        default="LogisticRegression",
                        choices=['LogisticRegression', 'NaiveBayes'],
                        help="Choose the model architecture you'd like to use (default=LogisticRegression).")
    parser.add_argument('--window-size',
                        default=0,
                        type=int,
                        help="Choose the size of the window on either side of a target token from which to generate "
                             "features (default=0).")
    parser.add_argument('--train-split',
                        default=0.9,
                        choices=np.arange(0.1, 0.9),
                        type=float,
                        metavar="[0.1-0.9]",
                        help="Choice the percentage of data you'd like in the training set (default=0.9).")
    parser.add_argument('--prediction-type',
                        default='unfiltered',
                        choices=['unfiltered', 'filtered'],
                        help="Use the possible synsets as a filter for a token to limit the possible labels only to those"
                             " that align with possible basic types (default=unfiltered).")
    args = parser.parse_args()
    mode = args.mode

    main(model_name=args.model,
         window=args.window_size,
         train_split=args.train_split,
         classifier_type=args.architecture,
         prediction_type=args.prediction_type)