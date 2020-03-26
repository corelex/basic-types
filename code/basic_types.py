import argparse
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import pickle
import sys, os

from modules import cltypes
from semcor import Semcor, SemcorFile

from bert_embedding import BertEmbedding
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call
from tqdm import tqdm


TIME = str(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
ALL_OUTPUT =  '../_experiments/model_all/' + TIME + '/'
TEST_OUTPUT = '../_experiments/model_test/' + TIME + '/'
SC_SENT = '../data/semcor.sent.tsv'
SC_TOKEN_FT = '../data/semcor.token.tsv'
SC_TOKEN_FT_SMALL = '../data/semcor.token.tsv.10000'
SC_TOKEN_DEP = '../data/semcor.token.fv'
POS = ['NN', 'NNP']
BASIC_TYPES = {key:[type.split('.')[0] for type in value[0][1].split()]
               for (key, value) in cltypes.BASIC_TYPES_3_1.items()}

KF = KFold(n_splits=10)
sys.path.append(os.getcwd())


class SemcorCorpus:
    def __init__(self, file, model_name):
        self.tsv_ids = self.extract_tsv_data(file)
        self.semcor = self.create_semcor(model_name)
        self.sen_list = [file.get_sentences() for file in self.semcor.files]
        self.sentences = list(itertools.chain.from_iterable([file.get_sentences() for file in self.semcor.files]))
        # self.sen_idx = self.semcor.sent_idx
        self.elements = self.extract_basic_type_objects()
        self.sen_dict = self.extract_sentences()

    def test_files(self):
        # for file in self.semcor.files:
        file = self.semcor.files[0]
        print(file.fname)
        sens = file.get_sentences()
        first_sen = sens[0].as_string()
        print(first_sen)

        for word in sens[0].elements:
            if word.is_word_form():
                print(word.text)
                print(word.lemma)
            else:
                print(word.text)

        print(self.sentences[0].as_string())


    def extract_tsv_data(self, file):
        with open(file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            data = [row for row in reader]
            tsv_ids = {}
            for semcor_token_item in data:
                token_identifier = (semcor_token_item[1], semcor_token_item[2]) #(sen_id, int_token_id)
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


    def extract_sentences(self):
        sen_dict = {}
        for sentences in self.sen_list:
            for sentence in sentences:
                # print(sentence)
                sid = sentence
                snt = [item for item in sentence.elements if item.is_word_form()]


            #     snt = [item.text for item in sentence.elements]
            #     sen_dict[sid] = [" ".join(snt)]
                # sen_dict[sid] = snt
        return sen_dict


    def extract_basic_type_objects(self):
        objects_list = []
        for sen_id in range(0, len(self.sentences)):
            sentence = self.sentences[sen_id]
            word_forms = sentence.elements

            new_sentence = []
            for word_form in word_forms:
                if word_form.is_word_form():
                    basic_type = None
                    if word_form.synset is not None:
                        basic_type = word_form.synset.btypes

                    tsv_id = (str(sen_id+1),str(int(word_form.position)+1))
                    b_type = BasicTypeObject(surface=word_form.text,
                                             sentence=sentence,
                                             lemma=word_form.lemma,
                                             basic_type=basic_type,
                                             sid=sen_id,
                                             pos=word_form.pos,
                                             sentence_position=word_form.position,
                                             synset=word_form.synset)

                    tsv_features = self.tsv_ids.get(tsv_id)

                    if tsv_features:
                        b_type.update_linking_rel(tsv_features['rel'])
                        dom_id = tsv_features['int_dom_token_no']
                        dom_features = self.tsv_ids.get((str(sen_id+1), str(dom_id)))

                        if dom_features:
                            b_type.update_dom_lemma(dom_features['lemma'])
                            b_type.update_dom_surface(dom_features['surface'])
                            b_type.update_dom_pos(dom_features['pos'])
                            b_type.update_dom_linking_rel(dom_features['rel'])

                    #set possible corelex types
                    synsets = wn.synsets(b_type.surface_form, pos=wn.NOUN)
                    path_hypernyms = [list(itertools.chain.from_iterable(synset.hypernym_paths())) for synset in
                                      synsets]
                    path_hyp = [hypernym.name().split('.')[0] for hypernym_list in path_hypernyms for hypernym in
                                hypernym_list]

                    corelex = [key for (key, value) in BASIC_TYPES.items() for hym_name in value if
                               hym_name in path_hyp]
                    b_type.update_corelex(corelex)

                    new_sentence.append(b_type)
                else:
                    new_sentence.append(word_form)

            objects_list.append(new_sentence)

        return objects_list


class BasicTypeObject:
    def __init__(self, surface, sentence, lemma, basic_type, sid, pos, sentence_position, synset):
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
        self.dom_surface = None
        self.dom_linking_rel = None
        self.dom_pos = None
        self.corelex = None
        self.synset = synset

    def __str__(self):
        b_type = "None"
        if self.basic_type is not None:
            b_type = str(self.basic_type)
        return "<BasicTypeObj (" + self.surface_form + ", " + b_type + ")>"

    def __repr__(self):
        b_type = "None"
        if self.basic_type is not None:
            b_type = str(self.basic_type)
        return "<BasicTypeObj (" + self.surface_form + ", " + b_type + ")>"

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

    def update_dom_surface(self, dom_surface):
        self.dom_surface = dom_surface

    def update_dom_linking_rel(self, dom_linking_rel):
        self.dom_linking_rel = dom_linking_rel

    def update_dom_pos(self, dom_pos):
        self.dom_pos = dom_pos

    def update_corelex(self, corelex):
        self.corelex = corelex

    def get_corelex(self):
        return self.corelex

    def get_feature_attributes(self):
        attribute_dict = {'surface_form' : self.surface_form,
                         'lemma': self.lemma,
                         'basic_type': self.basic_type,
                         'pos': self.pos,
                         'linking_rel': self.linking_rel,
                         'dom_surface': self.dom_surface,
                         'dom_lemma': self.dom_lemma,
                         'dom_pos': self.dom_pos,
                         'dom_linking_rel': self.dom_linking_rel,
                         'corelex': self.corelex}
        return attribute_dict

    def get_id_attributes(self):
        id_attributes = {
        'surface_form': self.surface_form,
        'lemma': self.lemma,
        'basic_type': self.basic_type,
        'wn_synset': str(self.synset),
        'corelex': self.corelex,
        'sid': self.sid,
        'sentence_position': self.sentence_position,
        'source_sentence': self.sentence.as_string()}
        return id_attributes


class BasicTyper:
    def __init__(self, feature_extractor, classifier_type, model_name):
        self.feature_extractor = feature_extractor
        self.model_name = model_name
        self.classifier = self.set_classifier(classifier_type)
        self.featurized_data = {}
        self.feature_mapping = None
        self.feature_names = None

    def set_classifier(self, classifier_type):
        if classifier_type == "NaiveBayes":
            return MultinomialNB()
        elif classifier_type == "DecisionTree":
            return DecisionTreeClassifier()
        else:
            return LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')

    def get_data(self):
        return self.featurized_data

    def featurize_corpus(self, semcor_corpus, print_features):
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

        features, labels, instances = self.naive_debias(feature_dictionaries, label_list, instance_list)

        self.features = features
        self.metadata = {
                    'metadata': {
                        'model': self.model_name,
                        'date_run': TIME,
                        'features': str(self.feature_extractor.get_feature_extractors()),
                        'window_size': self.feature_extractor.get_window_size(),
                        'synset_cap': self.feature_extractor.get_cap()
                    }
                }
        print(self.metadata)
        if print_features:
            if self.model_name == 'all':
                feature_output_file = ALL_OUTPUT + 'features_' + '.jsonl'
            else:
                feature_output_file = TEST_OUTPUT + 'features_' + '.jsonl'
            with open(feature_output_file, 'a+') as jsonl:
                json.dump(self.metadata, jsonl, indent=4)
                for i in range(len(instances)):
                    feat = {}
                    feat.update({'instance_id': instances[i].get_id_attributes()})
                    feat.update(features[i])
                    json.dump(feat, jsonl, indent=4)


        vectorizer = DictVectorizer(dtype=np.uint8, sparse=True)
        self.vectorizer = vectorizer
        X = vectorizer.fit_transform(features)
        self.feature_mapping = vectorizer.vocabulary_
        self.feature_names = vectorizer.feature_names_
        y = np.asarray(labels)

        self.X = X
        self.y = y
        self.instance_list = instances

        # row_count = self.X.shape[0]
        # print("instances: ", row_count)

        # train_idx = int(row_count*train_split)
        # dev_idx = train_idx + int((row_count-train_idx)/2)
        # self.featurized_data.update({'X_train': X[:train_idx],
        #                              'X_dev': X[train_idx:dev_idx],
        #                              'X_test': X[dev_idx:],
        #                              'y_train': y[:train_idx],
        #                              'y_dev':  y[train_idx:dev_idx],
        #                              'y_test': y[dev_idx:],
        #                              'train_instances': instance_list[:train_idx],
        #                              'dev_instances': instance_list[train_idx:dev_idx],
        #                              'test_instances': instance_list[dev_idx:]})


    def naive_debias(self, features, labels, instances):
        label_count = {}
        debiased_features = []
        debiased_labels = []
        debiases_instances = []
        for i in range(len(labels)):
            label = labels[i]
            if label in label_count.keys():
                if self.model_name == 'test':
                    if label_count[label] < 150:
                        label_count[label] += 1
                        debiased_features.append(features[i])
                        debiased_labels.append(labels[i])
                        debiases_instances.append(instances[i])
                else:
                    if label_count[label] < 1000:
                        label_count[label] += 1
                        debiased_features.append(features[i])
                        debiased_labels.append(labels[i])
                        debiases_instances.append(instances[i])
            else:
                label_count.update({label:1})
                debiased_features.append(features[i])
                debiased_labels.append(labels[i])
                debiases_instances.append(instances[i])

        print(label_count)
        return debiased_features, debiased_labels, debiases_instances

    def train(self, train_split):
        row_count = self.X.shape[0]
        train_idx = int(row_count * train_split)
        X_train, X_test = self.X[:train_idx], self.X[train_idx:]
        y_train, y_test = self.y[:train_idx], self.y[train_idx:]
        print("Training classifier ...")
        self.classifier.fit(X=X_train, y=y_train)
        predictions = self.predict(X_test)
        print("\nraw_predictions: ", self.validate(predictions, y_test)
              )
        export_graphviz(self.classifier, out_file='tree.dot', feature_names=self.feature_names,
                        class_names=self.classifier.classes_)
        # call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])

    def k_fold(self):
        KF.get_n_splits(self.X)
        print("Training classifier ...")
        for train_index, test_index in tqdm(KF.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.classifier.fit(X=X_train, y=y_train)
            predictions = self.predict(X_test)
            # filter_predictions, id_attributes = self.predict_probablities(X_test)

            relevant_probablities = self.filtered_probabilities(X_test, y_test, predictions)

            print("\nraw_predictions: ", self.validate(predictions, y_test))
            # print("filtered_predictions: ", self.validate(filter_predictions, y_test))

        if self.model_name == 'all':
            prediction_output_file = ALL_OUTPUT + 'predictions_' + '.jsonl'
        else:
            prediction_output_file = TEST_OUTPUT + 'predictions_' + '.jsonl'

        with open(prediction_output_file, 'a') as jsonl:
            test_features = self.vectorizer.inverse_transform(X_test)
            json.dump(self.metadata, jsonl, indent=4)
            for i in range(len(test_features)):
                if y_test[i] == predictions[i]:
                    misclassified = False
                else:
                    misclassified = True
                prediction_validation = {
                    'features': str(test_features[i]),
                    'true_label': y_test[i],
                    'predicted_label': predictions[i],
                    'probabilities': relevant_probablities[i].to_dict(),
                    'misclassified': misclassified
                }
                json.dump(prediction_validation, jsonl, indent=4)

            print(classification_report(predictions, y_test))
            self.most_informative_features()

    def predict(self, X):
        # X = self.featurized_data['X_dev']
        # print("Predicting labels ...")
        return self.classifier.predict(X=X)


    def predict_probablities(self, X):
        #TODO: figure out how to extract Corelex
        predicted_labels = []
        id_attributes = []
        probabilities = pd.DataFrame(self.classifier.predict_proba(X), columns=self.classifier.classes_)
        row_count = probabilities.shape[0]

        for i in range(0, row_count):
            prob_instance = probabilities.iloc[i]
            corelex = self.instance_list[i].get_corelex()
            instance_id = self.instance_list[i].get_id_attributes()
            if corelex:
                predicted_label = prob_instance.reindex(corelex).idxmax(axis=1)
            else:
                predicted_label = prob_instance.idxmax(axis=1)

            predicted_labels.append(predicted_label)
            id_attributes.append(instance_id)

        return predicted_labels, id_attributes

    def filtered_probabilities(self, X, y_true, y_predicted):
        predicted_prob_labels = []
        probabilities = pd.DataFrame(self.classifier.predict_proba(X), columns=self.classifier.classes_)
        row_count = probabilities.shape[0]

        for i in range(0, row_count):
            prob_instance = probabilities.iloc[i]
            categories = list(set([y_true[i], y_predicted[i]]))
            predicted_prob_labels.append(prob_instance.reindex(categories))

        return predicted_prob_labels


    def validate(self, predicted_labels, true_labels):
        # y_true = self.featurized_data['y_dev']
        return accuracy_score(true_labels, predicted_labels)

    def most_informative_features(self):
        for i, class_label in enumerate(self.classifier.classes_):
            top_features = np.argsort(self.classifier.coef_[i])[-10:]
            print("%s:\n\t%s" % (class_label,
                              "\n\t".join(self.feature_names[j] for j in top_features)))

    def save_classifier(self):
        filename = '../data/classifier-%s.pickle' % self.model_name
        print("Saving %s" % filename)
        with open(filename, 'wb') as fh:
            pickle.dump(self.model, fh)

class BertWordEmbeddings:
    def __init__(self, corpus, model_name):
        self.word_embeddings = None
        self.filename = '../data/embeddings-%s.pickle' % model_name
        self.sen_dict = corpus.sen_dict


    def create_embeddings(self):
        print("Creating embeddings from BERT...")
        bert_embedding = BertEmbedding()
        word_embeddings = {}
        for i in tqdm(range(1, len(self.sen_dict)+1)):
            word_embeddings[str(i)] = bert_embedding(self.sen_dict[str(i)])
        self.word_embeddings = word_embeddings
        self.save_embeddings(word_embeddings)

    def save_embeddings(self, embeddings):
        print("Saving %s" % self.filename)
        with open(self.filename, 'wb') as fh:
            pickle.dump(embeddings, fh)

    def load_embeddings(self):
        print("Loading %s" % self.filename)
        with open(self.filename, 'rb') as fh:
            embeddings = pickle.load(fh)
            self.word_embeddings = embeddings

    def get_embeddings(self):
        return self.word_embeddings

class WindowedFeatureExtractor:
    def __init__(self, feature_extractors, window_size: int, cap: int):
        self.feature_extractors = feature_extractors
        self.window_size = window_size
        self.cap = cap

    def get_feature_extractors(self):
        return self.feature_extractors

    def get_window_size(self):
        return self.window_size

    def get_cap(self):
        return self.cap

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
                                feature_extractor.extract(cur_instance, window_idx, self.cap, feature_dict)
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
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and instance.pos:
            features.update({'pos[' + str(window_idx) + ']=' + instance.pos: 1.0})

class RelFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and window_idx == 0:
            rel = instance.linking_rel
            if rel:
                features.update({'rel=' + rel: 1.0})

class DependencyFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and window_idx == 0:
            dom = instance.dom_lemma
            rel = instance.linking_rel
            if dom:
                features.update({'dep=' + rel + '_'+ dom: 1.0})

class DependencySynsetFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and window_idx == 0:
            rel = instance.linking_rel
            dom = instance.dom_lemma
            if dom:
                synsets = wn.synsets(dom)
                synset_names = list(set([synset.name() for synset in synsets]))
                if cap > len(synset_names):
                    cap = len(synset_names)
                feature_values = ['depsynset=' + rel + '_'+ synset_names[i] for i in range(0, cap)]
                features.update(zip(feature_values, itertools.repeat(1.0)))

class DependencyHyperFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and window_idx == 0:
            dom = instance.dom_surface
            if dom:
                synsets = wn.synsets(dom)
                hypernyms = list(set([synset.name() for synset in
                             list(itertools.chain.from_iterable([synset.hypernyms() for synset in synsets]))]))
                if cap > len(hypernyms):
                    cap = len(hypernyms)
                feature_values = ['dephyperid=' + hypernyms[i] for i in range(0, cap)]
                features.update(zip(feature_values, itertools.repeat(1.0)))

class DomLemmaFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and window_idx == 0:
            dom = instance.dom_lemma
            if dom:
                features.update({'dom=' + dom: 1.0})

class DomSynsetFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        if not instance.is_punctuation() and window_idx == 0:
            dom = instance.dom_surface
            if dom:
                synsets = wn.synsets(dom)
                synset_names = [synset.name() for synset in synsets]
                feature_values = ['domssid='+synset_names[i] for i in range(0, len(synset_names))]
                features.update(zip(feature_values, itertools.repeat(1.0)))

class DomHyperFeatureExtractor:
    def extract(self, instance, window_idx, cap,  features):
        if not instance.is_punctuation() and window_idx == 0:
            dom = instance.dom_surface
            if dom:
                synsets = wn.synsets(dom)
                synset_names = [synset.name() for synset in synsets]
                hypernyms = list(set([synset.name() for synset in
                             list(itertools.chain.from_iterable([synset.hypernyms() for synset in synsets]))]))
                feature_values = ['domssid=' + hypernyms[i] for i in range(0, len(synset_names))]
                features.update(zip(feature_values, itertools.repeat(1.0)))

class ContextFeatureExtractor:
    #TODO: maybe allow this feature to behavior differently w/r/t windows
    def extract(self, instance, window_idx, cap,  features):
        if not instance.is_punctuation() and window_idx != 0:
            # features.update({'context[' + str(window_idx) + ']=' + instance.surface_form: 1.0})
            features.update({'context=' + instance.surface_form: 1.0})
        elif instance.is_punctuation() and window_idx != 0:
            # features.update({'context[' + str(window_idx) + ']=PUNC' : 1.0})
            features.update({'context=PUNC': 1.0})

class SynsetFeatureExtractor:
    def extract(self, instance, window_idx, cap, features):
        #if not instance.is_punctuation() and window_idx == 0:
        if not instance.is_punctuation():
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            synset_names = [synset.name() for synset in synsets]
            feature_values = ['ssid' + '[' + str(window_idx) + ']=' + synset_names[i] for i in range(0, len(synset_names))]
            features.update(zip(feature_values, itertools.repeat(1.0)))

class HypernymFeatureExtractor:
    def extract(self, instance, window_idx, cap,  features):
        if not instance.is_punctuation() and window_idx == 0:
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            hypernyms = [synset.name() for synset in
                         list(itertools.chain.from_iterable([synset.hypernyms() for synset in synsets]))]
            feature_values = ['hnid=' + hypernyms[i] for i in range(0, len(hypernyms))]
            features.update(zip(feature_values, itertools.repeat(1.0)))

class HypernymPathFeatureExtractor:
    def extract(self, instance, window_idx, cap,  features):
        if not instance.is_punctuation() and window_idx == 0:
            synsets = wn.synsets(instance.surface_form, pos=wn.NOUN)
            path_hypernyms = [list(itertools.chain.from_iterable(synset.hypernym_paths())) for synset in synsets]

            if len(path_hypernyms) > 1:
                path_hyp = [hypernym.name() for hypernym_list in path_hypernyms for hypernym in hypernym_list]
                feature_values = ['hpid=' + path_hyp[i] for i in range(0, len(path_hyp))]
                features.update(zip(feature_values, itertools.repeat(1.0)))

class CorelexFeatureExtractor:
    def extract(self, instance, window_idx, cap,  features):
        if not instance.is_punctuation():
            corelex = instance.get_corelex()
            feature_values = ["corelex" + '[' + str(window_idx) + ']=' + corelex[i] for i in range(0, len(corelex))]
            features.update(zip(feature_values, itertools.repeat(1.0)))

class WordVectorFeature():
    def __init__(self, bert, scaling: float = 1.0) -> None:
        self.scaling = scaling
        self.word_vectors = bert.get_embeddings()
        # self.indexes = ["v" + str(i) for i in range(0, self.word_vectors.dim)]

    def extract(self, instance, window_idx, cap, features) -> None:
        if not instance.is_punctuation() and window_idx == 0:
            print(self.word_vectors['1'])
            print(instance.sid)

            sen_vector = self.word_vectors.get(instance.sid)
            # print(type(sen_vector[0]))
            # print(instance)
            # vector = sen_vector[instance.sentence_position][1]
            # exit(0)
            # vector_feature = [np.multiply(vector[i], self.scaling) for i in range(0, vector.size)]
            # print(vector)
            # features.update(zip(self.indexes, vector_feature))

def main(model_name, window, train_split, classifier_type, prediction_type, cap, print_features, use_embeddings=False):
    if model_name == 'test':
        if not os.path.exists(TEST_OUTPUT):
            os.makedirs(TEST_OUTPUT)
    else:
        if not os.path.exists(ALL_OUTPUT):
            os.makedirs(ALL_OUTPUT)

    features_in = SC_TOKEN_FT
    sc = SemcorCorpus(file=features_in, model_name=model_name)
    if use_embeddings:
        bert = BertWordEmbeddings(sc, model_name)
        if Path(bert.filename).exists():
            bert.load_embeddings()
        else:
            bert.create_embeddings()

    model = BasicTyper(
        WindowedFeatureExtractor(
            [
                # POSFeatureExtractor(),
                # RelFeatureExtractor(),
                DependencyFeatureExtractor(),
                # DependencySynsetFeatureExtractor(),
                # DependencyHyperFeatureExtractor(),
                # ContextFeatureExtractor(),
                # SynsetFeatureExtractor(),
                # HypernymFeatureExtractor(),
                # HypernymPathFeatureExtractor(),
                # CorelexFeatureExtractor(),
                # DomLemmaFeatureExtractor(),
                # DomSynsetFeatureExtractor(),
                # WordVectorFeature(bert)

            ],
            window,
            cap),
        classifier_type,
        model_name)

    model.featurize_corpus(sc, print_features)

    model.train(train_split)
    # model.k_fold()

    # if prediction_type == "filtered":
    #     predictions = model.predict_probablities()
    # else:
    #     predictions = model.predict()

    # predictions = model.predict()
    # model.validate(predictions)
    # model.most_informative_features()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A classifier to predict a noun's Corelex Basic Type.")
    parser.add_argument('--mode',
                        default='train-test',
                        choices=['train-test'])
    parser.add_argument('--model',
                        default='test',
                        choices=['all', 'test'],
                        help="Choose the the model size you'd like to train and test against (default=all).")
    parser.add_argument('--architecture',
                        default="DecisionTree",
                        choices=['LogisticRegression', 'NaiveBayes', 'DecisionTree'],
                        help="Choose the model architecture you'd like to use (default=LogisticRegression).")
    parser.add_argument('--window-size',
                        default=2,
                        type=int,
                        help="Choose the size of the window on either side of a target token from which to generate "
                             "features (default=0).")
    parser.add_argument('--train-split',
                        default=0.8,
                        choices=np.arange(0.1, 0.9),
                        type=float,
                        metavar="[0.1-0.9]",
                        help="Choice the percentage of data you'd like in the training set (default=0.9).")
    parser.add_argument('--prediction-type',
                        default='unfiltered',
                        choices=['unfiltered', 'filtered'],
                        help="Use the possible synsets as a filter for a token to limit the possible labels only to those"
                             " that align with possible basic types (default=unfiltered).")
    parser.add_argument('--cap',
                        default=3,
                        type=int,
                        help="Cap for top synset and dependency synset and hypernym features")
    parser.add_argument('-print-features',
                        help='Prints features to a .jsonl file.',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    mode = args.mode

    main(model_name=args.model,
         window=args.window_size,
         train_split=args.train_split,
         classifier_type=args.architecture,
         prediction_type=args.prediction_type,
         cap=args.cap,
         print_features=args.print_features,
         use_embeddings=False)