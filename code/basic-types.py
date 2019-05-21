# from bs4 import BeautifulSoup
from semcor import Semcor, SemcorFile
import nltk
import csv

SC_SENT = '../data/semcor.sent.tsv'
SC_TOKEN_FT = '../data/semcor.token.tsv'
SC_TOKEN_DEP = '../data/semcor.token.fv'

def data_prep(file):
    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        data = [row for row in reader]
        return data

def extract_types(sc):
    types = []
    map = {}
    i = 0
    sen_list = [file.get_sentences() for file in sc.files]
    for list in sen_list:
        for item in list:
            wf = item.wfs
            for form in wf:
                i += 1
                if form.is_word_form():
                    if form.lemma is not None:
                        map.update({str(i): form.lemma})
                    else:
                        map.update({str(i): form.text})
                    if form.synset is not None:
                            types.append(form.synset.btypes)
                    else:
                        types.append(None)
                else:
                    map.update({str(i): form.text})
                    types.append(None)

    # print(map)
    return types, map


def feature_set(types, token_features, map):
    mapped = zip(token_features, types)
    feature_set = []
    for i in mapped:
        if i[1] is not None:
            features = i[0]
            # indexs: [1] token_id, [2] sent_id, [3] token_no, [4] surface,[5] lemma, [6] pos, [7] sense_no, [8] sense_key, [9] ssid,
            # [10] int_dom_token_no, [11] dom_token_id, [12] rel
            if features[5] != 'VB':
                feature_dict = {
                    # "surface" : features[3],
                    # "lemma" : features[4],
                    "pos" : features[5],
                    # "sense_no" : features[6],
                    "sense_key" : features[7],
                    # "ssid" : features[8],
                    "rel" : features[11],
                }
                # if features[9] != '0':
                #     feature_dict.update({
                #         "int_dom_token": map[features[9]],
                #         "dom_token": map[features[10]]
                #     })
                # else:
                #     feature_dict.update({
                #         "int_dom_token": None,
                #         "dom_token": None
                #     })
                # print((feature_dict, i[1]))
                feature_set.append((feature_dict, i[1]))

    return feature_set


def create_train_test_data(feature_set):
    index = int(len(feature_set) * .8)
    training_set, test_set = feature_set[:index], feature_set[index:]

    return training_set, test_set

def train_classifier(training_set):

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    return classifier

def evaluate_classifier(classifier, test_set):
    """
    :param classifier: classifier that has been trained on training set.
    :param test_set: 20% of the featureset which includes features and a label.
    :return: percentage accuracy of the classifier being able to label the data correctly based on features.
    """
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print(accuracy)

def type_count(types):
    count = 0
    for type in types:
        if type is not None:
            count += 1
    return count, len(types)

if __name__ == '__main__':

    sc = Semcor()
    token_features = data_prep(SC_TOKEN_FT)
    types, map = extract_types(sc)
    print(type_count(types))
    feature_data = feature_set(types, token_features, map)
    training_set, test_set = create_train_test_data(feature_data)
    classifier = train_classifier(training_set)
    evaluate_classifier(classifier, test_set)
    print(classifier.labels())
    # classifier.show_most_informative_features(10)



