# from bs4 import BeautifulSoup
from semcor import Semcor, SemcorFile
import nltk
from collections import defaultdict

def corpus(sc):
    text = {}
    sen_list = [file.get_sentences() for file in sc.files]
    for list in sen_list:
        for item in list:
            # print(item)
            #todo tuple with sentence id and wf
            wf = item.wfs
            for form in wf:
                if form.is_word_form():
                    if form.synset is not None:
                        #previous tag, previous word, path to top,
                        #run it on a minimal feature set for a baseline –compare how well,
                        # try word itself as a single feature and check baseline
                        #store as tsv with "label_name=LABEL" -> makes this not nltk dependant
                        #melat toolkit could be a good alternative.
                        # if form.synset.btypes != null:
                        print(form)
                        print(form.text, '- btype:',  form.synset.btypes)
                        print(form.synset.btypes)
                        data = {'token' : form.text,
                                # 'btype' : form.synset.btypes,
                                 'cat' : form.synset.cat,
                                 # 'gloss': form.synset.gloss
                        }

                        text.update({form.text : data})
                    # else:
                    #     text.extend([{form.text:[]}])
    return text

def label_data(data_dict):
    labels = [({k:v}, data_dict[k]['btype']) for k,v in data_dict.items()]

    return labels

def feature_set(labels):

    features = []
    for i in labels:
        for k,v in i[0].items():
            features.extend([({'token': v['token'], 'cat': v['cat']}, i[1])])

    # print(features)
    return features
    # print(features)

def create_train_test_data(feature_set):
    index = int(len(feature_set) * .8)
    training_set, test_set = feature_set[:index], feature_set[index:]

    return training_set, test_set

def train_classifier(training_set):

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # classifier = nltk.MaxentClassifier.train(training_set)

    return classifier

def evaluate_classifier(classifier, test_set):
    """
    :param classifier: classifier that has been trained on training set.
    :param test_set: 20% of the featureset which includes features and a label.
    :return: percentage accuracy of the classifier being able to label the data correctly based on features.
    """
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print(accuracy)

if __name__ == '__main__':

    sc = Semcor()
    text = corpus(sc)
    # labeled_data = label_data(text)
    # features = feature_set(labeled_data)
    # training_set, test_set = create_train_test_data(features)
    # classifier = train_classifier(training_set)
    # evaluate_classifier(classifier, test_set)
    # print(classifier.labels())
    # classifier.show_most_informative_features(10)



