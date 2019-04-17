# from bs4 import BeautifulSoup
from semcor import Semcor, SemcorFile
import nltk
from collections import defaultdict
import re

def corpus(sc):
    # text = {}
    text = []
    sen_list = [file.get_sentences() for file in sc.files]
    for list in sen_list:
        for item in list:
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

                        if form.pos != 'VB':
                            # print(form.text, '- btype:',  form.synset.btypes, '- pos', form.pos)
                            position = form.position

                            data = {'token' : form.text,
                                    'lemma' : form.lemma,
                                    'cat' : form.synset.cat,
                                    'w_2b': item.wfs[position-2],
                                    'w_1b': item.wfs[position-1],
                                    'w_1a': item.wfs[position+1],
                                    # 'gloss': form.synset.gloss
                                    }

                            if (len(item.wfs) > position+2):
                                data.update({'w_2a': item.wfs[position+2]})
                            else:
                                data.update({'w_2a' : 'EOS'})

                            text.append((data, form.synset.btypes))
    return text

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

if __name__ == '__main__':

    sc = Semcor()
    labeled_data = corpus(sc)
    # labeled_data = label_data(text)
    # features = feature_set(labeled_data)
    # print(labeled_data)
    training_set, test_set = create_train_test_data(labeled_data)
    classifier = train_classifier(training_set)
    evaluate_classifier(classifier, test_set)
    # print(classifier.labels())
    classifier.show_most_informative_features(10)



