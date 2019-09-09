# Basic Types Classification
## Approach
CoreLex and CoreLex 2.0 investigate systematic polysemy in nouns, establishing an ontology and semantic database of 126 semantic types, defining a large number of systematic polysemous classes through an analysis of WordNet sense distributions. 
In order to better understand these sense distributions and the correlations between them, the ability to associate a lexical item with its semantic type automatedly would be of use. By using the SemCor corpus which comprised of texts semantically annotated with their WordNet sense. We are working to build a classifier that can correctly predict a lexical item’s semantic type in novel texts. 


## Running the Code
The module for running the classifier is basic-types.py. Requirements for running this are Stanford's CoreNLP and the packages specified in requirements.txt.

Additional notes on usage:

$ python3 basic-types.py --train-semcor

    Train a classifier from all of Semcor and save it in
    ../data/classifier-all.pickle.

$ python3 basic-types.py --train-test

    Train a classifier from a fragemtn of Semcor (two files) and save it in
    ../data/classifier-002.pickle, for testing and debugging purposes.

$ python3 basic-types.py --test

    Test the classifier on a feature set and test the evaluation code.

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