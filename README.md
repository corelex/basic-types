# Basic Types Classification 

## Approach
CoreLex and CoreLex 2.0 investigate systematic polysemy in nouns, establishing an ontology and semantic database of 126 semantic types, defining a large number of systematic polysemous classes through an analysis of WordNet sense distributions. 

In order to better understand these sense distributions and the correlations between them, the ability to associate a lexical item with its semantic type automatedly would be of use. By using the SemCor corpus which comprised of texts semantically annotated with their WordNet sense. We are working to build a classifier that can correctly predict a lexical item’s semantic type in novel texts.

The current performance with the default configuration is .81445 accuracy.
## Running the Code

The module for running the classifier is basic_types.py. The packages required for running the classifier are documented
in requirements.txt. Information on usage is accessible via the --help command.

```
usage: basic_types.py [-h] [--mode {train-test}] [--model {all,test}]
                      [--architecture {LogisticRegression,NaiveBayes}]
                      [--window-size WINDOW_SIZE] [--train-split [0.1-0.9]]
                      [--prediction-type {unfiltered,filtered}]

A classifier to predict a noun's Corelex Basic Type.

optional arguments:
  -h, --help            show this help message and exit
  --mode {train-test}
  --model {all,test}    Choose the the model size you'd like to train and test
                        against (default=all).
  --architecture {LogisticRegression,NaiveBayes}
                        Choose the model architecture you'd like to use
                        (default=LogisticRegression).
  --window-size WINDOW_SIZE
                        Choose the size of the window on either side of a
                        target token from which to generate features
                        (default=0).
  --train-split [0.1-0.9]
                        Choice the percentage of data you'd like in the
                        training set (default=0.9).
  --prediction-type {unfiltered,filtered}
                        Use the possible synsets as a filter for a token to
                        limit the possible labels only to those that align
                        with possible basic types (default=unfiltered).
 ```

This classifier is built on top of the Semcor browser and its associated objects and compiled files. 
There should be no need to recompile the Semcor browser, but if you run into issue with Semcor indexing, recompiling is 
the usual solution. Details on the Semcor browser can be found below.  


### Semcor Browser

#### Prerequisites

Python 2.7 or Python 3 is required, Python 3 is strongly recommended because it runs the code much faster and we test the code less frequent on Python 2.7. The only non-standard module used is Beautiful Soup, so you need to install bs4:

```
$ pip install bs4
```

All data needed, including Semcor sources, are included in this repository

This code was not tested on Windows. One expected problem is that the browser and analysis scripts use ANSI escape sequences (at least, I found this to be an issue about 5 years ago).


#### Browser

To run the browser you first need to compile the semcor files and then you can start the browser:

```bash
$ python semcor.py --compile [-n MAXFILES]
$ python browse.py [-n MAXFILES]
```

The compile step only needs to be run once, but you may need to redo it every time you upgrade to a new version of the code. The optional `-n` flag allows you to compile or load only MAXFILES files, the default is to load/compile all files. After the above you will get the browser prompt, you can type `h` to get a listing of commands:

```
*> h

h          -  help
s LEMMA    -  show statistics for LEMMA
n LEMMA    -  search for noun LEMMA
v LEMMA    -  search for verb LEMMA
a LEMMA    -  search for adjective LEMMA
r LEMMA    -  search for adverb LEMMA
p SID      -  print paragraph with sentence SID
bt         -  show list of basic types that occur in potentially interesting pairs
bt NAME    -  show potentially interesting pairs for the basic type
btp        -  show list of potentially interesting basic type pairs
btp T1-T2  -  show examples for basic type pair

*>
```


### Interface

The code in `semcor.py` is a general interface to Semcor. The only other Python interfaces that I am aware of are the [NLTK SemcorCorpusReader class](https://www.nltk.org/_modules/nltk/corpus/reader/semcor.html) and [pysemcor](https://github.com/letuananh/pysemcor), which were not sufficient for our purposes.

The code integrates Semcor with WordNet synset information and Corelex basic types, but it does so by importing a data file with the information needed, this datafile is created by code in another repository (https://github.com/marcverhagen/corelex), but for convenience it is included in this repository.


**Loading Semcor**. Everything starts with loading Semcor, as noted above, the first time you do this you also need to compile Semcor:

```Python
>>> from semcor import compile_semcor, Semcor, SemcorFile
>>> compile_semcor()
>>> sc = Semcor()
```

Note that the second line above is code that is executed when you run semcor.py from the command line with the --compile flag and that therefore you do not need to do this if you have used the browser before. When compiling Semcor all source files are parsed and stored a pickle files, speeding  up loading significantly. The second time you load Semcor you do not have to include `compile_semcor`, however, when you upgrade to a new version you should recompile. Both functions above can take an optional argument that would limit the number of files being compiled or loaded.

**Creating a application-specific sentence index**. This allows you to use sentence offsets from the entire corpus and link to Semcor Sentence objects, which is useful when running all Semcor sentences separately through another processing component like the Stanford dependency parser.

```Python
>>> from semcor import Semcor, SemcorFile
>>> sc = Semcor()
>>> sc.create_sentence_index("files.txt")
```

This creates an index in the `sent_idx` attribute of the Semcor instance, see the Semcor docstring for more information. The file `files.txt` is provided by the user and it contains an ordered list with filenames separated by some whitespace, for example, either of the following will work:

```
br-a01 br-a02 br-a11 br-a12
```

```
br-a01
br-a02
br-a11
br-a12
```

Files in the list that are not available in Semcor will be ignored.
