import codecs, glob
from collections import Counter


def read_lemmas():
    lemmas = {}
    fnames = glob.glob('out/*.json')
    for fname in fnames[:25]:
        print(fname)
        with codecs.open(fname, encoding='utf8') as fh:
            for line in fh:
                lemma, types = line.strip().split('\t')
                if len(lemma) < 4:
                    continue
                lemma = lemma.lower()
                types = types.split()
                lemmas.setdefault(lemma, Counter())
                lemmas[lemma].update(Counter(types))
    return lemmas


def filter_lemmas(lemmas):
    filtered_lemmas = {}
    for l, c in lemmas.items():
        d = { btype: count for btype, count in c.items() if count > 3 }
        if len(d) > 1:
            c = Counter(d)
            filtered_lemmas[l] = c
            print l, c.most_common(8)
    return filtered_lemmas


def write_lemmas(lemmas):
    bt1 = codecs.open('btypes-lemma-type.txt', 'w', encoding='utf8')
    bt2 = codecs.open('btypes-type-lemma.txt', 'w', encoding='utf8')
    bt1_dict = {}
    bt2_dict = {}
    for l in sorted(filtered_lemmas):
        types = '-'.join([t[0] for t in filtered_lemmas[l].most_common(2)])
        bt1.write("%s\t%s\n" % (l, types))
        bt2_dict.setdefault(types, []).append(l)
    for t in sorted(bt2_dict):
        bt2.write("%s    %s\n" % (t, ' '.join(bt2_dict[t])))



if __name__ == '__main__':

    lemmas = read_lemmas()
    filtered_lemmas = filter_lemmas(lemmas)
    print(len(lemmas))
    print(len(filtered_lemmas))
    write_lemmas(filtered_lemmas)
