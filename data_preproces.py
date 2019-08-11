import os
import collections
import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from config_file import Config
import re
import random
import numpy as np
import sys
import os


def text_preprocessing(datasets):
    dataset_text, dataset_label = [], []
    for file in datasets:
        lines, labels = [], []
        with open(file) as f:
            for l in f:
                try:
                    words = re.split('\s|-', l.lower().split("|||")[0].strip())
                    label = int(l.lower().split("|||")[1].strip())
                    lines += [words]
                    labels += [label]
                except:
                    continue
        dataset_text += [lines]
        dataset_label += [labels]
    return dataset_text, dataset_label  # , dataset_c


# insert words of a file
def insert_word(dataset, all_words):
    for data in dataset:            # 16 datasets
        for lines in data:          # every lines of per dataset
            for l in lines:         # every word of per lines
                all_words += l      # take all words into the list


# convert words to numbers		# words_to_number
def convert_words_to_number(dataset_text, dataset_label, common_word):
    transformed_text, transformed_label = [], []
    for lines, labels in zip(dataset_text, dataset_label):
        new_x, new_label = [], []
        for l, label in zip(lines, labels):
            words = [common_word[w] if w in common_word else 1 for w in l]
            new_x += [words]
            new_label += [label]

        transformed_text += [new_x]
        transformed_label += [new_label]
    return transformed_text, transformed_label


def wordembedding(glove_filename, common_word, word_dim):
    glove_dict = {}
    with open(glove_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            glove_dict[word] = embedding
    word2vec = [np.random.normal(0, 0.1, word_dim).tolist(), np.random.normal(0, 0.1, word_dim).tolist()]  # 2
    print(np.array(word2vec).shape)
    # tolist()
    missing = 0
    for number, word in sorted(zip(common_word.values(), common_word.keys())):
        try:
            word2vec.append(glove_dict[word])
        except KeyError:
            word2vec.append(np.random.normal(0, 0.1, word_dim).tolist())
            missing += 1
    pickle.dump(word2vec, open(parsed_path + 'dataset_vectors', 'wb'))
    print('missing' + str(missing))
    print(np.array(word2vec).shape)


if __name__ == "__main__":
    # configs
    config = Config()
    filename_list = config.filename_list
    word_dim = config.word_dim
    data_path = config.dataset_path
    parsed_path = config.parsed_path
    if not os.path.exists(parsed_path):
        os.mkdir(parsed_path)

    text_data, labels_data = [], []
    for file in filename_list:
        print(file)
        datasets = [data_path+'new/' + file + '_trn', data_path+'new/' + file + '_dev',
                    data_path+'new/' + file + '_tst']
        # preprocess the texts
        dataset_text, dataset_label = text_preprocessing(datasets)  # , dataset_class
        text_data.append(dataset_text)
        labels_data.append(dataset_label)

    # insert all words
    all_words = []
    insert_word(text_data, all_words)
    # obtain frequent words
    counter = collections.Counter(all_words)

    vocab = len(counter)
    # print(vocab)
    vocab_size = vocab - 2
    common_word = dict(counter.most_common(vocab_size))
    c = 2
    for key in common_word:
        common_word[key] = c
        c += 1

    for i in range(len(filename_list)):  # type: int
        file = filename_list[i]
        dataset_text = text_data[i]
        dataset_label = labels_data[i]

        # write out filtering training test data
        transformed_text, transformed_label = convert_words_to_number(dataset_text, dataset_label, common_word)
        pickle.dump((transformed_text, transformed_label), open(parsed_path + '/' + file + '_dataset', 'wb'))
        print(file + ' has done !')

    # word embeddings
    glove_filename = data_path+"glove.6B." + str(word_dim) + "d.txt"
    wordembedding(glove_filename, common_word, word_dim)
    print("done!")
