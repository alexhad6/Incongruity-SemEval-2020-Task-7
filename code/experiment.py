# Run the experiment

import argparse
import pandas as pd
import numpy as np
import incongruity
import glove

def extract_original_phrase(headline):
    start = headline.index('<')
    end = headline.index('>')
    return headline[start+1:end-1]

def main(args):
    # Load train and test data
    train = pd.read_csv(args.train)    
    test = pd.read_csv(args.test)
    train['original_phrase'] = train.original.apply(extract_original_phrase)
    test['original_phrase'] = test.original.apply(extract_original_phrase)

    # Load GloVe vectors
    glove_vectors, glove_words = glove.load_glove_vectors(args.npyFILE)

    # Calculate cosine similarity metrics
    train['original_vs_edit'] = train.apply(lambda row: incongruity.original_vs_edit(row.original_phrase, row.edit, glove_words, glove_vectors), axis=1)
    
    # Some GloVe vector tests
    # print(f'original: {train.original_phrase[25]}, edit: {train.edit[25]}, {train.metric1[25]}')
    # print(f'original: {train.original_phrase[22]}, edit: {train.edit[22]}, {train.metric1[22]}')
    # print(f'original: {train.original_phrase[9647]}, edit: {train.edit[9647]}, {train.metric1[9647]}')
    # print(train[train.metric1 == train.metric1.min()])
    # print(train[train.metric1 == train.metric1.max()])
    # print(incongruity.metric1(train.original_phrase[1], train.edit[1], glove_words, glove_vectors))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE',
        type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    parser.add_argument('train',
        type=argparse.FileType('r'),
        help='a CSV file of the train data')
    parser.add_argument('test',
        type=argparse.FileType('r'),
        help='a CSV file of the test data')
    main(parser.parse_args())


# python experiment.py ../glove/glove.6B.50d.npy ../data/humicroedit/task-1/train.csv ../data/humicroedit/task-1/dev.csv

# python experiment.py ../data/funlines/task-1/train_funlines.csv ../data/humicroedit/task-1/dev.csv
