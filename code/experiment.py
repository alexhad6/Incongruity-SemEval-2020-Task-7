# Run the experiment

import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import incongruity
import glove

def extract_original_phrase(headline):
    start = headline.index('<')
    end = headline.index('>')
    return headline[start+1:end-1]

def combined_data(dataframes, metric):
    metric_values = []
    meanGrades = []
    for dataframe in dataframes:
        for _, row in dataframe.iterrows():
            if not pd.isna(row[metric]):
                metric_values.append(row[metric])
                meanGrades.append(row.meanGrade)
    return metric_values, meanGrades

def plot_metric(metric_values, meanGrades, ylabel, plotname):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(meanGrades, metric_values, c='b', s=12, alpha=0.1)
    plt.xlabel('Mean Humor Grade')
    plt.ylabel(ylabel)
    plt.savefig(plotname)

def main(args):
    # Load train and test data
    train = pd.read_csv('../data/task-1/train.csv')    
    test = pd.read_csv('../data/task-1/test.csv')  
    funlines = pd.read_csv('../data/task-1/train_funlines.csv')
    train['original_phrase'] = train.original.apply(extract_original_phrase)
    test['original_phrase'] = test.original.apply(extract_original_phrase)
    funlines['original_phrase'] = funlines.original.apply(extract_original_phrase)

    # Load GloVe vectors
    glove_vectors, glove_words = glove.load_glove_vectors(args.npyFILE)

    # Calculate cosine similarity metrics
    train['original_vs_edit'] = train.apply(lambda row: incongruity.original_vs_edit(row.original_phrase, row.edit, glove_words, glove_vectors), axis=1)
    test['original_vs_edit'] = test.apply(lambda row: incongruity.original_vs_edit(row.original_phrase, row.edit, glove_words, glove_vectors), axis=1)
    funlines['original_vs_edit'] = funlines.apply(lambda row: incongruity.original_vs_edit(row.original_phrase, row.edit, glove_words, glove_vectors), axis=1)

    # Plot original_vs_edit vs meanGrade
    original_vs_edit, meanGrades = combined_data([train, test, funlines], 'original_vs_edit')
    plot_metric(original_vs_edit, meanGrades, 'Original vs. Edit Cosine Similarity', args.plot)
    print(f'Pearson correlation: {pearsonr(original_vs_edit, meanGrades)}')
    print(f'Spearman correlation: {spearmanr(original_vs_edit, meanGrades)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE', type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    # parser.add_argument('train', type=str,
    #     help='a CSV file of the train data')
    # parser.add_argument('test', type=str,
    #     help='a CSV file of the test data')
    parser.add_argument('plot', type=str,
        help='name of file to write plot to')
    main(parser.parse_args())

# python experiment.py ../glove/glove.6B.300d.npy plot-all.png


# original_vs_edit w/ 300D GloVe
# Pearson correlation: (-0.17739550138691068, 1.0680352459856034e-143)
# Spearman correlation: SpearmanrResult(correlation=-0.16169603479087277, pvalue=2.093149320427472e-119)
