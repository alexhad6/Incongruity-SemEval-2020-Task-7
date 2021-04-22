#!/usr/bin/env python3

import glove
import numpy
import argparse
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def read_relations(fp):
    return [tuple(line.split()) for line in fp][1:]

def perform_pca(array, n_components):
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(array)
    return pc

def extract_words(vectors, word_list, relations):
    relations_existing = []
    for relation in relations:
        if all(word in word_list for word in relation):
            relations_existing.append(relation)
    num_relations = len(relations_existing)
    vec_len = vectors.shape[1]
    vectors1 = numpy.zeros((num_relations, vec_len))
    vectors2 = numpy.zeros((num_relations, vec_len))
    for i in range(num_relations):
        word1, word2 = relations_existing[i]
        vectors1[i] = glove.get_vec(word1, word_list, vectors)
        vectors2[i] = glove.get_vec(word2, word_list, vectors)
    return vectors1, vectors2, relations_existing

def plot_relations(pca_first, pca_second, pca_relations, filename='plot.png'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pca_first[:,0], pca_first[:,1], c='r', s=50)
    ax.scatter(pca_second[:,0], pca_second[:,1], c='b', s=50)
    for i in range(len(pca_first)):
        (x1, y1) = pca_first[i]
        (x2, y2) = pca_second[i]
        plt.annotate(pca_relations[i][0], xy=(x1, y1), color='black')
        plt.annotate(pca_relations[i][1], xy=(x2, y2), color='black')
        ax.plot((x1, x2), (y1,y2), linewidth=1, color='lightgray')
    plt.savefig(filename)

def main(args):
    array, words = glove.load_glove_vectors(args.npyFILE)
    relations_orig = read_relations(args.relationsFILE)
    vectors1, vectors2, relations = extract_words(array, words, relations_orig)
    vectors12 = numpy.vstack((vectors1, vectors2))
    pca_array = perform_pca(vectors12, 2)
    pca1, pca2 = numpy.vsplit(pca_array, 2)
    plot_filename = args.plot if args.plot else 'plot.png'
    plot_relations(pca1, pca2, relations, filename=plot_filename)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE',
        type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    parser.add_argument('relationsFILE',
        type=argparse.FileType('r'),
        help='a file containing pairs of relations')
    parser.add_argument('--plot', '-p',
        type=str,
        help='name of file to write plot to')
    main(parser.parse_args())
