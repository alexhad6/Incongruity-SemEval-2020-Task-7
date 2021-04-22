#!/usr/bin/env python3

import glove
import numpy
import argparse

def compute_length(a):
    return numpy.linalg.norm(a, axis=a.ndim-1)

def cosine_similarity(array1, array2):
    len1 = compute_length(array1)
    len2 = compute_length(array2)
    return numpy.dot(array2, array1) / (len1 * len2)

def closest_vectors(vector1, word_list, array, n):
    sims = cosine_similarity(vector1, array)
    word_sim_pairs = list(zip(sims, word_list))
    word_sim_pairs.sort(key=lambda pair: pair[0], reverse=True)
    return word_sim_pairs[:n]

def main(args):
    array, words = glove.load_glove_vectors(args.npyFILE)
    n = args.num if args.num else 3
    if args.word:
        vector = glove.get_vec(args.word, words, array)
        print(closest_vectors(vector, words, array, n))
    if args.file:
        for line in args.file:
            vector = glove.get_vec(line.strip(), words, array)
            print(closest_vectors(vector, words, array, n))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE',
        type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    parser.add_argument('--word', '-w',
        type=str,
        help='a single word')
    parser.add_argument('--file', '-f',
        type=argparse.FileType('r'),
        help='a text file with one word per line')
    parser.add_argument('--num', '-n',
        type=int,
        help='find the top n most similar words')
    main(parser.parse_args())
