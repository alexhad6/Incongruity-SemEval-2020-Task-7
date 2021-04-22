#!/usr/bin/env python3

import glove
import similarity
import visualize
import numpy
import random
import argparse

def average_difference(first_vectors, second_vectors):
    difference_vectors = first_vectors - second_vectors
    return sum(difference_vectors) / len(difference_vectors)

def do_experiment(args):
    # Load vectors and relations
    array, words = glove.load_glove_vectors(args.npyFILE)
    relations_orig = visualize.read_relations(args.relationsFILE)

    # Extract train and test data
    random.shuffle(relations_orig)
    vectors1, vectors2, relations = visualize.extract_words(array, words, relations_orig)
    train_size = int(round(0.8 * len(relations)))
    train_relations = relations[:train_size]
    train_vectors1 = vectors1[:train_size]
    train_vectors2 = vectors2[:train_size]
    test_relations = relations[train_size:]
    test_vectors1 = vectors1[train_size:]
    test_vectors2 = vectors2[train_size:]
    train_avg_diff = average_difference(train_vectors1, train_vectors2)

    num_first_most_similar = 0
    num_first_top10 = 0
    reciprocal_rank_sum = 0
    for i in range(train_size):
        word1 = train_relations[i][0]
        similar_pairs = similarity.closest_vectors(train_vectors2[i], words, array, 101)[1:]
        similar_words = [pair[1] for pair in similar_pairs]
        if word1 == similar_words[0]:
            num_first_most_similar += 1
        if word1 in similar_words[:10]:
            num_first_top10 += 1
        if word1 in similar_words:
            reciprocal_rank_sum += 1 / (similar_words.index(word1) + 1)
    freq_first_most_similar = num_first_most_similar / train_size
    freq_first_top10 = num_first_top10 / train_size
    MRR = reciprocal_rank_sum / train_size
    print('TRAINING SET')
    print(f'1st word is most similar to 2nd word: {freq_first_most_similar}')
    print(f'1st word in top 10 most similar to 2nd word: {freq_first_top10}')
    print(f'Mean reciprocal rank: {MRR}')
    print()

    print('PREDICTIONS')
    test_size = len(test_relations)
    num_first_most_similar = 0
    num_first_top10 = 0
    reciprocal_rank_sum = 0
    for i in range(test_size):
        word1 = test_relations[i][0]
        prediction_vector = test_vectors2[i] + train_avg_diff
        similar_pairs = similarity.closest_vectors(prediction_vector, words, array, 100)
        similar_words = [pair[1] for pair in similar_pairs]
        if word1 == similar_words[0]:
            num_first_most_similar += 1
        if word1 in similar_words[:10]:
            num_first_top10 += 1
        if word1 in similar_words:
            reciprocal_rank_sum += 1 / (similar_words.index(word1) + 1)
    freq_first_most_similar = num_first_most_similar / test_size
    freq_first_top10 = num_first_top10 / test_size
    MRR = reciprocal_rank_sum / test_size
    print(f'1st word is most similar to prediction: {freq_first_most_similar}')
    print(f'1st word in top 10 most similar to prediction: {freq_first_top10}')
    print(f'Mean reciprocal rank for predictions: {MRR}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE',
        type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    parser.add_argument('relationsFILE',
        type=argparse.FileType('r'),
        help='a file containing pairs of relations')
    do_experiment(parser.parse_args())
