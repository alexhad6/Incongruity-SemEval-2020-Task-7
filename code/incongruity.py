# Metrics for analyzing how incongruous an edit is to a sentence

import numpy as np
import argparse
import glove

def cosine_similarity(vector1, vector2):
    len1 = np.linalg.norm(vector1)
    len2 = np.linalg.norm(vector2)
    return np.dot(vector1, vector2) / (len1 * len2)

def original_vs_edit(original_phrase, edit, glove_words, glove_vectors):
    # calculate similarity (metric 1- original word vs edit word)
    total_similarity = 0
    edit_vector = glove.get_vec(edit.lower(), glove_words, glove_vectors)
    original_words = original_phrase.lower().split()
    for original_word in original_words:
        original_vector = glove.get_vec(original_word, glove_words, glove_vectors)
        if edit_vector is None or original_vector is None:
            return None
        total_similarity += cosine_similarity(original_vector, edit_vector)
    return total_similarity/len(original_words)  # Return the average similarity

def edit_vs_neighbors(headline, edit, glove_words, glove_vectors, window_size=None):
    # find neighbors
    if window_size is None:
        window_size = len(headline)
    start = headline.index('<')
    end = headline.index('>')
    words_before = headline[:start].lower().split()[-window_size:]
    words_after = headline[end+1:].lower().split()[:window_size]

    # calculate similarity (metric 2- edit vs neighbor)
    total_similarity = 0
    num_words = 0
    edit_vector = glove.get_vec(edit.lower(), glove_words, glove_vectors)
    if edit_vector is None:
        return None
    for word in words_before + words_after:
        vector = glove.get_vec(word, glove_words, glove_vectors)
        if vector is not None:
            total_similarity += cosine_similarity(vector, edit_vector)
            num_words += 1
    if num_words == 0:
        return None
    return total_similarity/num_words
