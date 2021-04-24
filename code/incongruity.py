# Metrics for analyzing how incongruous an edit is to a sentence

import numpy as np
import argparse
import glove

def cosine_similarity(vector1, vector2):
    len1 = np.linalg.norm(vector1)
    len2 = np.linalg.norm(vector2)
    return np.dot(vector1, vector2) / (len1 * len2)

def original_vs_edit(original_phrase, edit, glove_words, glove_vectors):
    total_similarity = 0
    edit_vector = glove.get_vec(edit.lower(), glove_words, glove_vectors)
    original_words = original_phrase.lower().split()
    for original_word in original_words:
        original_vector = glove.get_vec(original_word, glove_words, glove_vectors)
        if edit_vector is None or original_vector is None:
            return None
        total_similarity += cosine_similarity(original_vector, edit_vector)
    return total_similarity/len(original_words)  # Return the average similarity
