# Exract, save, and retrieve GloVe vectors

import argparse
import numpy as np

def load_text_vectors(fp):
    vec_len = len(fp.readline().split(' ')) - 1
    file_len = len(fp.readlines()) + 1
    array = np.zeros((file_len, vec_len))
    words = [None] * file_len
    fp.seek(0)
    for i, line in enumerate(fp):
        pieces = line[:-1].split(' ')
        array[i] = [float(n) for n in pieces[1:]]
        words[i] = pieces[0]
    return array, words

def save_glove_vectors(word_list, vectors, fp):
    np.save(fp, vectors)
    np.save(fp, word_list)

def load_glove_vectors(fp):
    array = np.load(fp)
    words = list(np.load(fp))
    return array, words

def get_vec(word, word_list, vectors):
    try:
        return vectors[word_list.index(word)]
    except ValueError:
        # print(f'No GloVe vector for {word}')
        return None

def main(args):
    array, words = load_text_vectors(args.GloVeFILE)
    save_glove_vectors(words, array, args.npyFILE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('GloVeFILE',
        type=argparse.FileType('r'),
        help='a GloVe text file to read from')
    parser.add_argument('npyFILE',
        type=argparse.FileType('wb'),
        help='an .npy file to write the saved numpy data to')
    main(parser.parse_args())
