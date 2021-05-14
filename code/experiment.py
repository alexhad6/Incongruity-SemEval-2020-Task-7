# Run the experiment

import math
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import incongruity
import glove

def extract_original_phrase(headline):
    start = headline.index('<')
    end = headline.index('>')
    return headline[start+1:end-1]

def plot_metric(metric_values, meanGrades, xlabel, ylabel, plotname):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(metric_values, meanGrades, c='b', s=17, alpha=0.5, edgecolors='none')
    p = np.poly1d(np.polyfit(metric_values, meanGrades, 1))
    plt.plot(metric_values, p(metric_values), 'r', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(plotname)

def statistics(metric_values, mean_grades):
    p = np.poly1d(np.polyfit(metric_values, mean_grades, 1))
    n = len(mean_grades)
    slope, intercept, r_value, p_value, std_err = linregress(metric_values, mean_grades)
    print(f'Linear regression equation: {p}')
    print(f'Pearson correlation: {pearsonr(metric_values, mean_grades)}')
    print(f'Spearman correlation: {spearmanr(metric_values, mean_grades)}')
    print(f'n: {n}')
    print(f'slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}')

def run_metric(train, test, metric, metric_name, axis_name):
    # Apply metric to training data
    train[metric_name] = train.apply(metric, axis=1)
    train_notnull = train[train[metric_name].notnull()]
    metric_values = train_notnull[metric_name].values.tolist()

    # Plot metric vs. mean humor grades
    print(f'{metric_name} vs. mean grades')
    mean_grades = train_notnull['meanGrade'].values.tolist()
    plot_metric(metric_values, mean_grades, 'Metric 1 Predictions', 'Mean Humor Grade', f'images/{metric_name}-mean_grade.png')
    statistics(metric_values, mean_grades)
    print()

    # Calculate metric on the test data
    model = np.poly1d(np.polyfit(metric_values, mean_grades, 1))
    test[metric_name] = test.apply(metric, axis=1)
    test_notnull = test[test[metric_name].notnull()]
    test_metric_values = test_notnull[metric_name].values.tolist()
    test_mean_grades = test_notnull['meanGrade'].values.tolist()
    duluth_orginal_vs_edit_pred = test_notnull['duluth_orginal_vs_edit'].values.tolist()
    duluth_edit_only_pred = test_notnull['duluth_edit_only'].values.tolist()


    # # Plot model predictions vs. test humor grades
    # plot_metric(test_metric_values, test_mean_grades, axis_name, 'Mean Humor Grade', f'images/{metric_name}-mean_grade_test.png')

    # Use model to predict humor grades in test data
    print(f'{metric_name} predictions')
    average_grade = sum(mean_grades)/len(mean_grades)
    n = len(test_mean_grades)
    RMSE = math.sqrt(sum((model(test_metric_values[i]) - test_mean_grades[i])**2 for i in range(n))/n)
    RMSE_avg = math.sqrt(sum((average_grade - test_mean_grades[i])**2 for i in range(n))/n)
    RMSE_Duluth_original_vs_edit = math.sqrt(sum((duluth_orginal_vs_edit_pred[i] - test_mean_grades[i])**2 for i in range(n))/n)
    RMSE_Duluth_edit_only = math.sqrt(sum((duluth_edit_only_pred[i] - test_mean_grades[i])**2 for i in range(n))/n)
    print(f'RMSE: {RMSE}')
    print(f'RMSE average: {RMSE_avg}')
    print(f'RMSE Duluth original vs. edit: {RMSE_Duluth_original_vs_edit}')
    print(f'RMSE Duluth edit only: {RMSE_Duluth_edit_only}')
    print()

    # Plot metric vs. Duluth original_vs_edit predictions
    print(f'{metric_name} vs. Duluth original-vs-edit')
    plot_metric(test_metric_values, duluth_orginal_vs_edit_pred, axis_name, 'Duluth Original Predictions', f'images/{metric_name}-Duluth_original_vs_edit.png')
    statistics(test_metric_values, duluth_orginal_vs_edit_pred)
    print()

    # Plot metric vs. Duluth original_vs_edit predictions
    print(f'{metric_name} vs. Duluth edit-only')
    plot_metric(test_metric_values, duluth_edit_only_pred, axis_name, 'Duluth Edit Predictions', f'images/{metric_name}-Duluth_edit_only.png')
    statistics(test_metric_values, duluth_edit_only_pred)
    print()

def main(args):
    # Load train and test data
    humicroedit_train = pd.read_csv('../data/task-1/train.csv')
    funlines_train = pd.read_csv('../data/task-1/train_funlines.csv')
    train = pd.concat([humicroedit_train, funlines_train])
    test = pd.read_csv('../data/task-1/test.csv')
    train['original_phrase'] = train.original.apply(extract_original_phrase)
    test['original_phrase'] = test.original.apply(extract_original_phrase)

    # Load in Duluth predictions to test DataFrame
    duluth_original_vs_edit = pd.read_csv('../data/duluth-task-1/original-vs-edit.csv')
    duluth_edit_only = pd.read_csv('../data/duluth-task-1/edit-only.csv')
    test = test.sort_values(by=['id']).reset_index(drop=True)
    duluth_original_vs_edit = duluth_original_vs_edit.sort_values(by=['id']).reset_index(drop=True)
    duluth_edit_only = duluth_edit_only.sort_values(by=['id']).reset_index(drop=True)
    test['duluth_orginal_vs_edit'] = duluth_original_vs_edit.pred
    test['duluth_edit_only'] = duluth_edit_only.pred

    # Load GloVe vectors
    glove_vectors, glove_words = glove.load_glove_vectors(args.npyFILE)

    # Run original vs. edit (metric 1)
    metric1 = lambda row: incongruity.original_vs_edit(row.original_phrase, row.edit, glove_words, glove_vectors)
    run_metric(train, test, metric1, 'original_vs_edit', 'Metric 1 [Original vs. Edit]')
    print()

    # # Run edit vs. neighbors (metric 2) with window 1
    # metric2_window1 = lambda row: incongruity.edit_vs_neighbors(row.original, row.edit, glove_words, glove_vectors, window_size=1)
    # run_metric(train, test, metric2_window1, 'edit_vs_neighbors_window1', 'Metric 2 [Edit vs. Neighbors] (window=1)')
    # print()

    # # Run edit vs. neighbors (metric 2) with window 3
    # metric2_window3 = lambda row: incongruity.edit_vs_neighbors(row.original, row.edit, glove_words, glove_vectors, window_size=3)
    # run_metric(train, test, metric2_window3, 'edit_vs_neighbors_window3', 'Metric 2 [Edit vs. Neighbors] (window=3)')
    # print()

    # # Calculate edit vs. neighbors (metric 2) with no window
    # metric2 = lambda row: incongruity.edit_vs_neighbors(row.original, row.edit, glove_words, glove_vectors)
    # run_metric(train, test, metric2, 'edit_vs_neighbors', 'Metric 2 [Edit vs. Neighbors] (unlimited window)')
    # print()

    # # Calculate RMSEs
    # print('RMSEs using all data')
    # n = len(test.meanGrade)
    # average_grade = sum(test.meanGrade)/n
    # RMSE_avg = math.sqrt(sum((average_grade - test.meanGrade[i])**2 for i in range(n))/n)
    # RMSE_Duluth_original_vs_edit = math.sqrt(sum((duluth_original_vs_edit.pred[i] - test.meanGrade[i])**2 for i in range(n))/n)
    # RMSE_Duluth_edit_only = math.sqrt(sum((duluth_edit_only.pred[i] - test.meanGrade[i])**2 for i in range(n))/n)
    # print(f'RMSE average: {RMSE_avg}')
    # print(f'RMSE Duluth original vs. edit: {RMSE_Duluth_original_vs_edit}')
    # print(f'RMSE Duluth edit only: {RMSE_Duluth_edit_only}')
    # print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE', type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    main(parser.parse_args())

# python experiment.py ../glove/glove.6B.300d.npy
