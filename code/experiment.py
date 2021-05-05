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

def plot_metric(metric_values, meanGrades, xlabel, plotname):
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(metric_values, meanGrades, c='b', s=12, alpha=0.1)
    p = np.poly1d(np.polyfit(metric_values, meanGrades, 1))
    plt.plot(metric_values, p(metric_values), 'r', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel('Mean Humor Grade')
    plt.savefig(plotname)

def statistics(metric_values, mean_grades):
    p = np.poly1d(np.polyfit(metric_values, mean_grades, 1))
    n = len(mean_grades)
    slope, intercept, r_value, p_value, std_err = linregress(metric_values, mean_grades)
    avg_grade = sum(mean_grades)/n
    print(f'Linear regression equation: {p}')
    print(f'Pearson correlation: {pearsonr(metric_values, mean_grades)}')
    print(f'Spearman correlation: {spearmanr(metric_values, mean_grades)}')
    print(f'n: {n}')
    print(f'slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}')

def export_to_csv(metric_values, mean_grades, metric_name):
    export = pd.DataFrame()
    export['mean_grades'] = mean_grades
    export[metric_name] = metric_values
    export.to_csv(metric_name+'.csv')

def run_metric(train, test, metric, metric_name, axis_name):
    train[metric_name] = train.apply(metric, axis=1)
    train_notnull = train[train[metric_name].notnull()]
    metric_values = train_notnull[metric_name].values.tolist()
    mean_grades = train_notnull['meanGrade'].values.tolist()
    plot_metric(metric_values, mean_grades, axis_name, f'{metric_name}.png')
    print(metric_name)
    statistics(metric_values, mean_grades)

    model = np.poly1d(np.polyfit(metric_values, mean_grades, 1))
    average_grade = sum(mean_grades)/len(mean_grades)

    test[metric_name] = test.apply(metric, axis=1)
    test_notnull = test[test[metric_name].notnull()]
    test_metric_values = test_notnull[metric_name].values.tolist()
    test_mean_grades = test_notnull['meanGrade'].values.tolist()
    n = len(test_mean_grades)
    RMSE = math.sqrt(sum((model(test_metric_values[i]) - test_mean_grades[i])**2 for i in range(n))/n)
    RMSE_avg = math.sqrt(sum((average_grade - test_mean_grades[i])**2 for i in range(n))/n)
    print(f'RMSE: {RMSE}')
    print(f'RMSE average: {RMSE_avg}')
    print()

def main(args):
    # Load train and test data
    humicroedit_train = pd.read_csv('../data/task-1/train.csv')
    funlines_train = pd.read_csv('../data/task-1/train_funlines.csv')
    train = pd.concat([humicroedit_train, funlines_train])
    test = pd.read_csv('../data/task-1/test.csv')
    train['original_phrase'] = train.original.apply(extract_original_phrase)
    test['original_phrase'] = test.original.apply(extract_original_phrase)

    # Load GloVe vectors
    glove_vectors, glove_words = glove.load_glove_vectors(args.npyFILE)

    # Run original vs. edit (metric 1)
    metric1 = lambda row: incongruity.original_vs_edit(row.original_phrase, row.edit, glove_words, glove_vectors)
    run_metric(train, test, metric1, 'original_vs_edit', 'Original vs. Edit Cosine Similarity')
    print()

    # Run edit vs. neighbors (metric 2) with window 1
    metric2_window1 = lambda row: incongruity.edit_vs_neighbors(row.original, row.edit, glove_words, glove_vectors, window_size=1)
    run_metric(train, test, metric2_window1, 'edit_vs_neighbors_window1', 'Edit vs. Neighbors Cosine Similarity (window=1)')
    print()

    # Run edit vs. neighbors (metric 2) with window 3
    metric2_window3 = lambda row: incongruity.edit_vs_neighbors(row.original, row.edit, glove_words, glove_vectors, window_size=3)
    run_metric(train, test, metric2_window3, 'edit_vs_neighbors_window3', 'Edit vs. Neighbors Cosine Similarity (window=3)')
    print()

    # Calculate edit vs. neighbors (metric 2) with no window
    metric2 = lambda row: incongruity.edit_vs_neighbors(row.original, row.edit, glove_words, glove_vectors)
    run_metric(train, test, metric2, 'edit_vs_neighbors', 'Edit vs. Neighbors Cosine Similarity (no window)')
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npyFILE', type=argparse.FileType('rb'),
        help='an .npy file to read the saved numpy data from')
    # parser.add_argument('train', type=str,
    #     help='a CSV file of the train data')
    # parser.add_argument('test', type=str,
    #     help='a CSV file of the test data')
    # parser.add_argument('plot', type=str,
    #     help='name of file to write plot to')
    main(parser.parse_args())

# python experiment.py ../glove/glove.6B.50d.npy plot-all.png

# original_vs_edit w/ 300D GloVe
# Pearson correlation: (-0.17739550138691068, 1.0680352459856034e-143)
# Spearman correlation: SpearmanrResult(correlation=-0.16169603479087277, pvalue=2.093149320427472e-119)

# 50D Glove, humicroedit & funlines
# original_vs_edit
# Linear regression equation:  
# -0.444 x + 1.169
# Pearson correlation: (-0.1715648018191982, 6.488998669799704e-115)
# Spearman correlation: SpearmanrResult(correlation=-0.16304199571293118, pvalue=8.038090987863553e-104)
# n: 17379
# slope: -0.44402903916586006, intercept: 1.1691089639268526, r_value: -0.17156480181919806, p_value: 6.488998669867637e-115, std_err: 0.019342304195490097
# RMSE: 0.5870853388736246
# RMSE average: 0.5931697355562818

# -----------------------------------

# 300D Glove, humicroedit & funlines

# original_vs_edit
# Linear regression equation:  
# -0.7467 x + 1.149
# Pearson correlation: (-0.18115837184710443, 4.072014702863987e-128)
# Spearman correlation: SpearmanrResult(correlation=-0.16347894968689716, pvalue=2.2433095660085374e-104)
# n: 17379
# slope: -0.7466566918314562, intercept: 1.1489431955957161, r_value: -0.18115837184710432, p_value: 4.07201470290964e-128, std_err: 0.030748871050325997
# RMSE: 0.5862352315243626
# RMSE average: 0.5931697355562818

# edit_vs_neighbors_window1
# Linear regression equation:  
# -0.4588 x + 1.128
# Pearson correlation: (-0.11093829083263165, 1.4843859979422796e-48)
# Spearman correlation: SpearmanrResult(correlation=-0.1052615945145242, pvalue=7.3036907375955415e-44)
# n: 17318
# slope: -0.4587882821819596, intercept: 1.1280719576001372, r_value: -0.11093829083263154, p_value: 1.4843859979821853e-48, std_err: 0.031233307641369508
# RMSE: 0.589962526014791
# RMSE average: 0.5937898849608764

# edit_vs_neighbors_window3
# Linear regression equation:  
# -0.6395 x + 1.134
# Pearson correlation: (-0.13144075331495067, 6.9808207125084545e-68)
# Spearman correlation: SpearmanrResult(correlation=-0.1315526842123962, pvalue=5.375392992140799e-68)
# n: 17394
# slope: -0.6394837242422527, intercept: 1.134273121353309, r_value: -0.1314407533149507, p_value: 6.980820712272878e-68, std_err: 0.03657129385855845
# RMSE: 0.5888148605096918
# RMSE average: 0.5933295193064491

# edit_vs_neighbors
# Linear regression equation:  
# -0.8018 x + 1.137
# Pearson correlation: (-0.14850156053117664, 2.4663018291391694e-86)
# Spearman correlation: SpearmanrResult(correlation=-0.14980314690348023, pvalue=7.736849463456234e-88)
# n: 17394
# slope: -0.8018024814984126, intercept: 1.1372279724022023, r_value: -0.14850156053117664, p_value: 2.4663018290518952e-86, std_err: 0.040487348709439736
# RMSE: 0.5878118266449132
# RMSE average: 0.5933295193064491
