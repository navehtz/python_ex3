import time
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List
from scipy.spatial.distance import cdist


def scale(df: DataFrame) -> (DataFrame, DataFrame):
    if 'class' in df.columns:
        x = df.drop(['class'], axis=1)
        y = df['class']
    else:
        x = df
        y = None

    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(x)), y

def find_best_radius(x_trn: DataFrame, y_trn: DataFrame, y_vld: DataFrame, distances: np.ndarray) -> float:

    mean_of_distances = np.mean(distances, axis=1)
    mean_of_distances_scalar = np.mean(mean_of_distances)

    min_distance = np.min(distances)
    best_radius = 0
    best_accuracy = 0

    for radius in np.arange(min_distance, mean_of_distances_scalar, 0.1):
        predictions = predict_by_specific_radius(x_trn, y_trn, radius, distances)
        accuracy = np.count_nonzero(predictions == y_vld)
        if accuracy > best_accuracy:
            best_radius = radius
            best_accuracy = accuracy

    return best_radius

def predict_by_specific_radius(x_trn: DataFrame, y_trn: DataFrame, radius: float, distances: np.ndarray) -> List:
    result = []
    for distance in distances:
        result.append(find_label_with_most_appearences_in_specific_radius(y_trn, distance, radius))
    return result

def find_label_with_most_appearences_in_specific_radius(y_trn: DataFrame, distances: np.ndarray, radius: float):
    neighbors_in_specific_radius = np.where(distances <= radius)[0]
    labels_of_lines = y_trn[neighbors_in_specific_radius]
    if not labels_of_lines.empty:
        unique_labels, counts = np.unique(labels_of_lines, return_counts=True)
        max_count_index = np.argmax(counts)
        return unique_labels[max_count_index]
    else:
        return y_trn[np.argmin(distances)]


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    df_train = pd.read_csv(data_trn)
    df_valid = pd.read_csv(data_vld)

    x_trn_scaled, y_trn = scale(df_train)
    x_vld_scaled, y_vld = scale(df_valid)

    distances_from_validations = cdist(x_vld_scaled.values, x_trn_scaled.values, metric='euclidean')
    best_radius = find_best_radius(x_trn_scaled, y_trn, y_vld, distances_from_validations)

    tst_scaled, y_tst = scale(df_tst)
    distances_from_tests = cdist(tst_scaled.values, x_trn_scaled.values, metric='euclidean')

    predictions =  predict_by_specific_radius(x_trn_scaled, y_trn, best_radius, distances_from_tests)
    return predictions


# todo: fill in your student ids
students = {'id1': '207080185' , 'id2': '208077214'}

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
