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
    """
    Scales features using standardization.
    input:
        df: A DataFrame containing features.
    output:
        x_scaled: DataFrame of scaled features (without 'class').
        y: Series of labels if 'class' column exists, otherwise None.
    """
    if 'class' in df.columns:
        x = df.drop(['class'], axis=1)
        y = df['class']
    else:
        x = df
        y = None

    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(x)), y

def find_best_radius(x_trn: DataFrame, y_trn: DataFrame, y_vld: DataFrame, distances: np.ndarray) -> float:
    """
    Finds the optimal radius that gives the highest accuracy on the validation set.
    input:
        x_trn: Scaled training features.
        y_trn: Training labels.
        y_vld: Validation labels.
        distances: Matrix of Euclidean distances from validation to training samples.
    output:
        best_radius: Radius with the best validation accuracy.
    """
    mean_of_distances = np.mean(distances, axis=1)
    mean_of_distances_scalar = np.mean(mean_of_distances)

    min_distance = np.min(distances)
    best_radius = 0
    best_accuracy = 0

    # Try radius values from minimum distance up to average distance
    for radius in np.arange(min_distance, mean_of_distances_scalar, 0.1):
        predictions = predict_by_specific_radius(x_trn, y_trn, radius, distances)
        accuracy = np.count_nonzero(predictions == y_vld)  # Number of correct predictions
        if accuracy > best_accuracy:
            best_radius = radius
            best_accuracy = accuracy

    return best_radius

def predict_by_specific_radius(x_trn: DataFrame, y_trn: DataFrame, radius: float, distances: np.ndarray) -> List:
    """
    Predicts labels for each sample using a fixed radius.
    input:
        x_trn: Scaled training features.
        y_trn: Training labels.
        radius: Radius for nearest neighbor classification.
        distances: Matrix of distances from samples to training points.
    output:
        List of predicted labels.
    """
    result = []
    for distance in distances:
        result.append(find_label_with_most_appearences_in_specific_radius(y_trn, distance, radius))
    return result

def find_label_with_most_appearences_in_specific_radius(y_trn: DataFrame, distances: np.ndarray, radius: float):
    """
    Finds the most frequent label among neighbors within a given radius.
    input:
        y_trn: Training labels.
        distances: Distances from a single test sample to all training samples.
        radius: Distance threshold.
    output:
        Most common label within radius. If none, returns label of nearest neighbor.
    """
    neighbors_in_specific_radius = np.where(distances <= radius)[0]

    # Convert to 1D array if needed
    y_array = y_trn.values if isinstance(y_trn, pd.Series) else y_trn.iloc[:, 0].values

    if len(neighbors_in_specific_radius) > 0:
        labels_of_neighbors = y_array[neighbors_in_specific_radius]
        unique_labels, counts = np.unique(labels_of_neighbors, return_counts=True)
        max_count_index = np.argmax(counts)
        return unique_labels[max_count_index]
    else:
        # No neighbors found in radius – use nearest neighbor
        closest_index = np.argmin(distances)
        return y_array[closest_index]

def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    """
    Performs radius-based nearest neighbor classification.

    input:
        data_trn: Path to training CSV file.
        data_vld: Path to validation CSV file.
        df_tst: Test data (features only, no labels).
    output:
        List of predicted labels for test samples.
    """
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # Load datasets
    df_train = pd.read_csv(data_trn)
    df_valid = pd.read_csv(data_vld)

    # Scale training and validation sets
    x_trn_scaled, y_trn = scale(df_train)
    x_vld_scaled, y_vld = scale(df_valid)

    # Compute distance matrix from validation to training samples
    distances_from_validations = cdist(x_vld_scaled.values, x_trn_scaled.values, metric='euclidean')

    # Find best radius using validation set
    best_radius = find_best_radius(x_trn_scaled, y_trn, y_vld, distances_from_validations)

    # Scale test set
    tst_scaled, y_tst = scale(df_tst)

    # Compute distance matrix from test to training samples
    distances_from_tests = cdist(tst_scaled.values, x_trn_scaled.values, metric='euclidean')

    # Predict test labels using best radius
    predictions = predict_by_specific_radius(x_trn_scaled, y_trn, best_radius, distances_from_tests)
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
