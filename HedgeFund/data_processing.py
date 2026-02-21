import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

def open_training_set():
    """
    Open the training dataset and split into train and validation sets
    """

    train_df = pd.read_parquet('train.parquet', engine='pyarrow')
    train_data, val_data = train_test_split(train_df, test_size=0.10, random_state=42)

    return train_data, val_data



def remove_y_target_outliers(dataset):
    """
    Based on data exploration, we want to remove top and bottom 30% of the y_target extremes.
    These values had low weight values compared to the data points centered around 0
    """
    
    print(f'Number of entries before removing y_target extremes: {len(dataset)}')
    lower_bound = dataset['y_target'].quantile(0.3)
    upper_bound = dataset['y_target'].quantile(0.7)
    dataset = dataset[dataset['y_target'].between(lower_bound, upper_bound)]
    print(f'Number of entries after removing y_target extremes: {len(dataset)}')

    return dataset



def fill_empty_values(dataset):
    """
    There are a lot of columns that have a high amount of missing values. A lot of these
    columns also have outliers that like to pull the mean, thus we will fill empty values
    with the median of that column
    """

    ignore_cols = ['id', 'code', 'sub_code', 'sub_category', 'horizon', 'y_target', 'weight']
    feature_cols = [c for c in dataset.columns if c not in ignore_cols]
    num_missing = dataset[feature_cols].isnull().sum().sum()
    print(f"Number of missing values prior to filling with median: {num_missing}")
    for col in dataset.columns:
        if col not in ignore_cols:
            median_val = dataset[col].median()
            dataset[col] = dataset[col].fillna(median_val)

    num_missing = dataset[feature_cols].isnull().sum().sum()
    print(f"Number of missing values after filling with median: {num_missing}")

    return dataset


def encoding(dataset):
    """
    One hot encoding sub_category and horizon - there are only 4 horizon values and 5? for sub_category
    so we don't need to do target encoding (it wont make the dataset that much bigger)

    Target encoding code, sub_code, and ts_index - there are too many different possible values here for each,
    so we will fill missing values with the y_target median of that specific code, sub_code, or ts_index in a new col
    """

    dataset = pd.get_dummies(dataset, columns=['sub_category', 'horizon'], dtype=int)

    encode_cols = ['code', 'sub_code', 'ts_index']
    for col in encode_cols:
        dataset[f'{col}_median'] = dataset.groupby(col)['y_target'].transform('median')
        dataset.drop(columns=[col], inplace=True)

    return dataset


def corr_drop(dataset):
    """
    Dropping columns with low correlation to y_target
    Can undo this later, but want to decrease dimensionality

    """

    drop_cols = ['id']
    existing_cols = [c for c in drop_cols if c in dataset.columns]
    dataset.drop(columns=existing_cols, inplace=True)
    corr = dataset.drop(columns=['y_target', 'weight', 'horizon'], errors='ignore')
    target_corr = corr.corrwith(dataset['y_target']).abs()

    print(f'Number of features before dropping low correlation: {len(dataset.columns)}')
    low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
    dataset.drop(columns=low_corr_cols, inplace=True)
    print(f'Number of features after dropping low correlation: {len(dataset.columns)}')

    return dataset


def cluster_drop(dataset):
    """
    Dropping features that show high correlation to another feature
    Like correlation we can remove this later, but want to decrease dimenstionality
    in data that won't help the model
    """


    cluster_df = dataset.drop(columns=['y_target', 'weight', 'horizon'], errors='ignore')
    sample_corr = cluster_df.corr().astype('float32').abs()

    dissimilarity = 1 - sample_corr
    distance_matrix = squareform(dissimilarity, checks=False)

    Z = linkage(distance_matrix, method='complete')

    cutoff=0.2
    cluster_labels = fcluster(Z, t=cutoff, criterion='distance')

    cluster_map = {}
    feature_names = sample_corr.columns
    for feature, cluster_id in zip(feature_names, cluster_labels):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(feature)

    corr = dataset.drop(columns=['y_target', 'weight', 'horizon'], errors='ignore')
    target_corr = corr.corrwith(dataset['y_target']).abs()
    
    features_to_drop = []
    for cluster_id, features in cluster_map.items():
        if len(features) > 1:
            corrs = {f: target_corr.get(f, 0) for f in features}

            winner = max(corrs, key=corrs.get)

            for f in features:
                if f != winner:
                    features_to_drop.append(f)

    print(f'Number of features before dropping clustered features: {len(dataset.columns)}')
    dataset.drop(columns=features_to_drop, inplace=True)
    print(f'Number of features after dropping clustered features: {len(dataset.columns)}')

    return dataset


def main():
    train_data, val_data = open_training_set()
    val_data = remove_y_target_outliers(val_data)
    val_data = fill_empty_values(val_data)
    val_data = encoding(val_data)
    val_data = corr_drop(val_data)
    val_data = cluster_drop(val_data)

if __name__ == "__main__":
    main()