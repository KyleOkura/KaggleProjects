import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage



def split_train_file():
    """
    Open the training dataset and split into train and validation sets
    """

    print("Opening and splitting train_parquet into train and validation datasets")
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



def get_median_fill_values(dataset):
    """
    Extract median values from the training dataset to use for filling test data
    Returns a dictionary mapping column names to their median values
    """
    ignore_cols = ['id', 'code', 'sub_code', 'sub_category', 'horizon', 'y_target', 'weight']
    median_values = {}
    for col in dataset.columns:
        if col not in ignore_cols:
            median_values[col] = dataset[col].median()
    return median_values


def fill_empty_values(dataset, median_values=None):
    """
    There are a lot of columns that have a high amount of missing values. A lot of these
    columns also have outliers that like to pull the mean, thus we will fill empty values
    with the median of that column.
    
    If median_values dict is provided (for test data), use those values.
    Otherwise calculate from the dataset itself (for training data).
    """

    ignore_cols = ['id', 'code', 'sub_code', 'sub_category', 'horizon', 'y_target', 'weight']
    feature_cols = [c for c in dataset.columns if c not in ignore_cols]
    num_missing = dataset[feature_cols].isnull().sum().sum()
    print(f"Number of missing values prior to filling with median: {num_missing}")
    
    if median_values is None:
        # Calculate from dataset (training mode)
        for col in dataset.columns:
            if col not in ignore_cols:
                median_val = dataset[col].median()
                dataset[col] = dataset[col].fillna(median_val)
    else:
        # Use provided values (test mode)
        for col in dataset.columns:
            if col not in ignore_cols and col in median_values:
                dataset[col] = dataset[col].fillna(median_values[col])

    num_missing = dataset[feature_cols].isnull().sum().sum()
    print(f"Number of missing values after filling with median: {num_missing}")

    return dataset


def get_encoding_mappings(dataset):
    """
    Extract encoding mappings from the training dataset for use with test data.
    Returns:
    - encoding_maps: dict mapping column names to their value->median mappings
    - one_hot_columns: list of expected one-hot encoded column names
    """
    encode_cols = ['code', 'sub_code', 'ts_index']
    encoding_maps = {}
    
    for col in encode_cols:
        # Calculate median y_target for each unique value in the column
        encoding_maps[col] = dataset.groupby(col)['y_target'].median().to_dict()
    
    # Get one-hot encoded column names by doing a dummy transformation
    temp_df = pd.get_dummies(dataset[['sub_category', 'horizon']], columns=['sub_category', 'horizon'], dtype=int)
    one_hot_columns = temp_df.columns.tolist()
    
    return encoding_maps, one_hot_columns


def encoding(dataset, encoding_maps=None, one_hot_columns=None):
    """
    One hot encoding sub_category and horizon - there are only 4 horizon values and 5? for sub_category
    so we don't need to do target encoding (it wont make the dataset that much bigger)

    Target encoding code, sub_code, and ts_index - there are too many different possible values here for each,
    so we will fill missing values with the y_target median of that specific code, sub_code, or ts_index in a new col
    
    If encoding_maps is provided (for test data), use those mappings.
    Otherwise calculate from the dataset itself (for training data).
    """

    # One-hot encoding
    dataset = pd.get_dummies(dataset, columns=['sub_category', 'horizon'], dtype=int)
    
    # Ensure all expected one-hot columns exist (for test data)
    if one_hot_columns is not None:
        for col in one_hot_columns:
            if col not in dataset.columns:
                dataset[col] = 0
        # Reorder to match training data
        existing_cols = [c for c in dataset.columns if c not in one_hot_columns]
        dataset = dataset[existing_cols + one_hot_columns]

    # Target encoding
    encode_cols = ['code', 'sub_code', 'ts_index']
    
    if encoding_maps is None:
        # Calculate from dataset (training mode)
        for col in encode_cols:
            dataset[f'{col}_median'] = dataset.groupby(col)['y_target'].transform('median')
            dataset.drop(columns=[col], inplace=True)
    else:
        # Use provided mappings (test mode)
        for col in encode_cols:
            # Map values to their medians, use overall median for unknown values
            overall_median = sum(encoding_maps[col].values()) / len(encoding_maps[col]) if encoding_maps[col] else 0
            dataset[f'{col}_median'] = dataset[col].map(encoding_maps[col]).fillna(overall_median)
            dataset.drop(columns=[col], inplace=True)

    return dataset


def get_corr_drop_columns(dataset):
    """
    Identify columns with low correlation to y_target.
    Returns list of column names to drop.
    """
    drop_cols = ['id']
    corr = dataset.drop(columns=['y_target', 'weight', 'horizon'] + drop_cols, errors='ignore')
    target_corr = corr.corrwith(dataset['y_target']).abs()
    low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
    
    return drop_cols + low_corr_cols


def corr_drop(dataset, drop_columns=None):
    """
    Dropping columns with low correlation to y_target
    Can undo this later, but want to decrease dimensionality
    
    If drop_columns is provided (for test data), use that list.
    Otherwise calculate from the dataset itself (for training data).
    """

    if drop_columns is None:
        # Calculate from dataset (training mode)
        drop_cols = ['id']
        existing_cols = [c for c in drop_cols if c in dataset.columns]
        dataset.drop(columns=existing_cols, inplace=True)
        corr = dataset.drop(columns=['y_target', 'weight', 'horizon'], errors='ignore')
        target_corr = corr.corrwith(dataset['y_target']).abs()

        print(f'Number of features before dropping low correlation: {len(dataset.columns)}')
        low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
        dataset.drop(columns=low_corr_cols, inplace=True)
        print(f'Number of features after dropping low correlation: {len(dataset.columns)}')
    else:
        # Use provided list (test mode)
        print(f'Number of features before dropping low correlation: {len(dataset.columns)}')
        existing_cols = [c for c in drop_columns if c in dataset.columns]
        dataset.drop(columns=existing_cols, inplace=True)
        print(f'Number of features after dropping low correlation: {len(dataset.columns)}')

    return dataset


def get_cluster_drop_columns(dataset):
    """
    Identify features that show high correlation to another feature.
    Returns list of column names to drop.
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
    
    return features_to_drop


def cluster_drop(dataset, drop_columns=None):
    """
    Dropping features that show high correlation to another feature
    Like correlation we can remove this later, but want to decrease dimenstionality
    in data that won't help the model
    
    If drop_columns is provided (for test data), use that list.
    Otherwise calculate from the dataset itself (for training data).
    """

    if drop_columns is None:
        # Calculate from dataset (training mode)
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
    else:
        # Use provided list (test mode)
        print(f'Number of features before dropping clustered features: {len(dataset.columns)}')
        existing_cols = [c for c in drop_columns if c in dataset.columns]
        dataset.drop(columns=existing_cols, inplace=True)
        print(f'Number of features after dropping clustered features: {len(dataset.columns)}')

    return dataset


def get_preprocessing_config(train_data):
    """
    Extract all preprocessing configurations from training data.
    This should be called after processing training data and used for test data.
    
    Returns a dictionary containing:
    - median_values: for filling empty values
    - encoding_maps: for target encoding
    - one_hot_columns: expected one-hot encoded columns
    - corr_drop_cols: columns to drop from correlation analysis
    - cluster_drop_cols: columns to drop from clustering
    """
    # Process a copy to extract configurations
    data = train_data.copy()
    
    # Get median values before any transformations
    median_values = get_median_fill_values(data)
    
    # Fill empty values
    data = fill_empty_values(data)
    
    # Get encoding mappings before encoding
    encoding_maps, one_hot_columns = get_encoding_mappings(data)
    
    # Apply encoding
    data = encoding(data)
    
    # Get correlation drop columns
    corr_drop_cols = get_corr_drop_columns(data)
    
    # Apply correlation drop
    data = corr_drop(data)
    
    # Get cluster drop columns
    cluster_drop_cols = get_cluster_drop_columns(data)
    
    config = {
        'median_values': median_values,
        'encoding_maps': encoding_maps,
        'one_hot_columns': one_hot_columns,
        'corr_drop_cols': corr_drop_cols,
        'cluster_drop_cols': cluster_drop_cols
    }
    
    return config


def process_data():
    train_data, val_data = split_train_file()
    val_data = remove_y_target_outliers(val_data)
    val_data = fill_empty_values(val_data)
    val_data = encoding(val_data)
    val_data = corr_drop(val_data)
    val_data = cluster_drop(val_data)
    
    return val_data



if __name__ == "__main__":
    train_data, val_data = split_train_file()
    del train_data
    val_data = remove_y_target_outliers(val_data)
    val_data = fill_empty_values(val_data)
    val_data = encoding(val_data)
    val_data = corr_drop(val_data)
    val_data = cluster_drop(val_data)