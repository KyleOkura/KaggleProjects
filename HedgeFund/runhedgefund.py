import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def prepare_training_data(file_path):
    """
    Loads training data, performs cleaning, target encoding, and feature selection.
    Returns: X_train, y_train, and an 'artifacts' dictionary containing 
    maps/medians needed for the test set.
    """
    print(f"--- Loading and Processing Training Data: {file_path} ---")
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    # 1. Outlier Removal (Target)
    # Dropping top and bottom 2.5% of y_target as per notebook logic
    q_low = df['y_target'].quantile(0.025)
    q_high = df['y_target'].quantile(0.975)
    df = df[(df['y_target'] >= q_low) & (df['y_target'] <= q_high)].copy()
    
    # 2. Missing Value Handling
    # Calculate medians for numerical columns to fill NAs
    # We save these to apply to the test set later
    fill_values = df.median(numeric_only=True).to_dict()
    df = df.fillna(fill_values)
    
    # 3. Target Encoding (Corrected)
    # Calculating median y_target for each unique category in code, sub_code, ts_index
    encoding_maps = {}
    cols_to_encode = ['code', 'sub_code', 'ts_index']
    global_target_median = df['y_target'].median()
    
    for col in cols_to_encode:
        if col in df.columns:
            # Generate the map: Category -> Median Target
            mapping = df.groupby(col)['y_target'].median()
            encoding_maps[col] = mapping
            
            # Map the values to a new column
            df[f'{col}_encoded'] = df[col].map(mapping)
            
            # Fill any gaps (if a category didn't have a target, though unlikely in train)
            df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(global_target_median)
            
            # Drop original categorical column
            df = df.drop(columns=[col])

    # 4. Feature Selection via Hierarchical Clustering
    # Remove features that are highly correlated (redundant)
    print("--- Running Feature Selection (Clustering) ---")
    y = df['y_target']
    # Exclude non-feature columns
    X_temp = df.drop(columns=['y_target', 'id', 'sub_category'], errors='ignore')
    
    # Calculate correlation matrix
    corr_matrix = X_temp.corr().abs()
    
    # Create distance matrix (1 - correlation)
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix.values, 0)
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Hierarchical clustering (Ward's method)
    Z = linkage(condensed_dist, method='ward')
    
    # Create clusters (threshold t=0.2 roughly corresponds to high correlation)
    cluster_labels = fcluster(Z, t=0.2, criterion='distance')
    
    # Select one feature per cluster (the one with highest corr to target)
    selected_features = []
    feat_corr_to_target = X_temp.corrwith(y).abs()
    
    cluster_df = pd.DataFrame({'feature': X_temp.columns, 'cluster': cluster_labels})
    
    for cluster_id in cluster_df['cluster'].unique():
        feats_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id]['feature']
        # Pick feature with max correlation to target in this cluster
        best_feat = feat_corr_to_target[feats_in_cluster].idxmax()
        selected_features.append(best_feat)
        
    print(f"Selected {len(selected_features)} features out of {len(X_temp.columns)}.")
    
    X = df[selected_features]
    
    # Save artifacts for Test set processing
    artifacts = {
        'fill_values': fill_values,
        'encoding_maps': encoding_maps,
        'global_target_median': global_target_median,
        'selected_features': selected_features
    }
    
    return X, y, artifacts

def prepare_test_data(file_path, artifacts):
    """
    Loads test data and applies the EXACT transformations defined in the artifacts.
    """
    print(f"--- Processing Test Data: {file_path} ---")
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    # Store IDs for output file
    if 'id' in df.columns:
        row_ids = df['id']
    else:
        row_ids = df.index
        
    # Extract True Test Labels if they exist (for accuracy scoring)
    y_test = None
    if 'y_target' in df.columns:
        y_test = df['y_target']
        
    # 1. Missing Values (Use Train Medians)
    df = df.fillna(artifacts['fill_values'])
    
    # 2. Target Encoding (Use Train Maps)
    for col, mapping in artifacts['encoding_maps'].items():
        if col in df.columns:
            # Map using the dictionary from training
            df[f'{col}_encoded'] = df[col].map(mapping)
            
            # CRITICAL: Fill unseen categories in test with global train median
            df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(artifacts['global_target_median'])
            
    # 3. Feature Selection (Select same columns as Train)
    # Ensure all selected features exist (create with 0 if missing, strictly for safety)
    for feat in artifacts['selected_features']:
        if feat not in df.columns:
            df[feat] = 0
            
    X_test = df[artifacts['selected_features']]
    
    return X_test, y_test, row_ids

def run_model_and_evaluate(X_train, y_train, X_test, y_test, row_ids, output_filename='test_results.csv'):
    """
    Trains XGBoost, predicts on Test, calculates Score, saves Output.
    """
    print("--- Training Model (XGBoost) ---")
    # Using standard XGBoost Regressor parameters appropriate for this data size
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist',  # Faster training
        device='cuda',       # Use GPU if available
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("--- Predicting on Test Data ---")
    predictions = model.predict(X_test)
    
    # Calculate Score if targets are available
    if y_test is not None:
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("\n" + "="*40)
        print(f" ACCURACY SCORE (RMSE): {rmse:.5f}")
        print("="*40 + "\n")
    else:
        print("\n[INFO] No target labels in test file. Skipping accuracy score.\n")
        
    # Generate Output File
    output_df = pd.DataFrame({
        'id': row_ids,
        'prediction': predictions
    })
    
    output_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to: {output_filename}")

if __name__ == "__main__":
    # Define paths
    TRAIN_PATH = 'train.parquet'
    TEST_PATH = 'test.parquet' # Assuming test file is named this
    
    # 1. Process Train
    X_train, y_train, artifacts = prepare_training_data(TRAIN_PATH)
    
    # 2. Process Test
    try:
        X_test, y_test, ids = prepare_test_data(TEST_PATH, artifacts)
        
        # 3. Run & Score
        run_model_and_evaluate(X_train, y_train, X_test, y_test, ids)
        
    except FileNotFoundError:
        print(f"Error: {TEST_PATH} not found. Please ensure the test file exists.")