import pandas as pd
import train_data_processing as train_proc



def open_test_file():
    """
    Open the test dataset
    """
    print("Opening test_parquet")
    test_df = pd.read_parquet('test.parquet', engine='pyarrow')
    return test_df


def process_test_data(test_df, preprocessing_config):
    """
    Process test data using preprocessing configuration from training data.
    
    Args:
        test_df: The test dataframe
        preprocessing_config: Dictionary containing:
            - median_values: for filling empty values
            - encoding_maps: for target encoding
            - one_hot_columns: expected one-hot encoded columns
            - corr_drop_cols: columns to drop from correlation analysis
            - cluster_drop_cols: columns to drop from clustering
    
    Returns:
        Processed test dataframe ready for model prediction
    """
    print("\n=== Processing Test Data ===")
    
    # Fill empty values using training medians
    test_df = train_proc.fill_empty_values(
        test_df, 
        median_values=preprocessing_config['median_values']
    )
    
    # Apply encoding using training mappings
    test_df = train_proc.encoding(
        test_df,
        encoding_maps=preprocessing_config['encoding_maps'],
        one_hot_columns=preprocessing_config['one_hot_columns']
    )
    
    # Drop columns based on training correlation analysis
    test_df = train_proc.corr_drop(
        test_df,
        drop_columns=preprocessing_config['corr_drop_cols']
    )
    
    # Drop columns based on training clustering
    test_df = train_proc.cluster_drop(
        test_df,
        drop_columns=preprocessing_config['cluster_drop_cols']
    )
    
    print(f"\nFinal test data shape: {test_df.shape}")
    print("=== Test Data Processing Complete ===\n")
    
    return test_df


if __name__ == "__main__":
    print("Processing training data to extract configuration...")
    train_data, val_data = train_proc.split_train_file()
    val_data = train_proc.remove_y_target_outliers(val_data)
    
    preprocessing_config = train_proc.get_preprocessing_config(val_data)
  
    print("\nLoading test data...")
    
    test_df = open_test_file()
    test_df = process_test_data(test_df, preprocessing_config)
    
    print(f"Test data is ready for prediction with shape: {test_df.shape}")