def preprocess_data(data):
    """Example preprocessing: drop unnecessary columns if they exist."""
    columns_to_drop = ['UnnecessaryColumn']  # Adjust based on your data

    # Check if the columns to be dropped exist in the DataFrame
    existing_columns = data.columns
    valid_columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

    # Drop only the columns that exist in the DataFrame
    if valid_columns_to_drop:
        data = data.drop(columns=valid_columns_to_drop, errors='ignore')
    
    return data
