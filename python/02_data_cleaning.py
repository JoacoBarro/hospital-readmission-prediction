import pandas as pd
import numpy as np


def load_data(path="data/02_mapped/diabetic_data_mapped.csv"):
    df = pd.read_csv(path)
    return df


def replace_question_marks(df):
    """
    Replace '?' with proper NaN values.
    """
    df.replace("?", np.nan, inplace=True)
    return df


if __name__ == "__main__":
    df = load_data()
    df = replace_question_marks(df)

    print("Missing values after replacement:")
    print(df.isna().sum())

def drop_high_missing_columns(df):
    """
    Drop columns with excessive missing values that are not useful.
    """
    cols_to_drop = [
        "weight",
        "payer_code",
        "medical_specialty",
        "max_glu_serum",
        "A1Cresult"
    ]
    
    df.drop(columns=cols_to_drop, inplace=True)
    return df

if __name__ == "__main__":
    df = load_data()
    df = replace_question_marks(df)
    df = drop_high_missing_columns(df)

    print("Remaining missing values:")
    print(df.isna().sum())

def remove_invalid_readmissions(df):
    """
    Remove patients who died or went to hospice.
    They cannot be readmitted -> prevents data leakage.
    """
    
    invalid_dispositions = [
        11,  # Expired
        13,  # Hospice home
        14,  # Hospice medical facility
        19,  # Expired at home
        20,  # Expired in medical facility
        21   # Expired place unknown
    ]
    
    df = df[~df["discharge_disposition_id"].isin(invalid_dispositions)]
    
    return df

if __name__ == "__main__":
    df = load_data()
    df = replace_question_marks(df)
    df = drop_high_missing_columns(df)
    df = remove_invalid_readmissions(df)

    print("Shape after removing expired/hospice patients:")
    print(df.shape)
    print("\nRemaining missing values:")
    print(df.isna().sum())

def impute_categorical(df):
    """
    Fill remaining missing categorical values.
    """
    
    # race → Unknown
    df["race"].fillna("Unknown", inplace=True)
    
    # admission_type_id → mode
    df["admission_type_id"].fillna(df["admission_type_id"].mode()[0], inplace=True)
    
    # admission_source_id → mode
    df["admission_source_id"].fillna(df["admission_source_id"].mode()[0], inplace=True)
    
    return df

def impute_diagnoses(df):
    """
    Fill missing diagnosis codes.
    """
    
    df["diag_1"].fillna("Unknown", inplace=True)
    df["diag_2"].fillna("Unknown", inplace=True)
    df["diag_3"].fillna("Unknown", inplace=True)
    
    return df

def format_change(df):
    df["change"] = df["change"].map({
        "Ch": 1,
        "No": 0
    })
    return df

def remove_invalid_gender(df):
    df = df[~df["gender"].isin(["Unknown", "Invalid", "Unknown/Invalid"])]
    return df

def remove_unknown_race(df):
    df = df[df["race"] != "Unknown"]
    return df

def fix_race_labels(df):
    df["race"] = df["race"].replace("AfricanAmerican", "African American")
    return df

if __name__ == "__main__":
    df = load_data()
    df = replace_question_marks(df)
    df = drop_high_missing_columns(df)
    df = remove_invalid_readmissions(df)
    df = impute_categorical(df)
    df = impute_diagnoses(df)
    df = format_change(df)
    df = remove_invalid_gender(df)
    df = remove_unknown_race(df)
    df = fix_race_labels(df)
    
    print("Missing values after full cleaning:")
    print(df.isna().sum())
    print("\nFinal shape:", df.shape)

df.to_csv("data/03_cleaned/diabetic_data_cleaned.csv", index=False)
print("Data cleaning completed and saved.")