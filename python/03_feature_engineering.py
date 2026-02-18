import pandas as pd

def load_clean_data(path="data/03_cleaned/diabetic_data_cleaned.csv"):
    return pd.read_csv(path)

def encode_target(df):
    """
    1 = readmitted within 30 days
    0 = otherwise
    """
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    return df

# Remove identifiers (data leakage)
def drop_identifiers(df):
    df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)
    return df

# Convert AGE ranges → numeric
age_map = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95
}

def encode_age(df):
    df["age"] = df["age"].map(age_map)
    return df

# Create a useful utilization feature
def create_utilization_feature(df):
    df["total_visits"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )
    return df

def categorize_diagnosis(code):
    """
    Convert ICD9 codes into disease groups.
    """
    try:
        code = str(code)
        
        # remove V and E codes (special ICD codes)
        if code.startswith(("V", "E")):
            return "Other"

        code = float(code)

        if 390 <= code <= 459 or code == 785:
            return "Circulatory"
        elif 460 <= code <= 519 or code == 786:
            return "Respiratory"
        elif 520 <= code <= 579 or code == 787:
            return "Digestive"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif 580 <= code <= 629 or code == 788:
            return "Genitourinary"
        elif 140 <= code <= 239:
            return "Neoplasms"
        else:
            return "Other"

    except:
        return "Other"
    
def group_diagnoses(df):
    df["diag_1"] = df["diag_1"].apply(categorize_diagnosis)
    df["diag_2"] = df["diag_2"].apply(categorize_diagnosis)
    df["diag_3"] = df["diag_3"].apply(categorize_diagnosis)
    return df

medication_cols = [
    "metformin","repaglinide","nateglinide","chlorpropamide",
    "glimepiride","acetohexamide","glipizide","glyburide",
    "tolbutamide","pioglitazone","rosiglitazone","acarbose",
    "miglitol","troglitazone","tolazamide","examide",
    "citoglipton","insulin","glyburide-metformin",
    "glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone"
]

def create_medication_feature(df):
    """
    Count how many diabetes medications the patient is taking.
    Any value different from 'No' counts as a medication.
    """
    df["num_diabetes_meds"] = (df[medication_cols] != "No").sum(axis=1)
    return df

def drop_medication_columns(df):
    df.drop(columns=medication_cols, inplace=True)
    return df

def simplify_admission_type(df):
    emergency = ["Emergency", "Urgent", "Trauma Center"]
    elective = ["Elective"]

    df["admission_type_id"] = df["admission_type_id"].apply(
        lambda x: "Emergency" if x in emergency
        else ("Elective" if x in elective else "Other")
    )
    return df

def simplify_discharge(df):
    home = [
        "Discharged to home",
        "Discharged/transferred to home with home health service",
        "Hospice / home"
    ]

    facility_keywords = [
        "hospital", "facility", "care", "rehab",
        "nursing", "SNF", "ICF", "psychiatric"
    ]

    def group_discharge(val):
        val_lower = str(val).lower()

        if val in home:
            return "Home"
        elif any(word in val_lower for word in facility_keywords):
            return "Care Facility"
        else:
            return "Other"

    df["discharge_disposition_id"] = df["discharge_disposition_id"].apply(group_discharge)
    return df

def simplify_admission_source(df):
    def group_source(val):
        val = str(val).lower()

        if "emergency" in val:
            return "Emergency"
        elif "referral" in val:
            return "Referral"
        elif "transfer" in val:
            return "Transfer"
        else:
            return "Other"

    df["admission_source_id"] = df["admission_source_id"].apply(group_source)
    return df

def encode_binary_columns(df):
    binary_map = {"Yes": 1, "No": 0}
    df["diabetesMed"] = df["diabetesMed"].map(binary_map)
    return df

def drop_visit_columns(df):
    df.drop(
        columns=["number_outpatient", "number_emergency", "number_inpatient"],
        inplace=True
    )
    return df

def rename_hospital_columns(df):
    df.rename(columns={
        "admission_type_id": "admission_type",
        "discharge_disposition_id": "discharge_disposition",
        "admission_source_id": "admission_source"
    }, inplace=True)
    return df

def save_final_data(df, path="data/04_featured_engineered/diabetic_data_fe.csv"):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_clean_data()

    df = encode_target(df)
    df = drop_identifiers(df)
    df = encode_age(df)
    df = create_utilization_feature(df)

    # group diagnoses into categories
    df = group_diagnoses(df)

    # medication engineering
    df = create_medication_feature(df)
    df = drop_medication_columns(df)

    # reduce hospital categories
    df = simplify_admission_type(df)
    df = simplify_discharge(df)
    df = simplify_admission_source(df)

    # final polish
    df = encode_binary_columns(df)
    df = drop_visit_columns(df)
    df = rename_hospital_columns(df)

    save_final_data(df)

    print("Feature engineering completed!")
    print("Final shape:", df.shape)
    

