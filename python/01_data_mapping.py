import pandas as pd

df = pd.read_csv("data/01_raw/diabetic_data.csv")

admission_map = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: None,
    7: "Trauma Center",
    8: "Not Mapped"
}

discharge_map = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: None,
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab facility",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility (Medicaid only)",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: "Transferred to psychiatric hospital/unit",
    29: "Transferred to Critical Access Hospital (CAH)",
    30: "Transferred to another Health Care Institution"
}

admission_source_map = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: None,
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpatient (separate claim)",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice"
}


df["admission_type_id"] = df["admission_type_id"].map(admission_map)
df["discharge_disposition_id"] = df["discharge_disposition_id"].map(discharge_map)
df["admission_source_id"] = df["admission_source_id"].map(admission_source_map)

# Any unmapped IDs → Not Mapped
df["admission_type_id"].fillna("Not Mapped", inplace=True)
df["discharge_disposition_id"].fillna("Not Mapped", inplace=True)
df["admission_source_id"].fillna("Not Mapped", inplace=True)


df.to_csv("data/02_mapped/diabetic_data_mapped.csv", index=False)
print("Data mapping completed and saved.")
