LABELS = [
    "O", 
    "B-Section_Header", "I-Section_Header",
    "B-Column_Header", "I-Column_Header",
    "B-Test_Context_Name", "I-Test_Context_Name",
    "B-Test_Name", "I-Test_Name",
    "B-Test_Value", "I-Test_Value",
    "B-Test_Unit", "I-Test_Unit",
    "B-Test_Norm", "I-Test_Norm",
    "B-Patient_Name", "I-Patient_Name",
    "B-Patient_DOB", "I-Patient_DOB",
    "B-Patient_Weight", "I-Patient_Weight",
    "B-Patient_Height", "I-Patient_Height",
    "B-Patient_Gender", "I-Patient_Gender",
]

LABEL_COLORS = {
    "Section_Header": "darkblue",
    "Column_Header": "darkgreen",
    "Test_Context_Name": "red",
    "Test_Name": "green",
    "Test_Value": "orange",
    "Test_Unit": "cyan",
    "Test_Norm": "yellow",
    "Patient_Name": "purple",
    "Patient_DOB": "brown",
    "Patient_Weight": "pink",
    "Patient_Height": "blue",
    "Patient_Gender": "grey"
}

CRITICAL_LABELS = {
    "Patient_Name",
    "Patient_DOB",
    "Patient_Gender",
    "Patient_Weight",
    "Patient_Height",
}

BASE_MODEL_PATH = "./models/base_model" 

DATASET_PATH = "./data/dataset"

MODEL_PATH = "./models/custom_v7"

MODEL_VERSION = "custom_v7"

JSON_MIN_PATH = "./data/export.json"

IMAGES_PATH = "./data/images"

LABEL_STUDIO_URL = "http://localhost:8080" 

PROJECT_ID = 1

SESSION_ID = ".eJxVj8uOhCAQRf-FdUugQBCXs59vIIUUyrSBjmgyj8y_j05608tb5z5SP-zIkY3MBuoJle4SIXSaZOoGrWUH1gEoQ8EFwW6sbjOW_I17rsU_7myUN7Zi2_1a51xOaa2E3mppeS81AJzc47Ev_mi0-f8pyV5uAac7lQvEDyxz5VMt-5YDvyz8SRt_r5HWt6f3pWDBtpxpZZGcjsaFGIUycrBp0sJhFCkNSRFGcmBFCiZqqchFYdSZSCZImCjAVdqotesz-nzk7YuN0DsQgovfP-4SXKQ:1vs15e:8yrv8rvcK43Uk6KkHtoM_QhRhOYxZt65-jnvasaS7pQ" 

TASKS_JSON_PATH = "./data/project_tasks.json"

PRIORITY_FOLDER = "./data/priority_cases"