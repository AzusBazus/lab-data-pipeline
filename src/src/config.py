JSON_MIN_PATH = "src/data/export.json"

IMAGE_DIR = "src/data/images"

DATASET_PATH = "src/data/dataset_processed"

BASE_MODEL_ID = "./src/models/base_model" 

V1_MODEL_ID = "./src/models/custom_v1/final"

MODELS_DIR = "./src/models"

LABELS = [
    "O", 
    "B-Section_Header", "I-Section_Header",
    "B-Test_Context_Name", "I-Test_Context_Name",
    "B-Test_Name", "I-Test_Name",
    "B-Test_Value", "I-Test_Value",
    "B-Test_Unit", "I-Test_Unit",
    "B-Test_Norm", "I-Test_Norm",
    "B-Patient_Name", "I-Patient_Name",
    "B-Patient_DOB", "I-Patient_DOB",
    "B-Patient_Weight", "I-Patient_Weight",
    "B-Patient_Height", "I-Patient_Height",
]

LABEL_COLORS = {
    "Table_Context": "darkblue",
    "Section_Header": "red",
    "Test_Name": "green",
    "Test_Value": "orange",
    "Test_Unit": "cyan",
    "Patient_Name": "purple",
    "Patient_ID": "grey"
}