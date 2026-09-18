import os
import pytest
from pathlib import Path
import joblib
import pandas as pd
test_csv_expected_responses = os.environ.get("TEST_CSV_EXPECTED_RESPONSES", "tests/unmodified_data.csv")
model_weights_path =  os.environ.get("MODEL_WEIGHTS_PATH", "models/heart_pipeline.joblib")
model = joblib.load(model_weights_path)
minim_threshold = 0.9 #Define the minimum threshold for classification
def test_model_prediction():
    # Load the test CSV file
    test_data = pd.read_csv(test_csv_expected_responses)
   

    # Iterate through each row in the test data
    nr_correct_answers = 0
    for index, row in test_data.iterrows():
        predicted_truth = 0
        row_data = row.drop(["id", "Heart Disease"]).to_dict() 
        prediction = model.predict_proba(pd.DataFrame([row_data]))[0][1]
        if prediction > 0.5:
            predicted_truth = 1
        if  predicted_truth == row["Heart Disease"]:
            nr_correct_answers += 1

    print(f"Number of correct predictions: {nr_correct_answers} out of {len(test_data)}")        
    if nr_correct_answers / len(test_data) < minim_threshold:
        pytest.fail(f"Model accuracy is below the minimum threshold of {minim_threshold}. Accuracy: {nr_correct_answers / len(test_data)}")    
            
        

