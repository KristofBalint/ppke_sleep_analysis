from team_code import *
from helper_code import *
import os

# 1. Setup paths
data_folder = "training_set"
model_folder = "model"

# 2. Train the model (This creates model.sav)
train_model(data_folder, model_folder, verbose=True)

# 3. Load the model back into memory
trained_model_object = load_model(model_folder, verbose=True)

# 4. Correctly identify a "record"
# We need to find the list of patients from the demographics file
patient_data_file = os.path.join(data_folder, "demographics.csv")
patient_metadata_list = find_patients(patient_data_file)

print(patient_metadata_list)
results=[]
for i in range(len(patient_metadata_list)):
    # Grab the first patient record (this is a dictionary/object)
    single_record = patient_metadata_list[i]
    #print("Available keys in record:", single_record.keys())
    # 5. Run the model on that specific record
    binary_out, prob_out = run_model(
        model=trained_model_object,
        record=single_record,
        data_folder=data_folder,
        verbose=True
    )

    print(
        f"Prediction for {single_record[HEADERS['bids_folder']]}, {single_record[HEADERS['site_id']]},{single_record[HEADERS['session_id']]}: {binary_out} (Prob: {prob_out:.2f})")
    results.append({
        'BDSPPatientID': single_record[HEADERS['bids_folder']],
        'Cognitive_Impairment': int(binary_out),
        'Cognitive_Impairment_Probability': float(prob_out)
    })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

#python evaluate_model.py -d training_set/demographics.csv -o predictions.csv -s scores.txt