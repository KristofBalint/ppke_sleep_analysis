from team_code import *
from helper_code import *

data_folder = "training_set"
model_folder = "model"
csv_path = "channel_table.csv"

patient_data_file = os.path.join(data_folder, "demographics.csv")
patient_metadata_list = find_patients(patient_data_file)
if not patient_metadata_list:
    print("No patients found. Check your data_folder path.")
    exit()
'''
for record in patient_metadata_list:
  patient_id = record[HEADERS['bids_folder']]
  site_id    = record[HEADERS['site_id']]
  session_id = record[HEADERS['session_id']]

  print(f"--- Analyzing Feature Extraction for Patient: {patient_id} ---")

  patient_data = load_demographics(patient_data_file, patient_id, session_id)
  dem_feat = extract_demographic_features(patient_data)

  phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
  phys_data, phys_fs = load_signal_data(phys_file)
  phys_feat = extract_physiological_features(phys_data, phys_fs, csv_path=csv_path)

  algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
  algo_data, algo_fs = load_signal_data(algo_file)
  algo_feat = extract_algorithmic_annotations_features(algo_data)

  feature_structure = [
      ("Demographics", dem_feat.shape[0], dem_feat),
      ("Physiological", phys_feat.shape[0], phys_feat),
      ("Algorithmic", algo_feat.shape[0], algo_feat)
  ]

  print("\n Feature Vector Breakdown ")
  total_size = 0
  for name, size, vals in feature_structure:
      print(f"{name:15} | Size: {size:3} | Values: {vals[:5]}...") # Showing first 5 values
      total_size += size

  print("-" * 40)
  print(f"Total Combined Feature Length: {total_size}")
  print("\nActual content of the features")
  print(f"Physiological features {phys_feat}, type {type(phys_feat)}")
  print(f"Demographic features {dem_feat}, type {type(dem_feat)}")
  print(f"Algorithmic features {algo_feat}, {type(algo_feat)}")
  print("\n Detailed Demographic Mapping ")
  race_labels = ['Asian', 'Black', 'Other', 'Unavailable', 'White']
  print(f"Age:  {dem_feat[0]}")
  print(f"Sex:  Female={dem_feat[1]}, Male={dem_feat[2]}, Other={dem_feat[3]}")
  for i, label in enumerate(race_labels):
      print(f"Race - {label}: {dem_feat[4+i]}")
  print(f"BMI:  {dem_feat[9]}")
  '''

train_model(data_folder, model_folder, verbose=True)
trained_model_object = load_model(model_folder, verbose=True)

print(patient_metadata_list)
results = []
for i in range(len(patient_metadata_list)):
    single_record = patient_metadata_list[i]

    binary_out, prob_out = run_model(
        model=trained_model_object,
        record=single_record,
        data_folder=data_folder,
        verbose=True
    )

    # print(f"Prediction for {single_record[HEADERS['bids_folder']]}, {single_record[HEADERS['site_id']]},{single_record[HEADERS['session_id']]}: {binary_out} (Prob: {prob_out:.2f})")
    results.append({
        'BDSPPatientID': single_record[HEADERS['bids_folder']],
        'Cognitive_Impairment': int(binary_out),
        'Cognitive_Impairment_Probability': float(prob_out)
    })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

# python evaluate_model.py -d training_set/demographics.csv -o predictions.csv -s scores.txt