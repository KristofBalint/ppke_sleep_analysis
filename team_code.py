#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from tqdm import tqdm

from helper_code import *
import pandas as pd
from feature_extraction import assess_signal_quality, ALL_FEATURE_COLUMNS, \
    get_master_feature_names, UNIQUE_CHANNELS, get_category_map, valid_features, \
    stratified_pca_feature_selection, feature_debug
import mne
from xgboost import XGBClassifier
from sklearn import metrics
import traceback
from utils import DebugMessageCollector
from SleepFMIntegrator import *


team_debug = DebugMessageCollector("team_code")

################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')
#SleepFM setup
checkpoint_path = Path(SCRIPT_DIR) / "sleepfm" / "checkpoints" / "model_diagnosis" / "best.pth"
config = SleepFMConfig(checkpoint_path=checkpoint_path)
extractor = FeatureExtractorStrategy(config)
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    # Find the data files.
    team_debug.update_success("entered", message="train_model execution started")
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # Iterate over the records to extract the features and labels.
    features = list()
    labels = list()
    sleepfm_embeddings_array=list()

    category_lookup = get_category_map(DEFAULT_CSV_PATH)

    pbar = tqdm(range(num_records), desc="Extracting Features", unit="record", disable=not verbose)
    backup = "backup.csv"
    if os.path.exists(backup):  # If by any accident already exists, removes it
        os.remove(backup)
    header_written = False
    for i in pbar:
        try:
            team_debug.update_success("start", iteration=i)
            # Extract identifiers for this specific record
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({"patient": patient_id})

            # Load the patient data.
            patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            team_debug.update_success("patient_loaded", patient_id=patient_id, iteration=i)
            demographic_features = extract_demographic_features(patient_data)
            team_debug.update_success("demographics loaded", patient_id=patient_id, iteration=i)

            # Load signal data.

            # Load the physiological signal.
            physiological_data_file = os.path.join(data_folder,
                                                   PHYSIOLOGICAL_DATA_SUBFOLDER,
                                                   site_id,
                                                   f"{patient_id}_ses-{session_id}.edf")
            # --- Check if the file actually exists before proceeding ---
            if not os.path.exists(physiological_data_file):
                if verbose:
                    print(
                        f"  ! Missing physiological data for {patient_id}. Skipping...")
                continue  # skip record

            # Load the algorithmic annotations. It needs to be loaded before extracting phyisiological features in our code, as hour code requires the hypnogram from the algorithmic annotations
            algorithmic_annotations_file = os.path.join(data_folder,
                                                        ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                                                        site_id,
                                                        f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            algorithmic_annotations, algorithmic_fs = load_signal_data(
                algorithmic_annotations_file)
            algorithmic_features = extract_algorithmic_annotations_features(
                algorithmic_annotations)
            team_debug.update_success("algorithmic annotations loaded", patient_id=patient_id, iteration=i)

            # Load physiological data
            physiological_data, physiological_fs = load_signal_data(
                physiological_data_file)
            team_debug.update_success("physiological loaded", patient_id=patient_id, iteration=i)
            hypno = None
            if "stage_caisr" in algorithmic_annotations:
                hypno = algorithmic_annotations["stage_caisr"]
                # print(f"Hypnogram length before processing{len(hypno)}")
                hypno = np.round(hypno.flatten()).astype(int)
            else:
                hypno = None
            physiological_features = extract_physiological_features(
                physiological_data, physiological_fs,
                csv_path=csv_path, category_lookup=category_lookup,
                hypnogram=hypno, patient_id=patient_id,
                session_id=session_id)  # This function can rename, re-reference, resample, etc. the signal data.
            team_debug.update_success("feature extraction returned", patient_id=patient_id, iteration=i)
            pd.DataFrame({
                "patient": [patient_id]
            }).to_csv(backup, mode="a", header=not header_written, index=False)
            header_written = True
            team_debug.update_success("backup written", patient_id=patient_id, iteration=i)
            # Load the human annotations; these data will not be available in the hidden validation and test sets.
            human_annotations_file = os.path.join(data_folder,
                                                  HUMAN_ANNOTATIONS_SUBFOLDER,
                                                  site_id,
                                                  f"{patient_id}_ses-{session_id}_expert_annotations.edf")
            human_annotations, human_fs = load_signal_data(
                human_annotations_file)
            human_features = extract_human_annotations_features(
                human_annotations)

            # Load the diagnoses; these data will not be available in the hidden validation and test sets.
            diagnosis_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            label = load_diagnoses(diagnosis_file, patient_id)
            team_debug.update_success("diagnosis loaded", patient_id=patient_id, iteration=i)

            # Store the features and labels, but
            # the human annotations are not available on the hidden validation and test sets, but you
            # may want to consider how to use them for training.
            print("CONTENT AND TYPE OF LABEL")
            print(label)
            print(type(label))
            if int(label) in (0, 1):
                # physiological_vector=physiological_features.to_numpy().flatten().astype(np.float32)
                feat_df = physiological_features.copy()
                feat_df["Cognitive_Impairment"] = label
                feat_df["Age"] = demographic_features[0]
                feat_df["Sex_Female"] = demographic_features[1]
                feat_df["Sex_Male"] = demographic_features[2]
                feat_df["Race_Asian"] = demographic_features[4]
                feat_df["Race_Black"] = demographic_features[5]
                feat_df["Race_Others"] = demographic_features[6]
                feat_df["Race_Unavailable"] = demographic_features[7]
                feat_df["Race_White"] = demographic_features[8]
                feat_df["BMI"] = demographic_features[9]
                feat_df["Ahi_auto"] = algorithmic_features[0]
                feat_df["Ar_auto"] = algorithmic_features[1]
                feat_df["Limb_auto"] = algorithmic_features[2]
                feat_df["W_pct"] = algorithmic_features[3]
                feat_df["N1_pct"] = algorithmic_features[4]
                feat_df["N2_pct"] = algorithmic_features[5]
                feat_df["N3_pct"] = algorithmic_features[6]
                feat_df["R_pct"] = algorithmic_features[7]
                feat_df["efficiency"] = algorithmic_features[8]
                feat_df["W_prob"] = algorithmic_features[9]
                feat_df["N3_prob"] = algorithmic_features[10]
                feat_df["Ar_prob"] = algorithmic_features[11]
                # feat_df.to_csv(backup,mode="a",header=not header_written,index=False)
                # header_written=True
                features.append(feat_df)
                team_debug.update_success("patient appended", patient_id=patient_id, iteration=i)
                if len(features) == 0:
                    team_debug.update_failure("patient appended", patient_id=patient_id, iteration=i)

                # features.append(np.hstack(    [demographic_features, physiological_vector,algorithmic_features]))
                labels.append(label)
            '''
            modality_signals = {}
    
            for raw_channel_name, channel_signal in physiological_data.items():
                # Retrieve modality category (e.g., 'BAS', 'RESP', 'EKG', 'EMG')
                category = category_lookup.get(raw_channel_name)
                    
                if category:
                    if category not in modality_signals:
                        modality_signals[category] = {}
                    # Store signal under its category using the raw channel name as key
                    modality_signals[category][raw_channel_name] = channel_signal    
            '''
            #--------------------------------------------------------------
            # Extract embeddings, currently non-used, until problems with our code is identified
            #sleepfm_emb = extractor.extract_features(modality_signals)
            #sleep_fm_embeddings_array.append(sleepfm_emb)
            # --------------------------------------------------------------
            # if 'physiological_data' in locals(): del physiological_data
            # if 'algorithmic_annotations' in locals(): del algorithmic_annotations

        except Exception as e:
        # If an error occurs (e.g., a record is corrupted), log it and move to the next
            error_stack = traceback.format_exc()
            team_debug.update_failure("start", patient_id=patient_id if 'patient_id' in locals() else "Unknown", iteration=i,
                                  message=str(e), exc=e)
            tqdm.write(f"  !!! Error processing record {i + 1} ({patient_id}): {e}")
            tqdm.write(error_stack)
            continue

    pbar.close()

    params = {'max_depth': 7, 'learning_rate': 0.030842680992267595, 'n_estimators': 963, 'subsample': 0.668562285184131,
          'colsample_bytree': 0.6664590708996059, 'gamma': 1.0645185377275237, 'min_child_weight': 16,
          'reg_alpha': 0.0600354240141498, 'reg_lambda': 35.771321493090454}

    print(f"DEBUG: shape of features {len(features)}")

    if len(features) > 0:
        full_df = pd.concat(features, ignore_index=True)
        team_debug.update_success("feature dataframe created", message="Aggregated feature vector complete")
    else:
        # List of all required backup CSVs
        required_backups = [
            "backup_eeg.csv", "backup_emg.csv", "backup_ecg.csv",
            "backup_resp.csv", "backup_eog.csv", "backup_transitions.csv",
            "backup_synchrony.csv"
        ]

        # Check if any backup files are missing and raise a specific exception
        missing_backups = [b for b in required_backups if not os.path.exists(b)]
        if missing_backups:
            err_msg = (
                f"Extraction failed! The following backup CSV files were NOT created/found: {missing_backups}\n"
                f"\n--- COLLECTOR DEBUG SUMMARY ---\n{team_debug.summary()}"
            )
            raise RuntimeError(
                err_msg
            )
    
        print("In-memory extraction produced no features. Reconstructing full_df from backup CSVs...")

        # 1. Load base signal CSVs
        dfs = []
        for cat_file in ["backup_eeg.csv", "backup_emg.csv", "backup_ecg.csv", "backup_resp.csv", "backup_eog.csv"]:
            df_cat = pd.read_csv(cat_file)
            if not df_cat.empty:
                dfs.append(df_cat)
    
        if not dfs:
            err_msg = f"Backup CSV files exist but are all empty.\n\n--- COLLECTOR DEBUG SUMMARY ---\n{team_debug.summary()}"
            raise RuntimeError(err_msg)
    
        combined_signal_df = pd.concat(dfs, ignore_index=True)
    
        trans_df = pd.read_csv("backup_transitions.csv")
        if not trans_df.empty:
            merge_keys = ["subject", "session", "channel", "stage", "transition_id", "transition_type"]
            combined_signal_df = combined_signal_df.merge(trans_df, on=merge_keys, how="left", suffixes=("", "_dup"))
            combined_signal_df = combined_signal_df.loc[:, ~combined_signal_df.columns.str.endswith("_dup")]
    
        sync_df = pd.read_csv("backup_synchrony.csv")
        if not sync_df.empty:
            sync_keys = ["subject", "session", "channel", "stage"]
            combined_signal_df = combined_signal_df.merge(sync_df, on=sync_keys, how="left", suffixes=("", "_dup"))
            combined_signal_df = combined_signal_df.loc[:, ~combined_signal_df.columns.str.endswith("_dup")]
    
        # 4. Pivot and rename (identical to extract_physiological_features logic)
        quality_df = combined_signal_df.reindex(columns=ALL_FEATURE_COLUMNS)
        feature_df = quality_df.pivot(
            index=['subject', 'session'],
            columns=['category', 'channel', 'stage', 'transition_id']
        )
    
        valid_cols = [col for col in feature_df.columns if col[0] in valid_features.get(col[1], [])]
        feature_df = feature_df[valid_cols]
    
        feature_df.columns = [
            f"{col[1]}-{col[2]}-{col[3]}-{col[4]}-{col[0]}" if isinstance(col, tuple) else col
            for col in feature_df.columns.values
        ]
    
        MASTER_FEATURE_NAMES = get_master_feature_names(category_lookup)
        feature_df = feature_df.reindex(columns=MASTER_FEATURE_NAMES, fill_value=np.nan)
    
        feature_df["subject"] = feature_df.index.get_level_values(0)
        feature_df["session"] = feature_df.index.get_level_values(1)
    
        full_df = feature_df.reset_index(drop=True)
        team_debug.update_success("feature dataframe created",
                                  message="Reconstructed feature vector from backup files complete")

    
    y = full_df["Cognitive_Impairment"]
    
    ranked_features = stratified_pca_feature_selection(full_df, y, params)
    
    percentages = list(range(50, 4, -5))
    
    best_auc = 0
    best_features = None
    
    print(f"Lenght of the ranked features{len(ranked_features)}")
    for p in percentages:
    
        n_keep = int(len(ranked_features) * (p / 100))
        selected = ranked_features[:n_keep]
    
        X = full_df[selected]
        y = full_df["Cognitive_Impairment"]
    
        print(f"Shape of X {X.shape} and y {y.shape} in percentage {p}")
        model = XGBClassifier(**params, tree_method="hist", objective="binary:logistic",
                              eval_metric="auc", enable_categorical=True, random_state=42)
        model.fit(X, y)
    
        preds = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, preds)
    
        if auc > best_auc:
            best_auc = auc
            best_features = selected
    
    lengths = [len(f) for f in features]
    print(f"Unique feature lengths found: {set(lengths)}")
    
    selected_features = best_features
    #selected_features = [f for f in full_df.columns if "resp" in f.lower()] # temporary replacement, until I find wtf is wrong with stratified_pca_selection

    #X_tabular=full_df[selected_features].to_numpy()
    #X_combined=np.hstack([X_tabular,sleepfm_embeddings_array])

    X = full_df[selected_features]
    y = full_df["Cognitive_Impairment"]
    
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    
    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')
    
    print(X.shape)
    print(y.shape)
    
    X_np = X.to_numpy().astype(np.float32)
    y_np = y.to_numpy().astype(bool)
    print(X_np.shape)
    print(y_np.shape)
    
    team_debug.update_success("training started", message="Starting model fitting")
    model = XGBClassifier(**params,objective="binary:logistic",eval_metric="auc",tree_method="hist",enable_categorical=True,random_state=42).fit(X_np, y_np)
    #model = XGBClassifier(**params,objective="binary:logistic",eval_metric="auc",tree_method="hist",enable_categorical=True,random_state=42).fit(X_combined, y_np)
    
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Save the model.
    save_model(model_folder, model, selected_features)
    
    if verbose:
        print('Done.')
        print()


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    # Load the model.
    feature_list = model['feature_list']
    model = model['model']

    # Extract identifiers from the record dictionary
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Load the patient data.
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    demographic_features = extract_demographic_features(patient_data)

    # Load Algorithmic Annotations, this needs to be loaded first as the hypnogram is needed for the physiological features
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id,
                             f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    hypno = None
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        if "stage_caisr" in algo_data:
            hypno = np.round(algo_data["stage_caisr"].flatten()).astype(int)
        algorithmic_features = extract_algorithmic_annotations_features(
            algo_data)
    else:
        # Fallback to zeros (length 12)
        algorithmic_features = np.zeros(12)

    # Load signal data.
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
                             f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        # Ensure csv_path is accessible or defined
        physiological_features = extract_physiological_features(phys_data,
                                                                phys_fs,
                                                                hypnogram=hypno, patient_id=patient_id,
                                                                session_id=session_id)
    else:
        # Fallback to zeros if file is missing (length 49 in original code)
        # physiological_features = np.zeros(49)
        MASTER_FEATURE_NAMES = get_master_feature_names(get_category_map(DEFAULT_CSV_PATH))
        physiological_features = np.zeros(
            len(MASTER_FEATURE_NAMES))

    feat_df = physiological_features.copy()
    feat_df["Age"] = demographic_features[0]
    feat_df["Sex_Female"] = demographic_features[1]
    feat_df["Sex_Male"] = demographic_features[2]
    feat_df["Race_Asian"] = demographic_features[4]
    feat_df["Race_Black"] = demographic_features[5]
    feat_df["Race_Others"] = demographic_features[6]
    feat_df["Race_Unavailable"] = demographic_features[7]
    feat_df["Race_White"] = demographic_features[8]
    feat_df["BMI"] = demographic_features[9]
    feat_df["Ahi_auto"] = algorithmic_features[0]
    feat_df["Ar_auto"] = algorithmic_features[1]
    feat_df["Limb_auto"] = algorithmic_features[2]
    feat_df["W_pct"] = algorithmic_features[3]
    feat_df["N1_pct"] = algorithmic_features[4]
    feat_df["N2_pct"] = algorithmic_features[5]
    feat_df["N3_pct"] = algorithmic_features[6]
    feat_df["R_pct"] = algorithmic_features[7]
    feat_df["efficiency"] = algorithmic_features[8]
    feat_df["W_prob"] = algorithmic_features[9]
    feat_df["N3_prob"] = algorithmic_features[10]
    feat_df["Ar_prob"] = algorithmic_features[11]
    selected_features = [f for f in full_df.columns if "resp" in f.lower()]
    feat_df=feat_df[selected_features]
    features = feat_df[feature_list].to_numpy().flatten().astype(np.float32)
    
    # features = np.hstack([demographic_features, physiological_features,algorithmic_features]).reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.

    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).

    Returns:
        np.array: A feature vector of length 11:
            - [0]: Age (Continuous)
            - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
            - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
            - [9]: BMI (Continuous)
    """
    # 1. Age Feature (1 dimension)
    # Convert 'Age' to a float; default to 0 if missing
    age = np.array([load_age(data)])

    # 2. Sex One-Hot Encoding (3 dimensions: Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data)
    sex_vec = np.zeros(3)
    if sex == 'Female':
        sex_vec[0] = 1  # Index 0: Female
    elif sex == 'Male':
        sex_vec[1] = 1  # Index 1: Male
    else:
        sex_vec[2] = 1  # Index 2: Other/Unknown

        # 3. Race One-Hot Encoding (6 dimensions)
        # Standardizes the raw text into one of six categories using the helper function
        # race_category = get_standardized_race(data).lower() #FOR TWO FUCKING MONTHS I DID NOT NOTICE IT WAS REFACTORED
        race_category = load_race(data, standardize=True).lower()
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
    race_vec[race_mapping.get(race_category, 2)] = 1

    # 4. Body Mass Index (BMI) Feature (1 dimension)
    # Extracts the pre-calculated mean BMI; handles strings, NaNs, and missing keys
    bmi = np.array([load_bmi(data)])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)

    return np.concatenate([age, sex_vec, race_vec, bmi])


def extract_physiological_features(physiological_data, physiological_fs,
                                   csv_path=DEFAULT_CSV_PATH,
                                   category_lookup=get_category_map(DEFAULT_CSV_PATH),
                                   hypnogram=None, patient_id="Unknown", session_id="Unknown"):
    """
    Standardizes channels and extracts statistical/spectral features.
    """
    feature_debug.update_success("extract_physiological_features", patient_id=patient_id,
                                 message="Starting physiological extraction")
    original_labels = list(physiological_data.keys())

    # Step 1: Load rules and standardize names
    # Note: Use script-relative path or absolute path for robustness
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

    # Step 2: Apply renaming to BOTH signals and their corresponding FS
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        processed_fs[new_label] = physiological_fs.get(old_label, 200.0)  # Default to 200 if missing

    feature_debug.update_success("channels standardized", patient_id=patient_id, message="Renamed channels standard")

    class FakeRaw:
        def __init__(self, data_dict, fs_dict):
            self._data = data_dict
            self._fs = fs_dict
            self.ch_names = list(data_dict.keys())
            self.info = {"sfreq": list(fs_dict.values())[0] if fs_dict else 200}

        def __getitem__(self, key):
            sig = self._data[key]
            return sig.reshape(1, -1), None

    raw_psg = FakeRaw(processed_channels, processed_fs)
    quality_df = assess_signal_quality(raw_psg, category_lookup, patient_id=patient_id, session_id=session_id,
                                       hypnogram=hypnogram)
    # quality_df["subject"]=patient_id
    # quality_df["session"]=session_id
    print(f"Shape originally {quality_df.shape}")
    quality_df = quality_df.reindex(columns=ALL_FEATURE_COLUMNS)
    if 'subject' not in quality_df.columns:
        print("No subject name")
    if 'session' not in quality_df.columns:
        print("No session name")
    print(f"Shape after reindexing {quality_df.shape}")
    feature_df = quality_df.pivot(index=['subject', 'session'],
                                  columns=['category', 'channel', 'stage', 'transition_id'])
    feature_debug.update_success("pivot complete", patient_id=patient_id,
                                 message="Pivot transformation matrix constructed")
    print(f"Shape after pivoting {feature_df.shape}")
    valid_cols = [col for col in feature_df.columns
                  if col[0] in valid_features.get(col[1], [])]
    feature_df = feature_df[valid_cols]
    print(f"Shape after filtering invalid columns {feature_df.shape}")
    feature_df.columns = [f"{col[1]}-{col[2]}-{col[3]}-{col[4]}-{col[0]}" if isinstance(col, tuple) else col
                          for col in feature_df.columns.values]

    MASTER_FEATURE_NAMES = get_master_feature_names(category_lookup)
    feature_df = feature_df.reindex(columns=MASTER_FEATURE_NAMES, fill_value=np.nan)

    feature_df["subject"] = feature_df.index.get_level_values(0)
    feature_df["session"] = feature_df.index.get_level_values(1)
    feature_debug.update_success("reindex complete", patient_id=patient_id,
                                 message="Feature columns reindexed to master format")
    return feature_df.reset_index(drop=True)


def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR outputs.
    Output vector length: 12
    """
    if not algo_data:
        return np.zeros(12)

    features = []

    # --- 1. Respiratory & Arousal Event Densities ---
    # Total duration in hours (assuming 1Hz for event traces)
    # If the signal exists, we calculate events per hour (Index)
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0

    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return 0.0

        sig = algo_data[key].astype(float)
        # Create a binary mask: 1 if there is an event, 0 if not
        binary_sig = (sig > 0).astype(int)

        # Detect rising edges: 0 to 1 transition
        # diff will be 1 at the start of an event, -1 at the end
        diff = np.diff(binary_sig, prepend=0)
        num_events = np.count_nonzero(diff == 1)

        return num_events / total_hours

    ahi_auto = count_discrete_events(
        'resp_caisr')  # Automated Apnea-Hypopnea Index
    arousal_auto = count_discrete_events(
        'arousal_caisr')  # Automated Arousal Index
    limb_auto = count_discrete_events(
        'limb_caisr')  # Automated Limb Movement Index

    features.extend([ahi_auto, arousal_auto, limb_auto])

    # --- 2. Sleep Architecture (from stage_caisr) ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R (or similar mapping)
    stages = algo_data.get('stage_caisr', np.array([]))
    # Filter out invalid/background values (like the 9.0 in your sample)
    valid_stages = stages[stages < 9.0]

    if len(valid_stages) > 0:
        total_epochs = len(valid_stages)
        # Percentage of each stage
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)

        # Sleep Efficiency: (N1+N2+N3+R) / Total
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Model Confidence / Uncertainty ---
    # Mean probability of Wake and REM (indicators of sleep stability)
    # We use the raw probability traces
    prob_w = np.mean(algo_data.get('caisr_prob_w', [0]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [0]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [0]))

    # Standardize '9.0' or other filler values to 0
    clean_prob = lambda x: x if x < 1.0 else 0.0
    features.extend(
        [clean_prob(prob_w), clean_prob(prob_n3), clean_prob(prob_arous)])

    return np.array(features)


def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    Output vector length: 12 (to match algorithmic feature length)
    """
    # If data is missing (common in hidden test sets), return a zero vector
    if not human_data or 'resp_expert' not in human_data:
        return np.zeros(12)

    features = []

    # --- 1. Human Event Indices (Events per Hour) ---
    # Total duration in hours based on 1Hz signal
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours = total_seconds / 3600.0

    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return 0.0
        sig = (human_data[key] > 0).astype(int)
        # Identify the start of each continuous event block
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    ahi_human = count_discrete_events('resp_expert')  # Human AHI
    arousal_human = count_discrete_events(
        'arousal_expert')  # Human Arousal Index
    limb_human = count_discrete_events('limb_expert')  # Human PLMI

    features.extend([ahi_human, arousal_human, limb_human])

    # --- 2. Human Sleep Architecture ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R, 5=Unknown/Movement
    stages = human_data.get('stage_expert', np.array([]))

    # Filter out label 5 (often used by experts for movement/unscored)
    valid_mask = (stages < 9.0)
    valid_stages = stages[valid_mask]

    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Fragmentation & Stability (Replacing Probabilities) ---
    # These metrics quantify how "broken" the sleep is, which is a key marker.
    if len(valid_stages) > 1:
        # Number of stage transitions
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        # Wake After Sleep Onset (WASO) proxy: non-zero stages followed by zero
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        # REM Latency (epochs until first REM)
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = rem_indices[0] if len(rem_indices) > 0 else 0.0
    else:
        transitions = waso_minutes = rem_latency = 0.0

    features.extend([transitions, waso_minutes, rem_latency])

    return np.array(features)


# Save your trained model.
def save_model(model_folder, model, feature_list):
    d = {'model': model, 'feature_list': feature_list}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
