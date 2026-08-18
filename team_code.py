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
from dataclasses import asdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
import traceback
from pathlib import Path
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
checkpoint_path = Path(SCRIPT_DIR) / "SleepFM" / "checkpoints" / "model_diagnosis" / "best.pth"
HYBRID_MODEL_FILENAME = "hybrid_model.pt"
HYBRID_METADATA_FILENAME = "hybrid_metadata.sav"
XGB_MODEL_FILENAME = "xgb_model.sav"
XGB_METADATA_FILENAME = "xgb_metadata.sav"

# Pipeline switches
# 1 = Handcrafted feature extraction -> XGBoost (no stratified PCA/RFE ranking)
# 2 = Handcrafted feature extraction -> stratified PCA/RFE selection -> XGBoost
# 3 = Full handcrafted + stratified selection + SleepFM hybrid pipeline
PIPELINE_MODE = 3

# Used only by PIPELINE_MODE == 1. Valid examples:
FEATURE_CATEGORY = None
# When FEATURE_CATEGORY is set, retain demographics and CAISR annotation features too.
INCLUDE_PATIENT_LEVEL_FEATURES_WITH_CATEGORY = True

PIPELINE_NAMES = {
    1: "handcrafted_xgboost",
    2: "stratified_xgboost",
    3: "handcrafted_sleepfm_hybrid",
}

PATIENT_LEVEL_FEATURE_COLUMNS = [
    "Age", "Sex_Female", "Sex_Male",
    "Race_Asian", "Race_Black", "Race_Others", "Race_Unavailable", "Race_White",
    "BMI",
    "Ahi_auto", "Ar_auto", "Limb_auto",
    "W_pct", "N1_pct", "N2_pct", "N3_pct", "R_pct", "efficiency",
    "W_prob", "N3_prob", "Ar_prob",
]

SLEEPFM_MODALITY_MAP = {
    "eeg": "BAS",
    "eog": "BAS",
    "resp": "RESP",
    "ecg": "EKG",
    "leg emg": "EMG",
    "chin emg": "EMG",
    "emg": "EMG",
}
config = SleepFMConfig(checkpoint_path=checkpoint_path)

#helper functions

def add_patient_level_features(physiological_features,demographic_features,algorithmic_features,label=None,):
    feat_df = physiological_features.copy()

    if label is not None:
        feat_df["Cognitive_Impairment"] = int(label)

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

    return feat_df
    
def choose_feature_subset(full_df,ranked_features,params,):
    groups = full_df["subject"].astype(str)
    y = full_df["Cognitive_Impairment"].astype(int)
    splitter = GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42,)
    
    train_idx, val_idx = next(splitter.split(full_df, y, groups))
    percentages = list(range(50, 4, -5))

    best_auc = -np.inf
    best_features = None

    for percentage in percentages:
        n_keep = max(1,int(len(ranked_features) * percentage / 100))
        selected = list(ranked_features[:n_keep])
        # Remove anything that somehow disappeared.
        selected = [f for f in selected if f in full_df.columns]
        if not selected:
            continue
        X_train = full_df.iloc[train_idx][selected]
        X_val = full_df.iloc[val_idx][selected]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        selector_model = XGBClassifier(
            **params,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="auc",
            enable_categorical=True,
            random_state=42,
        )

        selector_model.fit(X_train,y_train,)
        probs = selector_model.predict_proba(X_val)[:, 1]
        auc = metrics.roc_auc_score(y_val,probs,)
        if auc > best_auc:
            best_auc = auc
            best_features = selected

    if not best_features:
        raise RuntimeError(
            "No handcrafted features survived feature selection."
        )

    return best_features
    
def build_sleepfm_modalities(physiological_data,physiological_fs,category_lookup,csv_path=DEFAULT_CSV_PATH,):
    """
    Convert Challenge physiological signal dictionaries into the modality
    dictionaries expected by SleepFMPreprocessor.

    Returns
    -------
    modality_signals
        {
            "BAS":  {channel: signal},
            "RESP": {...},
            "EKG":  {...},
            "EMG":  {...},
        }
    modality_fs
        Same hierarchy, containing channel sampling frequencies.
    """

    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    original_labels = list(physiological_data.keys())
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules,)

    modality_signals = {
        "BAS": {},
        "RESP": {},
        "EKG": {},
        "EMG": {},
    }

    modality_fs = {
        "BAS": {},
        "RESP": {},
        "EKG": {},
        "EMG": {},
    }

    for old_name, signal in physiological_data.items():
        if old_name in cols_to_drop:
            continue

        standard_name = rename_map.get(old_name,old_name.lower(), )
        category = category_lookup.get(standard_name)

        if category is None:
            continue

        sleepfm_modality = SLEEPFM_MODALITY_MAP.get(str(category).lower())
        if sleepfm_modality is None:
            continue

        modality_signals[sleepfm_modality][standard_name] = signal
        modality_fs[sleepfm_modality][standard_name] = (physiological_fs.get(old_name, 200.0))

    return modality_signals, modality_fs

def validate_pipeline_configuration():
    if PIPELINE_MODE not in PIPELINE_NAMES:
        raise ValueError( f"PIPELINE_MODE must be one of {sorted(PIPELINE_NAMES)}, got {PIPELINE_MODE}. Debug messages: {team_debug.summary()}")
    if PIPELINE_MODE != 1 and FEATURE_CATEGORY is not None:
        print(
            f"Warning: FEATURE_CATEGORY={FEATURE_CATEGORY!r} is ignored in "
            f"PIPELINE_MODE={PIPELINE_MODE}."
        )


def get_xgb_params():
    """Single source of truth for the XGBoost hyperparameters."""
    return {
        'max_depth': 7,
        'learning_rate': 0.030842680992267595,
        'n_estimators': 963,
        'subsample': 0.668562285184131,
        'colsample_bytree': 0.6664590708996059,
        'gamma': 1.0645185377275237,
        'min_child_weight': 16,
        'reg_alpha': 0.0600354240141498,
        'reg_lambda': 35.771321493090454,
    }


def get_available_handcrafted_features(full_df, category=None, include_patient_level=True):
    """
    Return usable handcrafted columns from the patient-level dataframe.

    If category is None, all physiological + patient-level features are used.
    If category is set, only physiological columns beginning with that category
    are retained, optionally together with demographics/CAISR features.
    """
    excluded = {"Cognitive_Impairment", "subject", "session"}
    all_features = [c for c in full_df.columns if c not in excluded]

    if category is None:
        selected = all_features
    else:
        category = str(category).strip().lower()
        if category == "emg":
            physiological = [c for c in all_featuresif c.lower().startswith("leg emg-") or c.lower().startswith("chin emg-")]
        else:
            prefix = category + "-"
            physiological = [c for c in all_features if c.lower().startswith(prefix)]

        selected = physiological
        if include_patient_level:
            selected += [c for c in PATIENT_LEVEL_FEATURE_COLUMNS if c in full_df.columns]

    # Preserve dataframe column order and remove duplicate names.
    wanted = set(selected)
    selected = [c for c in full_df.columns if c in wanted and c not in excluded]
    # Empty/all-NaN columns provide no information and can upset downstream preprocessing.
    selected = [c for c in selected if full_df[c].notna().any()]

    if not selected:
        raise RuntimeError(f"No handcrafted features available for category={category!r}.")
    return selected


def fit_xgboost_pipeline(full_df, labels, selected_features, params):
    """Fit an imputer + XGBoost model for modes 1 and 2."""
    X = full_df[selected_features].copy()
    y = np.asarray(labels, dtype=np.int64)

    # Convert bools explicitly so train and inference representations are stable.
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(np.int8)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imputed = imputer.fit_transform(X)

    model = XGBClassifier(
        **params,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_imputed, y)
    return model, imputer


def save_xgb_model(model_folder, model, selected_features, imputer,pipeline_mode, feature_category):
    os.makedirs(model_folder, exist_ok=True)
    # Prevent load_model() from seeing stale artifacts from another mode.
    for stale_name in (HYBRID_MODEL_FILENAME, HYBRID_METADATA_FILENAME):
        stale_path = os.path.join(model_folder, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)
    joblib.dump(model, os.path.join(model_folder, XGB_MODEL_FILENAME), protocol=0)
    metadata = {
        "model_type": "XGBClassifier",
        "pipeline_mode": int(pipeline_mode),
        "pipeline_name": PIPELINE_NAMES[int(pipeline_mode)],
        "feature_category": feature_category,
        "include_patient_level_features_with_category": INCLUDE_PATIENT_LEVEL_FEATURES_WITH_CATEGORY,
        "selected_features": list(selected_features),
        "imputer": imputer,
        "threshold": 0.5,
    }
    joblib.dump(metadata, os.path.join(model_folder, XGB_METADATA_FILENAME), protocol=0)


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    validate_pipeline_configuration()
    pipeline_name = PIPELINE_NAMES[PIPELINE_MODE]
    team_debug.update_success(
        "entered",
        message=(
            f"train_model execution started; mode={PIPELINE_MODE} "
            f"({pipeline_name}); category={FEATURE_CATEGORY}"
        ),
    )

    if verbose:
        print(f"Pipeline mode {PIPELINE_MODE}: {pipeline_name}")
        if PIPELINE_MODE == 1:
            print(f"Feature category: {FEATURE_CATEGORY or 'all'}")
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)
    team_debug.update_success("patients found", message=f"num_records={num_records}")
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    category_lookup = get_category_map(csv_path)
    handcrafted_rows = []
    labels = []
    sleepfm_rows = []

    # SleepFM exists only in the full hybrid pipeline.
    config = None
    sleepfm_encoder = None
    sleepfm_preprocessor = None
    if PIPELINE_MODE == 3:
        try:
            config = SleepFMConfig(
                checkpoint_path=checkpoint_path,
                freeze_backbone=True,
                debug=False,
                verbose=verbose,
            )
            # SleepFMEncoder currently checks config.load_pretrained although
            # SleepFMConfig does not declare that field. load_sleepfm() performs
            # checkpoint loading itself, so keep constructor-side loading off.
            setattr(config, "load_pretrained", False)
            sleepfm_encoder = load_sleepfm(config)
            sleepfm_preprocessor = SleepFMPreprocessor(config)
            team_debug.update_success(
                "sleepfm initialized",
                message=(
                    f"checkpoint={checkpoint_path}; device={config.device}; "
                    f"embedding_dim={config.embedding_dimension}"
                ),
            )
        except Exception as e:
            team_debug.update_failure("sleepfm initialized", message=str(e), exc=e)
            raise RuntimeError(
                "SleepFM initialization failed.\n" + team_debug.summary()
            ) from e

    if verbose:
        print('Extracting features and labels from the data...')

    pbar = tqdm(
        range(num_records),
        desc="Extracting Features",
        unit="record",
        disable=not verbose,
    )
    backup = "backup.csv"
    if os.path.exists(backup):
        os.remove(backup)
    header_written = False

    for i in pbar:
        patient_id = "Unknown"
        try:
            team_debug.update_success("start", iteration=i)
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]
            team_debug.update_success( "patient started", patient_id=patient_id, iteration=i, message=f"site={site_id}; session={session_id}")
            if verbose:
                pbar.set_postfix({"patient": patient_id})

            # Demographics.
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            demographic_features = extract_demographic_features(patient_data)
            team_debug.update_success("demographics loaded", patient_id=patient_id, iteration=i )

            # Algorithmic annotations / hypnogram.
            algorithmic_annotations_file = os.path.join( data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id,
                                                        f"{patient_id}_ses-{session_id}_caisr_annotations.edf",)
            algorithmic_annotations, _ = load_signal_data(algorithmic_annotations_file)
            algorithmic_features = extract_algorithmic_annotations_features(algorithmic_annotations)
            if "stage_caisr" in algorithmic_annotations:
                hypno = np.round(algorithmic_annotations["stage_caisr"].flatten()).astype(int)
            else:
                hypno = None
            team_debug.update_success(
                "algorithmic annotations loaded",patient_id=patient_id,iteration=i,
                message=f"hypnogram_length={0 if hypno is None else len(hypno)}",
            )

            # Physiological signals.
            physiological_data_file = os.path.join(data_folder,PHYSIOLOGICAL_DATA_SUBFOLDER,site_id,f"{patient_id}_ses-{session_id}.edf" )
            if not os.path.exists(physiological_data_file):
                if verbose:
                    tqdm.write( f"  ! Missing physiological data for {patient_id}. Skipping...")
                continue

            physiological_data, physiological_fs = load_signal_data(physiological_data_file)
            physiological_features = extract_physiological_features(
                physiological_data,
                physiological_fs,
                csv_path=csv_path,
                category_lookup=category_lookup,
                hypnogram=hypno,
                patient_id=patient_id,
                session_id=session_id,
                category_filter=(FEATURE_CATEGORY if PIPELINE_MODE == 1 else None),
            )
            team_debug.update_success(
                "feature extraction returned",
                patient_id=patient_id,
                iteration=i,
                message=(
                    f"shape={physiological_features.shape}; "
                    f"non_nan={int(physiological_features.notna().sum().sum())}"
                ),
            )

            # Label is required for every training branch.
            label = load_diagnoses(patient_data_file, patient_id)
            try:
                label = int(label)
            except (TypeError, ValueError):
                continue
            if label not in (0, 1):
                continue

            feat_df = add_patient_level_features( physiological_features,demographic_features,algorithmic_features, label=label,)
            if len(feat_df) != 1:
                raise RuntimeError(f"Expected one handcrafted row for {patient_id}, got {len(feat_df)}. Debug messages: {team_debug.summary()}")

            # Mode 3 alone computes SleepFM embeddings. This is deliberately
            # before appending the handcrafted row so the arrays stay aligned
            # if SleepFM fails for a patient.
            sleepfm_embedding = None
            if PIPELINE_MODE == 3:
                modality_signals, modality_fs = build_sleepfm_modalities(
                    physiological_data,
                    physiological_fs,
                    category_lookup,
                    csv_path=csv_path,
                )
                modality_counts = {modality: len(channels)for modality, channels in modality_signals.items()}
                team_debug.update_success("sleepfm modalities built", patient_id=patient_id,iteration=i, message=f"channels={modality_counts}", )
                sleepfm_embedding = extract_sleepfm_patient_embedding(sleepfm_encoder,sleepfm_preprocessor, modality_signals,modality_fs)
                if sleepfm_embedding.shape != (config.embedding_dimension,):
                    raise RuntimeError(
                        "SleepFM embedding shape mismatch: "
                        f"got {sleepfm_embedding.shape}, expected "
                        f"({config.embedding_dimension},)."
                    )

            handcrafted_rows.append(feat_df)
            labels.append(label)
            if PIPELINE_MODE == 3:
                sleepfm_rows.append(sleepfm_embedding)

            pd.DataFrame({"patient": [patient_id]}).to_csv( backup, mode="a", header=not header_written, index=False)
            header_written = True
            team_debug.update_success(
                "patient appended",
                patient_id=patient_id,
                iteration=i,
                message=(
                    f"HC={len(handcrafted_rows)}; labels={len(labels)}; "
                    f"SleepFM={len(sleepfm_rows)}"
                ),
            )

        except Exception as e:
            error_stack = traceback.format_exc()
            team_debug.update_failure("start",patient_id=patient_id, iteration=i,message=str(e), exc=e,)
            tqdm.write(f"  !!! Error processing record {i + 1} ({patient_id}): {e}")
            tqdm.write(error_stack)
            continue

    pbar.close()

    if not handcrafted_rows:
        raise RuntimeError(
            "No patients produced usable handcrafted data."
            f"\n--- COLLECTOR DEBUG SUMMARY ---\n{team_debug.summary()}"
        )

    full_df = pd.concat(handcrafted_rows, ignore_index=True)
    y = np.asarray(labels, dtype=np.int64)
    if len(full_df) != len(y):
        raise RuntimeError( f"Patient alignment failed: HC={len(full_df)}, labels={len(y)}, debug messages: {team_debug.summary()}" )

    # Existing PCA code explicitly needs these columns to be bool.
    all_bool_cols = [c for c in full_df.columns if "is_flat" in c or "amp_out_of_bounds" in c]
    for col in all_bool_cols:
        full_df[col] = full_df[col].astype(bool)

    params = get_xgb_params()

    
    # MODE 1: feature extraction -> XGBoost, no stratified PCA/RFE.
    if PIPELINE_MODE == 1:
        selected_features = get_available_handcrafted_features( full_df,category=FEATURE_CATEGORY,include_patient_level=INCLUDE_PATIENT_LEVEL_FEATURES_WITH_CATEGORY )
        team_debug.update_success(
            "mode 1 features selected",
            message=(
                f"category={FEATURE_CATEGORY}; "
                f"selected_features={len(selected_features)}"
            ),
        )
        model, imputer = fit_xgboost_pipeline( full_df, y, selected_features, params)
        save_xgb_model( model_folder, model,selected_features, imputer,pipeline_mode=1,feature_category=FEATURE_CATEGORY, )
        if verbose:
            print(
                f"Mode 1 training complete: {len(selected_features)} features, "
                f"category={FEATURE_CATEGORY or 'all'}."
            )
        return

    # Modes 2 and 3 share the existing stratified PCA/RFE ranking + subset step.
    team_debug.update_success(
        "feature selection started",
        message=f"input_shape={full_df.shape}; labels={len(y)}",
    )
    ranked_features = stratified_pca_feature_selection( full_df, pd.Series(y), params)
    selected_features = choose_feature_subset( full_df, ranked_features, params)
    selected_features = [ feature for feature in selected_features if feature in full_df.columns and full_df[feature].notna().any()]
    if not selected_features:
        raise RuntimeError(
            "No selected handcrafted features remain."
            f"\n--- COLLECTOR DEBUG SUMMARY ---\n{team_debug.summary()}"
        )

    # MODE 2: stratified PCA/RFE selection -> XGBoost.
    if PIPELINE_MODE == 2:
        model, imputer = fit_xgboost_pipeline( full_df, y, selected_features, params )
        save_xgb_model(
            model_folder,
            model,
            selected_features,
            imputer,
            pipeline_mode=2,
            feature_category=None,
        )
        if verbose:
            print( f"Mode 2 training complete: {len(selected_features)} selected features.")
        return

    # MODE 3: selected handcrafted features + SleepFM hybrid.
    sleepfm_matrix = np.stack(sleepfm_rows, axis=0).astype(np.float32)
    if not (len(full_df) == len(sleepfm_matrix) == len(y)):
        raise RuntimeError(
            "Hybrid patient alignment failed: "
            f"HC={len(full_df)}, SleepFM={len(sleepfm_matrix)}, labels={len(y)}"
        )

    handcrafted_raw = full_df[selected_features].to_numpy()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    handcrafted_imputed = imputer.fit_transform(handcrafted_raw)
    handcrafted_matrix = scaler.fit_transform( handcrafted_imputed).astype(np.float32)

    if not np.isfinite(handcrafted_matrix).all():
        raise RuntimeError(
            "Non-finite handcrafted values remain after final preprocessing."
        )

    model = HybridStrategy(
        config=config,
        handcrafted_dim=handcrafted_matrix.shape[1],
        num_classes=1,
    )
    model.backbone = sleepfm_encoder
    freeze_backbone(model.backbone)
    model = train_hybrid_head(
        model,
        handcrafted_matrix,
        sleepfm_matrix,
        y,
        config,
    )

    save_model(
        model_folder=model_folder,
        model=model,
        selected_features=selected_features,
        handcrafted_imputer=imputer,
        handcrafted_scaler=scaler,
        config=config,
        pipeline_mode=3,
    )
    if verbose:
        print("Mode 3 hybrid training complete.")


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    """Load whichever model type was saved during training."""
    xgb_metadata_path = os.path.join(model_folder, XGB_METADATA_FILENAME)
    hybrid_metadata_path = os.path.join(model_folder, HYBRID_METADATA_FILENAME)

    if os.path.exists(xgb_metadata_path):
        metadata = joblib.load(xgb_metadata_path)
        model_path = os.path.join(model_folder, XGB_MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        xgb_model = joblib.load(model_path)
        return {
            "model_type": "xgboost",
            "pipeline_mode": metadata["pipeline_mode"],
            "pipeline_name": metadata["pipeline_name"],
            "feature_category": metadata.get("feature_category"),
            "model": xgb_model,
            "selected_features": metadata["selected_features"],
            "imputer": metadata["imputer"],
            "threshold": metadata["threshold"],
        }

    if not os.path.exists(hybrid_metadata_path):
        raise FileNotFoundError(
            f"Neither {XGB_METADATA_FILENAME} nor {HYBRID_METADATA_FILENAME} "
            f"was found in {model_folder}."
        )

    model_path = os.path.join(model_folder, HYBRID_MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    metadata = joblib.load(hybrid_metadata_path)
    config_dict = metadata["sleepfm_config"]
    config_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict["checkpoint_path"] = None
    config_dict["debug"] = False
    config = SleepFMConfig(**config_dict)
    setattr(config, "load_pretrained", False)

    hybrid_model = HybridStrategy(
        config=config,
        handcrafted_dim=metadata["handcrafted_dim"],
        num_classes=1,
    )
    state_dict = torch.load(model_path, map_location=config.device)
    hybrid_model.load_state_dict(state_dict, strict=True)
    hybrid_model.to(config.device)
    hybrid_model.eval()

    return {
        "model_type": "hybrid",
        "pipeline_mode": metadata.get("pipeline_mode", 3),
        "pipeline_name": metadata.get("pipeline_name", PIPELINE_NAMES[3]),
        "model": hybrid_model,
        "config": config,
        "selected_features": metadata["selected_features"],
        "handcrafted_imputer": metadata["handcrafted_imputer"],
        "handcrafted_scaler": metadata["handcrafted_scaler"],
        "threshold": metadata["threshold"],
    }


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    pipeline_mode = int(model.get("pipeline_mode", 3))
    selected_features = model["selected_features"]
    threshold = model["threshold"]

    patient_id = record[HEADERS["bids_folder"]]
    site_id = record[HEADERS["site_id"]]
    session_id = record[HEADERS["session_id"]]
    category_lookup = get_category_map(DEFAULT_CSV_PATH)

    # Demographics.
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(
        patient_data_file, patient_id, session_id
    )
    demographic_features = extract_demographic_features(patient_data)

    # Algorithmic annotations / hypnogram.
    algo_file = os.path.join(
        data_folder,
        ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
        site_id,
        f"{patient_id}_ses-{session_id}_caisr_annotations.edf",
    )
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algorithmic_features = extract_algorithmic_annotations_features(algo_data)
        if "stage_caisr" in algo_data:
            hypno = np.round(algo_data["stage_caisr"].flatten()).astype(int)
        else:
            hypno = None
    else:
        algorithmic_features = np.zeros(12, dtype=np.float32)
        hypno = None

    # Physiological signals are needed by every requested pipeline mode.
    phys_file = os.path.join(
        data_folder,
        PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id,
        f"{patient_id}_ses-{session_id}.edf",
    )
    if not os.path.exists(phys_file):
        if verbose:
            print(
                f"Missing physiological data for {patient_id}; "
                "returning neutral probability."
            )
        return 0, 0.5

    physiological_data, physiological_fs = load_signal_data(phys_file)
    physiological_features = extract_physiological_features(
        physiological_data,
        physiological_fs,
        csv_path=DEFAULT_CSV_PATH,
        category_lookup=category_lookup,
        hypnogram=hypno,
        patient_id=patient_id,
        session_id=session_id,
        category_filter=(model.get("feature_category") if pipeline_mode == 1 else None),
    )
    feat_df = add_patient_level_features(
        physiological_features,
        demographic_features,
        algorithmic_features,
        label=None,
    )

    # Reindex rather than direct indexing: if a training column is absent in
    # a hidden record it becomes NaN and the fitted imputer handles it.
    handcrafted_raw = feat_df.reindex(columns=selected_features).copy()
    for col in handcrafted_raw.columns:
        if handcrafted_raw[col].dtype == bool:
            handcrafted_raw[col] = handcrafted_raw[col].astype(np.int8)

    # Modes 1 and 2 terminate here with XGBoost.
    if pipeline_mode in (1, 2):
        imputer = model["imputer"]
        X = imputer.transform(handcrafted_raw)
        probability_output = float(model["model"].predict_proba(X)[0, 1])
        binary_output = int(probability_output >= threshold)
        return binary_output, probability_output

    # Mode 3: handcrafted + SleepFM hybrid.
    hybrid_model = model["model"]
    config = model["config"]
    imputer = model["handcrafted_imputer"]
    scaler = model["handcrafted_scaler"]

    handcrafted_imputed = imputer.transform(handcrafted_raw)
    handcrafted_matrix = scaler.transform(
        handcrafted_imputed
    ).astype(np.float32)
    if not np.isfinite(handcrafted_matrix).all():
        raise RuntimeError(
            f"Non-finite handcrafted values for patient {patient_id}."
        )
    handcrafted_tensor = torch.tensor(
        handcrafted_matrix,
        dtype=torch.float32,
        device=config.device,
    )

    modality_signals, modality_fs = build_sleepfm_modalities(
        physiological_data,
        physiological_fs,
        category_lookup,
        csv_path=DEFAULT_CSV_PATH,
    )
    sleepfm_preprocessor = SleepFMPreprocessor(config)
    sleepfm_embedding = extract_sleepfm_patient_embedding(
        hybrid_model.backbone,
        sleepfm_preprocessor,
        modality_signals,
        modality_fs,
    )
    sleepfm_tensor = torch.tensor(
        sleepfm_embedding,
        dtype=torch.float32,
        device=config.device,
    ).unsqueeze(0)

    hybrid_model.eval()
    with torch.no_grad():
        logits = hybrid_model.forward_from_embeddings(
            sleepfm_tensor, handcrafted_tensor
        )
        probability_output = torch.sigmoid(logits).item()
    binary_output = int(probability_output >= threshold)
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
    # 1. Age
    age = load_age(data)
    age = np.array([age])

    # 2. Sex feature (one-hot encoding for Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data, standardize=True)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1 # Index 0: Female
    elif sex == 'Male': 
        sex_vec[1] = 1 # Index 1: Male
    else: 
        sex_vec[2] = 1 # Index 2: Other/Unknown

    # 3. Race One-Hot Encoding (5 dimensions)
    # Standardizes the raw text into one of five categories using the helper function
    race = load_race(data, standardize=True)
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    if race == 'Asian':
        race_vec[0] = 1
    elif race == 'Black':
        race_vec[1] = 1
    elif race == 'Others':
        race_vec[2] = 1
    elif race == 'Unavailable':
        race_vec[3] = 1
    elif race == 'White':
        race_vec[4] = 1
    else:
        race_vec[2] = 1 # Default to 'Others' for any unrecognized

    # 4. Body mass index (BMI)
    bmi = load_bmi(data)
    bmi = np.array([bmi])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)
    return np.concatenate([age, sex_vec, race_vec, bmi])


def extract_physiological_features(physiological_data, physiological_fs,
                                   csv_path=DEFAULT_CSV_PATH,
                                   category_lookup=get_category_map(DEFAULT_CSV_PATH),
                                   hypnogram=None, patient_id="Unknown", session_id="Unknown",
                                   category_filter=None):
    """
    Standardizes channels and extracts statistical/spectral features.
    """
    feature_debug.update_success("extract_physiological_features", patient_id=patient_id,message="Starting physiological extraction")
    original_labels = list(physiological_data.keys())

    
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

    
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())

        if category_filter is not None:
            requested = str(category_filter).strip().lower()
            actual = str(category_lookup.get(new_label, "unknown")).strip().lower()
            if requested == "emg":
                keep_channel = actual in {"leg emg", "chin emg", "emg"}
            else:
                keep_channel = actual == requested
            if not keep_channel:
                continue

        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        processed_fs[new_label] = physiological_fs.get(old_label, 200.0)  # Default to 200 if missing

    feature_debug.update_success(
        "channels standardized",
        patient_id=patient_id,
        message=(
            f"Renamed channels standard; category_filter={category_filter}; "
            f"kept_channels={len(processed_channels)}"
        ),
    )
    if not processed_channels:
        raise RuntimeError(f"No physiological channels remained after category_filter={category_filter!r}. Debug messages: {team_debug.summary()}")

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
    quality_df = assess_signal_quality(raw_psg, category_lookup, patient_id=patient_id, session_id=session_id,hypnogram=hypnogram)
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
    valid_cols = [col for col in feature_df.columns if col[0] in valid_features.get(col[1], [])]
    feature_df = feature_df[valid_cols]
    print(f"Shape after filtering invalid columns {feature_df.shape}")
    feature_df.columns = [f"{col[1]}-{col[2]}-{col[3]}-{col[4]}-{col[0]}" if isinstance(col, tuple) else col
                          for col in feature_df.columns.values]

    MASTER_FEATURE_NAMES = get_master_feature_names(category_lookup)
    feature_df = feature_df.reindex(columns=MASTER_FEATURE_NAMES, fill_value=np.nan)

    feature_df["subject"] = feature_df.index.get_level_values(0)
    feature_df["session"] = feature_df.index.get_level_values(1)
    feature_debug.update_success("reindex complete", patient_id=patient_id, message="Feature columns reindexed to master format")
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
def save_model(model_folder, model, selected_features, handcrafted_imputer, handcrafted_scaler, config, pipeline_mode=3):
    os.makedirs(model_folder, exist_ok=True)
    # Prevent load_model() from seeing stale XGBoost artifacts from another mode.
    for stale_name in (XGB_MODEL_FILENAME, XGB_METADATA_FILENAME):
        stale_path = os.path.join(model_folder, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)
    model_path = os.path.join( model_folder, HYBRID_MODEL_FILENAME, )
    metadata_path = os.path.join( model_folder, HYBRID_METADATA_FILENAME,)

    # The complete HybridStrategy state_dict includes: SleepFM backbone , handcrafted projection, fusion classifier
    torch.save( model.state_dict(), model_path, )
    config_dict = asdict(config)

    config_dict["checkpoint_path"] = None

    metadata = {
        "model_type": "HybridStrategy",
        "pipeline_mode": int(pipeline_mode),
        "pipeline_name": PIPELINE_NAMES[int(pipeline_mode)],
        "selected_features": list(selected_features),
        "handcrafted_dim": len(selected_features),
        "handcrafted_imputer": handcrafted_imputer,
        "handcrafted_scaler": handcrafted_scaler,
        "sleepfm_config": config_dict,
        "threshold": 0.5,
    }

    joblib.dump(metadata,metadata_path,protocol=0,)
