# import os
# import glob
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from helper_code import *
import neurokit2 as nk
import yasa
from utils import HiddenPrints
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
import wfdb#wfdb is needed for pecg
from pecg.ecg import Biomarkers as Bm
from get_hrv_bm import get_hrv_biomarkers
from joblib import Parallel, delayed
from get_spo2_bm import get_spo2_biomarkers  # currently problematic due to using an old scipy version
from feature_extraction import *

# ---------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------
BASE_DIR = "training_set"
# DEMO_PATH = os.path.join(BASE_DIR, "demographics.csv")
DEMO_PATH = os.path.join(BASE_DIR, DEMOGRAPHICS_FILE)  # Uses helper constant
CHANNEL_OUTPUT = "channel_quality_long.csv"


if not os.path.exists(CHANNEL_OUTPUT):
    header_written = False
    processed_subjects = set(

    )
else:
    header_written = True
    processed_subjects = set(
        pd.read_csv(CHANNEL_OUTPUT, usecols=["subject"])["subject"].unique())

# Data types and their filename suffixes


def ensure_channel_output_schema(output_path, ALL_FEATURE_COLUMNS):
    """
    Ensures CHANNEL_OUTPUT.csv matches ALL_FEATURE_COLUMNS.
    If columns differ, the file is rewritten with the correct header
    and missing columns filled with zeros.
    """

    if not os.path.exists(output_path):
        # Create empty file with correct header
        df = pd.DataFrame(columns=ALL_FEATURE_COLUMNS)
        df.to_csv(output_path, index=False)
        print(f"Created new CHANNEL_OUTPUT with correct schema.")
        return

    # Load existing CSV
    df = pd.read_csv(output_path)

    existing_cols = set(df.columns)
    expected_cols = set(ALL_FEATURE_COLUMNS)

    missing_cols = expected_cols - existing_cols
    extra_cols = existing_cols - expected_cols

    if not missing_cols and not extra_cols:
        # print("CHANNEL_OUTPUT schema OK.")
        return

    print("CHANNEL_OUTPUT schema mismatch detected.")
    print(f"Missing columns: {missing_cols}")
    print(f"Extra columns: {extra_cols}")

    # Add missing columns with zeros
    for col in missing_cols:
        df[col] = 0

    # Remove columns that should not exist
    if extra_cols:
        df = df.drop(columns=list(extra_cols))

    # Reorder columns to match ALL_FEATURE_COLUMNS
    df = df.reindex(columns=ALL_FEATURE_COLUMNS)

    # Rewrite CSV
    df.to_csv(output_path, index=False)

    print("CHANNEL_OUTPUT schema updated.")


# ensure_channel_output_schema(CHANNEL_OUTPUT, ALL_FEATURE_COLUMNS)

rename_rules = load_rename_rules("channel_table.csv")


# print(rename_rules)

# ---------------------------------------------------
# Load Demographics and Find Files
# ---------------------------------------------------

# df_demo = pd.read_csv(DEMO_PATH)


def process_patient_signals(edf_path):  # uses helper functions
    # Use helper to load raw signals into a dict: { 'c3-m2': np.array([...]) }
    channel_dict, fs_dict = load_edf_to_nparrays(edf_path)

    if not channel_dict:
        return None

    # Use helper to find the mapping
    # rename_map: { 'c3-m2': 'c3' } | cols_to_drop: ['c3-a2'] (if duplicates exist)
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        list(channel_dict.keys()),
        rename_rules
    )

    standardized_features = []

    # Now iterate through the CLEAN names you actually want
    for original_name, standard_name in rename_map.items():
        if original_name in cols_to_drop:
            continue

        signal = channel_dict[original_name]
        fs = fs_dict[original_name]


        feat = {
            "channel": standard_name,
            "mean": np.mean(signal),
            "hf_ratio": high_frequency_ratio(signal, fs),
            "dominant_freq": dominant_frequency(signal, fs)
        }
        standardized_features.append(feat)

    return pd.DataFrame(standardized_features)


def get_challenge_paths(row):
    """
    Uses the HEADERS mapping to safely get IDs and construct BIDS paths
    """
    bids_id = row[HEADERS['bids_folder']]
    session_id = row[HEADERS['session_id']]
    # Helper logic for SiteID extraction
    site_id = bids_id.split('-')[1][:5]

    paths = {
        "psg": os.path.join(BASE_DIR, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
                            f"{bids_id}_ses-{session_id}.edf"),
        "human": os.path.join(BASE_DIR, HUMAN_ANNOTATIONS_SUBFOLDER, site_id,
                              f"{bids_id}_ses-{session_id}_expert_annotations.edf"),
        "algo": os.path.join(BASE_DIR, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id,
                             f"{bids_id}_ses-{session_id}_caisr_annotations.edf")
    }

    # Verify existence
    return {k: v for k, v in paths.items() if os.path.exists(v)}


if not os.path.exists(DEMO_PATH):
    raise FileNotFoundError(f"Could not find demographics file at {DEMO_PATH}")





# ---------------------------------------------------
# 3. Main Processing Loop
# ---------------------------------------------------




all_stage_accuracies = []
all_limb_accuracies = []
all_resp_accuracies = []
all_arousal_accuracies = []
# all_quality_dfs=[]

df_demo = pd.read_csv(DEMO_PATH)

for index, row in df_demo.iterrows(): #using the helper functions
    found_paths = get_challenge_paths(row)

    if "psg" not in found_paths or "human" not in found_paths:
        print(
            f"Skipping {row[HEADERS['bids_folder']]}: Missing critical files.")
        continue

    print(f"\nProcessing: {row[HEADERS['bids_folder']]}")


    try:

        # 1. Load using MNE (for signal processing)
        raw_psg = mne.io.read_raw_edf(found_paths["psg"], preload=False,
                                      verbose=False)

        # 2. STANDARDIZE CHANNELS using helper_code
        # This fixes the "inconsistent naming" across different hospitals
        original_channels = raw_psg.ch_names
        rename_map, cols_to_drop = standardize_channel_names_rename_only(
            original_channels, rename_rules)

        # Apply the standardization
        raw_psg.rename_channels(rename_map)
        to_drop=[]
        if cols_to_drop:
            # Only drop if the columns actually exist in the MNE object
            to_drop = [c for c in cols_to_drop if c in raw_psg.ch_names]
            raw_psg.drop_channels(to_drop)

        print(
            f" - Standardized {len(rename_map)} channels. Dropped {len(to_drop)} duplicates.")

        #print(raw_psg.ch_names)

        subject_id = row[HEADERS["bids_folder"]]
        if subject_id in processed_subjects:
            print(f"Skipping {subject_id} (already processed)")
            continue


        # 3. Assess Quality on standardized names
        #quality_df = assess_signal_quality(raw_psg)
        #quality_df['subject'] = row[HEADERS['bids_folder']]
        #all_quality_dfs.append(quality_df)

        # 4. Load Annotations
        raw_human = mne.io.read_raw_edf(found_paths["human"], preload=True,
                                        verbose=False)
        raw_algo = mne.io.read_raw_edf(found_paths["algo"], preload=True, verbose=False)
        # Note: You can also standardize annotation channel names here if needed

        # ---------------------------------------------------
        # 5. Sleep Stage Analysis (Example)
        # ---------------------------------------------------
        # inspect_raw(raw_human, "HUMAN")
        # inspect_raw(raw_algo, "ALGO")
        if "stage_expert" in raw_human.ch_names and "stage_caisr" in raw_algo.ch_names:

            human_stage, _ = raw_human["stage_expert", :]
            algo_stage, _ = raw_algo["stage_caisr", :]

            human_stage = np.round(human_stage.flatten()).astype(int)
            algo_stage = np.round(algo_stage.flatten()).astype(int)

            # Align length
            min_len = min(len(human_stage), len(algo_stage))
            human_stage = human_stage[:min_len]
            algo_stage = algo_stage[:min_len]

            # Remove invalid labels (optional: adjust depending on dataset)
            valid_mask = (human_stage >= 0) & (algo_stage >= 0)
            human_stage = human_stage[valid_mask]
            algo_stage = algo_stage[valid_mask]

            # Accuracy
            accuracy = np.mean(human_stage == algo_stage)
            all_stage_accuracies.append(accuracy)

            print(f"\n - Sleep Stage Accuracy: {accuracy:.3f}")
            r, p = pearsonr(human_stage, algo_stage)
            ccc = concordance_cc(human_stage, algo_stage)
            print(
                f"Pearson r:{r}, mean squared error: {np.mean((human_stage - algo_stage) ** 2)}, concordance: {ccc}")
            corr = correlate(human_stage, algo_stage, mode='full')
            lag = np.argmax(corr) - len(human_stage) + 1
            print("Best lag (samples):", lag)

            # Confusion Matrix
            conf_mat = pd.crosstab(
                human_stage,
                algo_stage,
                rownames=["Human"],
                colnames=["Algorithm"]
            )

            print(" - Confusion Matrix stage:")
            #print(conf_mat)

            human_transitions = np.where(np.diff(human_stage) != 0)[0]
            human_indices = np.concatenate(
                ([0], human_transitions + 1, [len(human_stage)]))
            human_transitions_only = human_stage[human_indices[:-1]]
            human_sources = human_transitions_only[:-1]
            human_targets = human_transitions_only[1:]
            human_matrix = pd.crosstab(
                pd.Series(human_sources, name="from Stage"),
                pd.Series(human_targets, name="to Stage"), normalize="index")
            algo_transitions = np.where(np.diff(algo_stage) != 0)[0]
            algo_indices = np.concatenate(
                ([0], algo_transitions, [len(algo_stage)]))
            algo_transitions_only = algo_stage[algo_indices[:-1]]
            algo_sources = algo_transitions_only[:-1]
            algo_targets = algo_transitions_only[1:]
            algo_matrix = pd.crosstab(
                pd.Series(algo_sources, name="from Stage"),
                pd.Series(algo_targets, name="to Stage"), normalize="index")
            print("Transition matrices")
            print("Human")
            #print(human_matrix)
            print("Algorithmic")
            #print(algo_matrix)


        else:
            print(" - Sleep stage channels not found in both files.")

        # ---------------------------------------------------
        # 6. Respiratory Analysis (Example)
        # ---------------------------------------------------
        if "resp_expert" in raw_human.ch_names and "resp_caisr" in raw_algo.ch_names:
            human_resp, _ = raw_human["resp_expert", :]
            algo_resp, _ = raw_algo["resp_caisr", :]

            human_resp = human_resp.flatten()
            algo_resp = algo_resp.flatten()

            min_len = min(len(human_resp), len(algo_resp))
            human_resp = human_resp[:min_len]
            algo_resp = algo_resp[:min_len]

            # Binary event detection comparison
            human_binary = human_resp > 0
            algo_binary = algo_resp > 0

            resp_accuracy = np.mean(human_binary == algo_binary)
            all_resp_accuracies.append(resp_accuracy)

            print(f"\n - Respiratory Event Agreement: {resp_accuracy:.3f}")
            r, p = pearsonr(human_resp, algo_resp)
            ccc = concordance_cc(human_resp, algo_resp)
            print(
                f"Pearson r:{r}, mean squared error: {np.mean((human_resp - algo_resp) ** 2)}, concordance: {ccc}")
            # print(f" Human respiratory annotation statistics: mean {np.mean(human_resp)} median {np.median(human_resp)} Q1 {np.percentile(human_resp,0.25)} Q3 {np.percentile(human_resp,0.75)} max {np.max(human_resp)} min {np.min(human_resp)} std {np.std(human_resp)}")
            # print(f"Algorithmic respiratory annotation statistics: mean {np.mean(algo_resp)} median {np.median(algo_resp)} Q1 {np.percentile(algo_resp,0.25)} Q3 {np.percentile(algo_resp,0.75)} max {np.max(algo_resp)} min {np.min(algo_resp)} std {np.std(algo_resp)}")
            corr = correlate(human_resp, algo_resp, mode='full')
            lag = np.argmax(corr) - len(human_resp) + 1
            print("Best lag (samples):", lag)

            conf_mat = pd.crosstab(
                human_binary,
                algo_binary,
                rownames=["Human"],
                colnames=["Algorithm"]
            )

            print(" - Confusion Matrix respiration:")
            #print(conf_mat)

        # -----------------------------
        # 7. Limb analysis
        # ----------------------------
        if "limb_expert" in raw_human.ch_names and "limb_caisr" in raw_algo.ch_names:
            human_limb, _ = raw_human["limb_expert", :]
            algo_limb, _ = raw_algo["limb_caisr", :]

            human_limb = human_limb.flatten()
            algo_limb = algo_limb.flatten()

            min_len = min(len(human_limb), len(algo_limb))
            human_limb = human_limb[:min_len]
            algo_limb = algo_limb[:min_len]

            human_binary = human_limb > 0
            algo_binary = algo_limb > 0

            limb_accuracy = np.mean(human_binary == algo_binary)
            all_limb_accuracies.append(limb_accuracy)

            print(f"\n - Limb movement agreement {limb_accuracy:.3f}")
            r, p = pearsonr(human_limb, algo_limb)
            ccc = concordance_cc(human_limb, algo_limb)
            print(
                f"Pearson r:{r}, mean squared error: {np.mean((human_limb - algo_limb) ** 2)}, concordance: {ccc}")
            '''
'''
            print(
                f" Human limb movement annotation statistics: mean {np.mean(human_limb)} median {np.median(human_limb)} max {np.max(human_limb)} min {np.min(human_limb)} std {np.std(human_limb)}")
            print(
                f"Algorithmic limb movement annotation statistics: mean {np.mean(algo_limb)} median {np.median(algo_limb)} max {np.max(algo_limb)} min {np.min(algo_limb)} std {np.std(algo_limb)}")
            '''
'''
            corr = correlate(human_limb, algo_limb, mode='full')
            lag = np.argmax(corr) - len(human_limb) + 1
            print("Best lag (samples):", lag)

            conf_mat = pd.crosstab(
                human_binary,
                algo_binary,
                rownames=["Human"],
                colnames=["Algorithm"]
            )

            print(" - Confusion Matrix limb movement:")
            #print(conf_mat)

        # ----------------------------
        # 8. Arousal Analysis
        # ----------------------------
        if "arousal_expert" in raw_human.ch_names and "caisr_prob_arous" in raw_algo.ch_names and "caisr_prob_no-ar" in raw_algo.ch_names:
            human_arousal = raw_human["arousal_expert"][
                0].flatten()  # This sounds really dirty out of context
            algo_arousal = raw_algo["caisr_prob_arous"][0].flatten()
            no_arousal = raw_algo["caisr_prob_no-ar"][0].flatten()

            '''
'''
            r, p = pearsonr(algo_arousal, no_arousal)
            corr = correlate(algo_arousal, no_arousal, mode="full")
            lag = np.argmax(corr) - len(algo_arousal) + 1
            print(f"Comparisions of prob_arous and prob_no-ar, Pearson: {r}, "
                  f"mean squared error: {np.mean((algo_arousal - no_arousal) ** 2)}, concordance: {ccc}, best lag: {lag}")
            '''
'''
            min_len = min(len(algo_arousal), len(no_arousal))
            human_arousal = human_arousal[:min_len]
            algo_arousal = algo_arousal[:min_len]

            human_binary = human_arousal > 0.5
            algo_binary = algo_arousal > 0.5
            arousal_accuracy = np.mean(human_binary == algo_binary)
            all_arousal_accuracies.append(arousal_accuracy)

            print(f"\n -Arousal agreement: {arousal_accuracy:.3f}")
            r, p = pearsonr(algo_arousal,
                            human_arousal)  # the innuendos never end
            ccc = concordance_cc(human_arousal, algo_arousal)
            lag = np.argmax(corr) - len(algo_arousal) + 1
            print(
                f"Pearson: {r}, mean squared error {np.mean((human_arousal - algo_arousal) ** 2)}, concordance: {ccc}, lag: {lag}")

            conf_mat = pd.crosstab(
                human_binary,
                algo_binary,
                rownames=["Human"],
                colnames=["Algorithm"]
            )

            print(" - Confusion Matrix arousal:")
            #print(conf_mat)

        print("Assessing signal quality...")

        quality_df = assess_signal_quality(raw_psg, category_lookup)

        quality_df["subject"] = subject_id
        quality_df["session"] = row[HEADERS["session_id"]]
        quality_df = quality_df.reindex(columns=ALL_FEATURE_COLUMNS)
        quality_df.to_csv(
            CHANNEL_OUTPUT,
            mode="a",
            header=not header_written,
            index=False
        )
        header_written=True

        processed_subjects.add(subject_id)

        print(f"Saved quality metrics for {subject_id}")


        all_eeg = []
        all_emg = []
        all_ecg = []
        all_resp = []

        standardized_channels = raw_psg.ch_names

        for ch_name in standardized_channels:
            # Use the standardized name to find the category
            category = category_lookup.get(ch_name, "unknown")

            data, times = raw_psg[ch_name]
            data = data.flatten()
            fs = raw_psg.info['sfreq']

            if category == "eeg":
                print(f"EEG features for {ch_name}")
                # Apply EEG-specific features (bandpower, dominant freq)
                print(f"Root mean square amplitute {rms_amplitude(data)}")
                print(f"Peak to peak {peak_to_peak(data)}")
                print(f"Standard deviation {np.std(data)}")
                print(f"Maximum of amplitude{np.max(data)}")
                #per_above_thresh=per_above_thresh(data,)#need to think up pro
                print(f"Dominant frequency {dominant_frequency(data, fs)}")

                bands = {
                    "delta": [0.5, 4],
                    "theta": [4, 8],
                    "alpha": [8, 12],
                    "sigma": [12, 16],  # Vital for Sleep Spindles
                    "beta": [16, 30]
                }

                # Calculate relative power for each band
                rel_powers = {f"rel_{b}": bandpower(data, fs, freq, relative=True) for b, freq in bands.items()}

                # Brain Rhythm Ratios (e.g., Delta/Beta ratio is a huge sleep stage marker)
                db_ratio = rel_powers["rel_delta"] / (rel_powers["rel_beta"] + 1e-6)

                print(f"EEG Rel Powers: {rel_powers}")
                print(f"Delta/Beta Ratio: {db_ratio}")

            elif category == "ecg":
                # Apply ECG-specific features (max slope, HR)
                print(f"ECG features for {ch_name}")
                print(f"Root mean square amplitute {rms_amplitude(data)}")
                print(f"Peak to peak {peak_to_peak(data)}")
                print(f"Standard deviation {np.std(data)}")
                print(f"Maximum of amplitude{np.max(data)}")
                # per_above_thresh=per_above_thresh(data,)#need to think up pro

                print(f"Flatline duration {flatline_duration(data,fs,tolerance=1.5)}")
                print(f"Is it a flat signal? {is_flat_signal(data)}")
                print(f"Amplitude out of bound? {amplitude_out_of_bounds(data)}")
                print(f"High frequency ratio {high_frequency_ratio(data,fs)}")
                print(f"Maximal slope {max_slope(data,fs)}")

                ecg_cleaned = nk.ecg_clean(data, sampling_rate=fs)
                peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
                hr = nk.ecg_rate(peaks, sampling_rate=fs, desired_length=len(data))

                print(f"Mean Heart Rate: {np.mean(hr)}")

                if len(info['ECG_R_Peaks']) > 2:
                    sdnn = np.std(np.diff(info['ECG_R_Peaks']) / fs * 1000)
                    print(f"HRV (SDNN): {sdnn} ms")
                # ...
            elif category=="chin emg" or category=="leg emg":
                print(f"EMG feature for {ch_name}")
                print(f"Powerline ratio {powerline_ratio(data,fs)}")
                print(f"High frequency ratio {high_frequency_ratio(data,fs)}")
                print(f"Spectral peak to minimum ratio {spectral_peak_to_min_ratio(data,fs)}")
            elif category == "resp":
                resp_cleaned = nk.rsp_clean(data, sampling_rate=fs)
                peaks, info = nk.rsp_peaks(resp_cleaned, sampling_rate=fs)
                amplitude = nk.rsp_amplitude(resp_cleaned, peaks)
                intervals = np.diff(peaks['RSP_Peaks']) / fs
                resp_var = np.std(intervals) if len(intervals) > 0 else 0

                print(f"Mean Resp Rate: {60 / np.mean(intervals) if len(intervals) > 0 else 0}")
                print(f"Resp Variability: {resp_var}")

        raw_psg.close()
        raw_human.close()
        raw_algo.close()

    except Exception as e:
        print(f"Error processing {row[HEADERS['bids_folder']]}: {e}")


if all_stage_accuracies:
    print("\nOverall Sleep Stage Accuracy:")
    print(f"Mean: {np.mean(all_stage_accuracies):.3f}")
    print(f"Std:  {np.std(all_stage_accuracies):.3f}")

if all_limb_accuracies:
    print("\nOverall Limb Movement Accuracy:")
    print(f"Mean: {np.mean(all_limb_accuracies):.3f}")
    print(f"Std:  {np.std(all_limb_accuracies):.3f}")

if all_resp_accuracies:
    print("\nOverall Respitatory Movement Accuracy:")
    print(f"Mean: {np.mean(all_resp_accuracies):.3f}")
    print(f"Std:  {np.std(all_resp_accuracies):.3f}")

if all_arousal_accuracies:
    print("\nOverall Arousal Accuracy:")
    print(f"Mean: {np.mean(all_arousal_accuracies):.3f}")
    print(f"Std:  {np.std(all_arousal_accuracies):.3f}")


final_quality_df = pd.concat(all_quality_dfs, ignore_index=True)

channel_summary = (
    final_quality_df
    .groupby("channel")
    .agg(["mean", "std"])
)

print("\n=== Channel-Wise Population Statistics ===\n")
print(channel_summary)

# Flatten multiindex columns
channel_summary.columns = ['_'.join(col) for col in channel_summary.columns]
channel_summary = channel_summary.reset_index()

# Merge population stats back to individual data
merged = final_quality_df.merge(channel_summary, on="channel")

# Compute z-scores for each metric
metrics = ["std", "flatline_pct", "clipping_pct",
           "hf_power_ratio", "lf_power_ratio"]

for m in metrics:
    merged[f"{m}_z"] = (
        (merged[m] - merged[f"{m}_mean"]) /
        (merged[f"{m}_std"] + 1e-8)
    )

Z_THRESHOLD = 3

outliers = merged[
    (merged[[f"{m}_z" for m in metrics]].abs() > Z_THRESHOLD).any(axis=1)
]

print(outliers)

print("\n=== Detected Signal Quality Outliers ===\n")
print(outliers[["subject", "channel"] +
               [f"{m}_z" for m in metrics]])

merged["quality_score"] = merged[
    [f"{m}_z" for m in metrics]
].abs().mean(axis=1)

worst_channels = merged.sort_values("quality_score", ascending=False)
print(worst_channels)

print("\nProcessing Complete.")