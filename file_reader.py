#import os
#import glob
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

# ---------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------
BASE_DIR = "training_set"
#DEMO_PATH = os.path.join(BASE_DIR, "demographics.csv")
DEMO_PATH = os.path.join(BASE_DIR, DEMOGRAPHICS_FILE) # Uses helper constant

# Data types and their filename suffixes
DATA_TYPES = {
    "physiological": "",
    "human": "_expert_annotations",
    "algo": "_caisr_annotations"
}

rename_rules=load_rename_rules("channel_table.csv")
print(rename_rules)

# ---------------------------------------------------
# Load Demographics and Find Files
# ---------------------------------------------------

#df_demo = pd.read_csv(DEMO_PATH)

'''#old way
def get_edf_path(site_id, bids_id, session_id, data_group):
    """
    Constructs the path and checks if the file exists.
    data_group: 'physiological_data', 'human_annotations', or 'algorithmic_annotations'
    """
    suffix = ""
    if "human" in data_group:
        suffix = DATA_TYPES["human"]
    elif "algorithmic" in data_group:
        suffix = DATA_TYPES["algo"]

    filename = f"{bids_id}_ses-{session_id}{suffix}.edf"
    full_path = os.path.join(BASE_DIR, data_group, site_id, filename)

    return full_path if os.path.exists(full_path) else None
'''

def process_patient_signals(edf_path): #uses helper functions
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

        # --- Your homebrew feature extraction fits here ---
        # Because standard_name is now "clean" (e.g., 'c3', 'f4', 'eog_l')
        # your downstream analysis remains consistent across the whole dataset.

        feat = {
            "channel": standard_name,
            "mean": np.mean(signal),
            "hf_ratio": high_frequency_ratio(signal, fs),
            # Your existing function
            "dominant_freq": dominant_frequency(signal, fs)
            # Your existing function
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

def get_category_map(csv_path):
    df = pd.read_csv(csv_path)
    cat_map = {}
    for _, row in df.iterrows():
        if pd.isna(row['Channel_Names']):
            continue
        # The standard name is the first item in the semicolon-separated list
        standard_name = row['Channel_Names'].split(';')[0].strip().lower()
        cat_map[standard_name] = row['Category']
    return cat_map

# Create the map once
category_lookup = get_category_map('channel_table.csv')

# ---------------------------------------------------
# 3. Main Processing Loop
# ---------------------------------------------------

#3.1 helper functions-------------------------------------
def inspect_raw(raw, label):
    print(f"\n===== {label} INFO =====")
    print(raw)
    print("\nChannel Names:")
    print(raw.ch_names)
    print("\nChannel Types:")
    print(set(raw.get_channel_types()))
    print("\nSampling Frequency:")
    print(raw.info["sfreq"])

    print("\nAnnotations:")
    print(raw.annotations)
    print("============================\n")

def concordance_cc(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)


def assess_signal_quality(raw):
    sfreq = raw.info["sfreq"]
    quality_metrics = []

    for ch_name in raw.ch_names:
        data, _ = raw[ch_name]
        data = data.flatten()

        # Remove NaNs for stats
        valid = ~np.isnan(data)
        data_clean = data[valid]

        if len(data_clean) < 10:
            continue

        # Basic metrics
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean)
        var_val = np.var(data_clean)
        skew_val = skew(data_clean)
        kurt_val = kurtosis(data_clean)

        # Flatline detection
        flat_pct = np.mean(np.abs(np.diff(data_clean)) < 1e-6)

        # Clipping detection
        clip_pct = np.mean(
            (data_clean == np.max(data_clean)) |
            (data_clean == np.min(data_clean))
        )

        # Spectral power estimate
        f, Pxx = welch(data_clean, sfreq, nperseg=min(4096, len(data_clean)))

        total_power = np.sum(Pxx)
        hf_power = np.sum(Pxx[f > 30]) if len(f) > 0 else 0
        lf_power = np.sum(Pxx[(f >= 0.5) & (f <= 4)]) if len(f) > 0 else 0

        hf_ratio = hf_power / total_power if total_power > 0 else 0
        lf_ratio = lf_power / total_power if total_power > 0 else 0

        quality_metrics.append({
            "channel": ch_name,
            "mean": mean_val,
            "std": std_val,
            "variance": var_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "flatline_pct": flat_pct,
            "clipping_pct": clip_pct,
            "hf_power_ratio": hf_ratio,
            "lf_power_ratio": lf_ratio
        })

    return pd.DataFrame(quality_metrics)

#3.2 Helper functions for EEG----------------------

def rms_amplitude(x):
    return np.sqrt(np.mean(x**2))

def peak_to_peak(x):
    return np.max(x) - np.min(x)

def signal_std(x):
    return np.std(x)

def max_abs_amplitude(x):
    return np.max(np.abs(x))

def percent_above_threshold(x, threshold):
    return np.mean(np.abs(x) > threshold)

def dominant_frequency(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), fs*4))
    return f[np.argmax(Pxx)]

def bandpower(x, fs, band, relative=False):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), fs*4))
    idx = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.trapezoid(Pxx[idx], f[idx])

    if relative:
        total_power = np.trapezoid(Pxx, f)
        return band_power / total_power if total_power > 0 else 0

    return band_power

#3.3 ECG helper functions-----------------------------------------------
def flatline_duration(x, fs, tolerance=1e-6):
    diff = np.abs(np.diff(x))
    flat = diff < tolerance
    flat_samples = np.sum(flat)
    return flat_samples / fs

def is_flat_signal(x, threshold=1e-8):
    return np.var(x) < threshold

def amplitude_out_of_bounds(x, low=0.2, high=15.0):
    ptp = peak_to_peak(x)
    return ptp < low or ptp > high

def max_slope(x, fs):
    dx = np.diff(x) * fs
    return np.max(np.abs(dx))

def high_frequency_ratio(x, fs, cutoff=40):
    f, Pxx = welch(x, fs=fs)
    hf_power = np.trapezoid(Pxx[f > cutoff], f[f > cutoff])
    total_power = np.trapezoid(Pxx, f)
    return hf_power / total_power if total_power > 0 else 0

#3.4 EMG helper functions-------------------------------------
def powerline_ratio(x, fs, line_freq=50, bandwidth=1):
    f, Pxx = welch(x, fs=fs)
    idx_line = np.logical_and(f >= line_freq - bandwidth,
                              f <= line_freq + bandwidth)
    line_power = np.trapezoid(Pxx[idx_line], f[idx_line])
    total_power = np.trapezoid(Pxx, f)
    return line_power / total_power if total_power > 0 else 0

def signal_to_high_freq_ratio(x, fs, hf_cut=100):
    f, Pxx = welch(x, fs=fs)
    hf_power = np.trapezoid(Pxx[f > hf_cut], f[f > hf_cut])
    signal_power = np.trapezoid(Pxx[f <= hf_cut], f[f <= hf_cut])
    return signal_power / hf_power if hf_power > 0 else np.inf

def spectral_peak_to_min_ratio(x, fs):
    f, Pxx = welch(x, fs=fs)
    return np.max(Pxx) / (np.min(Pxx) + 1e-12)

def spectral_peak_to_min_ratio(x, fs):
    f, Pxx = welch(x, fs=fs)
    return np.max(Pxx) / (np.min(Pxx) + 1e-12)

def windowed_feature(x, fs, window_sec, feature_fn, **kwargs):
    win_samples = int(window_sec * fs)
    n_windows = len(x) // win_samples
    values = []

    for i in range(n_windows):
        segment = x[i*win_samples:(i+1)*win_samples]
        values.append(feature_fn(segment, fs=fs, **kwargs))

    return np.array(values)


all_stage_accuracies = []
all_limb_accuracies = []
all_resp_accuracies = []
all_arousal_accuracies = []
all_quality_dfs=[]

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

        # 3. Assess Quality on standardized names
        quality_df = assess_signal_quality(raw_psg)
        quality_df['subject'] = row[HEADERS['bids_folder']]
        all_quality_dfs.append(quality_df)

        # 4. Load Annotations
        raw_human = mne.io.read_raw_edf(found_paths["human"], preload=True,
                                        verbose=False)
        raw_algo = mne.io.read_raw_edf(found_paths["algo"], preload=True, verbose=False)
        # Note: You can also standardize annotation channel names here if needed

        # ... [Your existing sleep stage/resp analysis code] ...

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
            print(conf_mat)

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
            print(human_matrix)
            print("Algorithmic")
            print(algo_matrix)


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
            print(conf_mat)

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
            print(conf_mat)

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
            print(conf_mat)

        print("Assessing signal quality...")
        quality_df = assess_signal_quality(raw_psg)
        all_quality_dfs.append(quality_df)
        pd.set_option('display.max_columns', None)
        print(quality_df)

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
                print(f"Band power 10-20 {data,fs,[10,20]}")
                # ...
            elif category == "ecg":
                # Apply ECG-specific features (max slope, HR)
                print(f"ECG features for {ch_name}")
                print(f"Flatline duration {flatline_duration(data,fs,tolerance=1.5)}")
                print(f"Is it a flat signal? {is_flat_signal(data)}")
                print(f"Amplitude out of bound? {amplitude_out_of_bounds(data)}")
                print(f"High frequency ratio {high_frequency_ratio(data,fs)}")
                print(f"Maximal slope {max_slope(data,fs)}")
                # ...
            elif category=="chin emg" or category=="leg emg":
                print(f"EMG feature for {ch_name}")
                print(f"Powerline ratio {powerline_ratio(data,fs)}")
                print(f"High frequency ratio {high_frequency_ratio(data,fs)}")
                print(f"Spectral peak to minimum ratio {spectral_peak_to_min_ratio(data,fs)}")
            #elif category == "resp":

        # Apply Respiratory features
        # ...

        raw_psg.close()
        raw_human.close()
        raw_algo.close()

    except Exception as e:
        print(f"Error processing {row[HEADERS['bids_folder']]}: {e}")

'''#older
# This iterates through the demographics rows to pair data automatically
for index, row in df_demo.iterrows():
    bids_folder = row['BidsFolder']  # e.g., sub-I0002150000686
    session_id = row['SessionID']  # e.g., 1

    # Extract SiteID (The letter + 4 numbers from the BidsFolder)
    # Logic: "sub-I0002..." -> SiteID is I0002
    site_id = bids_folder.split('-')[1][:5]

    # Construct paths for the pair
    psg_path = get_edf_path(site_id, bids_folder, session_id,
                            "physiological_data")
    ann_path = get_edf_path(site_id, bids_folder, session_id,
                            "human_annotations")
    algo_path = get_edf_path(site_id, bids_folder, session_id,
                             "algorithmic_annotations")


    if not psg_path or not ann_path:
        print(f"Skipping {bids_folder}: Missing PSG or Annotation file.")
        continue

    print(
        f"\nProcessing: {bids_folder} (Site: {site_id}, Session: {session_id})")

    try:
        # -----------------------------
        # 4. Data Loading & Analysis
        # -----------------------------
        raw_psg = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        raw_human = mne.io.read_raw_edf(ann_path, preload=True, verbose=False)
        raw_algo = mne.io.read_raw_edf(algo_path, preload=True, verbose=False)

        # Basic Alignment Check
        dur_psg = raw_psg.times[-1]
        dur_human = raw_human.times[-1]
        dur_algo = raw_algo.times[-1]
        diff = abs(dur_psg - dur_human)

        print(
            f" - Duration PSG: {dur_psg:.1f}s | Human: {dur_human:.1f}s | Algorithmic: {dur_algo:.1f}s| Diff: {diff:.2f}s")

        if diff > 30.0:
            print(" [!] Warning: Significant duration mismatch.")

        # ---------------------------------------------------
        # 5. Sleep Stage Analysis (Example)
        # ---------------------------------------------------
        #inspect_raw(raw_human, "HUMAN")
        #inspect_raw(raw_algo, "ALGO")
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
            print(conf_mat)

            human_transitions=np.where(np.diff(human_stage)!=0)[0]
            human_indices=np.concatenate(([0],human_transitions+1,[len(human_stage)]))
            human_transitions_only=human_stage[human_indices[:-1]]
            human_sources=human_transitions_only[:-1]
            human_targets=human_transitions_only[1:]
            human_matrix=pd.crosstab(pd.Series(human_sources,name="from Stage"),pd.Series(human_targets,name="to Stage"),normalize="index")
            algo_transitions=np.where(np.diff(algo_stage)!=0)[0]
            algo_indices=np.concatenate(([0],algo_transitions,[len(algo_stage)]))
            algo_transitions_only=algo_stage[algo_indices[:-1]]
            algo_sources=algo_transitions_only[:-1]
            algo_targets=algo_transitions_only[1:]
            algo_matrix=pd.crosstab(pd.Series(algo_sources,name="from Stage"),pd.Series(algo_targets,name="to Stage"),normalize="index")
            print("Transition matrices")
            print("Human")
            print(human_matrix)
            print("Algorithmic")
            print(algo_matrix)


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
            print(f"Pearson r:{r}, mean squared error: {np.mean((human_resp - algo_resp) ** 2)}, concordance: {ccc}")
            #print(f" Human respiratory annotation statistics: mean {np.mean(human_resp)} median {np.median(human_resp)} Q1 {np.percentile(human_resp,0.25)} Q3 {np.percentile(human_resp,0.75)} max {np.max(human_resp)} min {np.min(human_resp)} std {np.std(human_resp)}")
            #print(f"Algorithmic respiratory annotation statistics: mean {np.mean(algo_resp)} median {np.median(algo_resp)} Q1 {np.percentile(algo_resp,0.25)} Q3 {np.percentile(algo_resp,0.75)} max {np.max(algo_resp)} min {np.min(algo_resp)} std {np.std(algo_resp)}")
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
            print(conf_mat)


        #-----------------------------
        # 7. Limb analysis
        #----------------------------
        if "limb_expert" in raw_human.ch_names and "limb_caisr" in raw_algo.ch_names:
            human_limb, _ = raw_human["limb_expert", :]
            algo_limb, _ = raw_algo["limb_caisr", :]

            human_limb=human_limb.flatten()
            algo_limb=algo_limb.flatten()

            min_len= min(len(human_limb), len(algo_limb))
            human_limb = human_limb[:min_len]
            algo_limb = algo_limb[:min_len]

            human_binary = human_limb > 0
            algo_binary = algo_limb > 0

            limb_accuracy= np.mean(human_binary == algo_binary)
            all_limb_accuracies.append(limb_accuracy)

            print(f"\n - Limb movement agreement {limb_accuracy:.3f}")
            r, p = pearsonr(human_limb, algo_limb)
            ccc=concordance_cc(human_limb, algo_limb)
            print(f"Pearson r:{r}, mean squared error: {np.mean((human_limb-algo_limb)**2)}, concordance: {ccc}")
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
            print(conf_mat)

        #----------------------------
        # 8. Arousal Analysis
        #----------------------------
        if "arousal_expert" in raw_human.ch_names and "caisr_prob_arous" in raw_algo.ch_names and "caisr_prob_no-ar" in raw_algo.ch_names:
            human_arousal = raw_human["arousal_expert"][0].flatten() #This sounds really dirty out of context
            algo_arousal = raw_algo["caisr_prob_arous"][0].flatten()
            no_arousal = raw_algo["caisr_prob_no-ar"][0].flatten()

            '''
'''         r, p = pearsonr(algo_arousal, no_arousal)
            corr = correlate(algo_arousal, no_arousal, mode="full")
            lag = np.argmax(corr) - len(algo_arousal) + 1
            print(f"Comparisions of prob_arous and prob_no-ar, Pearson: {r}, "
                f"mean squared error: {np.mean((algo_arousal - no_arousal) ** 2)}, concordance: {ccc}, best lag: {lag}")
            '''
'''
            min_len= min(len(algo_arousal), len(no_arousal))
            human_arousal=human_arousal[:min_len]
            algo_arousal=algo_arousal[:min_len]

            human_binary=human_arousal>0.5
            algo_binary=algo_arousal>0.5
            arousal_accuracy=np.mean(human_binary==algo_binary)
            all_arousal_accuracies.append(arousal_accuracy)

            print(f"\n -Arousal agreement: {arousal_accuracy:.3f}")
            r,p=pearsonr(algo_arousal, human_arousal)   #the innuendos never end
            ccc=concordance_cc(human_arousal, algo_arousal)
            lag=np.argmax(corr)-len(algo_arousal) +1
            print(f"Pearson: {r}, mean squared error {np.mean((human_arousal-algo_arousal)**2)}, concordance: {ccc}, lag: {lag}")

            conf_mat=pd.crosstab(
                human_binary,
                algo_binary,
                rownames=["Human"],
                colnames=["Algorithm"]
            )

            print(" - Confusion Matrix arousal:")
            print(conf_mat)

        print("Assessing signal quality...")
        quality_df = assess_signal_quality(raw_psg)
        all_quality_dfs.append(quality_df)
        pd.set_option('display.max_columns', None)
        print(quality_df)

        # IMPORTANT: Close/Clear to save memory in a loop
        raw_psg.close()
        raw_human.close()
        raw_algo.close()
        plt.close('all')

    except Exception as e:
        print(f"Error processing {bids_folder}: {e}")
'''
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

'''
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
'''
'''
print("\n=== Detected Signal Quality Outliers ===\n")
print(outliers[["subject", "channel"] +
               [f"{m}_z" for m in metrics]])

merged["quality_score"] = merged[
    [f"{m}_z" for m in metrics]
].abs().mean(axis=1)

worst_channels = merged.sort_values("quality_score", ascending=False)
print(worst_channels)
'''
print("\nProcessing Complete.")