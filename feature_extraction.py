import pandas as pd
import mne
import numpy as np
from scipy.stats import pearsonr, kurtosis, skew
from scipy.signal import correlate, find_peaks, welch, hilbert
from helper_code import *
import neurokit2 as nk
import yasa
from utils import HiddenPrints
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
import wfdb  # wfdb is needed for pecg
from pecg.ecg import Biomarkers as Bm
from get_hrv_bm import get_hrv_biomarkers
from joblib import Parallel, delayed
from get_spo2_bm import \
    get_spo2_biomarkers  # currently problematic due to using an old scipy version #even after throwing out the complexity metics, it was still slow as dicks
from utils import unfuck_old
from tsfresh.feature_extraction import feature_calculators as tsf
import time
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn import metrics
import csv
import traceback

unfuck_old()

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
# category_lookup = get_category_map('channel_table.csv')


DATA_TYPES = {
    "physiological": "",
    "human": "_expert_annotations",
    "algo": "_caisr_annotations"
}

STAGE_MAP = {
    1: "N3",
    2: "N2",
    3: "N1",
    4: "REM",
    5: "Wake"
}

metadata = ["subject", "session", "channel", "stage", "category","transition_id","transition_type"]
generic_features = ["abs_energy","abs_sum_changes", "mean_abs_change", "root_mean_square",
                    "variation_coeff", "median","skewness", "kurtosis","longest_strike_above_mean", 
                    "longest_strike_below_mean","quantile_0.1","quantile_0.25","quantile_0.75",
                    "quantile_0.9","autocorr_lag_1","autocorr_lag_2","autocorr_lag_5",
                    "autocorr_lag_10","ratio_beyond_r_sigma_2","flatline_pct", "clipping_pct", 
                    "mean", "std", "variance", "hf_power_ratio","lf_power_ratio",
                    "trans_mean_before","trans_mean_after","trans_std_before","trans_std_after",
                    "trans_recovery","trans_variability_change"]
emg_only_features = ['hf_ratio_signal', 'powerline_ratio',
                     'spec_peak_min_ratio', 'energy_ratio_chunks']
ecg_only_features_1 = ['rms', 'ptp', 'max_amp', 'flatline_duration', 'is_flat',
                       'amp_out_of_bounds',
                       'hf_ratio_signal', 'max_slope', 'mean_hr',
                       "mean_2nd_deriv_central"]
ecg_only_features_2 = ["AVNN", "SDNN", "RMSSD", "PNN50", "SEM", "PIP", "IALS",
                       "PSS", "PAS",
                       "SD1", "SD2", "alpha_1", "alpha_2", "MSE", "TOTAL_POWER",
                       "VLF_POWER", "LF_POWER", "HF_POWER", "VLF_NORM",
                       "LF_NORM", "HF_NORM", "LF_HF_RATIO"]         
eeg_only_features_1 = ['rms', 'ptp', 'max_amp', 'dominant_freq',
                       'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma',
                       'rel_beta', 'delta_beta_ratio',
                       'time_rev_asym', 'num_peaks']
eeg_only_features_2 = ['spindle_count', 'spindle_density', 'spindle_mean_amp',
                       'spindle_amp_std',
                       'spindle_mean_duration', 'spindle_duration_std',
                       'spindle_mean_freq',
                       'spindle_mean_relpower', 'spindle_mean_rms',
                       'spindle_total_duration']
eeg_only_features_3=["sw_density","sw_mean_amp","sw_amp_std","sw_mean_duration",
                      "sw_duration_std","sw_mean_freq","sw_total_duration"]  
resp_only_features = ['mean_resp_rate', 'resp_variability', 'linear_trend']
synchrony_only_features = ["r", "ccc", "lag", "peak_lag", "amp_ratio","phase_lock"]
transition_only_features=["trans_mean_before","trans_mean_after","trans_std_before","trans_std_after","trans_recovery","trans_variability_change"]
annotation_only_features=["ahi_auto","arousal_auto","limb_auto","w_pct","r_pct","n1_pct","n2_pct","n3_pct","efficiency","prob_w","prob_n3","prob_arous"] 
ALL_FEATURE_COLUMNS = list(
    set(metadata + generic_features + synchrony_only_features + emg_only_features +
        ecg_only_features_1 + ecg_only_features_2 +
        eeg_only_features_1 + eeg_only_features_2 + eeg_only_features_3+ resp_only_features))
valid_features = {
    'eeg': eeg_only_features_1 + eeg_only_features_2 + eeg_only_features_3 + generic_features + synchrony_only_features,
    'leg emg': emg_only_features + generic_features + synchrony_only_features,
    'chin emg': emg_only_features + generic_features,
    'ecg': ecg_only_features_1 + ecg_only_features_2 + generic_features,
    'resp': resp_only_features + generic_features + synchrony_only_features,
    'eog': generic_features + synchrony_only_features  
}

UNIQUE_CHANNELS = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2',
                   'o2-m1', 'e1', 'e2', 'chin1-chin2', 'ptaf',
                   'airflow', 'chest', 'abd', 'lat', 'rat',
                   'ecg', 'spo2', 'cflow', 'cpres', 'm1',
                   'c3', 'm2', 'c4', 'o2', 'o1',
                   'chin 2', 'chin 1', 'f3', 'f4', 'e1-m2',
                   'e2-m1', 'rleg+', 'rleg-', 'lleg+', 'lleg-']

FEATURES_TO_USE = [c for c in ALL_FEATURE_COLUMNS if
                   c not in ['subject', 'session', 'channel', 'category']]

VALID_STAGE_NAMES = list(STAGE_MAP.values()) + ["All"]


def get_master_feature_names(category_lookup):
    """
    Generates consistent column names containing id.
    Format: category-channel-stage-id-feature
    """
    MASTER_FEATURE_NAMES = []
    for ch in UNIQUE_CHANNELS:
        cat = category_lookup.get(ch)
        if cat in valid_features:
            for stage in VALID_STAGE_NAMES:
              for feat in valid_features[cat]:
                MASTER_FEATURE_NAMES.append(f"{cat}-{ch}-{stage}-{feat}")
    return MASTER_FEATURE_NAMES


# 3.1 helper functions-------------------------------------
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
    """Concordance correlation coefficient measures the agreement between two variables

    :x: time series of data
    :y: time series of data

    :return: Concordance
    """
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)


def extract_tsfresh_generic(data):
    """Calculates the specific generic features requested using tsfresh."""
    res = {
        "abs_energy": tsf.abs_energy(data),
        "abs_sum_changes": tsf.absolute_sum_of_changes(data),
        "mean_abs_change": tsf.mean_abs_change(data),
        "root_mean_square": tsf.root_mean_square(data),
        "variation_coeff": tsf.variation_coefficient(data),
        "median": np.median(data),
        "skewness": tsf.skewness(data),
        "kurtosis": tsf.kurtosis(data),
        "longest_strike_above_mean": tsf.longest_strike_above_mean(data),
        "longest_strike_below_mean": tsf.longest_strike_below_mean(data),
    }

    # Quantiles
    for q in [0.1, 0.25, 0.75, 0.9]:
        res[f"quantile_{q}"] = tsf.quantile(data, q)

    # Autocorrelations
    for lag in [1, 2, 5, 10]:
        res[f"autocorr_lag_{lag}"] = tsf.autocorrelation(data, lag)

    # Ratio beyond sigma (standard 1, 2, 3)
    res["ratio_beyond_r_sigma_2"] = tsf.ratio_beyond_r_sigma(data, 2)

    return res


# 3.2 Helper functions for EEG----------------------

def rms_amplitude(x):
    return np.sqrt(np.mean(x ** 2))


def peak_to_peak(x):
    return np.max(x) - np.min(x)


def signal_std(x):
    return np.std(x)


def max_abs_amplitude(x):
    return np.max(np.abs(x))


def percent_above_threshold(x, threshold):
    return np.mean(np.abs(x) > threshold)


def bandpower_from_psd(f, Pxx, band):
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(Pxx[idx], f[idx])


# 3.3 ECG helper functions-----------------------------------------------
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


def high_frequency_ratio(Pxx, f, cutoff=40):
    # f, Pxx = welch(x, fs=fs)
    hf_power = np.trapezoid(Pxx[f > cutoff], f[f > cutoff])
    total_power = np.trapezoid(Pxx, f)
    return hf_power / total_power if total_power > 0 else 0


# 3.4 EMG helper functions-------------------------------------
def powerline_ratio(Pxx, f, line_freq=50, bandwidth=1):
    # f, Pxx = welch(x, fs=fs)
    idx_line = np.logical_and(f >= line_freq - bandwidth,
                              f <= line_freq + bandwidth)
    line_power = np.trapezoid(Pxx[idx_line], f[idx_line])
    total_power = np.trapezoid(Pxx, f)
    return line_power / total_power if total_power > 0 else 0


# 3.5 Cross channel synchrony helper functions------------------------------------
def peak_lag(x, y, fs):
    """
    Calculates the average time difference between the closest peaks of two time series. It's the lag of peaks, and not the peak of lags.

    :x: time series of data
    :y: time series of data
    :fs: sampling frequency

    :return: average lag time between the two data
    """
    peaks_x, _ = find_peaks(x, distance=fs * 1)
    peaks_y, _ = find_peaks(y, distance=fs * 1)
    if len(peaks_x) == 0 or len(peaks_y) == 0:
        return np.nan
    lags = []
    for px in peaks_x:
        nearest = peaks_y[np.argmin(np.abs(peaks_y - px))]
        lags.append((nearest - px) / fs)
    return np.mean(lags)


def crosscorr_lag(x, y, fs):
    "Cross correlation of lags"
    corr = correlate(x - np.mean(x), y - np.mean(y), mode="full")
    lag_samples = np.argmax(corr) - (len(x) - 1)
    return lag_samples / fs


def phase_locking(x, y):
    """
    Difference of phase calculated from hibert transform
    """
    hx = hilbert(x)
    hy = hilbert(y)
    phase_x = np.angle(hx)
    phase_y = np.angle(hy)
    phase_diff = phase_x - phase_y
    return np.abs(np.mean(np.exp(1j * phase_diff)))


def synchrony_features(chest, abd, fs):
    """
    Calculation of the above synchrony features

    :chest: archaic variable name, when this was only used for respiratory data, can be any time series
    :abd: archaic variable name, when this was only used for respiratory data, can be any time series
    :fs: sampling frequency

    :return: all the calculated synchrony features in one variable
    """
    r, _ = pearsonr(chest, abd)
    ccc = concordance_cc(chest, abd)
    lag = crosscorr_lag(chest, abd, fs)
    peak_delay = peak_lag(chest, abd, fs)
    amp_ratio = np.std(chest) / (np.std(abd) + 1e-8)
    phase_lock = phase_locking(chest, abd)
    return {"r": r, "ccc": ccc, "lag": lag, "peak_lag": peak_delay,
            "amp_ratio": amp_ratio, "phase_lock": phase_lock}


def process_synchrony_pair(pair, raw_psg, STAGE_MAP, hypnogram):
    """
    Function that processes the synchrony pairs with the data divided into sleep stages

    :pair: a list with two elements containing the names of channels
    :raw_psg: polyosomnographic data read in from the edf files
    :STAGE_MAP: numerical coding of sleep stages
    :hypnogram: time mapping of sleep stage

    :return: the synchrony pair data coupled with the sleep stages
    """
    if not (pair[0] in raw_psg.ch_names and pair[1] in raw_psg.ch_names):
        return []

    fs = raw_psg.info["sfreq"]
    left, _ = raw_psg[pair[0]]
    right, _ = raw_psg[pair[1]]
    left, right = left.ravel(), right.ravel()

    # Align hypnogram
    hyp = np.repeat(hypnogram, int(fs * 30))
    min_len = min(len(left), len(right), len(hyp))
    left, right, hyp = left[:min_len], right[:min_len], hyp[:min_len]

    pair_results = []
    for stage_val, stage_name in STAGE_MAP.items():
        mask = (hyp == stage_val)
        if np.sum(mask) < 5:
            continue

        sync_feats = synchrony_features(left[mask], right[mask], fs)
        # We store these with metadata so we can merge them later
        pair_results.append({
            "pair": tuple(pair),
            "stage": stage_name,
            **sync_feats
        })
    return pair_results

# 3.6 Stage transition helper functions
def extract_local_transition_features(data, boundary_idx, fs, window_sec=60):
    """
    Calculates generic transition features around a single specific boundary point.
    """
    window_samples = int(window_sec * fs)
    
    before_start = max(0, boundary_idx - window_samples)
    before_end = boundary_idx

    after_start = boundary_idx
    after_end = min(len(data), boundary_idx + window_samples)

    before = data[before_start:before_end]
    after = data[after_start:after_end]

    if len(before) < fs or len(after) < fs:
        return {}

    return {
        "trans_mean_before": np.mean(before),
        "trans_mean_after": np.mean(after),
        "trans_std_before": np.std(before),
        "trans_std_after": np.std(after),
        "trans_recovery": np.mean(after) - np.mean(before),
        "trans_variability_change": np.std(after) - np.std(before)
    }


def extract_features(data, fs, ch_name, category,subject_id,session_id, stage_id,transition_id,transition_type, hypnogram_subset=None,processed=True):
    """
    Extracts features of a single channel in a category dependent manner. Every channel has generic features, 
    and category dependent features with the exception of EOG channels.

    :data: time series of data
    :fs: sampling frequency
    :ch_name: normalised name of the channel
    :category: what type of device was the channel recorded with
    :hypnogram_subset: determines what sleep stage it was in
    :processed: determines if the dataset was normalised or not

    :return: returns the calculated features in one variable
    """
    # -----------------------------
    # Generic Features (all channels)
    # -----------------------------

    time_1 = time.perf_counter()

    row_dict = extract_tsfresh_generic(data)
    row_dict.update({
        "subject": subject_id,
        "session": session_id,
        "channel": ch_name,
        "category": category,
        "stage": stage,
        "transition_id":transition_id,
        "transition_type":transition_type,
        "flatline_pct": np.mean(np.abs(np.diff(data)) < 1e-6),
        "clipping_pct": np.mean((data == data.min()) | (data == data.max()))
    })

    mean_val = data.mean()
    std_val = data.std()
    var_val = std_val ** 2

    f, Pxx = welch(data, fs, nperseg=min(4096, len(data)))
    total_power = np.trapezoid(Pxx, f)
    hf_power = np.sum(Pxx[f > 30]) if len(f) > 0 else 0
    lf_power = np.sum(Pxx[(f >= 0.5) & (f <= 4)]) if len(f) > 0 else 0
    hf_ratio = hf_power / total_power if total_power > 0 else 0
    lf_ratio = lf_power / total_power if total_power > 0 else 0
    row_dict.update({
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "hf_power_ratio": hf_ratio,
        "lf_power_ratio": lf_ratio,
    })

    time_2 = time.perf_counter()

    # Initialize row
    # print("Calculating generic features")

    #print(f"Generic feature calculation time: {time_2 - time_1}")

    # -----------------------------
    # EEG-specific Features
    # -----------------------------
    if category == "eeg":
        # print("Calculating eeg features")
        time_3 = time.perf_counter()
        # row_dict["sample_entropy"] = tsf.sample_entropy(data)
        # row_dict["approx_entropy"] = tsf.approximate_entropy(data, m=2,r=2 * std_val)
        # row_dict["perm_entropy"] = tsf.permutation_entropy(data)
        # row_dict["lempel_ziv"] = tsf.lempel_ziv_complexity(data, bins=5)
        # row_dict["time_rev_asym"] = tsf.time_reversal_asymmetry_statistic(data,lag=int(fs * 0.02))
        row_dict["num_peaks"] = tsf.number_peaks(data, n=5)
        # row_dict["fourier_entropy"] = tsf.fourier_entropy(data, bins=10)
        # row_dict["rms"] = rms_amplitude(data)
        #row_dict["rms"] = tsf.root_mean_square(data)#already in generic features
        row_dict["ptp"] = peak_to_peak(data)
        row_dict["max_amp"] = np.max(data)
        # row_dict["dominant_freq"] = dominant_frequency(data, fs)
        row_dict["dominant_freq"] = f[np.argmax(Pxx)]

        bands = {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 12],
            "sigma": [12, 16],
            "beta": [16, 30],
        }

        for band_name, band_range in bands.items():
            bp = bandpower_from_psd(f, Pxx, band_range)
            row_dict[
                f"rel_{band_name}"] = bp / total_power if total_power > 0 else 0

        row_dict["delta_beta_ratio"] = (
                row_dict["rel_delta"] / (row_dict["rel_beta"] + 1e-6)
        )
        # print("eeg feature calculations done")

        try:
            # print("calculating spindle features")
            relevant_stages = [1, 2]
            if hypnogram_subset is not None and np.any(
                    np.isin(hypnogram_subset, relevant_stages)):

                sp = yasa.spindles_detect(
                    data * (1 if processed else 1e6),
                    fs,
                    hypno=hypnogram_subset,
                    include=(1, 2)
                )

                sw = yasa.sw_detect(
                    data * (1 if processed else 1e6),
                    fs,
                    hypno=hypnogram_subset,
                    include=(1, 2),
                    coupling=True
                )

            else:
                sp = None
                sw = None

            if sp is not None:
                sp_summary = sp.summary()

                n_sp = len(sp_summary)
                recording_minutes = len(data) / fs / 60

                row_dict["spindle_count"] = n_sp
                row_dict["spindle_density"] = n_sp / recording_minutes

                row_dict["spindle_mean_amp"] = sp_summary["Amplitude"].mean()
                row_dict["spindle_amp_std"] = sp_summary["Amplitude"].std()

                row_dict["spindle_mean_duration"] = sp_summary["Duration"].mean()
                row_dict["spindle_duration_std"] = sp_summary["Duration"].std()

                row_dict["spindle_mean_freq"] = sp_summary["Frequency"].mean()
                row_dict["spindle_mean_relpower"] = sp_summary["RelPower"].mean()
                row_dict["spindle_mean_rms"] = sp_summary["RMS"].mean()

                row_dict["spindle_total_duration"] = sp_summary["Duration"].sum()

                
            else:
                print("The sp was None")
                row_dict["spindle_count"] = 0
                row_dict["spindle_density"] = 0
                row_dict["spindle_mean_amp"] = 0
                row_dict["spindle_amp_std"] = 0
                row_dict["spindle_mean_duration"] = 0
                row_dict["spindle_duration_std"] = 0
                row_dict["spindle_mean_freq"] = 0
                row_dict["spindle_mean_relpower"] = 0
                row_dict["spindle_mean_rms"] = 0
                row_dict["spindle_total_duration"] = 0
            # print("Spindle feature calculations done")

            if sw is not None:

                sw_summary = sw.summary()

                n_sw = len(sw_summary)
                recording_minutes = len(data) / fs / 60

                row_dict["sw_count"] = n_sw
                row_dict["sw_density"] = n_sw / recording_minutes

                row_dict["sw_mean_amp"] = sw_summary["PTP"].mean()
                row_dict["sw_amp_std"] = sw_summary["PTP"].std()

                row_dict["sw_mean_duration"] = sw_summary["Duration"].mean()
                row_dict["sw_duration_std"] = sw_summary["Duration"].std()

                row_dict["sw_mean_freq"] = sw_summary["Frequency"].mean()

                row_dict["sw_total_duration"] = sw_summary["Duration"].sum()

                # Coupling metrics
                if "SpindlePhase" in sw_summary.columns:

                    coupled = sw_summary.dropna(subset=["SpindlePhase"])

                    row_dict["sw_spindle_coupling_count"] = len(coupled)

                    row_dict["sw_spindle_coupling_ratio"] = (
                        len(coupled) / n_sw if n_sw > 0 else 0
                    )

                    row_dict["sw_spindle_phase_mean"] = coupled[
                        "SpindlePhase"].mean()
                    row_dict["sw_spindle_phase_std"] = coupled[
                        "SpindlePhase"].std()

                    row_dict["sw_spindle_delay_mean"] = coupled[
                        "SpindleDelay"].mean()
                    row_dict["sw_spindle_delay_std"] = coupled["SpindleDelay"].std()

                    savable=row_dict.copy()
                    savable=savable.reindex(valid_features["ecg"])
                    savable.to_csv("backup_ecg.csv",mode="a",header=False,index=False)
                else:

                    row_dict["sw_count"] = 0
                    row_dict["sw_density"] = 0
                    row_dict["sw_mean_amp"] = 0
                    row_dict["sw_amp_std"] = 0
                    row_dict["sw_mean_duration"] = 0
                    row_dict["sw_duration_std"] = 0
                    row_dict["sw_mean_freq"] = 0
                    row_dict["sw_total_duration"] = 0

                    row_dict["sw_spindle_coupling_count"] = 0
                    row_dict["sw_spindle_coupling_ratio"] = 0
                    row_dict["sw_spindle_phase_mean"] = 0
                    row_dict["sw_spindle_phase_std"] = 0
                    row_dict["sw_spindle_delay_mean"] = 0
                    row_dict["sw_spindle_delay_std"] = 0
                   
            else:

                row_dict["sw_count"] = 0
                row_dict["sw_density"] = 0
                row_dict["sw_mean_amp"] = 0
                row_dict["sw_amp_std"] = 0
                row_dict["sw_mean_duration"] = 0
                row_dict["sw_duration_std"] = 0
                row_dict["sw_mean_freq"] = 0
                row_dict["sw_total_duration"] = 0

                row_dict["sw_spindle_coupling_count"] = 0
                row_dict["sw_spindle_coupling_ratio"] = 0
                row_dict["sw_spindle_phase_mean"] = 0
                row_dict["sw_spindle_phase_std"] = 0
                row_dict["sw_spindle_delay_mean"] = 0
                row_dict["sw_spindle_delay_std"] = 0
                
            time_4 = time.perf_counter()
            #print(f"EEG calculation time: {time_4 - time_3}")
            #print(f"Total calculation time: {time_4 - time_1}")
            savable=row_dict.copy()
            savable=savable.reindex(valid_features["eeg"])
            savable.to_csv("backup_eeg.csv",mode="a",header=False,index=False)
        
        except Exception as e:
            print(f"EEG extraction error: {e}")
            traceback.print_exc()
            row_dict["spindle_count"] = 0
            row_dict["spindle_density"] = 0
            row_dict["spindle_mean_amp"] = 0
            row_dict["spindle_amp_std"] = 0
            row_dict["spindle_mean_duration"] = 0
            row_dict["spindle_duration_std"] = 0
            row_dict["spindle_mean_freq"] = 0
            row_dict["spindle_mean_relpower"] = 0
            row_dict["spindle_mean_rms"] = 0
            row_dict["spindle_total_duration"] = 0
            row_dict["sw_spindle_coupling_count"] = 0
            row_dict["sw_spindle_coupling_ratio"] = 0
            row_dict["sw_spindle_phase_mean"] = 0
            row_dict["sw_spindle_phase_std"] = 0
            row_dict["sw_spindle_delay_mean"] = 0
            row_dict["sw_spindle_delay_std"] = 0

            row_dict["sw_count"] = 0
            row_dict["sw_density"] = 0
            row_dict["sw_mean_amp"] = 0
            row_dict["sw_amp_std"] = 0
            row_dict["sw_mean_duration"] = 0
            row_dict["sw_duration_std"] = 0
            row_dict["sw_mean_freq"] = 0
            row_dict["sw_total_duration"] = 0

            row_dict["sw_spindle_coupling_count"] = 0
            row_dict["sw_spindle_coupling_ratio"] = 0
            row_dict["sw_spindle_phase_mean"] = 0
            row_dict["sw_spindle_phase_std"] = 0
            row_dict["sw_spindle_delay_mean"] = 0
            row_dict["sw_spindle_delay_std"] = 0
            savable=row_dict.copy()
            savable=savable.reindex(valid_features["eeg"])
            savable.to_csv("backup_eeg.csv",mode="a",header=False,index=False)

    # -----------------------------
    # ECG-specific Features
    # -----------------------------
    elif category == "ecg":
        # print("Calculating ecg features")
        time_5 = time.perf_counter()
        row_dict["cid_ce"] = tsf.cid_ce(data, normalize=True)
        row_dict["mean_2nd_deriv_central"] = tsf.mean_second_derivative_central(data)
        # row_dict["sample_entropy"] = tsf.sample_entropy(data)

        #row_dict["rms"] = rms_amplitude(data)
        row_dict["ptp"] = peak_to_peak(data)
        row_dict["max_amp"] = np.max(data)

        row_dict["flatline_duration"] = flatline_duration(data, fs,
                                                          tolerance=1.5)
        row_dict["is_flat"] = is_flat_signal(data)
        row_dict["amp_out_of_bounds"] = amplitude_out_of_bounds(data)
        row_dict["hf_ratio_signal"] = high_frequency_ratio(Pxx, f)
        row_dict["max_slope"] = max_slope(data, fs)

        # print("Ecg feature calculations done")
        try:
            '''
            ecg_cleaned = nk.ecg_clean(data, sampling_rate=fs)
            peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
            hr = nk.ecg_rate(peaks, sampling_rate=fs, desired_length=len(data))

            row_dict["mean_hr"] = np.mean(hr)

            if len(info["ECG_R_Peaks"]) > 2:
                sdnn = np.std(np.diff(info["ECG_R_Peaks"]) / fs * 1000)
            else:
                sdnn = np.nan
            #row_dict["hrv_sdnn"] = sdnn
            '''
            # print("Calculating HRV features")
            pre = Pre.Preprocessing(data, fs)
            filtered_signal = pre.notch(n_freq=60)  # 50 Hz for european powerline, 60 Hz for USA
            filtered_signal = Pre.Preprocessing(filtered_signal, fs).bpfilt()

            fp = Fp.FiducialPoints(filtered_signal, fs)
            # Two different peak detector algorithms:
            with HiddenPrints():  # to avoid long verbose of the peak detector functions
                jqrs_peaks = fp.jqrs()
                # xqrs_peaks = fp.xqrs()

            rr = np.diff(jqrs_peaks) / fs
            if len(rr) > 0:
                hr = 60 / rr
                row_dict["mean_hr"] = np.mean(hr)
            else:
                row_dict["mean_hr"] = np.nan

            hrv_biomarker = get_hrv_biomarkers(jqrs_peaks, fs)
            row_dict["AVNN"] = hrv_biomarker["AVNN"]
            row_dict["SDNN"] = hrv_biomarker["SDNN"]
            row_dict["RMSSD"] = hrv_biomarker["RMSSD"]
            row_dict["PNN50"] = hrv_biomarker["PNN50"]
            row_dict["SEM"] = hrv_biomarker["SEM"]
            row_dict["PIP"] = hrv_biomarker["PIP"]
            row_dict["IALS"] = hrv_biomarker["IALS"]
            row_dict["PSS"] = hrv_biomarker["PSS"]
            row_dict["PAS"] = hrv_biomarker["PAS"]

            row_dict["SD1"] = hrv_biomarker["SD1"]
            row_dict["SD2"] = hrv_biomarker["SD2"]
            row_dict["alpha_1"] = hrv_biomarker["alpha_1"]
            row_dict["alpha_2"] = hrv_biomarker["alpha_2"]
            row_dict["MSE"] = hrv_biomarker["MSE"]

            row_dict["TOTAL_POWER"] = hrv_biomarker["TOTAL_POWER"]
            row_dict["VLF_POWER"] = hrv_biomarker["VLF_POWER"]
            row_dict["LF_POWER"] = hrv_biomarker["LF_POWER"]
            row_dict["HF_POWER"] = hrv_biomarker["HF_POWER"]
            row_dict["VLF_NORM"] = hrv_biomarker["VLF_NORM"]
            row_dict["LF_NORM"] = hrv_biomarker["LF_NORM"]
            row_dict["HF_NORM"] = hrv_biomarker["HF_NORM"]
            row_dict["LF_HF_RATIO"] = hrv_biomarker["LF_HF_RATIO"]
            # print("HRV feature calculations done")
            time_6 = time.perf_counter()
            #print(f"ECG claculation time: {time_6 - time_5}")
            #print(f"Total time: {time_6 - time_1}")

        savable=row_dict.copy()
        savable=savable.reindex(valid_features["ecg"])
        savable.to_csv("backup_ecg.csv",mode="a",header=False,index=False)
        except Exception as e:
            print(f"ECG extraction error: {e}")
            traceback.print_exc()
            row_dict["mean_hr"] = np.nan
            row_dict["AVNN"] = np.nan
            row_dict["SDNN"] = np.nan
            row_dict["RMSSD"] = np.nan
            row_dict["PNN50"] = np.nan
            row_dict["SEM"] = np.nan
            row_dict["PIP"] = np.nan
            row_dict["IALS"] = np.nan
            row_dict["PSS"] = np.nan
            row_dict["PAS"] = np.nan

            row_dict["SD1"] = np.nan
            row_dict["SD2"] = np.nan
            row_dict["alpha_1"] = np.nan
            row_dict["alpha_2"] = np.nan
            row_dict["MSE"] = np.nan

            row_dict["TOTAL_POWER"] = np.nan
            row_dict["VLF_POWER"] = np.nan
            row_dict["LF_POWER"] = np.nan
            row_dict["HF_POWER"] = np.nan
            row_dict["VLF_NORM"] = np.nan
            row_dict["LF_NORM"] = np.nan
            row_dict["HF_NORM"] = np.nan
            row_dict["LF_HF_RATIO"] = np.nan
            savable=row_dict.copy()
            savable=savable.reindex(valid_features["ecg"])
            savable.to_csv("backup_ecg.csv",mode="a",header=False,index=False)

    # -----------------------------
    # EMG-specific Features
    # -----------------------------
    elif category in ["chin emg", "leg emg"]:
        # print("Calculating EMG features")
        try:
            time_7 = time.perf_counter()
            row_dict["powerline_ratio"] = powerline_ratio(Pxx, f)
            row_dict["hf_ratio_signal"] = high_frequency_ratio(Pxx, f)
            row_dict["spec_peak_min_ratio"] = np.max(Pxx) / (np.min(Pxx) + 1e-12)
            #row_dict["rms"] = tsf.root_mean_square(data)
            row_dict["energy_ratio_chunks"] = \
            tsf.energy_ratio_by_chunks(data, param=[{"num_segments": 10,
                                                     "segment_focus": 5}])[0][1]
            row_dict["num_peaks"] = tsf.number_peaks(data, n=3)
            time_8 = time.perf_counter()
            savable=row_dict.copy()
            savable=savable.reindex(valid_features["leg_emg"])
            savable.to_csv("backup_emg.csv",mode="a",header=False,index=False)
        except Exception as e:
            print(f"EMG extraction error: {e}")
            traceback.print_exc()
        #print(f"EMG feature calculations time{time_8 - time_7}")
        #print(f"Total_time{time_8 - time_1}")
    # -----------------------------
    # Respiratory Features
    # -----------------------------
    elif category == "resp":
        #time_9 = time.perf_counter()
        try:
            # print("Calculating respiratory features")
            if ch_name == "spo2":
                # SpO2 specific: Simple artifact removal, NO bio-respiratory cleaning
                resp_cleaned = nk.signal_filter(data, sampling_rate=fs,
                                                highcut=0.1,
                                                method='butterworth')
            elif ch_name in ["cflow", "cpres"]:
                # CPAP specific: Gentle lowpass
                resp_cleaned = nk.signal_filter(data, sampling_rate=fs,
                                                highcut=2.0)
            else:
                # ABD, CHEST, PTAF, AIRFLOW: Standard respiratory cleaning
                resp_cleaned = nk.rsp_clean(data, sampling_rate=fs)

            peaks, info = nk.rsp_peaks(resp_cleaned, sampling_rate=fs)

            intervals = np.diff(peaks["RSP_Peaks"]) / fs
            mean_int = np.mean(intervals) if len(intervals) > 0 else 0
            mean_rr = 60 / mean_int if (
                        len(intervals) > 0 and mean_int != 0) else np.nan
            resp_var = np.std(intervals) if len(intervals) > 0 else np.nan

            row_dict["mean_resp_rate"] = mean_rr
            row_dict["resp_variability"] = resp_var
            row_dict["linear_trend"] = tsf.linear_trend(data, param=[{"attr": "slope"}])[0][1]
            savable=row_dict.copy()
            savable=savable.reindex(valid_features["resp"])
            savable.to_csv("backup_resp.csv",mode="a",header=False,index=False)
            # row_dict["approx_entropy"] = tsf.approximate_entropy(data, m=2,r=2 * std_val)
            # currently problematic due to using an old scipy version
            #time_10 = time.perf_counter()
            #print(f"Respiratory feature calculations time:{time_10 - time_9}")
            #print(f"Total time: {time_10 - time_1}")
            '''
            if ch_name == "spo2":
                print("Calculating spo2 features")
                spo2_df = get_spo2_biomarkers(data, fs, patient_name="unknown")

                if len(spo2_df) > 0:

                    spo2_row = spo2_df.iloc[0]

                    for col in spo2_row.index:
                        if col not in ["patient", "time begin", "time end"]:
                            row_dict[col] = spo2_row[col]
                    #print("Spo2 feature calculations done")
            else:
                for col in [
                    "AV", "MED", "Min", "SD", "RG", "P",
                    "M", "ZC", "DI", "ODI", "DL_u", "DL_sd", "DA100_u",
                    "DA100_sd", "DAmax_u", "DAmax_sd","DD100_u", "DD100_sd",
                    "DDmax_u", "DDmax_sd", "DS_u","DS_sd", "TD_u", "TD_sd",
                    "CA", "CT", "POD", "AODmax","AOD100",
                       #"ApEn", "LZ", "CTM", "SampEn", "DFA",
                    "PRSAc", "PRSAad","PRSAos", "PRSAsb", "PRSAsa",
                    "AC", "PSD_total","PSD_band", "PSD_ratio", "PSD_peak"
                ]:
                    row_dict[col] = 0
            '''

        except Exception as e:
            print(f"SPO2 extraction error: {e}")
            traceback.print_exc()
            row_dict["mean_resp_rate"] = np.nan
            row_dict["resp_variability"] = np.nan
            row_dict["linear_trend"] = np.nan
            savable=row_dict.copy()
            savable=savable.reindex(valid_features["resp"])
            savable.to_csv("backup_resp.csv",mode="a",header=False,index=False)
            # row_dict["approx_entropy"] = np.nan
            '''
            for col in [
                "AV", "MED", "Min", "SD", "RG", "P",
                "M", "ZC", "DI", "ODI", "DL_u", "DL_sd", "DA100_u",
                "DA100_sd", "DAmax_u", "DAmax_sd",
                "DD100_u", "DD100_sd", "DDmax_u", "DDmax_sd", "DS_u",
                "DS_sd", "TD_u", "TD_sd", "CA", "CT", "POD", "AODmax","AOD100",
                #"ApEn", "LZ", "CTM", "SampEn", "DFA",
                "PRSAc", "PRSAad","PRSAos", "PRSAsb", "PRSAsa", "AC", "PSD_total",
                "PSD_band", "PSD_ratio", "PSD_peak"
            ]:
                row_dict[col] = 0
            '''
    elif category=="eog":
        savable=row_dict.copy()
        savable=savable.reindex(valid_features["eog"])
        savable.to_csv("backup_eog.csv",mode="a",header=False,index=False)
    return row_dict


def process_channel(ch_name, raw_psg, category_lookup, subject_id,session_id, hypnogram=None, processed=True,aggregate=True):
    try:
      data, _ = raw_psg[ch_name]
      data = data.ravel()
      fs = raw_psg.info["sfreq"]
    
      data = data[~np.isnan(data)]
      if len(data) < 10:
        return None
    
      category = category_lookup.get(ch_name, "unknown")
    
      if hypnogram is None:
        res = extract_features(data, fs, ch_name, category, None, processed)
        return [{**res, "stage": "All", "transition_id": 0, "transition_type": "None"}] if res else []
    
      # Align hypnogram and data
      hypnogram_expanded = np.repeat(hypnogram, int(fs * 30))
      min_len = min(len(data), len(hypnogram_expanded))
      data = data[:min_len]
      hypnogram_expanded = hypnogram_expanded[:min_len]
    
      # Find where stages change in the expanded hypnogram, this gives us the boundaries of continuous blocks
      stage_changes = np.where(np.diff(hypnogram_expanded) != 0)[0]
      block_boundaries = [0] + list(stage_changes + 1) + [min_len]
        
      stage_results = []
      transition_counter = 0
    
      # Iterate through continuous blocks chronologically
      for i in range(len(block_boundaries) - 1):
        start_idx = block_boundaries[i]
        end_idx = block_boundaries[i + 1]
            
        stage_val = hypnogram_expanded[start_idx]
        if stage_val not in STAGE_MAP:
          continue  # Skip undefined stages
                
        stage_name = STAGE_MAP[stage_val]
            
        # Determine transition type
        if end_idx < min_len:
          next_stage_val = hypnogram_expanded[end_idx]
          next_stage_name = STAGE_MAP.get(next_stage_val, "Unknown")
          transition_type = f"{stage_name}_to_{next_stage_name}"
        else:
          transition_type = f"{stage_name}_to_End"
    
            # Slice data for this specific continuous block
        block_data = data[start_idx:end_idx]
        block_hypno = hypnogram_expanded[start_idx:end_idx]
    
        if len(block_data) < 5: 
          continue
    
        features = extract_features(block_data, fs, ch_name, category, subject_id,session_id, stage_name,transition_counter,transition_type, block_hypno, processed)
        trans_features = extract_local_transition_features(data=data,boundary_idx=end_idx,fs=fs, window_sec=60)
        if trans_features:
            trans_row = {
                "subject": subject_id,
                "session": session_id,
                "channel": ch_name,
                "stage": stage_name,
                "category": category,
                "transition_id": transition_counter,
                "transition_type": transition_type,
                **trans_features
            }
            
            # Immediately append transition features to CSV
            trans_headers = metadata + transition_only_features
            trans_df = pd.DataFrame([trans_row]).reindex(columns=trans_headers)
            trans_df.to_csv("backup_transitions.csv", mode="a", header=False, index=False)
          
        # Merge the static block features with the dynamic transition features
        features.update(trans_features)
        stage_results.append(features)
                
        transition_counter += 1
    
      if not aggregate:
        return stage_results
      else:
        df = pd.DataFrame(stage_results)
        normal_cols = [c for c in df.columns if c not in (["transition_id", "transition_type"] +transition_only_features    )]
        normal_df = (df.groupby("stage")[normal_cols].mean(numeric_only=True).reset_index())
        transition_df = (df.groupby(["stage", "transition_type"])[transition_only_features].mean(numeric_only=True).reset_index())
        merged = transition_df.merge(normal_df,on="stage",how="left")
        merged["transition_id"] = -1
        
        return merged.to_dict("records")
                
    except Exception as e:
      print(f"Something went wrong in channel processing for {ch_name}: {e}")
      # Fallback: construct default single-row DataFrame filled with zeros
      try:
        master_feature_names = get_master_feature_names(category_lookup)
        fallback_dict = {feat: 0.0 for feat in master_feature_names}
      except Exception:
        fallback_dict = {}

      fallback_dict["channel"] = ch_name
      fallback_dict["stage"] = "All"
      fallback_dict["transition_id"] = 0
      fallback_dict["transition_type"] = "None"

      return pd.DataFrame([fallback_dict])


def assess_signal_quality(raw_psg, category_lookup, subject_id,session_id, hypnogram=None, processed=True):
    """Two part assessment of signal quality.

    First it processes each channel of the polysomnographic data in parallel.
    Then it processes the synchronity of various pairs in parallel.

    :raw_psg: polysomnographic data read in from the edf files
    :category_lookup: the channel_table.csv containing the various unique
    channel names and categories :hypnogram: time mapping of sleep stage
    :processed: determines if the dataset was normalised or not
    """
    try:
        #creating backup files
        category_features = {
            "emg": list(set(valid_features["leg emg"] + valid_features["chin emg"])),
            "eeg": valid_features["eeg"],
            "ecg": valid_features["ecg"],
            "resp": valid_features["resp"],
            "eog": valid_features["eog"],
        }
        for category, features in category_features.items():
            headers = metadata + list(dict.fromkeys(features))
            filepath = f"backup_{category}.csv"
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"Initialized {filepath} with {len(headers)} columns.")
        trans_filepath = "backup_transitions.csv"
        if not os.path.exists(trans_filepath):
            trans_headers = metadata + transition_only_features
            with open(trans_filepath, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(trans_headers)
        sync_filepath = "backup_synchrony.csv"
        if not os.path.exists(sync_filepath):
            sync_headers = ["subject", "session", "channel", "stage", "transition_id", "transition_type"] + synchrony_only_features
            with open(sync_filepath, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(sync_headers)
        #actually processing the channels
        results = Parallel(n_jobs=4, prefer="threads")(
            delayed(process_channel)(
                ch, raw_psg, category_lookup,subject_id,session_id, hypnogram, processed
            )
            for ch in raw_psg.ch_names
        )
    except Exception as e:
        print(f"Error during parallel channel processing: {e}")
        results = []

    # Filter out None or empty DataFrames, then concatenate into a single DataFrame
    df_list = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

    if df_list:
        flattened = pd.concat(df_list, ignore_index=True)
    else:
        # Fallback: create default zero-rows manually for every channel in raw_psg
        fallback_rows = []
        try:
            master_feature_names = get_master_feature_names(category_lookup)
        except Exception:
            master_feature_names = []

        for ch in raw_psg.ch_names:
            row = {feat: 0.0 for feat in master_feature_names}
            row["channel"] = ch
            row["stage"] = "All"
            row["transition_id"] = 0
            row["transition_type"] = "None"
            fallback_rows.append(row)

        flattened = pd.DataFrame(fallback_rows)

    if hypnogram is not None and not flattened.empty:
        synchron_pairs = [
            # in-modality features
            ["chest", "abd"],["e1", "e2"],["e1-m2", "e2-m1"],["f3", "f4"],
            ["f3-m2", "f4-m1"], ["c3", "c4"], ["c3-m2", "c4-m1"],["o1", "o2"],
            ["o1-m2", "o2-m1"], ["rleg+", "lleg+"],["rleg-", "lleg-"],
            # cross-modal features
            ["abd", "ecg"],["e2", "o2"],["e2-m1", "o2-m1"],  # eog compared to occipital
            ["f4", "chest"],["f4-m1", "chest"],["c4", "chest"],
            ["c4-m1", "chest"],["o1", "chest"],["o1-m2", "chest"],  # !very arbitrary selection of chest!
        ]

        try:
            sync_results = Parallel(n_jobs=-2, prefer="threads")(
                delayed(process_synchrony_pair)(
                    pair, raw_psg, STAGE_MAP, hypnogram
                )
                for pair in synchron_pairs
            )
        except Exception as e:
            print(f"Error during parallel synchrony processing: {e}")
            sync_results = []

        sync_flattened = [
            item for sublist in sync_results if sublist for item in sublist
        ]

        if sync_flattened:
            sync_rows = []
            for res in sync_flattened:
                p1, p2 = res["pair"]
                stage = res["stage"]
                feats = {k: v for k, v in res.items() if k not in ["pair", "stage"]}
        
                # Append to main dataframe in memory
                mask = (flattened["stage"] == stage) & (flattened["channel"] == p1)
                for col, val in feats.items():
                    flattened.loc[mask, col] = val
                
                # Build independent row for backup
                sync_rows.append({
                    "subject": subject_id,
                    "session": session_id,
                    "channel": p1,
                    "stage": stage,
                    "transition_id": -1,  # Standard placeholder for stage-level synchrony
                    "transition_type": "None",
                    **feats
                })
        
            # Immediately write synchrony features to disk
            if sync_rows:
                sync_headers = ["subject", "session", "channel", "stage", "transition_id", "transition_type"] + synchrony_only_features
                sync_df = pd.DataFrame(sync_rows).reindex(columns=sync_headers)
                sync_df.to_csv("backup_synchrony.csv", mode="a", header=False, index=False)

    return flattened


def stratified_pca_feature_selection(df, y, params):
    """
    Stratified feature selection based on NaN count filtering and RFE/PCA dimensionality reduction, 
    and the feature importances multiplied by the AUROCs of various stratas.

    :df: pandas dataframe
    :y: target variable
    :params: parameters of the XGBoost classifier

    :return: global vector of various features and weights calculated
    """
    base_meta_cols = ['BMI', 'Age', 'Race_Asian', 'Race_Black',
                      'Race_Others', 'Race_Unavailable',
                      'Race_White', 'Sex_Female', 'Sex_Male']
    base_annotation_cols = ["Ahi_auto", "Ar_auto", "Limb_auto", "W_pct",
                            "N1_pct", "N2_pct", "N3_pct", "R_pct", "efficiency",
                            "W_prob", "N3_prob", "Ar_prob"]

    categories = ['ecg', 'eeg', 'eog', 'emg', 'resp']
    stages = ["N1", "N2", "N3", "Wake", "REM"]

    groups = df["subject"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups))
    gkf = GroupKFold(n_splits=5)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df.drop(
        columns=["Cognitive_Impairment", "subject", "session"])
    X_test = test_df.drop(
        columns=["Cognitive_Impairment", "subject", "session"])

    y_train = y.iloc[train_idx]
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze() 
    y_test = y.iloc[test_idx]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze() 

    results = []

    component_nums = [10, 20, 30, 40, 50, 100, 200, 250, 300, 400, 500, 750]
    limits = [1, 2, 4, 6, 16, 17, 20, 25, 26, 29, 61, 129, 202, 265, 334, 451,
              461, 567, 597, 599]
    counter=0
    strata_list = []
    for cat in categories:
        strata_list.append(('category', cat))
    for stg in stages:
        strata_list.append(('stage', stg))
    for cat in categories:
        for stg in stages:
            strata_list.append(('both', (cat, stg)))

    for type, val in strata_list:

        if type == 'category':
            cols = [c for c in X_train.columns if val in c.lower()]

        elif type == 'stage':
            cols = [c for c in X_train.columns if val in c]

        else:
            cat, stg = val
            cols = [c for c in X_train.columns if cat in c.lower() and stg in c]

        if len(cols) == 0:
            continue

        unique_cols = list(dict.fromkeys(cols + base_meta_cols + base_annotation_cols))
        cols = [c for c in unique_cols if c in X_train.columns]
        #cols = [c for c in (cols + base_meta_cols + base_annotation_cols) if c in X_train.columns]

        X_tr_sub = X_train[cols]
        X_te_sub = X_test[cols]
        if isinstance(X_tr_sub, np.ndarray):
            X_tr_sub = pd.DataFrame(X_tr_sub)
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train) 

        strata_model = XGBClassifier(**params, enable_categorical=True,
                                     tree_method="hist", random_state=42)
        strata_model.fit(X_tr_sub, y_train)

        # probs = strata_model.predict_proba(X_te_sub)[:,1]
        # auc = metrics.roc_auc_score(y_test, probs)

        aucs = []

        for tr_idx, val_idx in gkf.split(X_tr_sub, y_train, groups[train_idx]):
            if isinstance(X_tr_sub, np.ndarray):
                X_tr_sub = pd.DataFrame(X_tr_sub)
            if isinstance(y_train, np.ndarray):
                y_train = pd.Series(y_train)
            X_tr_sub = X_tr_sub.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True) 
            X_tr_fold = X_tr_sub.iloc[tr_idx]
            X_val_fold = X_tr_sub.iloc[val_idx]
            y_tr_fold = y_train.iloc[tr_idx]
            y_val_fold = y_train.iloc[val_idx]

            strata_model.fit(X_tr_fold, y_tr_fold)
            probs = strata_model.predict_proba(X_val_fold)[:, 1]
            aucs.append(metrics.roc_auc_score(y_val_fold, probs))

        auc = np.mean(aucs)

        results.append((cols, strata_model.feature_importances_, auc))

        best_auc = auc
        best_tr_subset = X_tr_sub
        best_te_subset = X_te_sub

        
        # NaN filtering based
        nan_counts_per_col = X_tr_sub.isna().sum()
        unique_subsets = {}
        distinct_counts = sorted(nan_counts_per_col.unique())
    
        for thresh in distinct_counts:
            cols_to_keep = nan_counts_per_col[ nan_counts_per_col <= thresh].index.tolist()
            subset_key = tuple(sorted(cols_to_keep))
            if subset_key and subset_key not in unique_subsets:
                unique_subsets[subset_key] = thresh
        
        '''for limit in limits:  # pre filtering using NaN counts
            if limit > len(X_tr_sub):
                continue
            X_tr_filt = X_tr_sub.dropna(thresh=limit, axis=1)
            X_te_filt = X_te_sub[X_tr_filt.keys()]
           '''
        #many limits in the old method had the exact same feature set, which is both redundant and makes the end product biased
        for subset_cols, actual_limit in unique_subsets.items():
            X_tr_filt = X_tr_sub[list(subset_cols)]
            X_te_filt = X_te_sub[list(subset_cols)]
            filter_model = XGBClassifier(**params, random_state=42,
                                         enable_categorical=True,
                                         tree_method="hist")
            # filter_model.fit(X_tr_filt, y_train)
            # probs = filter_model.predict_proba(X_te_filt)[:, 1]
            # auc = metrics.roc_auc_score(y_test, probs)

            aucs = []

            for tr_idx, val_idx in gkf.split(X_tr_filt, y_train, groups[train_idx]):
                if isinstance(X_tr_filt,np.ndarray):
                    X_tr_filt = pd.DataFrame(X_tr_filt)
                if isinstance(y_train,np.ndarray):
                    y_train = pd.Series(y_train)
                X_tr_filt=X_tr_filt.reset_index(drop=True)
                y_train=y_train.reset_index(drop=True) 
                X_tr_fold = X_tr_filt.iloc[tr_idx]
                X_val_fold = X_tr_filt.iloc[val_idx]
                y_tr_fold = y_train.iloc[tr_idx]
                y_val_fold = y_train.iloc[val_idx]

                filter_model.fit(X_tr_fold, y_tr_fold)
                probs = filter_model.predict_proba(X_val_fold)[:, 1]
                aucs.append(metrics.roc_auc_score(y_val_fold, probs))

            auc = np.mean(aucs)

            results.append((X_tr_filt.keys(), filter_model.feature_importances_, auc))
            if best_auc < auc:
                best_auc = auc
                best_tr_subset = X_tr_filt
                best_te_subset = X_te_filt
        # PCA and RFE versions
        for comp in component_nums:

            if best_tr_subset.shape[1] < comp:
                continue

            imputer = SimpleImputer(strategy='median', add_indicator=True)
            scaler = StandardScaler()
            pca = PCA(n_components=comp, random_state=42)

            X_tr = scaler.fit_transform(imputer.fit_transform(best_tr_subset))
            X_te = scaler.transform(imputer.transform(best_te_subset))

            n_samples, n_features = X_tr.shape
            max_possible_components = min(n_samples, n_features)

            if comp > max_possible_components:
                continue

            '''corr_matrix = best_tr_subset.corr().abs()
            avg_corr = np.nanmean(
                corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)])
'''#takes too much memory to calculate
            X_filled = best_tr_subset.fillna(best_tr_subset.mean()).values
            mean = X_filled.mean(axis=0)
            std = X_filled.std(axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
            Z = (X_filled - mean) / std
            N, M = Z.shape
            row_sums = Z.sum(axis=1)
            sum_of_all_elements = np.sum(row_sums ** 2) / N
            sum_off_diagonal = sum_of_all_elements - M
            sum_upper_tri = sum_off_diagonal / 2
            num_upper_elements = M * (M - 1) / 2
            avg_corr = sum_upper_tri / num_upper_elements


            if avg_corr > 0.6 or type == "stage" or val == "resp":
                # PCA because stages and respiratory data are multimodal
                X_tr_pca = pca.fit_transform(X_tr)
                X_te_pca = pca.transform(X_te)

                pca_model = XGBClassifier(**params, tree_method="hist",
                                          enable_categorical=True,
                                          random_state=42)
                # pca_model.fit(X_tr_pca, y_train)
                # probs = pca_model.predict_proba(X_te_pca)[:,1]
                # auc = metrics.roc_auc_score(y_test, probs)

                aucs = []

                for tr_idx, val_idx in gkf.split(X_tr_pca, y_train, groups[train_idx]):
                    if isinstance(X_tr_pca,np.ndarray):
                        X_tr_pca = pd.DataFrame(X_tr_pca)
                    if isinstance(y_train,np.ndarray):
                        y_train = pd.Series(y_train)
                    X_tr_pca=X_tr_pca.reset_index(drop=True)
                    y_train=y_train.reset_index(drop=True) 
                    X_tr_fold = X_tr_pca.iloc[tr_idx]
                    X_val_fold = X_tr_pca.iloc[val_idx]
                    y_tr_fold = y_train.iloc[tr_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    pca_model.fit(X_tr_fold, y_tr_fold)
                    probs = pca_model.predict_proba(X_val_fold)[:, 1]
                    aucs.append(metrics.roc_auc_score(y_val_fold, probs))

                auc = np.mean(aucs)

                loadings = np.abs(pca.components_)
                weighted_loadings = np.dot(pca_model.feature_importances_,
                                           loadings)

                results.append(
                    (best_tr_subset.columns.tolist(), weighted_loadings, auc))
            else:
                # RFE, because in our experiences gave better results on data of singular modality
                step_size = max(1, int((len(best_tr_subset.columns) - comp) / 10))  # Recursion is very slow on high dimensional data
                RFE_model = RFE(XGBClassifier(**params, tree_method="hist",
                                              enable_categorical=True,
                                              random_state=42),
                                n_features_to_select=comp, step=step_size)
                # RFE_model.fit(X_tr, y_train)
                # probs = RFE_model.predict_proba(X_te)[:, 1]
                # auc = metrics.roc_auc_score(y_test, probs)
                aucs = []

                for tr_idx, val_idx in gkf.split(X_tr, y_train, groups[train_idx]):
                    if isinstance(X_tr,np.ndarray):
                        X_tr=pd.DataFrame(X_tr)
                    if isinstance(y_train,np.ndarray):
                        y_train=pd.Series(y_train)
                    X_tr=X_tr.reset_index(drop=True)
                    y_train=y_train.reset_index(drop=True) 
                    X_tr_fold = X_tr.iloc[tr_idx]
                    X_val_fold = X_tr.iloc[val_idx]
                    y_tr_fold = y_train.iloc[tr_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    RFE_model.fit(X_tr_fold, y_tr_fold)
                    probs = RFE_model.predict_proba(X_val_fold)[:, 1]
                    aucs.append(metrics.roc_auc_score(y_val_fold, probs))

                auc = np.mean(aucs)

                selected_mask = RFE_model.support_
                selected_features = best_tr_subset.columns[
                    selected_mask].tolist()

                results.append((best_tr_subset.columns.tolist(),
                                RFE_model.estimator_.feature_importances_, auc))

    # accumulate global weights
    feature_weights = {}

    for cols, weights, auc in results:

        if auc < 0.53:
            continue

        for f, w in zip(cols, weights):
            feature_weights[f] = feature_weights.get(f, 0) + w * auc

    if not feature_weights:
        print("!!! Applying fallback: returning all available columns.")
            # Get every column except the protected metadata/labels
        all_potential_cols = [c for c in df.columns if c not in ["Cognitive_Impairment", "subject","session"]]
        return all_potential_cols 
    sorted_features = sorted(feature_weights.items(),key=lambda x: x[1],reverse=True)

    ranked = [f[0] for f in sorted_features]

    return ranked
