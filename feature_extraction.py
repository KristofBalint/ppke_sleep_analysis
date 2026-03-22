import pandas as pd
import mne
import numpy as np
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
import wfdb  # wfdb is needed for pecg
from pecg.ecg import Biomarkers as Bm
from get_hrv_bm import get_hrv_biomarkers
from joblib import Parallel, delayed
from get_spo2_bm import get_spo2_biomarkers  # currently problematic due to using an old scipy version #even after throwing out the complexity metics, it was still slow as dicks
from utils import unfuck_old

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

ALL_FEATURE_COLUMNS = ["subject", "session", "channel", "stage", "category",
                       # generic
                       "mean", "std", "variance", "skewness", "kurtosis",
                       "flatline_pct", "clipping_pct", "hf_power_ratio",
                       "lf_power_ratio",
                       # EEG
                       "rms", "ptp", "max_amp", "dominant_freq", "rel_delta",
                       "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
                       "delta_beta_ratio",
                       # Spindle Features (still in EEG)
                       "spindle_count", "spindle_density", "spindle_mean_amp",
                       "spindle_amp_std", "spindle_mean_duration",
                       "spindle_duration_std", "spindle_mean_freq",
                       "spindle_mean_relpower", "spindle_mean_rms",
                       "spindle_total_duration",
                        # Slow Wave Features (still in EEG)
                        "sw_count", "sw_density", "sw_mean_amp", "sw_amp_std",
                        "sw_mean_duration", "sw_duration_std", "sw_mean_freq",
                        "sw_total_duration",
                        # Slow Wave–Spindle Coupling (still in EEG)
                        "sw_spindle_coupling_count",
                        "sw_spindle_coupling_ratio",
                        "sw_spindle_phase_mean",
                        "sw_spindle_phase_std",
                        "sw_spindle_delay_mean",
                        "sw_spindle_delay_std",
                       # ECG
                       "flatline_duration", "is_flat", "amp_out_of_bounds",
                       "hf_ratio_signal", "max_slope", "mean_hr",
                       # HRV metrics (still in ECG)
                       "AVNN", "SDNN", "RMSSD", "PNN50", "SEM", "PIP", "IALS",
                       "PSS", "PAS", "SD1", "SD2", "alpha_1", "alpha_2", "MSE",
                       "TOTAL_POWER", "VLF_POWER", "LF_POWER", "HF_POWER",
                       "VLF_NORM", "LF_NORM", "HF_NORM", "LF_HF_RATIO",
                       # EMG
                       "powerline_ratio", "spec_peak_min_ratio",
                       # Resp
                       "mean_resp_rate", "resp_variability"
                       # SpO2 biomarkers (still in Resp) #compatibility issues with old version

                       #                  "AV", "MED", "Min", "SD", "RG", "P",
                       # "M", "ZC", "DI", "ODI", "DL_u", "DL_sd", "DA100_u",
                       # "DA100_sd", "DAmax_u", "DAmax_sd", "DD100_u", "DD100_sd",
                       # "DDmax_u", "DDmax_sd", "DS_u", "DS_sd", "TD_u", "TD_sd",
                       # "CA", "CT", "POD", "AODmax", "AOD100",
                       # "ApEn", "LZ", "CTM", "SampEn","DFA",
                       # "PRSAc", "PRSAad", "PRSAos", "PRSAsb", "PRSAsa", "AC",
                       # "PSD_total", "PSD_band", "PSD_ratio", "PSD_peak"

                       ]

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
MASTER_FEATURE_NAMES = [
    f"{ch}_{stage}_{feat}"
    for ch in UNIQUE_CHANNELS
    for stage in VALID_STAGE_NAMES
    for feat in FEATURES_TO_USE
]


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
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)


def extract_features(data, fs, ch_name, category, hypnogram_subset=None,
                     processed=True):
    # -----------------------------
    # Generic Features (all channels)
    # -----------------------------
    mean_val = data.mean()
    std_val = data.std()
    var_val = std_val ** 2
    skew_val = skew(data)
    kurt_val = kurtosis(data)

    flat_pct = np.mean(np.abs(np.diff(data)) < 1e-6)

    clip_pct = np.mean(
        (data == data.min()) |
        (data == data.max())
    )

    f, Pxx = welch(data, fs, nperseg=min(4096, len(data)))
    total_power = np.trapezoid(Pxx, f)
    hf_power = np.sum(Pxx[f > 30]) if len(f) > 0 else 0
    lf_power = np.sum(Pxx[(f >= 0.5) & (f <= 4)]) if len(f) > 0 else 0

    hf_ratio = hf_power / total_power if total_power > 0 else 0
    lf_ratio = lf_power / total_power if total_power > 0 else 0

    # Initialize row
    # print("Calculating generic features")
    row_dict = {
        "channel": ch_name,
        "category": category,
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "flatline_pct": flat_pct,
        "clipping_pct": clip_pct,
        "hf_power_ratio": hf_ratio,
        "lf_power_ratio": lf_ratio,
    }
    # print("Generic feature calculation done")

    # -----------------------------
    # EEG-specific Features
    # -----------------------------
    if category == "eeg":
        # print("Calculating eeg features")
        row_dict["rms"] = rms_amplitude(data)
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

                row_dict["spindle_mean_duration"] = sp_summary[
                    "Duration"].mean()
                row_dict["spindle_duration_std"] = sp_summary["Duration"].std()

                row_dict["spindle_mean_freq"] = sp_summary["Frequency"].mean()
                row_dict["spindle_mean_relpower"] = sp_summary[
                    "RelPower"].mean()
                row_dict["spindle_mean_rms"] = sp_summary["RMS"].mean()

                row_dict["spindle_total_duration"] = sp_summary[
                    "Duration"].sum()

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

                    row_dict["sw_spindle_phase_mean"] = coupled["SpindlePhase"].mean()
                    row_dict["sw_spindle_phase_std"] = coupled["SpindlePhase"].std()

                    row_dict["sw_spindle_delay_mean"] = coupled["SpindleDelay"].mean()
                    row_dict["sw_spindle_delay_std"] = coupled["SpindleDelay"].std()

                else:
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

        except Exception as e:
            print(f"EEG extraction error: {e}")
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

    # -----------------------------
    # ECG-specific Features
    # -----------------------------
    elif category == "ecg":
        # print("Calculating ecg features")
        row_dict["rms"] = rms_amplitude(data)
        row_dict["ptp"] = peak_to_peak(data)
        row_dict["max_amp"] = np.max(data)

        row_dict["flatline_duration"] = flatline_duration(data, fs,
                                                          tolerance=1.5)
        row_dict["is_flat"] = is_flat_signal(data)
        row_dict["amp_out_of_bounds"] = amplitude_out_of_bounds(data)
        row_dict["hf_ratio_signal"] = high_frequency_ratio(data, fs)
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
            filtered_signal = pre.notch(
                n_freq=60)  # 50 Hz for european powerline, 60 Hz for USA
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


        except Exception as e:
            print(f"ECG extraction error: {e}")
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



    # -----------------------------
    # EMG-specific Features
    # -----------------------------
    elif category in ["chin emg", "leg emg"]:
        # print("Calculating EMG features")
        row_dict["powerline_ratio"] = powerline_ratio(data, fs)
        row_dict["hf_ratio_signal"] = high_frequency_ratio(data, fs)
        row_dict["spec_peak_min_ratio"] = spectral_peak_to_min_ratio(data, fs)
        # print("EMG feature calculations done")
    # -----------------------------
    # Respiratory Features
    # -----------------------------
    elif category == "resp":

        try:
            # print("Calculating respiratory features")
            resp_cleaned = nk.rsp_clean(data, sampling_rate=fs)
            peaks, info = nk.rsp_peaks(resp_cleaned, sampling_rate=fs)

            intervals = np.diff(peaks["RSP_Peaks"]) / fs
            mean_int = np.mean(intervals)
            mean_rr = 60 / mean_int if (
                    len(intervals) > 0 and mean_int != 0) else np.nan
            resp_var = np.std(intervals) if len(intervals) > 0 else np.nan

            row_dict["mean_resp_rate"] = mean_rr
            row_dict["resp_variability"] = resp_var
            # currently problematic due to using an old scipy version
            '''
            print("Respiratory feature calculations done")
            if ch_name == "spo2":
                print("Calculating spo2 features")
                spo2_df = get_spo2_biomarkers(data, fs, patient_name="unknown")

                if len(spo2_df) > 0:

                    spo2_row = spo2_df.iloc[0]

                    for col in spo2_row.index:
                        if col not in ["patient", "time begin", "time end"]:
                            row_dict[col] = spo2_row[col]
                    print("Spo2 feature calculations done")
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
            row_dict["mean_resp_rate"] = np.nan
            row_dict["resp_variability"] = np.nan
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
    return row_dict


def process_channel(ch_name, raw_psg, category_lookup, hypnogram=None,processed=True):
    data, _ = raw_psg[ch_name]
    data = data.ravel()
    fs = raw_psg.info["sfreq"]

    data = data[~np.isnan(data)]

    if len(data) < 10:
        return None

    category = category_lookup.get(ch_name, "unknown")

    if hypnogram is None:
        # Fallback to whole-signal processing if no hypnogram
        res = extract_features(data, fs, ch_name, category, None, processed)
        return [{**res, "stage": "All"}] if res else []

    hypnogram = np.repeat(hypnogram, int(fs*30)) #Not having this caused so much problems.

    min_len = min(len(data), len(hypnogram))
    data = data[:min_len]
    hypnogram = hypnogram[:min_len]

    #print(f"DEBUG: Channel {ch_name} data length: {len(data)}")
    #print(f"DEBUG: Hypnogram data length: {len(hypnogram)}")

    stage_results = []
    unique_stages = np.unique(hypnogram)
    #print(unique_stages)
    for stage_val in unique_stages:
        #stage_name = STAGE_MAP.get(stage_val, f"Unknown_{stage_val}")
        if stage_val not in STAGE_MAP:
            continue  # We don't need the undefined stages
        stage_name = STAGE_MAP[stage_val]

        # Mask data for this specific stage
        mask = (hypnogram == stage_val)
        stage_data = data[mask]
        stage_hypno = hypnogram[mask]

        #print(  f"DEBUG: {ch_name} Stage {stage_name} has {len(stage_data)} samples.")
        #print(f"DEBUG: Threshold is {fs * 10}")

        if len(stage_data) < fs * 10:  # Skip stages shorter than 10 seconds
        #if len(stage_data)<10: #I thought 10 second stages were too long, but it turned out the hypnograms were 30 seconds long in each datapoint
            continue

        features = extract_features(stage_data, fs, ch_name, category,
                                    stage_hypno, processed)
        if features:
            features["stage"] = stage_name
            stage_results.append(features)

    return stage_results


def assess_signal_quality(raw_psg, category_lookup, hypnogram=None,
                          processed=True):
    quality_metrics = []

    results = Parallel(n_jobs=2)(
        delayed(process_channel)(
            ch, raw_psg, category_lookup, hypnogram, processed
        )
        for ch in raw_psg.ch_names
    )

    flattened = []
    for r in results:
        if r:
            flattened.extend(r)

    return pd.DataFrame(flattened)


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


def dominant_frequency(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), fs * 4))
    return f[np.argmax(Pxx)]


def bandpower(x, fs, band, relative=False):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), fs * 4))
    idx = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.trapezoid(Pxx[idx], f[idx])

    if relative:
        total_power = np.trapezoid(Pxx, f)
        return band_power / total_power if total_power > 0 else 0

    return band_power


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


def high_frequency_ratio(x, fs, cutoff=40):
    f, Pxx = welch(x, fs=fs)
    hf_power = np.trapezoid(Pxx[f > cutoff], f[f > cutoff])
    total_power = np.trapezoid(Pxx, f)
    return hf_power / total_power if total_power > 0 else 0


# 3.4 EMG helper functions-------------------------------------
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
        segment = x[i * win_samples:(i + 1) * win_samples]
        values.append(feature_fn(segment, fs=fs, **kwargs))

    return np.array(values)