import numpy as np
import hrv_bms as hrv


def get_hrv_biomarkers(peaks, fs):
    """
           This function computes heart rate variability (HRV) biomarkers from peak indices.
           When applied to PPG signals, the function effectively computes Beat Rate Variability (BRV).

           :param peaks: Indices of heartbeat peaks in the signal in seconds(sample positions).
           :type peaks: array-like
           :param fs: Sampling frequency of the signal in Hz.
           :type fs: float

           :return: Dictionary of computed HRV metrics.
           """

    # Calculate RR intervals in seconds (differences between successive peaks)
    rr_intervals = np.diff(peaks) / fs

    if len(rr_intervals) < 5:
        # Return a dictionary of NaNs with the same keys
        return {
            "AVNN": np.nan, "SDNN": np.nan, "RMSSD": np.nan, "PNN50": np.nan,
            "SEM": np.nan, "PIP": np.nan, "IALS": np.nan, "PSS": np.nan,
            "PAS": np.nan, "SD1": np.nan, "SD2": np.nan, "alpha_1": np.nan,
            "alpha_2": np.nan, "MSE": np.nan, "TOTAL_POWER": np.nan,
            "VLF_POWER": np.nan, "LF_POWER": np.nan, "HF_POWER": np.nan,
            "VLF_NORM": np.nan, "LF_NORM": np.nan, "HF_NORM": np.nan,
            "LF_HF_RATIO": np.nan
        }

        # Extract all available HRV metrics from the RR intervals
    try:
        all_metrics = hrv.get_all_metrics(rr_intervals)
        return all_metrics
    except Exception as e:
        print(f"Internal HRV calculation error: {e}")
        # Return NaNs if a specific math operation still fails
        return {k: np.nan for k in
                ["AVNN", "SDNN", "RMSSD", "PNN50", "SEM", "PIP",
                 "IALS", "PSS", "PAS", "SD1", "SD2", "alpha_1",
                 "alpha_2", "MSE", "TOTAL_POWER", "VLF_POWER",
                 "LF_POWER", "HF_POWER", "VLF_NORM", "LF_NORM",
                 "HF_NORM", "LF_HF_RATIO"]}