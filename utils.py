import os, sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
import traceback


class HiddenPrints:
    """
        Context manager to temporarily suppress console output.

        """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def plot_leads(signals, fs, leads=range(12), labels=[''], peaks=None, figsize=(8, 4)):
    """
        Plots multiple ECG leads from one or more signals with optional peak markers.

        :param signals: List of 2D numpy arrays, each representing an ECG signal (samples x leads).
        :type signals: list of np.ndarray
        :param fs: Sampling frequency of the signals in Hz.
        :type fs: float
        :param leads: Indices of leads to plot. Defaults to all 12 leads.
        :type leads: iterable of int, optional
        :param labels: List of labels for each signal, used in the legend.
        :type labels: list of str, optional
        :param peaks: Dictionary of peak positions to overlay on the plots, keys are labels and values are 2D arrays (samples x leads).
        :type peaks: dict, optional
        :param figsize: Figure size in inches (width, height).
        :type figsize: tuple, optional

        :return: None
        """
    t = np.linspace(0, signals[0].shape[0] / fs - 1, signals[0].shape[0])
    fig, axes = plt.subplots(len(leads), 1, figsize=figsize)
    for ind, l in enumerate(leads):
        for s, signal_ in enumerate(signals):
            axes[ind].plot(t, signal_[:, l], label=labels[s])
        if peaks is not None:
            for key, value in peaks.items():
                peaks_i = value[:, l]
                axes[ind].plot(t[peaks_i], signal_[peaks_i, l], 'o', label=key)
    fig.supylabel('ECG (mV)')
    fig.supxlabel('Time (seconds)')

    plt.legend(loc='upper right')
    plt.show()

def unfuck_old():
    if not hasattr(scipy.stats,
                   'median_absolute_deviation'):  # silly little renames because pobm is an old package with outdated functions
        scipy.stats.median_absolute_deviation = scipy.stats.median_abs_deviation
    if not hasattr(scipy.signal, 'hamming'):
        scipy.signal.hamming = scipy.signal.windows.hamming
    if not hasattr(np, 'RankWarning'):
        np.RankWarning = np.exceptions.RankWarning
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    if not hasattr(np, 'infty'):
        np.infty = np.inf
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    if not hasattr(np, 'alltrue'):
        np.alltrue = np.all
    if not hasattr(np, 'sometrue'):
        np.sometrue = np.any
    if not hasattr(np, 'in1d'):
        np.in1d = np.isin
    try:
        from xgboost import XGBClassifier
        if not hasattr(XGBClassifier, "__sklearn_tags__"):
            def dummy_sklearn_tags(self):
                # Return scikit-learn's standard default tags for a classifier
                return {
                    "estimator_type": "classifier",
                    "binary_only": False,
                    "multilabel": False,
                    "multioutput": False,
                    "no_validation": False,
                    "poor_score": False,
                    "requires_y": True,
                    "requires_fit": True,
                }
            XGBClassifier.__sklearn_tags__ = dummy_sklearn_tags
    except ImportError:
        pass

class DebugMessageCollector:
    """
    Collects execution status and failures across stages and checkpoints.
    Designed to store debug metrics in a 2D matrix (Checkpoints x Attributes).
    """

    # Defined Checkpoints (Dimension 2)
    CHECKPOINTS = [
        # train_model setup
        "entered","patients found","sleepfm initialized",
        
        # per-patient input loading
        "patient started","patient loaded","demographics loaded",
        "algorithmic annotations loaded", "physiological loaded", "hypnogram processed",
        "diagnosis loaded",
    
        # handcrafted branch
        "feature extraction returned","patient feature row created",
    
        # SleepFM branch
        "sleepfm modalities built","sleepfm preprocessing started", "sleepfm embedding returned",
    
        # patient completion
        "patient appended",
    
        # dataset construction
        "full dataframe created","boolean conversion complete","sleepfm matrix created","alignment verified",
    
        # feature selection
        "feature selection started","feature ranking returned","feature subset selected",
    
        # final handcrafted preprocessing
        "final imputation complete","final scaling complete",
    
        # hybrid model
        "hybrid model created","hybrid training started","hybrid training complete",
    
        # saving
        "model saved",
    
        # feature_extraction internal stages
        "extract_physiological_features","channels standardized","quality assessment started",
        "EEG processed","ECG processed","EMG processed","resp processed","EOG processed",
        "parallel processing complete","parallel results flattened","synchrony complete",
        "pivot complete","reindex complete","returning dataframe",
    ]

    # Defined Metrics/Attributes per Checkpoint (Dimension 1)
    ATTRIBUTES = [
        "last_success_patient",
        "last_success_iteration",
        "first_failure_patient",
        "first_failure_iteration",
        "failure count",
        "last_message",
        "traceback",
    ]

    def __init__(self, name=""):
        self.name = name

        # 2D Data Structure: { checkpoint: { attribute: value } }
        self.matrix = {
            cp: {attr: None for attr in self.ATTRIBUTES}
            for cp in self.CHECKPOINTS
        }

        self.flags = {}
        self.children = []

    def update_success(self, checkpoint: str, patient_id=None, iteration=None, message=""):
        """Records a successful run for a given checkpoint."""
        if checkpoint not in self.matrix:
            # Handle dynamic checkpoints gracefully if needed
            self.matrix[checkpoint] = {attr: None for attr in self.ATTRIBUTES}

        self.matrix[checkpoint]["last_success_patient"] = patient_id
        self.matrix[checkpoint]["last_success_iteration"] = iteration
        self.matrix[checkpoint]["last_message"] = message

    def update_failure(self, checkpoint: str, patient_id=None, iteration=None, message="", exc=None):
        """Records the first failure details for a given checkpoint."""
        if checkpoint not in self.matrix:
            self.matrix[checkpoint] = {attr: None for attr in self.ATTRIBUTES}

        cp_data = self.matrix[checkpoint]
        
        # Store first failure patient and iteration if not already set
        if cp_data["first_failure_patient"] is None:
            cp_data["first_failure_patient"] = patient_id
            cp_data["first_failure_iteration"] = iteration
            if exc is not None:
                cp_data["traceback"] = traceback.format_exc()
            elif isinstance(message, Exception):
                cp_data["traceback"] = str(message)
        if cp_data["failure count"] is None:
            cp_data["failure count"]=0
        cp_data["failure count"]+=1
        cp_data["last_message"] = str(message)

    def flag(self, name, value=True):
        if name not in self.flags:
            self.flags[name] = value

    def merge(self, other):
        self.children.append(other)

    def to_dataframe(self):
        """Returns the 2D debug vector as a Pandas DataFrame (Checkpoints as rows, Attributes as columns)."""
        import pandas as pd
        return pd.DataFrame.from_dict(self.matrix, orient="index")

    def summary(self):
        parts = [f"=== Debug Summary [{self.name}] ==="]

        if self.flags:
            parts.append(f"Flags: {self.flags}")

        for cp, metrics in self.matrix.items():
            # Filter out checkpoints that have no recorded data
            if any(v is not None for v in metrics.values()):
                parts.append(f"\n[Checkpoint: {cp}]")
                for k, v in metrics.items():
                    if v is not None:
                        parts.append(f"  {k}: {v}")

        for child in self.children:
            parts.append("\n" + child.summary())

        return "\n".join(parts)
