import sys
import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# -----------------------------
# 1. Load EDF file
# -----------------------------
train_dir="training_set"
phys_dir="/physiological_data"

edf_path = "training_set/physiological_data/I0002/sub-I0002150000686_ses-1.edf"

raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

print("===================================")
print("Basic Information")
print("===================================")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Recording duration: {raw.times[-1]:.2f} seconds")
print("\nChannel names:")
print(raw.ch_names)

# -----------------------------
# 2. Plot first 30 seconds
# -----------------------------
duration = 30  # seconds
start_time = 0

eeg_channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})

raw.plot(
    start=start_time,
    duration=duration,
    n_channels=10,  # change depending on how many channels you want
    scalings={'eeg': 100e-6},
    #block=True,
    title=f"First {duration} seconds"
)

plt.show()

# Load data into memory (only do this for short segments!)
raw.load_data()

# Select first 10 seconds
start_sample = 0
stop_sec=300
stop_sample = int(stop_sec * raw.info['sfreq'])

data, times = raw[:, start_sample:stop_sample]

# Plot first 3 channels
plt.figure(figsize=(12, 6))

# Select the channels you want to plot
# Example for your first plot (First 8 channels)
n_channels_to_plot = min(8, data.shape[0])
channel_indices = range(n_channels_to_plot)
offset = 200e-6

tick_positions = []
tick_labels = []

for i in channel_indices:
    y_offset = i * offset
    plt.plot(times, data[i] + y_offset)

    # Store the center position and the name
    tick_positions.append(y_offset)
    tick_labels.append(raw.ch_names[i])

plt.yticks(tick_positions, tick_labels)

plt.xlabel("Time (s)")
plt.ylabel("Channels")
plt.title(f"First {stop_sec} Seconds - EEG and EOG Channels")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 12))

plt.subplot(2,1,1)
#for i in range(min(8, data.shape[0]),data.shape[0]):
emg_indices = [8, 13, 14]
offsets = [1 * 200e-6, 2 * 200e-6, 3 * 200e-6]
labels = [raw.ch_names[i] for i in emg_indices]

for i, y_off in zip(emg_indices, offsets):
    plt.plot(times, data[i] + y_off)

plt.yticks(offsets, labels) # This puts the name right next to the signal
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"First {stop_sec} Second - EMG Channels")

plt.subplot(2,1,2)
breath_indices=[9,10,16,17,18]
offsets=[90* 200e-6, 10000* 200e-6, 14000* 200e-6, 16000* 200e-6, 18000* 200e-6]
labels=[raw.ch_names[i] for i in breath_indices]
for i, y_off in zip(breath_indices, offsets):
    plt.plot(times, data[i] + y_off)

plt.yticks(offsets, labels)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"First {stop_sec} Seconds - Breathing Channels")
plt.show()

ann_path = "training_set/human_annotations/I0002/sub-I0002150000686_ses-1_expert_annotations.edf"

ann_raw = mne.io.read_raw_edf(ann_path, preload=True, verbose=False)

print("Annotation channels:")
print(ann_raw.ch_names)

print("\nChannel Statistics (first 19 channels):")
for ch in raw.ch_names[:19]:
    data, _ = raw[ch, :]
    print(f"{ch}: mean={data.mean():.6f}, min={data.min():.6f}, max={data.max():.6f} std={data.std():.6f}")

print("\n===================================")
print("Annotation Info")
print("===================================")

ann_path = "training_set/human_annotations/I0002/sub-I0002150000686_ses-1_expert_annotations.edf"
ann_raw = mne.io.read_raw_edf(ann_path, preload=True, verbose=False)

print(f"Annotation sampling frequency: {ann_raw.info['sfreq']} Hz")
print(f"Annotation duration: {ann_raw.times[-1]:.2f} seconds")

for ch in ann_raw.ch_names:
    data, _ = ann_raw[ch, :]
    unique_vals = np.unique(data)
    print(f"\nChannel: {ch}")
    print(f"Unique labels: {unique_vals}")
    print(f"Number of unique labels: {len(unique_vals)}")

# Example for stage_expert
stage_data, _ = ann_raw["stage_expert", :]
stage_data = stage_data.flatten()

unique, counts = np.unique(stage_data, return_counts=True)

#print("\nSleep Stage Distribution:")
#for u, c in zip(unique, counts):
#    print(f"Stage {int(u)}: {c} samples")

# -----------------------------
# Paths
# -----------------------------
psg_path = "training_set/physiological_data/I0002/sub-I0002150000686_ses-1.edf"
ann_path = "training_set/human_annotations/I0002/sub-I0002150000686_ses-1_expert_annotations.edf"

# -----------------------------
# Load Data
# -----------------------------
raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
ann_raw = mne.io.read_raw_edf(ann_path, preload=True, verbose=False)

print("\n==============================")
print("️1 Alignment Check")
print("==============================")

psg_duration = raw.times[-1]
ann_duration = ann_raw.times[-1]

print(f"PSG duration: {psg_duration:.2f} sec")
print(f"Annotation duration: {ann_duration:.2f} sec")
print(f"Difference: {abs(psg_duration - ann_duration):.4f} sec")

if abs(psg_duration - ann_duration) < 1:
    print(" Durations align well")
else:
    print("️ Durations DO NOT align")

# ---------------------------------------------------
# 2 Sleep Stage Imbalance (30-sec epochs expected)
# ---------------------------------------------------
print("\n==============================")
print("2 Sleep Stage Distribution")
print("==============================")

stage_data, _ = ann_raw["stage_expert", :]
stage_data = stage_data.flatten()
sfreq = ann_raw.info["sfreq"]
total_samples = len(stage_data)

stage_data_rounded = np.round(stage_data).astype(int)

# 2. Get unique stages and their total counts
unique, counts = np.unique(stage_data_rounded, return_counts=True)

total_samples = len(stage_data_rounded)
sfreq = ann_raw.info["sfreq"]

print(f"{'Stage ID':<10} | {'Total Minutes':<15} | {'Percentage':<10}")
print("-" * 45)

for u, c in zip(unique, counts):
    minutes = c / sfreq / 60
    percentage = (c / total_samples) * 100
    # This now shows the TOTAL time spent in each stage across the whole night
    print(f"{u:<10} | {minutes:>13.2f}m | {percentage:>9.2f}%")

# --- Setup for Visualization ---
# 1. Prepare the Timeline (Hypnogram)
times_hours = np.arange(len(stage_data_rounded)) / sfreq / 3600  # Convert to hours

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

# 2. Plot the Hypnogram
ax1.step(times_hours, stage_data_rounded, where='post', color='#2c3e50', linewidth=1)
ax1.set_title("Sleep Stage Timeline (Hypnogram)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Stage ID")
ax1.set_xlabel("Time (Hours)")
ax1.set_yticks(unique)  # Only show stages that actually exist
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.invert_yaxis()  # Standard practice: deeper stages/higher IDs go lower

# 3. Plot the Distribution (Pie Chart)
# Only label stages with > 1% to keep it clean
labels = [f"Stage {int(u)}" if (c/total_samples) > 0.01 else "" for u, c in zip(unique, counts)]
ax2.pie(counts, labels=labels, autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=140, colors=plt.cm.Paired(np.linspace(0, 1, len(unique))))
ax2.set_title("Overall Stage Distribution")

plt.tight_layout()
plt.show()

# Calculate transitions
diffs = np.diff(stage_data_rounded)
transitions = np.where(diffs != 0)[0]

# Add start and end points
indices = np.concatenate(([0], transitions + 1, [len(stage_data_rounded)]))

print(f"\n{'Sequence':<10} | {'Stage':<8} | {'Duration (min)':<15}")
print("-" * 40)

for i in range(len(indices) - 1):
    start = indices[i]
    end = indices[i + 1]
    stage = stage_data_rounded[start]
    duration_min = (end - start) / sfreq / 60

    # Optional: Filter out noise (e.g., transitions shorter than 2 seconds)
    if duration_min > (2 / 60):
        print(f"{i + 1:<10} | {stage:<8} | {duration_min:>13.2f}m")

transitions_only = stage_data_rounded[indices[:-1]]
sources = transitions_only[:-1]
targets = transitions_only[1:]

# Build a matrix (using pandas for ease)
df_matrix = pd.crosstab(pd.Series(sources, name='From Stage'),
                        pd.Series(targets, name='To Stage'),
                        normalize='index')

plt.figure(figsize=(10, 8))
sns.heatmap(df_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Probability of Sleep Stage Transitions")
plt.show()

# ---------------------------------------------------
# 3 Respiratory Event Frequency
# ---------------------------------------------------
print("\n==============================")
print("3 Respiratory Events")
print("==============================")

resp_data, _ = ann_raw["resp_expert", :]
resp_data = resp_data.flatten()

print("Min:", np.min(resp_data))
print("Max:", np.max(resp_data))
print("Mean:", np.mean(resp_data))
print("Std:", np.std(resp_data))

print(resp_data[:100])
print("Annotation sfreq:", ann_raw.info["sfreq"])
# Find where signal changes
changes = np.where(np.diff(resp_data) != 0)[0]

print("Number of transitions:", len(changes))

# Print first 20 transitions
print("First 20 transitions:")
for idx in changes[:20]:
    print(f"Index {idx}: {resp_data[idx]} → {resp_data[idx+1]}")
non_zero = np.sum(resp_data != 0)
print("Non-zero samples:", non_zero)
print("Percent non-zero:", 100 * non_zero / len(resp_data))

print(ann_raw.annotations)
print("Unique rounded values:")
print(np.unique(np.round(resp_data, 2)))
abs_signal = np.abs(resp_data)
threshold = 0.5   # experiment

event_mask = abs_signal > threshold

print("Percent event-like:", np.mean(event_mask) * 100)
sfreq = ann_raw.info["sfreq"]

start = 0
stop = int(120 * sfreq)

plt.figure(figsize=(12,4))
plt.plot(resp_data[start:stop])
plt.title("Respiratory Annotation Channel")
plt.show()

demo = pd.read_csv("training_set/demographics.csv")
print(demo.columns)
print(demo.head())

var = demo[
    (demo["BidsFolder"] == "sub-I0002150000686") & (demo["SessionID"] == 1)]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
print(var)

#unique_resp = np.unique(resp_data)

#print(len(unique_resp))

'''for cls in unique_resp:
    if cls == 0:
        continue
    count = np.sum(resp_data == cls)
    seconds = count / ann_raw.info["sfreq"]
    print(f"Class {int(cls)}: {count} samples ({seconds:.1f} sec total)")

# ---------------------------------------------------
# 4 Event Duration Distribution
# ---------------------------------------------------
print("\n==============================")
print("4 Event Duration Analysis (Respiratory)")
print("==============================")

def compute_event_durations(signal, sfreq, target_class):
    durations = []
    in_event = False
    length = 0

    for val in signal:
        if val == target_class:
            length += 1
            in_event = True
        else:
            if in_event:
                durations.append(length / sfreq)
                length = 0
                in_event = False

    if in_event:
        durations.append(length / sfreq)

    return durations

for cls in unique_resp:
    if cls == 0:
        continue
    durations = compute_event_durations(resp_data, ann_raw.info["sfreq"], cls)
    if len(durations) == 0:
        continue

    print(f"\nClass {int(cls)}:")
    print(f"  Number of events: {len(durations)}")
    print(f"  Mean duration: {np.mean(durations):.2f} sec")
    print(f"  Median duration: {np.median(durations):.2f} sec")
    print(f"  Max duration: {np.max(durations):.2f} sec")'''


