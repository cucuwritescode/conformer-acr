import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mir_eval
import os
import json

#configuration
#preferred local paths for the requested test
LOCAL_AUDIO = 'tracks/Let It Be.mp3'
LOCAL_JAMS = 'msaf-data/Isophonics/references/Isophonics_06_-_Let_It_Be.jams'

#fallback paths for standalone distribution
DEFAULT_AUDIO = 'audio.wav'
DEFAULT_LAB = 'ground_truth.lab'

HOP_LENGTH = 512
SR = 22050

def get_binary_templates():
    """
    generates 24 binary templates (12 major, 12 minor) for chord recognition.
    normalised for cosine similarity.
    """
    templates = np.zeros((24, 12))
    for root in range(12):
        #major
        templates[root, [root % 12, (root + 4) % 12, (root + 7) % 12]] = 1
        #minor
        templates[root + 12, [root % 12, (root + 3) % 12, (root + 7) % 12]] = 1
    
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    templates = templates / (norms + 1e-10)
    return templates

def classify_chroma(chroma, templates):
    """
    performs classification using cosine similarity.
    """
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    similarity_matrix = np.dot(templates, chroma_norm)
    predictions = np.argmax(similarity_matrix, axis=0)
    return similarity_matrix, predictions

def reduce_chord_to_triad(label):
    """
    manually reduces complex chord labels to major/minor triads.
    """
    if label == 'N': return 'N'
    parts = label.split(':')
    root = parts[0].split('/')[0]
    if len(parts) == 1:
        return f"{root}:maj"
    quality = parts[1].split('/')[0]
    return f"{root}:min" if 'min' in quality else f"{root}:maj"

def load_jams_chords(jams_path):
    """
    extracts chord intervals and labels from a JAMS file.
    """
    with open(jams_path, 'r') as f:
        data = json.load(f)
    chord_ann = next(ann for ann in data['annotations'] if ann['namespace'] == 'chord')
    intervals = []
    labels = []
    data_list = chord_ann['data']
    for i in range(len(data_list)):
        start = data_list[i]['time']
        value = data_list[i]['value']
        #ensure continuity
        end = data_list[i+1]['time'] if i + 1 < len(data_list) else start + data_list[i]['duration']
        if end > start:
            intervals.append([start, end])
            labels.append(value)
    return np.array(intervals), labels

def main():
    #file selection logic
    audio_path = LOCAL_AUDIO if os.path.exists(LOCAL_AUDIO) else DEFAULT_AUDIO
    
    if os.path.exists(LOCAL_JAMS):
        print(f"loading ground truth from JAMS: {LOCAL_JAMS}")
        ref_intervals, raw_labels = load_jams_chords(LOCAL_JAMS)
    elif os.path.exists(DEFAULT_LAB):
        print(f"loading ground truth from LAB: {DEFAULT_LAB}")
        ref_intervals, raw_labels = mir_eval.io.load_labeled_intervals(DEFAULT_LAB)
    else:
        print("error: no ground truth (.jams or .lab) found.")
        return

    if not os.path.exists(audio_path):
        print(f"error: audio file {audio_path} not found.")
        return

    print(f"loading {audio_path}...")
    y, sr = librosa.load(audio_path, sr=SR)
    duration = librosa.get_duration(y=y, sr=sr)

    #feature extraction
    print("extracting chroma (CQT)...")
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    n_frames = chroma_cqt.shape[1]

    #process labels
    ref_labels = [reduce_chord_to_triad(l) for l in raw_labels]

    #pipeline A: baseline (fixed-frame)
    print("running pipeline A (baseline)...")
    templates = get_binary_templates()
    similarity_A, preds_A_idx = classify_chroma(chroma_cqt, templates)

    #pipeline B: segmentation-based
    print("running pipeline B (segmentation-based)...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    segment_boundaries = np.unique(np.concatenate(([0], onset_frames, [n_frames])))
    
    chroma_segmented = np.zeros_like(chroma_cqt)
    for i in range(len(segment_boundaries) - 1):
        start, end = int(segment_boundaries[i]), int(segment_boundaries[i+1])
        if start >= end: continue
        avg_chroma = np.mean(chroma_cqt[:, start:end], axis=1, keepdims=True)
        chroma_segmented[:, start:end] = avg_chroma

    similarity_B, preds_B_idx = classify_chroma(chroma_segmented, templates)

    #evaluation
    chord_labels = [
        'C:maj', 'C#:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'Bb:maj', 'B:maj',
        'C:min', 'C#:min', 'D:min', 'Eb:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'Ab:min', 'A:min', 'Bb:min', 'B:min'
    ]
    labels_A = [chord_labels[i] for i in preds_A_idx]
    labels_B = [chord_labels[i] for i in preds_B_idx]

    times = librosa.frames_to_time(np.arange(n_frames + 1), sr=sr, hop_length=HOP_LENGTH)
    times[-1] = min(times[-1], duration)
    est_int_A = np.vstack([times[:-1], times[1:]]).T
    est_int_B = np.vstack([times[:-1], times[1:]]).T
    
    score_A = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_A, labels_A)
    score_B = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_B, labels_B)

    print("\n" + "="*20)
    print("EVALUATION RESULTS")
    print("="*20)
    print(f"pipeline A (fixed-frame) weighted accuracy (Mirex): {score_A['mirex']:.4f}")
    print(f"pipeline B (segmentation) weighted accuracy (Mirex): {score_B['mirex']:.4f}")
    print("="*20)

    #visualisation
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    imgA = librosa.display.specshow(similarity_A, x_axis='time', y_axis='chroma', sr=sr, hop_length=HOP_LENGTH, ax=axes[0], cmap='magma')
    axes[0].set_title(f'pipeline A (fixed-frame) similarity (acc: {score_A["mirex"]:.2f})')
    fig.colorbar(imgA, ax=axes[0])

    imgB = librosa.display.specshow(similarity_B, x_axis='time', y_axis='chroma', sr=sr, hop_length=HOP_LENGTH, ax=axes[1], cmap='magma')
    axes[1].set_title(f'pipeline B (segmentation) similarity (acc: {score_B["mirex"]:.2f})')
    fig.colorbar(imgB, ax=axes[1])

    librosa.display.waveshow(y, sr=sr, ax=axes[2], alpha=0.5)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    axes[2].vlines(onset_times, -1, 1, color='r', linestyle='--', alpha=0.7, label='detected onsets')
    axes[2].set_title('audio waveform & segmentation points')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('chord_analysis_results.png')
    print("\nvisualisation saved to chord_analysis_results.png")
    plt.show()

if __name__ == "__main__":
    main()
