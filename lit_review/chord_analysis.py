import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mir_eval
import os
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def get_harmonic_templates(num_harmonics=3, alpha=0.6):
    """
    generates 24 templates involving harmonics to better match real instruments.
    """
    templates = np.zeros((24, 12))
    for root in range(12):
        #major triad relative notes: fundamental, major third, perfect fifth
        for note_idx in [0, 4, 7]:
            note = (root + note_idx) % 12
            for h in range(1, num_harmonics + 1):
                #harmonic k frequency is k * f
                #chromatic shift is 12 * log2(k)
                shift = int(np.round(12 * np.log2(h)))
                harmonic_note = (note + shift) % 12
                #decaying weight for higher harmonics
                templates[root, harmonic_note] += (alpha ** (h-1))
                
        #minor triad relative notes: fundamental, minor third, perfect fifth
        for note_idx in [0, 3, 7]:
            note = (root + note_idx) % 12
            for h in range(1, num_harmonics + 1):
                shift = int(np.round(12 * np.log2(h)))
                harmonic_note = (note + shift) % 12
                templates[root + 12, harmonic_note] += (alpha ** (h-1))
                
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    templates = templates / (norms + 1e-10)
    return templates

def classify_chroma(chroma, templates, bass_chroma=None, bass_weight=0.1):
    """
    performs classification using cosine similarity with optional bass weighting.
    """
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    similarity_matrix = np.dot(templates, chroma_norm)
    
    if bass_chroma is not None:
        #give a bonus to the chord whose root matches the strongest bass note
        #templates are ordered [C:maj, C#:maj... C:min, C#:min...]
        #so row i and row i+12 both have root 'i'
        for i in range(12):
            similarity_matrix[i, :] += bass_weight * bass_chroma[i, :]
            similarity_matrix[i+12, :] += bass_weight * bass_chroma[i, :]
            
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

    #harmonic-percussive source separation (HPSS)
    #this filters out the drums/transients to leave only the musical pitches
    print("running HPSS to filter out percussion noise...")
    y_harmonic = librosa.effects.harmonic(y)

    #feature extraction with automatic tuning correction
    print("estimating tuning and extracting chroma...")
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    
    #standard chroma (all frequencies)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, tuning=tuning)
    
    #bass chroma (20Hz to 260Hz, approx C0 to C4)
    #n_octaves=3 starting from C1 (approx 32Hz)
    bass_chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, 
                                            tuning=tuning, fmin=librosa.note_to_hz('C1'), 
                                            n_octaves=3)
    
    n_frames = chroma_cqt.shape[1]

    #process labels
    ref_labels = [reduce_chord_to_triad(l) for l in raw_labels]

    #get templates (harmonic templates for better spectral matching)
    templates_binary = get_binary_templates()
    templates_harmonic = get_harmonic_templates()

    #log-compression of chroma features (FMP C3)
    #this balances the contribution of strong and weak components
    gamma = 100
    chroma_log = np.log(1 + gamma * chroma_cqt)
    bass_chroma_log = np.log(1 + gamma * bass_chroma)

    #run pipelines
    chord_labels = [
        'C:maj', 'C#:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'Bb:maj', 'B:maj',
        'C:min', 'C#:min', 'D:min', 'Eb:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'Ab:min', 'A:min', 'Bb:min', 'B:min'
    ]
    
    print(f"running pipeline A (baseline) with tuning: {tuning:.2f} cents and Bass Weighting...")
    #pipeline A uses binary templates and standard chroma (for comparison)
    similarity_A, preds_A_idx = classify_chroma(chroma_cqt, templates_binary, bass_chroma=bass_chroma)
    labels_A = [chord_labels[i] for i in preds_A_idx]

    print("running pipeline B (segmentation-based)...")
    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    onsets = np.unique(np.concatenate(([0], onset_frames, [n_frames])))
    
    labels_B = []
    similarity_B = np.zeros((24, n_frames))
    
    for i in range(len(onsets)-1):
        idx = np.arange(int(onsets[i]), int(onsets[i+1]))
        if len(idx) == 0: continue
        #average chroma for segment (use log-compressed)
        segment_chroma = np.mean(chroma_log[:, idx], axis=1, keepdims=True)
        #average bass chroma for segment
        segment_bass = np.mean(bass_chroma_log[:, idx], axis=1, keepdims=True)
        #classify using harmonic templates
        s_mat, s_pred = classify_chroma(segment_chroma, templates_harmonic, bass_chroma=segment_bass)
        chord = chord_labels[s_pred[0]]
        labels_B.extend([chord] * len(idx))
        similarity_B[:, idx] = s_mat #broadcast segment similarity to frames

    print("running pipeline C (HMM / Viterbi smoothing)...")
    #pipeline C uses harmonic templates, log-compression, and Viterbi decoding
    #first, compute local similarity matrix (emission probabilities proxy)
    similarity_C, _ = classify_chroma(chroma_log, templates_harmonic, bass_chroma=bass_chroma_log)
    
    #transition matrix: high probability to stay in same chord
    p_stay = 0.95
    p_move = (1 - p_stay) / 23
    trans_matrix = np.full((24, 24), p_move)
    np.fill_diagonal(trans_matrix, p_stay)
    
    #viterbi decoding
    #convert similarity to pseudo-probabilities using softmax
    #a higher beta makes the distribution more peaked
    beta = 10 
    exp_sim = np.exp(beta * similarity_C)
    prob_C = exp_sim / np.sum(exp_sim, axis=0, keepdims=True)
    
    preds_C_idx = librosa.sequence.viterbi(prob_C, trans_matrix)
    labels_C = [chord_labels[i] for i in preds_C_idx]

    times = librosa.frames_to_time(np.arange(n_frames + 1), sr=sr, hop_length=HOP_LENGTH)
    times[-1] = min(times[-1], duration)
    est_int_A = np.vstack([times[:-1], times[1:]]).T
    est_int_B = np.vstack([times[:-1], times[1:]]).T
    
    score_A = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_A, labels_A)
    score_B = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_B, labels_B)
    est_int_C = est_int_A #same time intervals for C
    score_C = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_C, labels_C)

    print("\n" + "="*20)
    print("EVALUATION RESULTS: ACCURACY")
    print("="*20)
    print(f"pipeline A (fixed-frame baseline)  WCSR: {score_A['mirex']:.4f}")
    print(f"pipeline B (segmentation-based)    WCSR: {score_B['mirex']:.4f}")
    print(f"pipeline C (HMM/Viterbi + Harm)    WCSR: {score_C['mirex']:.4f}")
    print("="*20)

    #visualisation
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    imgA = librosa.display.specshow(similarity_A, x_axis='time', y_axis='chroma', sr=sr, hop_length=HOP_LENGTH, ax=axes[0], cmap='magma')
    axes[0].set_title(f'pipeline A (fixed-frame) similarity (acc: {score_A["mirex"]:.2f})')
    fig.colorbar(imgA, ax=axes[0])

    imgB = librosa.display.specshow(similarity_B, x_axis='time', y_axis='chroma', sr=sr, hop_length=HOP_LENGTH, ax=axes[1], cmap='magma')
    axes[1].set_title(f'pipeline B (segmentation) similarity (acc: {score_B["mirex"]:.2f})')
    fig.colorbar(imgB, ax=axes[1])

    imgC = librosa.display.specshow(similarity_C, x_axis='time', y_axis='chroma', sr=sr, hop_length=HOP_LENGTH, ax=axes[2], cmap='magma')
    axes[2].set_title(f'pipeline C (Viterbi) similarity (acc: {score_C["mirex"]:.2f})')
    fig.colorbar(imgC, ax=axes[2])

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax2, alpha=0.5)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    ax2.vlines(onset_times, -1, 1, color='r', linestyle='--', alpha=0.7, label='detected onsets')
    axes[2].set_title('audio waveform & segmentation points')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('chord_analysis_results.png')
    print("\nvisualisation saved to chord_analysis_results.png")

    #generate confusion matrix for pipeline C (now the best)
    #first, we need frame-wise ground truth labels for comparison
    frame_centers = (times[:-1] + times[1:]) / 2
    ref_labels_frames = mir_eval.util.interpolate_intervals(ref_intervals, ref_labels, frame_centers, fill_value='N')
    plot_confusion_matrix(ref_labels_frames, labels_C, "C")
    plt.show()

def plot_confusion_matrix(ref_labels, est_labels, pipeline_name="B"):
    """
    visualises classification errors using a confusion matrix.
    filters for common chords and normalises by true class.
    """
    common_chords = ['C:maj', 'G:maj', 'A:min', 'E:min', 'F:maj', 'D:min']
    
    #filter both lists based on common_chords being the ground truth
    ref_filtered = []
    est_filtered = []
    for r, e in zip(ref_labels, est_labels):
        if r in common_chords:
            ref_filtered.append(r)
            est_filtered.append(e)
            
    if not ref_filtered:
        print("warning: no common chords found in the ground truth for confusion matrix.")
        return

    #compute confusion matrix
    cm = confusion_matrix(ref_filtered, est_filtered, labels=common_chords, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=common_chords, yticklabels=common_chords)
    plt.title(f'confusion matrix (normalised) - pipeline {pipeline_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{pipeline_name}.png')
    print(f"\nconfusion matrix saved to confusion_matrix_{pipeline_name}.png")

if __name__ == "__main__":
    main()
