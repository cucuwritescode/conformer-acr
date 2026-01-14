import librosa
import numpy as np
import os
import glob
from scipy.spatial.distance import cosine

#settings
AUDIO_FOLDER = 'audio_dataset/'  #folder containing audio files
HOP_LENGTH = 512
SR = 22050

def generate_templates():
    """
    generates 24 major/minor binary templates for chord recognition.
    """
    templates = {}
    roots = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    for i, root in enumerate(roots):
        #major: 0, 4, 7
        vec = np.zeros(12)
        vec[[i, (i+4)%12, (i+7)%12]] = 1
        templates[f'{root}:maj'] = vec
        #minor: 0, 3, 7
        vec = np.zeros(12)
        vec[[i, (i+3)%12, (i+7)%12]] = 1
        templates[f'{root}:min'] = vec
    return templates

TEMPLATES = generate_templates()

def get_chroma(audio_path):
    """
    loads audio and extracts CQT chromagram.
    """
    y, sr = librosa.load(audio_path, sr=SR)
    #CQT for better bass resolution (theory compliant)
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH))
    chroma = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    return chroma, y, sr

def classify_frame(chroma_vec):
    """
    classifies a single chroma vector based on cosine similarity.
    """
    best_chord = None
    max_sim = -1
    for label, template in TEMPLATES.items():
        #cosine similarity calculation (1 - cosine distance)
        #adding small epsilon to avoid division by zero
        sim = 1 - cosine(chroma_vec + 1e-10, template + 1e-10)
        if sim > max_sim:
            max_sim = sim
            best_chord = label
    return best_chord

def evaluate_track(file_path):
    """
    analyses a single track through both pipelines and calculates stability.
    """
    #pipeline A (fixed-frame)
    chroma, y, sr = get_chroma(file_path)
    est_a = [classify_frame(c) for c in chroma.T]
    
    #pipeline B (segmentation-based)
    #simple onset detection (spectral flux simulation)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, units='frames')
    onsets = np.concatenate(([0], onsets, [chroma.shape[1]]))
    
    est_b = []
    for i in range(len(onsets)-1):
        start, end = int(onsets[i]), int(onsets[i+1])
        if start >= end: continue
        #average pooling for the segment
        segment = np.mean(chroma[:, start:end], axis=1)
        chord = classify_frame(segment)
        #broadcast result back to all frames in the segment
        est_b.extend([chord] * (end-start))
        
    def calculate_stability(seq):
        """
        calculates stability score as a proxy for segmentation success.
        inverse of flux: (1 - changes / total_frames)
        """
        if not seq: return 0.0
        changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        return 1.0 - (changes / len(seq))

    score_a = calculate_stability(est_a)
    score_b = calculate_stability(est_b)
    
    return score_a, score_b

def main():
    results_a = []
    results_b = []

    #search for wav and mp3 files
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(glob.glob(os.path.join(AUDIO_FOLDER, ext)))

    if not audio_files:
        print(f"error: no audio files found in {AUDIO_FOLDER}")
        return

    print(f"processing tracks in '{AUDIO_FOLDER}'...")
    print("-" * 50)
    
    for file in sorted(audio_files):
        try:
            sa, sb = evaluate_track(file)
            results_a.append(sa)
            results_b.append(sb)
            print(f"{os.path.basename(file):<25}: Fixed={sa:.3f}, Seg={sb:.3f}")
        except Exception as e:
            print(f"error processing {file}: {e}")

    if results_a and results_b:
        print("\n" + "="*50)
        print("FINAL AGGREGATE RESULTS: STABILITY (S)")
        print("="*50)
        print(f"Pipeline A (Fixed)   Stability: {np.mean(results_a):.3f} \u00B1 {np.std(results_a):.3f}")
        print(f"Pipeline B (Segm.)   Stability: {np.mean(results_b):.3f} \u00B1 {np.std(results_b):.3f}")
        print("="*50)
        print("Note: Stability (S) measures label consistency over time, not accuracy.")

if __name__ == "__main__":
    main()
