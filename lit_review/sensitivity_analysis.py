import librosa
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import mir_eval
from scipy.spatial.distance import cosine

#configuration
AUDIO_FOLDER = 'audio_dataset/'
JAMS_FOLDER = 'msaf-data/Isophonics/references/'
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
SR = 22050
HOP_LENGTH = 512

#explicit mapping of audio to JAMS
DATA_MAP = {
    "A Day In The Life (Remastered 2009).mp3": "Isophonics_13_-_A_Day_In_The_Life.jams",
    "Blackbird (Remastered 2009).mp3": "Isophonics_CD1_-_11_-_Black_Bird.jams",
    "Come Together (Remastered 2015).mp3": "Isophonics_01_-_Come_Together.jams",
    "Eleanor Rigby (Remastered 2015).mp3": "Isophonics_02_-_Eleanor_Rigby.jams",
    "Here Comes The Sun (2019 Mix).mp3": "Isophonics_07_-_Here_Comes_The_Sun.jams",
    "In My Life (Remastered 2009).mp3": "Isophonics_11_-_In_My_Life.jams",
    "Let It Be.mp3": "Isophonics_06_-_Let_It_Be.jams",
    "Penny Lane (Remastered 2015).mp3": "Isophonics_09_-_Penny_Lane.jams",
    "Something (Remastered 2015).mp3": "Isophonics_02_-_Something.jams",
    "Strawberry Fields Forever.mp3": "Isophonics_08_-_Strawberry_Fields_Forever.jams",
    "While My Guitar Gently Weeps (2018 Mix).mp3": "Isophonics_CD1_-_07_-_While_My_Guitar_Gently_Weeps.jams",
    "Yesterday (Remastered 2009).mp3": "Isophonics_13_-_Yesterday.jams"
}

def generate_templates():
    templates = {}
    roots = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    for i, root in enumerate(roots):
        vec = np.zeros(12)
        vec[[i, (i+4)%12, (i+7)%12]] = 1
        templates[f'{root}:maj'] = vec
        vec = np.zeros(12)
        vec[[i, (i+3)%12, (i+7)%12]] = 1
        templates[f'{root}:min'] = vec
    return templates

TEMPLATES = generate_templates()

def reduce_chord(label):
    if label == 'N': return 'N'
    parts = label.split(':')
    root = parts[0].split('/')[0]
    if len(parts) == 1: return f"{root}:maj"
    quality = parts[1].split('/')[0]
    return f"{root}:min" if 'min' in quality else f"{root}:maj"

def load_jams_chords(jams_path):
    with open(jams_path, 'r') as f:
        data = json.load(f)
    chord_ann = next(ann for ann in data['annotations'] if ann['namespace'] == 'chord')
    intervals, labels = [], []
    for i, entry in enumerate(chord_ann['data']):
        start = entry['time']
        val = entry['value']
        end = chord_ann['data'][i+1]['time'] if i+1 < len(chord_ann['data']) else start + entry['duration']
        if end > start:
            intervals.append([start, end])
            labels.append(reduce_chord(val))
    return np.array(intervals), labels

def classify_frame(chroma_vec, bass_chroma_vec=None, bass_weight=0.1):
    best_chord, max_sim = None, -1
    #TEMPLATES is a dictionary
    for i, (label, template) in enumerate(TEMPLATES.items()):
        sim = 1 - cosine(chroma_vec + 1e-10, template + 1e-10)
        
        #add bass weight if applicable
        if bass_chroma_vec is not None:
            root_idx = i % 12
            sim += bass_weight * bass_chroma_vec[root_idx]
            
        if sim > max_sim:
            max_sim, best_chord = sim, label
    return best_chord

def calculate_stability(seq):
    if not seq: return 0.0
    changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    return 1.0 - (changes / len(seq))

def main():
    results = {t: {'acc': [], 'stab': []} for t in THRESHOLDS}
    
    print(f"starting sensitivity analysis on {len(DATA_MAP)} tracks...")
    
    for audio_name, jams_name in DATA_MAP.items():
        audio_path = os.path.join(AUDIO_FOLDER, audio_name)
        jams_path = os.path.join(JAMS_FOLDER, jams_name)
        
        if not os.path.exists(audio_path) or not os.path.exists(jams_path):
            print(f"skipping {audio_name} (missing files)")
            continue
            
        print(f"processing {audio_name}...")
        y, sr = librosa.load(audio_path, sr=SR)
        
        #apply HPSS to filter out percussion noise
        y_harmonic = librosa.effects.harmonic(y)
        
        #estimate tuning for better chroma mapping
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, tuning=tuning)
        bass_chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH, 
                                                tuning=tuning, fmin=librosa.note_to_hz('C1'), 
                                                n_octaves=3)
        
        ref_int, ref_labs = load_jams_chords(jams_path)
        
        duration = librosa.get_duration(y=y, sr=sr)
        times = librosa.frames_to_time(np.arange(chroma.shape[1] + 1), sr=sr, hop_length=HOP_LENGTH)
        times[-1] = min(times[-1], duration)
        est_int = np.vstack([times[:-1], times[1:]]).T
        
        onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH)
        
        for t in THRESHOLDS:
            #sweep threshold
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, pre_max=t, post_max=t)
            onsets = np.unique(np.concatenate(([0], onsets, [chroma.shape[1]])))
            
            est_labs = []
            for i in range(len(onsets)-1):
                start, end = int(onsets[i]), int(onsets[i+1])
                if start >= end: continue
                segment_avg = np.mean(chroma[:, start:end], axis=1)
                segment_bass = np.mean(bass_chroma[:, start:end], axis=1)
                chord = classify_frame(segment_avg, segment_bass)
                est_labs.extend([chord] * (end-start))
            
            score = mir_eval.chord.evaluate(ref_int, ref_labs, est_int, est_labs)
            results[t]['acc'].append(score['mirex'])
            results[t]['stab'].append(calculate_stability(est_labs))

    #reporting
    print("\n--- ABLATION STUDY RESULTS ---")
    print(f"{'Threshold':<10} | {'Stability (S)':<15} | {'Accuracy (WCSR)':<15}")
    print("-" * 45)
    
    final_stab, final_acc = [], []
    for t in THRESHOLDS:
        m_s, s_s = np.mean(results[t]['stab']), np.std(results[t]['stab'])
        m_a, s_a = np.mean(results[t]['acc']), np.std(results[t]['acc'])
        print(f"{t:<10.1f} | {m_s:0.3f} \u00B1 {s_s:0.3f} | {m_a:0.3f} \u00B1 {s_a:0.3f}")
        final_stab.append(m_s)
        final_acc.append(m_a)

    #plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Onset Threshold (Sensitivity)')
    ax1.set_ylabel('Stability (S)', color='tab:blue')
    ax1.plot(THRESHOLDS, final_stab, color='tab:blue', marker='o', linewidth=2, label='Stability')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (WCSR)', color='tab:orange')
    ax2.plot(THRESHOLDS, final_acc, color='tab:orange', marker='s', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Sensitivity Analysis: Stability vs. Accuracy Trade-off')
    fig.tight_layout()
    plt.savefig('sensitivity_ablation.png')
    print("\nsensitivity plot saved to sensitivity_ablation.png")

if __name__ == "__main__":
    main()
