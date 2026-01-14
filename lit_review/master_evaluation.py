import librosa
import numpy as np
import os
import json
import csv
import pandas as pd
import mir_eval
from tqdm import tqdm

#configuration
AUDIO_FOLDER = 'audio_dataset/'
JAMS_FOLDER = 'msaf-data/Isophonics/references/'
SR = 22050
HOP_LENGTH = 512

#locked dataset (12 tracks)
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

def generate_templates(harmonic=False, num_harmonics=3, alpha_harm=0.6):
    roots = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    templates = []
    
    #major triads
    for root_idx in range(12):
        vec = np.zeros(12)
        target_notes = [0, 4, 7] if not harmonic else [0, 4, 7] #logic inside loop
        for note_rel in [0, 4, 7]:
            note = (root_idx + note_rel) % 12
            if not harmonic:
                vec[note] = 1
            else:
                for h in range(1, num_harmonics + 1):
                    shift = int(np.round(12 * np.log2(h)))
                    h_note = (note + shift) % 12
                    vec[h_note] += (alpha_harm ** (h-1))
        templates.append(vec)
        
    #minor triads
    for root_idx in range(12):
        vec = np.zeros(12)
        for note_rel in [0, 3, 7]:
            note = (root_idx + note_rel) % 12
            if not harmonic:
                vec[note] = 1
            else:
                for h in range(1, num_harmonics + 1):
                    shift = int(np.round(12 * np.log2(h)))
                    h_note = (note + shift) % 12
                    vec[h_note] += (alpha_harm ** (h-1))
        templates.append(vec)
        
    t_array = np.array(templates)
    return t_array / (np.linalg.norm(t_array, axis=1, keepdims=True) + 1e-10)

TEMPLATES_BINARY = generate_templates(harmonic=False)
TEMPLATES_HARMONIC = generate_templates(harmonic=True)

CHORD_LABELS = [
    'C:maj', 'C#:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'Bb:maj', 'B:maj',
    'C:min', 'C#:min', 'D:min', 'Eb:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'Ab:min', 'A:min', 'Bb:min', 'B:min'
]

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

class DataCenter:
    def __init__(self, audio_name, jams_name):
        self.name = audio_name
        audio_path = os.path.join(AUDIO_FOLDER, audio_name)
        jams_path = os.path.join(JAMS_FOLDER, jams_name)
        
        y, _ = librosa.load(audio_path, sr=SR)
        self.duration = librosa.get_duration(y=y, sr=SR)
        self.ref_int, self.ref_labs = load_jams_chords(jams_path)
        
        #determine number of frames consistent with librosa's hop logic
        n_frames = int(np.floor(len(y) / HOP_LENGTH)) + 1
        
        #raw features (no HPSS, no tuning)
        self.chroma_raw = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_LENGTH)[:, :n_frames]
        self.onsets_raw = self._get_onsets(y)
        
        #harmonic features (HPSS)
        y_harmonic = librosa.effects.harmonic(y)
        self.chroma_hpss = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH)[:, :n_frames]
        self.onsets_hpss = self._get_onsets(y_harmonic)
        
        #tuning correction features
        tuning_raw = librosa.estimate_tuning(y=y, sr=SR)
        self.chroma_tuning = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_LENGTH, tuning=tuning_raw)[:, :n_frames]
        
        tuning_hpss = librosa.estimate_tuning(y=y_harmonic, sr=SR)
        self.chroma_hpss_tuning = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH, tuning=tuning_hpss)[:, :n_frames]
        
        #bass features
        self.bass_raw = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz('C1'), n_octaves=3)[:, :n_frames]
        self.bass_hpss_tuning = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH, tuning=tuning_hpss, fmin=librosa.note_to_hz('C1'), n_octaves=3)[:, :n_frames]

    def _get_onsets(self, y):
        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH)
        onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP_LENGTH)
        return np.unique(np.concatenate(([0], onsets, [int(len(y)/HOP_LENGTH)+1])))

def classify_chroma(chroma_norm, bass_chroma=None, alpha=0.0, harmonic=False):
    templs = TEMPLATES_HARMONIC if harmonic else TEMPLATES_BINARY
    sim = np.dot(templs, chroma_norm)
    if bass_chroma is not None and alpha > 0:
        #match lengths if they differ by 1 frame
        min_len = min(sim.shape[1], bass_chroma.shape[1])
        sim = sim[:, :min_len]
        bc = bass_chroma[:, :min_len]
        for i in range(12):
            sim[i, :] += alpha * bc[i, :]
            sim[i+12, :] += alpha * bc[i, :]
    return sim, np.argmax(sim, axis=0)

def evaluate_config(data_list, hpss, tuning, bass_alpha, harmonic=False, smoothing=False, log_comp=False):
    results = []
    gamma = 100 if log_comp else 0
    
    for dc in data_list:
        #choose correct features
        if hpss and tuning: chroma = dc.chroma_hpss_tuning
        elif hpss: chroma = dc.chroma_hpss
        elif tuning: chroma = dc.chroma_tuning
        else: chroma = dc.chroma_raw
        
        bass = dc.bass_hpss_tuning if (hpss and tuning) else dc.bass_raw
        
        if log_comp:
            chroma = np.log(1 + gamma * chroma)
            bass = np.log(1 + gamma * bass)
            
        n_frames = chroma.shape[1]
        est_labs = []
        
        if smoothing:
            #HMM / Viterbi
            sim, _ = classify_chroma(chroma, bass, bass_alpha, harmonic)
            #Softmax to pseudo-probs
            beta = 10
            exp_sim = np.exp(beta * sim)
            prob = exp_sim / np.sum(exp_sim, axis=0, keepdims=True)
            
            p_stay = 0.95
            p_move = (1 - p_stay) / 23
            trans_matrix = np.full((24, 24), p_move)
            np.fill_diagonal(trans_matrix, p_stay)
            
            preds = librosa.sequence.viterbi(prob, trans_matrix)
            est_labs = [CHORD_LABELS[p] for p in preds]
        else:
            #segmentation or frame-wise
            onsets = dc.onsets_hpss if hpss else dc.onsets_raw
            onsets = onsets[onsets < n_frames]
            onsets = np.unique(np.concatenate((onsets, [n_frames])))
            
            for i in range(len(onsets)-1):
                idx = np.arange(int(onsets[i]), int(onsets[i+1]))
                if len(idx) == 0: continue
                seg_chr = np.mean(chroma[:, idx], axis=1, keepdims=True)
                seg_chr_norm = seg_chr / (np.linalg.norm(seg_chr, axis=0, keepdims=True) + 1e-10)
                seg_bass = np.mean(bass[:, idx], axis=1, keepdims=True) if bass_alpha > 0 else None
                _, pred = classify_chroma(seg_chr_norm, seg_bass, bass_alpha, harmonic)
                est_labs.extend([CHORD_LABELS[pred[0]]] * len(idx))
            
        times = librosa.frames_to_time(np.arange(n_frames + 1), sr=SR, hop_length=HOP_LENGTH)
        times[-1] = min(times[-1], dc.duration)
        est_int = np.vstack([times[:-1], times[1:]]).T
        
        wcsr = mir_eval.chord.evaluate(dc.ref_int, dc.ref_labs, est_int, est_labs)['mirex']
        changes = sum(1 for i in range(1, len(est_labs)) if est_labs[i] != est_labs[i-1])
        stability = 1.0 - (changes / len(est_labs)) if len(est_labs) > 0 else 1.0
        
        results.append({'Track': dc.name, 'WCSR': wcsr, 'Stability': stability})
    return results

def main():
    print("loading data and pre-calculating features (this takes a moment)...")
    dataset = []
    for audio, jams in tqdm(DATA_MAP.items()):
        dataset.append(DataCenter(audio, jams))
        
    print("\noptimising alpha (bass weight)...")
    alpha_sweep = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    best_alpha, best_wcsr = 0.0, -1
    for a in alpha_sweep:
        res = evaluate_config(dataset, True, True, a)
        mean_wcsr = np.mean([r['WCSR'] for r in res])
        print(f"  alpha={a:0.2f}, Mean WCSR={mean_wcsr:0.4f}")
        if mean_wcsr > best_wcsr:
            best_wcsr, best_alpha = mean_wcsr, a
    print(f"Optimal Alpha: {best_alpha}")

    configs = [
        ("Baseline", False, False, 0.0, False, False, False),
        ("+HPSS+Tuning+BassCue", True, True, best_alpha, False, False, False),
        ("+LogComp", True, True, best_alpha, False, False, True),
        ("+HarmTempl", True, True, best_alpha, True, False, True),
        ("FMP_FINAL (Viterbi)", True, True, best_alpha, True, True, True)
    ]
    
    final_rows = []
    print("\nrunning final ablation...")
    for name, h, t, a, harm, smooth, logc in configs:
        res = evaluate_config(dataset, h, t, a, harm, smooth, logc)
        for r in res:
            final_rows.append({**r, 'Config': name})
            
    df = pd.DataFrame(final_rows)
    df.to_csv('per_track_results.csv', index=False)
    
    summary = df.groupby('Config').agg({'WCSR': ['mean', 'std'], 'Stability': ['mean', 'std']}).reindex([c[0] for c in configs])
    print("\nFINAL ABLATION TABLE")
    print(summary)
    
    #impact analysis
    c_prev = df[df['Config'] == '+HPSS+Tuning+BassCue'].set_index('Track')
    c_final = df[df['Config'] == 'FMP_FINAL (Viterbi)'].set_index('Track')
    diff = c_final['WCSR'] - c_prev['WCSR']
    
    print("\nFMP SYSTEM IMPACT ANALYSIS (vs HPSS+Tuning+Bass)")
    print(f"Improved on {(diff > 0).sum()}/12, Degraded on {(diff < 0).sum()}/12")
    print("\nTop 3 Winners:")
    print(diff.sort_values(ascending=False).head(3))
    print("\nTop 3 Losers:")
    print(diff.sort_values(ascending=True).head(3))

if __name__ == "__main__":
    main()
