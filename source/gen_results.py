#created by Facundo Franchino
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mir_eval
from scipy.spatial.distance import cdist

#config
AUDIO_FILE = 'Let_It_Be.wav'  #TODO dl a beatles test track
LAB_FILE = 'Let_It_Be.lab'    #TODO replace with LAB path
HOP_LENGTH = 512
FS = 22050

def get_binary_templates():
    """
    Generates the 24 binary templates (12 Major, 12 Minor) used in Meinard Müller's FMP book.
    """
    templates = np.zeros((24, 12))
    #major: root, +4 (major third), +7 (perfect fifth)
    #minor: root, +3 (minor third), +7 (perfect fifth)
    for root in range(12):
        #major (0-11)
        templates[root, [(root)%12, (root+4)%12, (root+7)%12]] = 1
        #minor (12-23)
        templates[root+12, [(root)%12, (root+3)%12, (root+7)%12]] = 1
    
    #normalise templates (critical for cosine similarity)
    templates /= np.linalg.norm(templates, axis=1)[:, np.newaxis]
    return templates.T #transpose for matrix multiplication

def compute_cosine_similarity(chroma, templates):
    """
    Computes similarity between chroma frames and templates.
    """
    #normalise input chroma vectors
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0) + 1e-10)
    #dot product = cosine similarity (since vectors are normalised)
    return np.dot(templates.T, chroma_norm)

#load audio & annotations
print(f"Loading {AUDIO_FILE}...")
y, sr = librosa.load(AUDIO_FILE, sr=FS)

#load ground truth (automatically parses the .lab file)
ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(LAB_FILE)
#reduce complex chords to Major/Minor (N:maj7 -> N:maj)
ref_labels = mir_eval.chord.reduce_chord_quality(ref_labels)

#feature extraction (use CQT as per FMP recommendation)
print("Extracting CQT Chroma...")
chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

#pipeline A: baseline (frame-wise recognition)
templates = get_binary_templates()
similarity_A = compute_cosine_similarity(chroma_cqt, templates)
preds_A_idx = np.argmax(similarity_A, axis=0)

#pipeline B: segmentation (your "adaptive windowing" method)
print("Running Segmentation...")
#onset detection (simulating spectral flux from Lab 3)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
#add start (0) and end frames
segment_boundaries = np.concatenate(([0], onset_frames, [chroma_cqt.shape[1]]))

chroma_segmented = np.zeros_like(chroma_cqt)

#average chroma within each segment
for i in range(len(segment_boundaries)-1):
    start = segment_boundaries[i]
    end = segment_boundaries[i+1]
    if start == end: continue
    
    #compute mean of this segment
    segment_mean = np.mean(chroma_cqt[:, start:end], axis=1)
    #broadcast back to all frames in this segment
    chroma_segmented[:, start:end] = segment_mean[:, np.newaxis]

#classify the segmented chroma
similarity_B = compute_cosine_similarity(chroma_segmented, templates)
preds_B_idx = np.argmax(similarity_B, axis=0)

#convert & evaluate
chord_map = ['C:maj','C#:maj','D:maj','D#:maj','E:maj','F:maj','F#:maj','G:maj','G#:maj','A:maj','A#:maj','B:maj',
             'C:min','C#:min','D:min','D#:min','E:min','F:min','F#:min','G:min','G#:min','A:min','A#:min','B:min']

#map indices to labels
labels_A = [chord_map[i] for i in preds_A_idx]
labels_B = [chord_map[i] for i in preds_B_idx]

#convert frame labels to intervals for mir_eval
times = librosa.frames_to_time(np.arange(len(labels_A)), sr=sr, hop_length=HOP_LENGTH)
est_int_A, est_lab_A = mir_eval.util.intervals_from_annotation_data(times, labels_A)
est_int_B, est_lab_B = mir_eval.util.intervals_from_annotation_data(times, labels_B)

#calculate weighted accuracy (Mirex score)
score_A = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_A, est_lab_A)
score_B = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_int_B, est_lab_B)

print("\n--- RESULTS ---")
print(f"Pipeline A (Baseline) Accuracy:     {score_A['mirex05']:.4f}")
print(f"Pipeline B (Segmentation) Accuracy: {score_B['mirex05']:.4f}")

#visualization (generates figure 1 for your paper)
plt.figure(figsize=(10, 8))

#plot A: baseline (noisy)
plt.subplot(3, 1, 1)
librosa.display.specshow(similarity_A, x_axis='time', hop_length=HOP_LENGTH, sr=sr, cmap='coolwarm')
plt.title(f'Pipeline A: Frame-wise Similarity (Acc: {score_A["mirex05"]:.2f})')
plt.ylabel('Chord Index')

#plot B: segmentation (clean)
plt.subplot(3, 1, 2)
librosa.display.specshow(similarity_B, x_axis='time', hop_length=HOP_LENGTH, sr=sr, cmap='coolwarm')
plt.title(f'Pipeline B: Segmented Similarity (Acc: {score_B["mirex05"]:.2f})')
plt.ylabel('Chord Index')

#plot C: waveform + onsets
plt.subplot(3, 1, 3)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
vlines = librosa.frames_to_time(segment_boundaries, sr=sr, hop_length=HOP_LENGTH)
plt.vlines(vlines, -1, 1, color='r', linestyle='--', alpha=0.5, label='Segments')
plt.title('Waveform & Adaptive Segmentation')
plt.legend()

plt.tight_layout()
plt.show()