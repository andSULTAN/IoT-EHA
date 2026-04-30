import os
import glob
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from data_loader import DataLoader
from ensemble import IncrementalEnsemble

# 1. Load latest models
MODELS_DIR = "models"
MODEL_PATTERN = os.path.join(MODELS_DIR, "iot_shield_ensemble_*.pkl")
model_files = glob.glob(MODEL_PATTERN)
if not model_files:
    print("Model topilmadi!")
    exit(1)
latest_model_file = max(model_files, key=os.path.getctime)
print(f"Yuklanmoqda: {latest_model_file}")

basename = os.path.basename(latest_model_file).replace('.pkl', '')
parts = basename.split('_')
timestamp = parts[-2] + '_' + parts[-1]
scaler_file = os.path.join(MODELS_DIR, f"scaler_{timestamp}.pkl")
encoder_file = os.path.join(MODELS_DIR, f"label_encoder_{timestamp}.pkl")
meta_file = os.path.join(MODELS_DIR, f"metadata_{timestamp}.json")

ensemble = IncrementalEnsemble.load(latest_model_file)
scaler = joblib.load(scaler_file)
encoder = joblib.load(encoder_file)
with open(meta_file, 'r') as f:
    meta = json.load(f)

class_names = meta['class_names']

# 2. Setup Data Loader and Test Set
with open('session.json', 'r', encoding='utf-8') as f:
    session = json.load(f)
dl = DataLoader()
dl.file_paths = session.get('file_paths', [])
dl.use_grouping = True
dl.scaler = scaler
dl.label_encoder = encoder
dl.feature_columns = meta['feature_names']
dl.label_column = "Label"
dl.class_names = class_names

print("Test set yaratilmoqda (bu biroz vaqt olishi mumkin)...")
# Extract a representative sample for robust testing
test_X, test_y = dl.build_stratified_test_set(rows_per_file=5000, max_rows_per_class=5000)

# 3. Overall Metrics
print("\n=== UMUMIY METRIKALAR ===")
y_pred = ensemble.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)
f1_weighted = f1_score(test_y, y_pred, average='weighted', zero_division=0)
f1_macro = f1_score(test_y, y_pred, average='macro', zero_division=0)
precision = precision_score(test_y, y_pred, average='weighted', zero_division=0)
recall = recall_score(test_y, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 weighted: {f1_weighted:.4f}")
print(f"F1 macro: {f1_macro:.4f}")
print(f"Precision weighted: {precision:.4f}")
print(f"Recall weighted: {recall:.4f}")

# 4. Individual Model Results
print("\n=== INDIVIDUAL MODEL NATIJALARI ===")
individual_acc = []
individual_f1 = []
for name, model in ensemble.models.items():
    y_ind = model.predict(test_X)
    acc_ind = accuracy_score(test_y, y_ind)
    f1_ind = f1_score(test_y, y_ind, average='weighted', zero_division=0)
    print(f"{name}: Accuracy={acc_ind:.4f}, F1_weighted={f1_ind:.4f}")
    individual_acc.append(acc_ind)
    individual_f1.append(f1_ind)

# 5. Classification Report
print("\n=== PER-CLASS CLASSIFICATION REPORT ===")
report = classification_report(test_y, y_pred, target_names=class_names, zero_division=0, output_dict=True)
print(classification_report(test_y, y_pred, target_names=class_names, zero_division=0))

# 6. Test Set Distribution
print("\n=== TEST SET TARKIBI ===")
unique, counts = np.unique(test_y, return_counts=True)
class_counts = {}
for u, c in zip(unique, counts):
    class_counts[class_names[u]] = c
    print(f"  {class_names[u]}: {c} namuna")
print(f"Jami: {len(test_y)} namuna, {len(unique)} sinf")

# ==========================================
# PLOTTING
# ==========================================
import matplotlib
matplotlib.use('Agg') # for headless

# Colors
COLOR_GOOD = '#2E7D32'
COLOR_MID = '#E65100'
COLOR_BAD = '#C62828'

def get_color(val):
    if val >= 0.85: return COLOR_GOOD
    elif val >= 0.50: return COLOR_MID
    else: return COLOR_BAD

# Diagram 1: Overall Metrics
plt.figure(figsize=(12, 7), dpi=200)
metrics_names = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'Precision', 'Recall']
metrics_vals = [accuracy, f1_weighted, f1_macro, precision, recall]
colors = [get_color(v) for v in metrics_vals]

bars = plt.bar(metrics_names, metrics_vals, color=colors)
plt.ylim(0, 1.1)
plt.title("IoT-Shield Ansambl Modelining Umumiy Samaradorlik Ko'rsatkichlari", fontsize=14, pad=20)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('results_overall_metrics.png')
plt.close()

# Diagram 2: Ensemble vs Individual
plt.figure(figsize=(12, 7), dpi=200)
model_names = ['Ensemble'] + list(ensemble.models.keys())
acc_vals = [accuracy] + individual_acc
f1_vals = [f1_weighted] + individual_f1

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
rects1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='#1f77b4')
rects2 = ax.bar(x + width/2, f1_vals, width, label='F1 Weighted', color='#ff7f0e')

ax.set_ylabel('Scores')
ax.set_title('Ansambl va Individual Modellar Qiyosi', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim(0, 1.1)

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
fig.tight_layout()
plt.savefig('results_ensemble_vs_individual.png')
plt.close()

# Diagram 3: Per-class F1
plt.figure(figsize=(12, 7), dpi=200)
per_class_f1 = {cls: report[cls]['f1-score'] for cls in class_names if cls in report}
sorted_f1 = sorted(per_class_f1.items(), key=lambda item: item[1]) # ascending for horizontal
names = [x[0] for x in sorted_f1]
f1s = [x[1] for x in sorted_f1]
colors = [get_color(v) for v in f1s]

bars = plt.barh(names, f1s, color=colors)
plt.xlim(0, 1.1)
plt.title("Sinflar bo'yicha F1-Score Ko'rsatkichlari", fontsize=14, pad=20)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.4f}", ha='left', va='center')
plt.tight_layout()
plt.savefig('results_per_class_f1.png')
plt.close()

# Diagram 4: Confusion Matrix
plt.figure(figsize=(12, 10), dpi=200)
cm = confusion_matrix(test_y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (16 sinf)", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results_confusion_matrix.png')
plt.close()

# Diagram 5: Class Distribution
plt.figure(figsize=(12, 7), dpi=200)
sorted_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
top5 = sorted_counts[:5]
others = sum(item[1] for item in sorted_counts[5:])

labels = [item[0] for item in top5] + ["Boshqalar"]
sizes = [item[1] for item in top5] + [others]
explode = [0.05]*5 + [0]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, pctdistance=0.85, explode=explode,
        colors=sns.color_palette("Set2"))
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Test To'plamidagi Sinf Taqsimoti", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('results_class_distribution.png')
plt.close()

# 7. Inference Speed Test
print("\n=== INFERENCE TEZLIK TESTI ===")
# Generate a dummy X or take first 1000 from test set
test_set_size = len(test_X)
sample_size = min(1000, test_set_size)
sample_X = test_X[:sample_size]

start = time.perf_counter()
for i in range(sample_size):
    _ = ensemble.predict(sample_X[i:i+1])
elapsed = time.perf_counter() - start
ms_per_sample = (elapsed / sample_size) * 1000
print(f"Inference tezligi: {ms_per_sample:.3f} ms/namuna")
print(f"Soniyada: {1000/(ms_per_sample/1000):.0f} bashorat")

# 8. Model Sizes
print("\n=== MODEL HAJMI ===")
total_size = 0
for f in os.listdir(MODELS_DIR):
    if timestamp in f:
        fpath = os.path.join(MODELS_DIR, f)
        size = os.path.getsize(fpath)
        total_size += size
        print(f"  {f}: {size/1024:.1f} KB")
print(f"Jami model hajmi: {total_size/1024:.1f} KB ({total_size/1024/1024:.3f} MB)")

print("\nBarcha grafiklar muvaffaqiyatli chizildi va .png formatda saqlandi.")
