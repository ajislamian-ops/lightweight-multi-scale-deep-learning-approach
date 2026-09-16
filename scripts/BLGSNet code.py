import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

# ✅ ADD THIS (you were using TF-Keras but not importing tensorflow)
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    concatenate, LSTM, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore")

# ================= PCA CORR SETTINGS (FOR PAPER FIGURE) =================
PLOT_CORR_TOP_N = 30           # if too crowded, set to 15–20 for Elsevier clarity
PLOT_PCA_LOADING_TOP_N = 25
# =======================================================================

# ============================================================
# Optional: avoid TF grabbing all GPU memory
# ============================================================
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# ============================================================
# Helper formatting
# ============================================================
def pct(x, decimals=2):
    return round(float(x) * 100.0, decimals)

def classification_report_percent(y_true, y_pred, fault_names, labels=None):
    from sklearn.metrics import precision_recall_fscore_support

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\nClassification Report (Percent %):")
    print(f"{'Class':<28} {'Prec%':>8} {'Recall%':>9} {'F1%':>8} {'Support':>9}")
    print("-" * 65)
    for i, lab in enumerate(labels):
        name = fault_names.get(lab, str(lab))
        print(f"{name:<28} {pct(p[i]):>8.2f} {pct(r[i]):>9.2f} {pct(f1[i]):>8.2f} {int(s[i]):>9d}")
    print("-" * 65)
    print(f"{'Macro avg':<28} {pct(p_macro):>8.2f} {pct(r_macro):>9.2f} {pct(f1_macro):>8.2f} {int(sum(s)):>9d}")
    print(f"{'Weighted avg':<28} {pct(p_w):>8.2f} {pct(r_w):>9.2f} {pct(f1_w):>8.2f} {int(sum(s)):>9d}")

# ============================================================
# ✅ NEW: PARAMS + FLOPs + CPU/GPU BENCHMARK HELPERS
# ============================================================
def print_param_counts(model):
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    total = int(model.count_params())
    print("\n=== PARAMETER COUNTS (BLGSNet) ===")
    print(f"Trainable params     : {trainable:,}")
    print(f"Non-trainable params : {non_trainable:,}")
    print(f"Total params         : {total:,}")

def _convert_to_frozen_graph(concrete_func):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    return frozen_func, graph_def

def _profile_graph_flops(graph_def):
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd="op",
            options=opts
        )
        return int(flops.total_float_ops) if flops is not None else None

def get_flops_tf2(model, input_shape_without_batch):
    """
    FLOPs for one forward pass with batch=1.
    input_shape_without_batch e.g. (num_features, 1)
    """
    try:
        @tf.function
        def forward(x):
            return model(x, training=False)

        spec = tf.TensorSpec([1, *input_shape_without_batch], tf.float32)
        concrete = forward.get_concrete_function(spec)
        _, graph_def = _convert_to_frozen_graph(concrete)
        flops = _profile_graph_flops(graph_def)
        return flops
    except Exception as e:
        print("\nFLOPs computation failed:", str(e))
        print("Tip: FLOPs depends on TF version; if it fails, tell me your TF version.")
        return None

def benchmark_inference_tf(model, sample_input_np, device="/CPU:0", warmup=50, runs=200):
    """
    Proper timing with sync (.numpy()) so GPU timing is real.
    Measures batch=1 inference by default (sample_input_np should be shape (1, F, 1)).
    """
    x = tf.convert_to_tensor(sample_input_np, dtype=tf.float32)

    @tf.function(jit_compile=False)
    def infer(inp):
        return model(inp, training=False)

    with tf.device(device):
        for _ in range(warmup):
            y = infer(x)
            _ = y.numpy()  # sync

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            y = infer(x)
            _ = y.numpy()  # sync
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return float(times.mean()), float(times.std()), float(np.median(times))

# ============================================================
# 0) LOAD TRAIN / TEST DATASETS
# ============================================================
train_path = r"C:\Users\WICOM_PC\Desktop\iforest_khan\train.csv"
test_path  = r"C:\Users\WICOM_PC\Desktop\iforest_khan\test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

fault_names = {
    0: "Fault-free system",
    1: "String fault",
    2: "String-to-ground fault",
    3: "String-to-string fault",
    4: "Component fault",
    5: "Partial shading fault"
}

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("\nTrain class distribution:")
print(train_df["class"].value_counts())

print("\nTest class distribution:")
print(test_df["class"].value_counts())

# ============================================================
# 0.1) COMMON FEATURES ONLY
# ============================================================
train_features = set(train_df.columns) - {"class"}
test_features  = set(test_df.columns) - {"class"}
common_features = sorted(list(train_features.intersection(test_features)))

if len(common_features) == 0:
    raise ValueError("No common feature columns found between train and test!")

X_train = train_df[common_features].copy()
y_train = train_df["class"].astype(int)

X_test  = test_df[common_features].copy()
y_test  = test_df["class"].astype(int)

print(f"\nUsing {len(common_features)} common feature columns.")

# ============================================================
# 1) STANDARDIZATION
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=common_features)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=common_features)

# ============================================================
# 1.1) TRAIN / VALIDATION SPLIT
# ============================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# ============================================================
# 2) FEATURE METHOD #1: MLP FEATURE RANKING (TOP-K)
# ============================================================
TOP_K = 15

print("\nTraining MLP for feature importance (permutation importance)...")
mlp_ranker = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp_ranker.fit(X_tr, y_tr)

perm = permutation_importance(
    mlp_ranker, X_val, y_val,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

mlp_importance = pd.Series(perm.importances_mean, index=common_features).sort_values(ascending=False)
top_mlp_features = mlp_importance.head(TOP_K).index.tolist()

print("\nTop features by MLP (Top-K):")
print(top_mlp_features)

# ============================================================
# ✅ ELSEVIER-STYLE CORRELATION MATRIX (TRAINING ONLY)
# ============================================================
if len(common_features) <= PLOT_CORR_TOP_N:
    corr_features = common_features
else:
    corr_features = top_mlp_features[:min(PLOT_CORR_TOP_N, len(top_mlp_features))]
    if len(corr_features) < PLOT_CORR_TOP_N:
        needed = PLOT_CORR_TOP_N - len(corr_features)
        extra = [f for f in common_features if f not in corr_features][:needed]
        corr_features = corr_features + extra

corr_features = [c for c in corr_features if c in X_train_scaled.columns]

print(f"\nPlotting Elsevier-style correlation matrix using {len(corr_features)} features...")

corr_matrix = X_train_scaled[corr_features].corr(method="pearson")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.set_context("paper", font_scale=1.0)
sns.set_style("white")

fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

hm = sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.6,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.9, "pad": 0.02},
    ax=ax
)

ax.set_title("Pearson Correlation Matrix (Training Set)", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig("Elsevier_Correlation_Matrix.tiff", dpi=600, bbox_inches="tight")
plt.savefig("Elsevier_Correlation_Matrix.png", dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# 3) PCA
# ============================================================
print("\nFitting PCA (Top-K components)...")
pca = PCA(n_components=TOP_K, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("Explained variance ratio (sum):", pca.explained_variance_ratio_.sum())

loadings = pd.DataFrame(
    pca.components_.T,
    index=common_features,
    columns=[f"PC{i+1}" for i in range(TOP_K)]
)

abs_loading_strength = loadings.abs().sum(axis=1).sort_values(ascending=False)
top_loading_features = abs_loading_strength.head(min(PLOT_PCA_LOADING_TOP_N, len(abs_loading_strength))).index.tolist()

print("\nTop original features contributing to PCA components (by summed |loading|):")
print(top_loading_features)

plt.figure(figsize=(12, 7), dpi=150)
sns.heatmap(loadings.loc[top_loading_features, :], cmap="vlag", center=0,
            linewidths=0.4, linecolor="white", cbar=True)
plt.title("PCA Loadings Heatmap (Top Contributing Features)")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.tight_layout()
plt.savefig("Elsevier_PCA_Loadings.tiff", dpi=600, bbox_inches="tight")
plt.savefig("Elsevier_PCA_Loadings.png", dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# ========================= REST OF YOUR PIPELINE =========================
# (Unchanged)
# ============================================================
def prepare_data_for_blgsnet(X_data):
    X_arr = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
    num_features = X_arr.shape[1]
    return X_arr.reshape(-1, num_features, 1)

X_tr_blgs   = prepare_data_for_blgsnet(X_tr)
X_val_blgs  = prepare_data_for_blgsnet(X_val)
X_test_blgs = prepare_data_for_blgsnet(X_test_scaled)

def create_blgsnet_model(num_features, num_classes):
    inp = Input(shape=(num_features, 1), name="blgsnet_input")

    c1 = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
    c1 = MaxPooling1D(pool_size=2)(c1)

    c2 = Conv1D(64, kernel_size=5, activation="relu", padding="same")(inp)
    c2 = MaxPooling1D(pool_size=2)(c2)

    c3 = Conv1D(64, kernel_size=7, activation="relu", padding="same")(inp)
    c3 = MaxPooling1D(pool_size=2)(c3)

    merged_cnn = concatenate([c1, c2, c3], axis=-1)
    blstm = Bidirectional(LSTM(64, return_sequences=False))(merged_cnn)

    x = Dense(256, activation="relu")(blstm)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="BLGSNet")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

num_classes = len(np.unique(y_train))
num_features = X_train_scaled.shape[1]
blgsnet_model = create_blgsnet_model(num_features, num_classes)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    "blgsnet_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

print("\nTraining BLGSNet model...")
history = blgsnet_model.fit(
    X_tr_blgs, y_tr,
    validation_data=(X_val_blgs, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("\nEvaluating BLGSNet on TEST set...")

if os.path.exists("blgsnet_model.h5"):
    size_mb = os.path.getsize("blgsnet_model.h5") / (1024 * 1024)
    print(f"\nBLGSNet model size: {size_mb:.2f} MB")

# ============================================================
# ✅ NEW OUTPUT: Params + FLOPs + CPU/GPU inference time
# ============================================================
print_param_counts(blgsnet_model)

flops = get_flops_tf2(blgsnet_model, input_shape_without_batch=(num_features, 1))
if flops is not None:
    print("\n=== FLOPs (Forward pass, batch=1) ===")
    print(f"FLOPs  : {flops:,}")
    print(f"GFLOPs : {flops / 1e9:.6f}")
    print("Note: TF reports float_ops. Some papers assume 1 MAC = 2 FLOPs.")

# CPU/GPU latency (batch=1)
x1 = X_test_blgs[:1]  # shape (1, num_features, 1)

cpu_mean, cpu_std, cpu_med = benchmark_inference_tf(blgsnet_model, x1, device="/CPU:0", warmup=50, runs=200)
print("\n=== INFERENCE LATENCY (batch=1) ===")
print(f"CPU mean   : {cpu_mean:.4f} ms | std: {cpu_std:.4f} | median: {cpu_med:.4f}")

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    gpu_mean, gpu_std, gpu_med = benchmark_inference_tf(blgsnet_model, x1, device="/GPU:0", warmup=50, runs=200)
    print(f"GPU mean   : {gpu_mean:.4f} ms | std: {gpu_std:.4f} | median: {gpu_med:.4f}")
else:
    print("(No GPU detected by TensorFlow — GPU latency not measured.)")

# ============================================================
# Your original latency function (kept)
# ============================================================
def measure_latency_ms(model, X, warmup=5):
    for _ in range(warmup):
        _ = model.predict(X[:1], verbose=0)

    t0 = time.perf_counter()
    _ = model.predict(X[:1], verbose=0)
    t1 = time.perf_counter()
    single_ms = (t1 - t0) * 1000

    t0 = time.perf_counter()
    _ = model.predict(X, verbose=0)
    t1 = time.perf_counter()
    per_sample_ms = ((t1 - t0) / len(X)) * 1000

    return single_ms, per_sample_ms

single_ms, avg_per_sample_ms = measure_latency_ms(blgsnet_model, X_test_blgs)

print("\n=== LATENCY RESULTS (BLGSNet) [predict-based] ===")
print(f"Single-sample latency: {single_ms:.4f} ms")
print(f"Average latency per sample (full test set): {avg_per_sample_ms:.4f} ms/sample")

y_pred_probs = blgsnet_model.predict(X_test_blgs)
y_pred = np.argmax(y_pred_probs, axis=1)

blgs_accuracy  = accuracy_score(y_test, y_pred)
blgs_precision = precision_score(y_test, y_pred, average="weighted")
blgs_recall    = recall_score(y_test, y_pred, average="weighted")
blgs_f1        = f1_score(y_test, y_pred, average="weighted")

print(f"\nBLGSNet - Acc: {blgs_accuracy:.4f} | Prec: {blgs_precision:.4f} | "
      f"Recall: {blgs_recall:.4f} | F1: {blgs_f1:.4f}")

# ============================================================
# ✅ AI-AGENTIC DECISION SUPPORT LAYER
# This does NOT change model training or accuracy.
# It converts BLGSNet predictions into actionable PV maintenance decisions.
# ============================================================

def pv_fault_agent(pred_class, confidence, feature_row_scaled, feature_names, fault_names):
    """
    Rule-based AI agent for PV fault interpretation and maintenance recommendation.

    Inputs:
        pred_class: predicted class label from BLGSNet
        confidence: softmax confidence of predicted class
        feature_row_scaled: one standardized test sample
        feature_names: feature names used by the model
        fault_names: dictionary of class names

    Output:
        dictionary containing fault interpretation, severity, priority, reason, and action.
    """

    fault = fault_names.get(int(pred_class), f"Class {pred_class}")

    # Identify most abnormal standardized features for explanation
    abs_vals = np.abs(feature_row_scaled)
    top_idx = np.argsort(abs_vals)[-5:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_deviation = [float(feature_row_scaled[i]) for i in top_idx]

    # Confidence-based reliability
    if confidence >= 0.95:
        confidence_level = "Very high"
    elif confidence >= 0.85:
        confidence_level = "High"
    elif confidence >= 0.70:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    # Fault-specific maintenance logic
    if pred_class == 0:
        severity = "Normal"
        priority = "No immediate action"
        reason = "The PV measurements are consistent with normal operating behavior."
        action = "Continue routine monitoring and scheduled inspection."
    elif pred_class == 1:
        severity = "Medium"
        priority = "Planned maintenance"
        reason = "The model detects string-level abnormality, usually related to current mismatch or disconnected string behavior."
        action = "Inspect string current, connectors, fuses, and string wiring continuity."
    elif pred_class == 2:
        severity = "High"
        priority = "Urgent inspection"
        reason = "The model detects string-to-ground fault behavior, which may indicate insulation degradation or leakage path."
        action = "Check grounding path, insulation resistance, DC cables, and protection devices."
    elif pred_class == 3:
        severity = "High"
        priority = "Urgent inspection"
        reason = "The model detects string-to-string fault behavior, often associated with abnormal electrical coupling between strings."
        action = "Inspect inter-string wiring, combiner box connections, and protection circuits."
    elif pred_class == 4:
        severity = "Medium to High"
        priority = "Maintenance required"
        reason = "The model detects component-level abnormality, which may be related to module, connector, diode, or inverter-side degradation."
        action = "Inspect PV modules, bypass diodes, connectors, and inverter-side electrical components."
    elif pred_class == 5:
        severity = "Low to Medium"
        priority = "Condition-based maintenance"
        reason = "The model detects partial shading behavior, usually caused by temporary or permanent obstruction."
        action = "Check nearby obstacles, dust/soiling, vegetation, module mismatch, and shading pattern."
    else:
        severity = "Unknown"
        priority = "Manual review"
        reason = "The predicted class is not included in the predefined agent rules."
        action = "Perform manual inspection and verify sensor measurements."

    return {
        "Predicted class": int(pred_class),
        "Predicted fault": fault,
        "Confidence (%)": round(float(confidence) * 100, 2),
        "Confidence level": confidence_level,
        "Severity": severity,
        "Maintenance priority": priority,
        "Likely reason": reason,
        "Recommended action": action,
        "Top contributing feature 1": top_features[0],
        "Top contributing feature 2": top_features[1],
        "Top contributing feature 3": top_features[2],
        "Top contributing feature 4": top_features[3],
        "Top contributing feature 5": top_features[4],
        "Feature deviation 1": round(top_deviation[0], 4),
        "Feature deviation 2": round(top_deviation[1], 4),
        "Feature deviation 3": round(top_deviation[2], 4),
        "Feature deviation 4": round(top_deviation[3], 4),
        "Feature deviation 5": round(top_deviation[4], 4)
    }


print("\n==============================")
print("AI-AGENTIC PV FAULT DECISION SUPPORT")
print("==============================")

agent_reports = []

# Use BLGSNet raw predictions and probabilities for agent interpretation
X_test_scaled_np = X_test_scaled.values

for i in range(len(X_test_scaled_np)):
    pred_class_i = int(y_pred[i])
    confidence_i = float(np.max(y_pred_probs[i]))

    report_i = pv_fault_agent(
        pred_class=pred_class_i,
        confidence=confidence_i,
        feature_row_scaled=X_test_scaled_np[i],
        feature_names=common_features,
        fault_names=fault_names
    )

    report_i["Sample index"] = i
    report_i["True class"] = int(y_test.iloc[i]) if hasattr(y_test, "iloc") else int(y_test[i])
    report_i["True fault"] = fault_names.get(report_i["True class"], f"Class {report_i['True class']}")
    agent_reports.append(report_i)

agent_df = pd.DataFrame(agent_reports)

# Save complete agentic decision report
agent_report_path = "BLGSNet_AI_Agent_Decision_Report.csv"
agent_df.to_csv(agent_report_path, index=False)

print(f"\nAI-agent decision report saved as: {agent_report_path}")

print("\nAgent summary by predicted fault:")
print(agent_df["Predicted fault"].value_counts())

print("\nAgent summary by maintenance priority:")
print(agent_df["Maintenance priority"].value_counts())

print("\nFirst 10 AI-agent decisions:")
print(agent_df[
    [
        "Sample index", "True fault", "Predicted fault", "Confidence (%)",
        "Severity", "Maintenance priority", "Recommended action"
    ]
].head(10).to_string(index=False))

# Optional: plot maintenance priority distribution
plt.figure(figsize=(8, 5), dpi=150)
agent_df["Maintenance priority"].value_counts().plot(kind="bar")
plt.xlabel("Maintenance Priority")
plt.ylabel("Number of Samples")
plt.title("AI-Agent Maintenance Priority Summary")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("AI_Agent_Maintenance_Priority_Summary.png", dpi=600, bbox_inches="tight")
plt.show()



# Traditional models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=40),
    "Random Forest": RandomForestClassifier(random_state=40),
    "Neural Network (MLP)": MLPClassifier(max_iter=1000, random_state=40)
}

def evaluate_model(model, X_train_fs, X_test_fs, y_train_fs, y_test_fs):
    model.fit(X_train_fs, y_train_fs)
    y_pred_local = model.predict(X_test_fs)
    y_prob = model.predict_proba(X_test_fs) if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test_fs, y_pred_local)
    prec = precision_score(y_test_fs, y_pred_local, average="weighted")
    rec  = recall_score(y_test_fs, y_pred_local, average="weighted")
    f1   = f1_score(y_test_fs, y_pred_local, average="weighted")

    if y_prob is not None:
        auc = roc_auc_score(y_test_fs, y_prob, multi_class="ovr", average="weighted")
    else:
        auc = 0.0

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1, "AUC": auc}, y_pred_local

results = {}
predictions = {}

print("\n==============================")
print("Traditional Models with MLP Top-K Features")
print("==============================")
results["MLP_TopK"] = {}
predictions["MLP_TopK"] = {}

X_train_mlp = X_train_scaled[top_mlp_features]
X_test_mlp  = X_test_scaled[top_mlp_features]

for model_name, model in models.items():
    metrics, ypred_local = evaluate_model(model, X_train_mlp, X_test_mlp, y_train, y_test)
    results["MLP_TopK"][model_name] = metrics
    predictions["MLP_TopK"][model_name] = ypred_local
    print(f"{model_name}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1-score']:.4f}, AUC={metrics['AUC']:.4f}")

print("\n==============================")
print("Traditional Models with PCA Top-K Components")
print("==============================")
results["PCA_TopK"] = {}
predictions["PCA_TopK"] = {}

for model_name, model in models.items():
    metrics, ypred_local = evaluate_model(model, X_train_pca, X_test_pca, y_train, y_test)
    results["PCA_TopK"][model_name] = metrics
    predictions["PCA_TopK"][model_name] = ypred_local
    print(f"{model_name}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1-score']:.4f}, AUC={metrics['AUC']:.4f}")

best_accuracy = -1.0
best_model_name = ""
best_fs_name = ""

for fs_name, fs_results in results.items():
    for model_name, metrics in fs_results.items():
        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]
            best_model_name = model_name
            best_fs_name = fs_name

if blgs_accuracy > best_accuracy:
    print(f"\n✅ Best overall: BLGSNet (Acc={pct(blgs_accuracy):.2f}%)")
    best_overall_pred = y_pred
else:
    print(f"\n✅ Best overall: {best_model_name} using {best_fs_name} (Acc={pct(best_accuracy):.2f}%)")
    best_overall_pred = predictions[best_fs_name][best_model_name]

print("\nAccuracy adjustment removed: using raw model predictions for final evaluation.")

all_classes = sorted(list(fault_names.keys()))
tick_labels = [fault_names[i] for i in all_classes]

classification_report_percent(y_test, best_overall_pred, fault_names, labels=all_classes)

cm = confusion_matrix(y_test, best_overall_pred, labels=all_classes)
final_accuracy = np.trace(cm) / np.sum(cm)

print("\n==============================")
print("FINAL TEST ACCURACY (OFFICIAL)")
print("==============================")
print(f"Correct predictions : {np.trace(cm)}")
print(f"Total samples       : {np.sum(cm)}")
print(f"Final Accuracy      : {final_accuracy:.6f} ({final_accuracy*100:.2f}%)")

plt.figure(figsize=(11, 9), dpi=150)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=tick_labels, yticklabels=tick_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (With Latency Measured)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4), dpi=150)

plt.subplot(1, 2, 1)
plt.plot(np.array(history.history["accuracy"]) * 100)
plt.plot(np.array(history.history["val_accuracy"]) * 100)
plt.title("BLGSNet Accuracy (%)")
plt.ylabel("Accuracy (%)")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("BLGSNet Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")

plt.tight_layout()
plt.show()
