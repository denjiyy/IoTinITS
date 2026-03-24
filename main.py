import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("traffic_data_sofia.csv")

FEATURES = ['Speed_kmh', 'Lane_Occupancy', 'Hour_of_Day']
TARGET   = 'Congestion_Level'
LABELS   = {0: "Free Flow", 1: "Moderate", 2: "Congested"}

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train model ────────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 55)
print("  Sofia Traffic Congestion Model — Evaluation")
print("=" * 55)
print(f"\n  Test set accuracy : {accuracy * 100:.1f}%")
print(f"  Training samples  : {len(X_train)}")
print(f"  Test samples      : {len(X_test)}")
print()
print(classification_report(
    y_test, y_pred,
    target_names=[LABELS[i] for i in sorted(LABELS)],
    zero_division=0
))

# ── Feature importance ─────────────────────────────────────────────────────────
importance = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("  Feature importance:")
for feat, score in importance:
    bar = "█" * int(score * 40)
    print(f"    {feat:<20} {bar} {score:.3f}")

# ── Live predictions on representative scenarios ───────────────────────────────
print("\n" + "=" * 55)
print("  Real-Time Predictions — Sofia Junctions")
print("=" * 55)

scenarios = [
    {"desc": "J-SOF-01  Mon 08:20 (rush hour)",  "Speed_kmh": 14,  "Lane_Occupancy": 91, "Hour_of_Day": 8},
    {"desc": "J-SOF-02  Mon 10:30 (mid-morning)", "Speed_kmh": 58,  "Lane_Occupancy": 38, "Hour_of_Day": 10},
    {"desc": "J-SOF-03  Mon 12:15 (lunch)",        "Speed_kmh": 40,  "Lane_Occupancy": 62, "Hour_of_Day": 12},
    {"desc": "J-SOF-01  Mon 17:05 (peak hour)",    "Speed_kmh": 9,   "Lane_Occupancy": 96, "Hour_of_Day": 17},
    {"desc": "J-SOF-02  Mon 20:00 (evening)",      "Speed_kmh": 64,  "Lane_Occupancy": 28, "Hour_of_Day": 20},
    {"desc": "J-SOF-03  Tue 02:00 (night)",        "Speed_kmh": 76,  "Lane_Occupancy": 13, "Hour_of_Day": 2},
]

STATUS_ICON = {0: "🟢", 1: "🟡", 2: "🔴"}

for s in scenarios:
    row = pd.DataFrame([{k: v for k, v in s.items() if k != "desc"}])
    pred   = model.predict(row)[0]
    proba  = model.predict_proba(row)[0]
    confidence = proba[pred] * 100
    icon   = STATUS_ICON[pred]
    label  = LABELS[pred]
    print(f"\n  {icon} {s['desc']}")
    print(f"     → {label} (confidence: {confidence:.0f}%)")
    print(f"       Free:{proba[0]*100:.0f}%  Moderate:{proba[1]*100:.0f}%  Congested:{proba[2]*100:.0f}%")

print("\n" + "=" * 55)
print("  Model is ready for predicting traffic jams in real time.")
print("=" * 55)

# ── Charts ─────────────────────────────────────────────────────────────────────
COLORS = ["#4ade80", "#facc15", "#f87171"]  # green / yellow / red
LEVEL_NAMES = [LABELS[i] for i in sorted(LABELS)]

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0f172a")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])   # Congestion over time  (wide)
ax2 = fig.add_subplot(gs[0, 2])    # Feature importance
ax3 = fig.add_subplot(gs[1, 0])    # Avg speed by hour
ax4 = fig.add_subplot(gs[1, 1])    # Lane occupancy distribution
ax5 = fig.add_subplot(gs[1, 2])    # Confusion matrix

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")

# 1. Congestion level over time ─────────────────────────────────────────────────
df_plot = df.copy()
df_plot["Timestamp"] = pd.to_datetime(df_plot["Timestamp"])
df_plot = df_plot.sort_values("Timestamp")

for level, color in zip([0, 1, 2], COLORS):
    mask = df_plot["Congestion_Level"] == level
    ax1.scatter(
        df_plot.loc[mask, "Timestamp"],
        df_plot.loc[mask, "Congestion_Level"],
        c=color, label=LABELS[level], s=45, alpha=0.85, zorder=3
    )
ax1.plot(df_plot["Timestamp"], df_plot["Congestion_Level"],
         color="#475569", linewidth=0.6, alpha=0.5, zorder=2)
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(LEVEL_NAMES, fontsize=8)
ax1.set_title("Congestion Level Over Time", fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Timestamp")
ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
fig.autofmt_xdate(rotation=30, ha="right")

# 2. Feature importance ─────────────────────────────────────────────────────────
feat_names, feat_scores = zip(*importance)
bars = ax2.barh(feat_names, feat_scores, color=["#38bdf8", "#818cf8", "#fb923c"])
ax2.set_xlim(0, max(feat_scores) * 1.25)
ax2.set_title("Feature Importance", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Score")
for bar, score in zip(bars, feat_scores):
    ax2.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{score:.3f}", va="center", color="#e2e8f0", fontsize=8)

# 3. Average speed by hour ──────────────────────────────────────────────────────
hourly_speed = df.groupby("Hour_of_Day")["Speed_kmh"].mean()
ax3.plot(hourly_speed.index, hourly_speed.values,
         color="#38bdf8", linewidth=2, marker="o", markersize=4)
ax3.fill_between(hourly_speed.index, hourly_speed.values,
                 alpha=0.15, color="#38bdf8")
ax3.set_title("Avg Speed by Hour", fontsize=11, fontweight="bold", pad=10)
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Speed (km/h)")
ax3.set_xticks(range(0, 24, 3))

# 4. Lane occupancy distribution by congestion level ───────────────────────────
for level, color in zip([0, 1, 2], COLORS):
    vals = df.loc[df["Congestion_Level"] == level, "Lane_Occupancy"]
    ax4.hist(vals, bins=10, alpha=0.65, color=color, label=LABELS[level], edgecolor="#0f172a")
ax4.set_title("Lane Occupancy Distribution", fontsize=11, fontweight="bold", pad=10)
ax4.set_xlabel("Lane Occupancy (%)")
ax4.set_ylabel("Count")
ax4.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

# 5. Confusion matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
im = ax5.imshow(cm, cmap="Blues")
ax5.set_xticks(range(len(LEVEL_NAMES)))
ax5.set_yticks(range(len(LEVEL_NAMES)))
ax5.set_xticklabels(LEVEL_NAMES, rotation=30, ha="right", fontsize=7)
ax5.set_yticklabels(LEVEL_NAMES, fontsize=7)
ax5.set_title("Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax5.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "#94a3b8", fontsize=10)

fig.suptitle("Sofia Traffic Congestion — Model Dashboard",
             fontsize=14, fontweight="bold", color="#f1f5f9", y=1.01)

plt.savefig("traffic_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\n  Chart saved → traffic_dashboard.png")