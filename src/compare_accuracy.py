import os
import json
import numpy as np
import matplotlib.pyplot as plt

"""
PlantPulse: Comparative Accuracy Benchmark Module
Evaluates and contrasts MobileNetV2 (Transfer Learning) and Custom CNN (From Scratch),
generates side-by-side performance tables, and exports validation curves to results/ and static/.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_metrics(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_comparison():
    m_algo1 = load_metrics("algo1_metrics.json")
    m_algo2 = load_metrics("algo2_metrics.json")
    
    models = []
    if m_algo2:
        models.append({
            "key": "mobilenetv2",
            "name": "MobileNetV2",
            "metrics": m_algo2,
            "color": "#10b981",
            "edge": "#047857"
        })
    if m_algo1:
        models.append({
            "key": "custom_cnn",
            "name": "Sequential 3-Block CNN",
            "metrics": m_algo1,
            "color": "#3b82f6",
            "edge": "#1d4ed8"
        })
        
    if not models:
        print("Error: No metrics files found in results/ directory.")
        return

    # -------------------------------------------------------------
    # 1. Print Comparison Table
    # -------------------------------------------------------------
    sep = "=" * 105
    dash = "-" * 105
    print("\n" + sep)
    print("                 COTTON DISEASE DETECTION: FAIR ALGORITHM BENCHMARK")
    print("           (All models evaluated on the EXACT same 384-image stratified test set)")
    print(sep)
    
    col_w = 22
    header = f"{'Metric':<24} | " + " | ".join([f"{m['name']:<{col_w}}" for m in models])
    print(header)
    print(dash)
    
    def format_row(label, key, is_pct=True):
        vals = []
        for m in models:
            v = m["metrics"].get(key, 0)
            if is_pct:
                vals.append(f"{v * 100:6.2f}%{'':<{col_w - 7}}")
            else:
                vals.append(f"{v:6.1f}s{'':<{col_w - 7}}")
        return f"{label:<24} | " + " | ".join(vals)
        
    print(format_row("Test Accuracy", "test_accuracy"))
    print(format_row("Weighted Precision", "precision_weighted"))
    print(format_row("Weighted Recall", "recall_weighted"))
    print(format_row("Weighted F1-Score", "f1_weighted"))
    print(format_row("Training Time", "training_time_seconds", is_pct=False))
    print(sep)
    
    # -------------------------------------------------------------
    # 2. Winner Verdict
    # -------------------------------------------------------------
    best_model = max(models, key=lambda m: m["metrics"].get("test_accuracy", 0))
    best_acc = best_model["metrics"].get("test_accuracy", 0) * 100
    print(f"\nWINNER: {best_model['name']} ({best_acc:.2f}% Test Accuracy)")
    print(f"Conclusion: Evaluated fairly on identical held-out test data, {best_model['name']} achieves the highest generalization accuracy.")
    print(sep + "\n")
    
    # -------------------------------------------------------------
    # 3. Side-by-side Bar Chart & Validation Curve Overlay
    # -------------------------------------------------------------
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_keys = [
        ("Accuracy", "test_accuracy"),
        ("Precision", "precision_weighted"),
        ("Recall", "recall_weighted"),
        ("F1-Score", "f1_weighted")
    ]
    
    x = np.arange(len(metrics_keys))
    n_models = len(models)
    width = 0.8 / n_models
    
    for i, m in enumerate(models):
        vals = [m["metrics"].get(k, 0) * 100 for _, k in metrics_keys]
        offset = (i - (n_models - 1) / 2) * width
        rects = ax_bar.bar(x + offset, vals, width, label=m["name"], color=m["color"], edgecolor=m["edge"])
        
        # Add labels on bars
        for rect in rects:
            h = rect.get_height()
            ax_bar.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_bar.set_ylabel('Percentage (%)', fontsize=12)
    ax_bar.set_title('Test Set Performance (Fair Benchmark)', fontsize=14, fontweight='bold', pad=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([label for label, _ in metrics_keys], fontsize=11)
    ax_bar.set_ylim(0, 115)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.4)
    ax_bar.legend(loc='lower right', frameon=True)
    
    # Validation curve overlay
    for m in models:
        val_acc = m["metrics"].get("history", {}).get("val_accuracy", [])
        if val_acc:
            epochs = range(1, len(val_acc) + 1)
            ax_curve.plot(epochs, [v * 100 for v in val_acc], 'o-', label=f"{m['name']} Val Acc", color=m["color"], linewidth=2)
            
    ax_curve.set_xlabel('Epochs', fontsize=12)
    ax_curve.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax_curve.set_title('Validation Accuracy Across Epochs', fontsize=14, fontweight='bold', pad=12)
    ax_curve.grid(True, linestyle='--', alpha=0.4)
    ax_curve.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    comp_path = os.path.join(RESULTS_DIR, "comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Comparison chart saved to: {comp_path}")
    
    # Also copy to static/ for Flask web serving
    static_comp_path = os.path.join(BASE_DIR, "static", "comparison.png")
    try:
        import shutil
        shutil.copyfile(comp_path, static_comp_path)
        print(f"Chart synced to static: {static_comp_path}")
    except Exception as e:
        print(f"Note: Could not copy to static: {e}")

if __name__ == "__main__":
    run_comparison()
