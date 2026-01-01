import os
import matplotlib.pyplot as plt
import math
from supabase import create_client
from dotenv import load_dotenv
from metaprompt import VISUALIZATION_CLASSNAME

# 1. Configuration & Connection
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Define the models and their assigned colors
MODELS = ['HistGradientBoost', 'LogisticRegression', 'RandomForest', VISUALIZATION_CLASSNAME]
COLORS = [
    "#9FD8FF",  # lightened #57B9FF (HistGradientBoost)
    "#D9A0D9",  # lightened #BF40BF (LogisticRegression)
    "#7FB7A7",  # lightened #097969 (RandomForest)
    "#ef4444",  # unchanged (ResidualMLPEnsemble)
]
def get_min_max_score(class_name):
    print(f"Fetching score for {class_name}...")
    response = supabase.table("algorithms") \
        .select("min_max_score") \
        .eq("class_name", class_name) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    
    if response.data:
        return math.floor(response.data[0].get('min_max_score', 0) * 10000) / 100

    return 0

try:
    # 2. Fetch scores for all models
    scores = [get_min_max_score(m) for m in MODELS]

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(MODELS, scores, color=COLORS)

    # Styling the chart
    plt.title(f"{VISUALIZATION_CLASSNAME} vs Sk-learn Baseline Agg. Min-Max Score", fontsize=16, fontweight='bold')
    plt.ylabel("Aggregate Score", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylim(0, 100.1)  # Scale from 0 to 1.1 to leave room for labels
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add accuracy labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.2f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold' if i == 3 else 'normal'  # only your model bold
        )


    plt.tight_layout()

    # 4. Save the plot
    filename = f"{VISUALIZATION_CLASSNAME}_barchart.png"
    plt.savefig(filename, dpi=300)
    print(f"Success! Plot saved as {filename}")

except Exception as e:
    print(f"Error: {e}")