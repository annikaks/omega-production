import os
import matplotlib.pyplot as plt
from supabase import create_client
from dotenv import load_dotenv

# 1. Configuration & Connection
load_dotenv() # Ensure you have SUPABASE_URL and SUPABASE_KEY in your .env file
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

TARGET_CLASS_NAME = "DimAwareForest"

# 2. Dataset Column Names (Must match your SQL column order)
datasets = [
    "iris_acc", "wine_acc", "breast_cancer_acc", "digits_acc", "balance_scale_acc",
    "blood_transfusion_acc", "haberman_acc", "seeds_acc", "teaching_assistant_acc", "zoo_acc",
    "planning_relax_acc", "ionosphere_acc", "sonar_acc", "glass_acc", "vehicle_acc",
    "liver_disorders_acc", "heart_statlog_acc", "pima_diabetes_acc", "australian_acc", "monks_1_acc"
]

# Display names for the X-axis
display_names = [d.replace('_acc', '').replace('_', ' ').title() for d in datasets]

def get_model_data(class_name):
    print(f"Fetching data for {class_name}...")
    response = supabase.table("algorithms") \
        .select("*") \
        .eq("class_name", class_name) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    
    if not response.data:
        raise ValueError(f"No model found with name: {class_name}")
    
    row = response.data[0]
    # Extract values in the correct order
    return [row.get(d, 0) for d in datasets]

# 3. Benchmark Data (Hardcoded for stability)
benchmark_1 = [0.9, 1.0, 0.9737, 0.9583, 0.75, 0.7733, 0.629, 0.8571, 0.6129, 0.9524, 0.5676, 0.9859, 0.8095, 0.6744, 0.7706, 0.9193, 0.8333, 0.7532, 0.9351, 1.0]
benchmark_2 = [0.9333, 0.9722, 0.9825, 0.9722, 0.65, 0.7667, 0.6774, 0.881, 0.6129, 0.9048, 0.7027, 0.9296, 0.8333, 0.6744, 0.8176, 0.343, 0.8519, 0.7143, 0.9237, 0.6429]
benchmark_3 = [0.9, 1.0, 0.9561, 0.9639, 0.75, 0.7467, 0.629, 0.9048, 0.6129, 0.9048, 0.7027, 0.9577, 0.881, 0.814, 0.7294, 0.9178, 0.8148, 0.7597, 0.9466, 1.0]

try:
    # 4. Fetch Target Data
    target_values = get_model_data(TARGET_CLASS_NAME)

    # 5. Plotting
    plt.figure(figsize=(15, 7))
    
    # Benchmarks (Blue)
    plt.plot(display_names, benchmark_1, color='#57B9FF', alpha=0.7, linestyle=':', label='HistGradientBoost')
    plt.plot(display_names, benchmark_2, color='#57B9FF', alpha=0.7, linestyle='--', label='LogisticRegression')
    plt.plot(display_names, benchmark_3, color='#57B9FF', alpha=0.7, linestyle='-.', label='RandomForest')

    # Target Model (Red)
    plt.plot(display_names, target_values, color='#ef4444', linewidth=3, marker='o', markersize=8, label=TARGET_CLASS_NAME)

    # Styling
    plt.title(f"{TARGET_CLASS_NAME} vs Sklearn Baseline Classifiers", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Accuracy")
    plt.xlabel("Sklearn/OpenML Datasets")
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='lower left')
    plt.tight_layout()

    # Save
    filename = f"comparison_{TARGET_CLASS_NAME.lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"Success! Plot saved as {filename}")

except Exception as e:
    print(f"Error: {e}")