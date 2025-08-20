from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

# fetch_california_housing, fetch_olivetti_faces
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score, cohen_kappa_score
# from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes, fetch_california_housing, fetch_olivetti_faces, fetch_20newsgroups, fetch_covtype, fetch_kddcup99

def evaluate_model(model, dataset_name, test_size=0.2):
    if dataset_name == "Iris":
        data = load_iris(return_X_y=True)
    elif dataset_name == "Wine":
        data = load_wine()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Digits":
        data = load_digits()
    elif dataset_name == "Diabetes":
        data = load_diabetes()
    elif dataset_name == "California Housing":
        data = fetch_california_housing()
    elif dataset_name == "Olivetti Faces":
        data = fetch_olivetti_faces()
    elif dataset_name == "Covertype":
        data = fetch_covtype()
    elif dataset_name == "KDD Cup 1999":
        data = fetch_kddcup99()
    elif dataset_name == "Abalone":
        data = fetch_openml(name='abalone', version=1, as_frame=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    # cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Initialize results dictionary
    results = {
        "dataset": dataset_name,
        "model": model.__class__.__name__,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "cohen_kappa": kappa,
        # "cross_val_scores": cv_scores,
        # "mean_cv_score": cv_scores.mean(),
        # "cv_score_std": cv_scores.std()
    }
    
    # Calculate ROC AUC and Average Precision for binary classification
    if len(np.unique(y)) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        results["average_precision"] = average_precision_score(y_test, y_pred_proba)
    
    return results



import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_california_housing, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator
from typing import List, Tuple, Dict, Union

class BenchmarkSuite:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.datasets = self._load_datasets()
        self.results = {}

    def _load_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        datasets = {
            'Iris': load_iris(return_X_y=True),
            'Wine': load_wine(return_X_y=True),
            'Breast Cancer': load_breast_cancer(return_X_y=True),
            'Digits': load_digits(return_X_y=True),
            'California Housing': fetch_california_housing(return_X_y=True),
            'Olivetti Faces': fetch_olivetti_faces(return_X_y=True),
        }

        split_datasets = {}
        for name, (X, y) in datasets.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            split_datasets[name] = (X_train_scaled, X_test_scaled, y_train, y_test)
        
        return split_datasets

    def run_benchmark(self, models: List[BaseEstimator]) -> Dict[str, Dict[str, float]]:
        for model in models:
            model_name = model.__class__.__name__
            self.results[model_name] = {}

            for dataset_name, (X_train, X_test, y_train, y_test) in self.datasets.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if dataset_name in ['California Housing', 'Olivetti Faces']:
                    score = r2_score(y_test, y_pred)
                    metric = 'R2 Score'
                else:
                    score = accuracy_score(y_test, y_pred)
                    metric = 'Accuracy'
                
                self.results[model_name][dataset_name] = {metric: score}

        return self.results

    def execute(filename, class_name, model, count=1):
        if count > 2:
            try:
                os.remove(filename)
                return
            except:
                pass
        
        EXECUTION_STRINGS = f"""importlib.reload(metaomni)
    ml_model = metaomni.{filename.split('.py')[0]}.{class_name}()
    ml_model.fit(x_train, y_train)
    preds = ml_model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    print(class_name, accuracy)"""
        
        try:
            exec(EXECUTION_STRINGS)
        except Exception as e:
            error_message = traceback.format_exc()
            print("Hit error: ", error_message)
            
            prompt = f"""
            Existing code:
            {open(f'metaomni/{filename}', 'r').read()}
        
            Error message on original execution:
            {e}
        
            Full traceback:
            {error_message}
        
            Given the original code and this error, rewrite a {model} classifier in the style of SciKit learn, with a {class_name} class that implements the methods fit(self, X_train, y_train) and predict(self, X_test)"""
            implementation = gen(prompt)
            
            snippets = extract_code_snippets(implementation)
            save_first_snippet(snippets, 'metaomni/' + filename)
            execute(filename, class_name, model, count+1)

    def print_results(self):
        for model, datasets in self.results.items():
            print(f"\nResults for {model}:")
            for dataset, scores in datasets.items():
                for metric, score in scores.items():
                    print(f"  {dataset} - {metric}: {score:.4f}")

# suite = BenchmarkSuite()
# models = [mo.MultiLevelAbstractionKNN()]
# results = suite.run_benchmark(models)
# suite.print_results()
# 
# gen("What are 10 professional visualizations to compare machine learning algorithm performaces to each other, and generally to visualize the performance of classifiers?")
# plot = """Confusion Matrix Heatmap:
#    - A color-coded matrix showing True Positives, True Negatives, False Positives, and False Negatives.
#    - Useful for multi-class classification problems.
#    - Can be normalized to show percentages instead of raw counts."""
# 
# plotting_code = gen("""Given python classifaction predictions y_test and preds,
# which contain ground truth test labels and machine learning algorithm predictions,
# write high quality visualization code using matplotlib, which is: %s"""%(plot))
# 
# plot_code = extract_code_snippets(plotting_code)
# 
# print(plot_code[0])
# 
# exec(plot_code[0])
# 
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# 
# def plot_decision_boundary(model, X, y, title="Decision Boundary", filename="decision_boundary.png"):
#     # Reduce to 2D for visualization
#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X)
#     
#     # Create a mesh grid
#     x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
#     y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                          np.arange(y_min, y_max, 0.1))
#     
#     # Make predictions on the mesh grid
#     Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
#     Z = Z.reshape(xx.shape)
#     
#     # Plot the decision boundary
#     plt.figure(figsize=(10, 8))
#     plt.contourf(xx, yy, Z, alpha=0.4)
#     scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.8)
#     plt.title(title)
#     plt.xlabel("First Principal Component")
#     plt.ylabel("Second Principal Component")
#     plt.colorbar(scatter)
#     
#     # Save the plot instead of showing it
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()  # Close the figure to free up memory
# 
# # Assuming x_test and y_test are already defined
# model = mo.MultiLevelAbstractionKNN(n_neighbors=5, n_levels=3)
# model.fit(x_test, y_test)
# plot_decision_boundary(model, x_test, y_test, "MultiLevelAbstractionKNN Decision Boundary", "multilevel_abstraction_knn_decision_boundary.png")
# 
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier
# from matplotlib.colors import ListedColormap
# 
# def plot_decision_boundaries(model1, model2, X, y, title="Decision Boundaries Comparison", filename="decision_boundaries.png"):
#     # Reduce to 2D for visualization
#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X)
#     
#     # Create a mesh grid
#     x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
#     y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                          np.arange(y_min, y_max, 0.1))
#     
#     # Make predictions on the mesh grid for both models
#     Z1 = model1.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
#     Z1 = Z1.reshape(xx.shape)
#     
#     Z2 = model2.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
#     Z2 = Z2.reshape(xx.shape)
#     
#     # Create custom colormap
#     cmap = ListedColormap(['#FF9999', '#66B2FF', '#99FF99'])
#     
#     # Plot the decision boundaries
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
#     
#     # Plot for MultiLevelAbstractionKNN
#     im1 = ax1.contourf(xx, yy, Z1, cmap=cmap, alpha=0.8)
#     scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, edgecolor='black', linewidth=1)
#     ax1.set_title("MultiLevelAbstractionKNN")
#     ax1.set_xlabel("First Principal Component")
#     ax1.set_ylabel("Second Principal Component")
#     
#     # Plot for standard KNN
#     im2 = ax2.contourf(xx, yy, Z2, cmap=cmap, alpha=0.8)
#     scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, edgecolor='black', linewidth=1)
#     ax2.set_title("Standard KNN")
#     ax2.set_xlabel("First Principal Component")
#     ax2.set_ylabel("Second Principal Component")
#     
#     # Add legend
#     classes = np.unique(y)
#     class_names = [f'Class {i}' for i in classes]
#     legend1 = ax1.legend(scatter1.legend_elements()[0], class_names, 
#                          loc="lower right", title="Classes")
#     ax1.add_artist(legend1)
#     
#     legend2 = ax2.legend(scatter2.legend_elements()[0], class_names, 
#                          loc="lower right", title="Classes")
#     ax2.add_artist(legend2)
#     
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()
# 
# # Assuming x_test and y_test are already defined
# model1 = mo.MultiLevelAbstractionKNN(n_neighbors=5, n_levels=3)
# model1.fit(x_test, y_test)
# model2 = KNeighborsClassifier(n_neighbors=5)
# model2.fit(x_test, y_test)
# plot_decision_boundaries(model1, model2, x_test, y_test, "MultiLevelAbstractionKNN vs Standard KNN")
# 
# # Table data to latex generator
# import pandas as pd
# table_df = pd.DataFrame(table_overall)
# table_df.columns = dataset_names
# table_df.index = [m[0] for m in models]
# latex_table = table_df.to_latex(index=False)
# # To save the LaTeX table to a file:
# with open('table2.tex', 'w') as f:
#     f.write(latex_table)
