# Omega: A Library of AI Invented Machine Learning Algorithms

![Omega Logo](omega_logo.png)

Omega is an open-source Python library containing novel machine learning algorithms generated through automated algorithmic synthesis techniques. These algorithms push the boundaries of traditional machine learning approaches and offer unique capabilities for tackling complex data analysis tasks.

## Key Features

- **Novel Algorithms**: Includes machine learning algorithms not found in other libraries, created through evolutionary computation and neural architecture search.
- **Simple Scikit-Learn Style Interface**: Follows scikit-learn API conventions for easy integration into existing ML pipelines.

## Installation

```bash
pip install omega-ml
```

## Quick Start

```python
from omega import HybridKNNClassifier

# Load your dataset
X, y = load_dataset()

# Initialize and train the model
model = HybridKNNClassifier()
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)
```

## Featured Algorithms

- **HybridKNNClassifier**: An advanced k-nearest neighbors algorithm incorporating multi-level abstraction.
- **MultiLevelAbstractionKNN**: KNN variant with enhanced feature space transformation.
- **EntropyGuidedKNN**: KNN algorithm guided by information-theoretic principles.
- **BiasVarianceOptimizedKNNEnsemble**: Ensemble method balancing bias and variance trade-offs.

## Performance

Omega's algorithms have demonstrated superior performance on a variety of benchmark datasets:

| Classifier | Wine | Breast Cancer | Digits | Diabetes | Covertype | Abalone |
|------------|------|---------------|--------|----------|-----------|---------|
| KNeighborsClassifier | 0.750 | 0.667 | 0.667 | 0.667 | 0.667 | 0.611 |
| HybridKNNClassifier | 0.972 | 0.861 | 0.944 | 1.000 | 0.972 | 0.972 |
| MultiLevelAbstractionKNN | 0.972 | 0.944 | 0.972 | 0.972 | 0.944 | 0.944 |

# Model Generation

The pipeline is designed for large-scale automated exploration of model ideas while maintaining reproducibility, fairness, and strict error handling.

1. **Model Generation**
  - Uses generative AI to propose new ML model architectures
  - Iteratively fixes errors until models are executable or discarded

2. **Model Evaluation**
  - Benchmarks all generated models on multiple datasets
  - Produces a normalized aggregate score that weights datasets equally
  - Outputs both console tables and LaTeX-ready tables

## Repository Structure
```
omega/ 
├── src/
│ └── algorithm_generator/
│ ├── main.py # Model generation entry point
│ ├── evaluate.py # Evaluation + benchmarking
│ ├── metaomni/ # Generated models (Python package)
│ │ ├── init.py
│ │ ├── model_1.py
│ │ ├── model_2.py
│ │ └── ...
│ └── results/
│ └── benchmark_multirow.tex
```

## 1. Generating Models

### Step 1: Configure generation

Open: `src/algorithm_generator/metaprompt.py`
Edit the following (or equivalent) parameters:
- **RESEARCH_PRINCIPLES** (high-level inductive biases)
- **MODELS** (what types of models are generated)
- **NUM_IDEAS** (how many ideas we come up with/model)

### Step 2: Run the generator
From the **project root**:

```bash
cd omega
python src/algorithm_generator/main.py
```

**If successful**: new model .py files are created in: `src/algorithm_generator/metaomni/` and added to `src/algorithm_generator/metaomni/__init__.py`

**If unsuccessful**: If no models appear or no imports are added, this usually means:
- The generated model raised runtime errors
- The iterative fixing loop exhausted its retries
- The model was discarded as non-executable

Review the printed error messages during generation and adjust prompts and retry generation.

## 2. Evaluating Models
Once models are generated and importable, they can be benchmarked automatically.

### Step 1: Evaluation Setup 
Open: `src/algorithm_generator/evaluate.py`

By default, models are loaded from: `src/algorithm_generator/metaomni/` (if you saved models elsewhere, ensure the directory appears in `models_dir` in `main()`)

### Step 2: Adding Datasets
The evaluation suite is **classification-only** by default, since the default generation prompt produces classifiers.

We default to classifcaiton datasets: Iris, Wine, Breast Cancer, Digits

To add datasets:
- Add their names to `dataset_names`
- Implement loading logic in: `BenchmarkSuite._load_datasets()`
*If a dataset is not implemented there, evaluation will fail.*

### Step 3: Execution
Run `python src/algorithm_generator/evaluate.py`

This will 
- Load all valid models from `metaomni/`
- Evaluate each model on each dataset
- Print a table of results to the console
- Compute an aggregate performance score
- Save a LaTeX table to: `src/algorithm_generator/results/benchmark_multirow.tex` (to compile please include \usepackage{booktabs}, \usepackage{multirow})

*Note: Any extra output (e.g. epoch logs, training losses) is printed directly by model implementations and must be handled inside the model .py files. If you see lots of ERR in your table, the generated model isn't able to handle the dataset properly. Set `logging` to true to see the exceptions/errors being thrown.*

#### Example output:
| Model                        | Iris  | Wine  | Breast Cancer | Digits | Aggregate |
|------------------------------|-------|-------|----------------|--------|-----------|
| CompressionDrivenLearner     | 1.0000 | 1.0000 | 0.9825 | 0.9722 | **0.978** |
| CompressionAwareLoss         | 1.0000 | 1.0000 | 0.9737 | 0.9778 | 0.958 |
| AdaptiveComplexityNet        | ERR | 0.9722 | 0.9825 | 0.9694 | 0.704 |
| DirectionalEnsembleTrees     | 1.0000 | 0.9722 | 0.8860 | 0.5972 | 0.617 |
| MultiResolutionPathwayFusion | 1.0000 | ERR | ERR | 0.9611 | 0.495 |
| HybridNeuronModel            | 0.6333 | 0.6944 | 0.9912 | 0.1139 | 0.250 |
| SimilarityAttention          | ERR | ERR | ERR | ERR | 0.000 |
| MultiLevelAbstractionNet     | ERR | ERR | ERR | ERR | 0.000 |




### Note on Aggregate Performance
Raw Accuracy or R² values are not directly comparable across datasets.
We compute a Relative Aggregate (RelAgg) score that weights each dataset equally.

$$n_{m,d} = \frac{s_{m,d} - \min_d}{\max_d - \min_d} \text{ where } (s_{m,d} \text{ is model accuracy on dataset d})$$
$$\text{RelAgg}_m = \frac{1}{|D|} \sum_{d \in D} n_{m,d}$$

#### Notes:
- If a model fails on a dataset, its contribution for that dataset is 0
- If all models achieve identical performance on a dataset, normalized scores default to 1

#### Interpretation: 
- $\text{RelAgg}\in [0, 1]$
- Measures how consistently close a model is to the best-performing model across datasets, 1 being the best and 0 being the worst

## Citation

If you use Omega in your research, please cite:

```
@article{nixon2024omega,
  title={Automating the Generation of Novel Machine Learning Algorithms},
  author={Nixon, Jeremy},
  journal={arXiv preprint arXiv:2404.00001},
  year={2024}
}
```

## License

Omega is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback:
- Issue Tracker: https://github.com/omniscience-research/omega/issues
- Email: jeremy@omniscience.tech

Let's push the boundaries of machine learning together with Omega!