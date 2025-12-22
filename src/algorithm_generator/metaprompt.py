RESEARCH_IDEAS = """Adaptive Compression KNN: Instead of using raw feature spaces, dynamically compress the data based on local density. In sparse regions, use less compression to maintain detail, while in dense regions, apply more compression to reduce noise. This approach balances simplicity and complexity adaptively.
Bias-Variance Optimized KNN: Implement an ensemble of KNN classifiers with different k values. Use a meta-learner to dynamically weight their contributions based on estimated bias and variance for each input, optimizing the bias-variance trade-off for each prediction.
Multi-level Abstraction KNN: Create a hierarchical KNN that operates at multiple levels of abstraction. Start with high-level features for initial neighborhood selection, then progressively refine using more detailed features. This could capture both coarse and fine-grained structures in the data.
Compression-based Similarity KNN: Instead of using traditional distance metrics, define similarity based on the compressibility of the concatenated feature vectors of two instances. This could capture complex, non-linear relationships between features.
Directional KNN: Introduce directionality into the KNN algorithm by giving more weight to neighbors that are "in front of" the query point in the direction of the decision boundary. This could improve performance in datasets with complex decision boundaries.
Continuous-Discrete Hybrid KNN: Develop a KNN variant that seamlessly handles both continuous and discrete features. Use appropriate distance metrics for each type and combine them using a learned weighting scheme, allowing for better handling of mixed data types.
Fractal Dimension KNN: Estimate the local fractal dimension of the feature space around each query point. Use this to adaptively set k or to weight neighbors, potentially capturing complex local structures in the data.
Similarity-Dissimilarity Balanced KNN: Instead of just finding similar neighbors, also consider dissimilar points. Make decisions based on a balance of both similar and dissimilar instances, potentially improving robustness and decision boundary placement.
Entropy-Guided KNN: Use local entropy estimates to guide the KNN algorithm. In high-entropy (more random) regions, increase k to reduce noise influence. In low-entropy (more structured) regions, decrease k to capture fine-grained patterns.
Interaction-Aware KNN: Develop a KNN variant that explicitly models feature interactions. Use techniques like feature crossing or learned feature interactions to create a richer representation space for finding neighbors, potentially capturing complex patterns that individual features miss.""".split('\n')

# linear regression
# logistic regression
# perceptron
# k-nearest neighbors
# naive bayes
# decision trees
# support vector machines
# k-means clustering
# random forests
# adaboost
# gradient boosting machines
# xgboost
# principal component analysis (pca)
# independent component analysis (ica)
# gaussian mixture models
# one-class svm
# isolation forest
# local outlier factor

# more principles

# abstraction: representing complex data or processes at different levels of detail or generality.
# compression: reducing the dimensionality or complexity of data while preserving essential information.
# sparsity: utilizing or encouraging solutions with few non-zero elements.
# smoothing: reducing noise or irregularities in data or model behavior.
# discretization: converting continuous data or processes into discrete counterparts.
# continuization: making discrete data or processes continuous for easier manipulation.
# similarity and distance: measuring how alike or different entities are in various spaces.
# hierarchy: organizing information or computations in levels of increasing complexity or abstraction.
# composition: combining simpler elements to create more complex structures or behaviors.
# decomposition: breaking complex problems or data into simpler, manageable parts.
# iteration: repeatedly applying processes to refine results or converge on solutions.
# randomization: introducing controlled randomness to improve robustness or exploration.
# regularization: constraining models to prevent overfitting and improve generalization.
# adaptation: modifying behavior based on feedback or changing conditions.
# ensemble: combining multiple models or perspectives to improve overall performance.
# transfer: applying knowledge from one domain or task to another.
# attention: focusing computational resources on the most relevant parts of the input.
# memory: storing and recalling information over time to inform current decisions.
# generalization: extracting patterns that apply beyond the specific instances seen during training.
# discretization: converting continuous data or processes into discrete counterparts.
# inference: drawing conclusions or making predictions based on incomplete information.
# representation: finding effective ways to encode information for specific tasks.
# optimization: finding the best solution within constraints.
# factorization: breaking down complex structures into simpler, constituent parts.
# augmentation: expanding available data or features to improve learning or robustness.
# adversarial thinking: improving models by considering potential oppositions or weaknesses.
# bootstrapping: using the model's own predictions to improve itself or to generate new data.
# marginalization: focusing on overall patterns by averaging out less important details.
# disentanglement: separating mixed or confounded factors in data or representations.
# locality: exploiting the idea that nearby points in input space should have similar outputs.

# abstraction: representing complex ideas or data in a simplified form.
# compression: reducing the size or dimensionality of data while preserving essential information.
# sparsity: utilizing or encouraging solutions with few non-zero elements.
# smoothing: reducing noise or irregularities in data or model behavior.
# discretization: converting continuous data or processes into distinct categories or steps.
# continuization: making discrete data or processes continuous for easier manipulation.
# similarity and distance: quantifying how alike or different entities are in various spaces.
# hierarchy: organizing information or computations in levels of increasing complexity or abstraction.
# composition: combining simpler elements to create more complex structures or behaviors.
# decomposition: breaking complex problems or data into simpler, manageable parts.
# iteration: repeatedly applying processes to refine results or converge on solutions.
# randomization: introducing controlled randomness to improve robustness or exploration.
# regularization: constraining models to prevent overfitting and improve generalization.
# adaptation: modifying behavior based on feedback or changing conditions.
# ensemble: combining multiple models or perspectives to improve overall performance.
# transfer: applying knowledge from one domain or task to another.
# attention: focusing computational resources on the most relevant parts of the input.
# memory: storing and recalling information over time to inform current decisions.
# generalization: extracting patterns that apply beyond the specific instances seen during training.
# inference: drawing conclusions or making predictions based on incomplete information.
# representation: finding effective ways to encode information for specific tasks.
# optimization: finding the best solution within given constraints.
# factorization: breaking down complex structures into simpler, constituent parts.
# augmentation: expanding available data or features to improve learning or robustness.
# adversarial thinking: improving models by considering potential oppositions or weaknesses.
# bootstrapping: using the model's own predictions to improve itself or to generate new data.
# marginalization: focusing on overall patterns by averaging out less important details.
# disentanglement: separating mixed or confounded factors in data or representations.
# locality: exploiting the idea that nearby points in input space should have similar outputs.
# dimensionality: considering the number of features or parameters in relation to the amount of data.
# invariance: maintaining consistent outputs despite certain transformations of the input.
# equivariance: transforming outputs in predictable ways when inputs are transformed.
# causality: identifying and leveraging cause-effect relationships in data.
# uncertainty: quantifying and propagating the degree of confidence in predictions or estimates.
# bias-variance tradeoff: balancing the model's ability to fit training data versus its ability to generalize.
# interpolation: estimating values between known data points.
# extrapolation: extending predictions beyond the range of known data.
# kernelization: implicitly mapping data to higher-dimensional spaces for easier separation.
# boosting: iteratively improving weak learners to create a strong learner.
# bagging: reducing variance by training models on random subsets of the data.
# stacking: layering multiple models to capture different aspects of the data.
# incrementalism: updating models or knowledge gradually as new information becomes available.
# batch processing: processing data in groups rather than individually or continuously.
# active sampling: selectively choosing the most informative data points for labeling or analysis.
# semi-supervision: leveraging both labeled and unlabeled data for training.
# self-supervision: creating supervised tasks from unlabeled data.
# multi-task integration: improving generalization by learning multiple related tasks simultaneously.
# curriculum design: presenting training data in a meaningful order to improve learning.
# few-shot adaptation: generalizing from very few examples of new classes or tasks.
# zero-shot inference: recognizing or generating examples of classes not seen during training.
# continual accumulation: accumulating knowledge over time without catastrophic forgetting.
# meta-learning: learning how to learn, or improving the learning process itself.
# embedding: mapping discrete entities to continuous vector spaces.
# quantization: discretizing continuous values for efficiency or generalization.
# pruning: removing unnecessary components of a model to improve efficiency.
# distillation: transferring knowledge from complex models to simpler ones.
# flow: ensuring effective propagation of information through complex systems.
# normalization: adjusting and scaling features or activations to improve stability and performance.
# residual learning: learning differences or corrections rather than absolute mappings.
# gating: controlling information flow in neural networks or decision processes.
# contrastive analysis: learning by comparing similar and dissimilar examples.
# generation: creating new examples that resemble the training data.
# discrimination: focusing on boundaries between classes rather than full data distribution.
# robustness: maintaining performance under varying or adverse conditions.
# multimodal fusion: integrating information from multiple types of data or sensors.
# anomaly detection: identifying patterns that deviate significantly from the norm.
# reinforcement: learning through interaction with an environment.
# exploration vs. exploitation: balancing the need to gather new information with the use of existing knowledge.
# credit assignment: determining which actions or decisions led to particular outcomes.
# temporal difference: updating estimates based on other estimates in sequential decision problems.
# imitation: learning behavior by mimicking demonstrations or examples.
# inverse problem solving: inferring causes or inputs from observed effects or outputs.
# domain adaptation: adjusting models to perform well on related but distinct data distributions.
# conditional manipulation: treat aspects of the input differently.
# information bottleneck: compressing inputs while preserving task-relevant information.
# mutual information: measuring the mutual dependence between variables.
# counterfactual reasoning: considering hypothetical scenarios to improve decision-making.
# intervention: incorporating the effects of actions or changes in causal models.
# distribution shift handling: adapting to changes in data distribution between training and deployment.
# interpretability: making model decisions understandable to humans.
# saliency: identifying the most relevant features or inputs for a particular output or decision.
# architecture search: automatically discovering effective model structures.
# hyperparameter tuning: efficiently optimizing model parameters that are not learned during training.
# mixup: combining training examples to regularize the model and increase robustness.
# curriculum complexity: gradually increasing the difficulty of tasks or examples during training.
# Accumulation: Aggregating information or updates over time to improve stability or performance.
# Distributed Learning: Collaboratively learning from distributed data sources or models.
# Symbolic-Subsymbolic Integration: Combining rule-based systems with neural networks.
# Graph Representation: Modeling relationships and interactions between entities as graphs.
# Amortization: Using learned models to speed up repeated computations or inferences.
# Energy Minimization: Framing learning and inference as minimizing an energy function.
# Denoising: Removing noise or unwanted variations from data or representations.
# Latent Space Manipulation: Operating on compressed or abstract representations of data.
# Prototype Learning: Learning representative examples or templates for classes or concepts.
# Modularity: Organizing systems into independent, interchangeable components.
# Emergence: Observing complex behaviors arising from simple rules or interactions.
# Self-Organization: Allowing systems to spontaneously form structured patterns or behaviors.
# Homeostasis: Maintaining stable internal states despite external fluctuations.
# Synchronization: Coordinating the behavior of multiple components or agents.
# Oscillation: Utilizing or managing cyclic patterns in data or model behavior.

RESEARCH_PRINCIPLES = """Simplicity vs. complexity
Bias - Variance Decomposition
Abstraction - level of abstraction at which more or less structure, or different types of structure are present
Framed as Compression
Degree of Compression
Directionality
Discrete vs. Continuous
Abstraction - fine vs. coarse grain structure
Similarity, say, with a feature or set of features
Randomness, degree to which there is structure, compressibility of data
Dimensionality - Interactions between features vs. single feature structure""".split('\n')

MODELS = """Random Forests
Neural Network Classifier
Bagged Decision Trees
Decision Trees for Classificaiton
K-Nearest Neighbors
Perceptron Classifier
Naive Bayes
Logistic Regression
Support Vector Machine
Gradient Boosting Machine
Quadratic Discriminant Analysis
Linear Discriminant Analysis""".split('\n')

# TODO: Saving/Generation Paths ##
LOG_FILE = "/Users/annikaks/Desktop/omega/src/algorithm_generator/metaomni/tst.csv"
GENERATION_DIRECTORY_PATH = "/Users/annikaks/Desktop/omega/src/algorithm_generator/metaomni"
SUMMARIZE_IMMEDIATELY = False
IMPORT_STRUCTURE_PREFIX = "metaomni."
# ie for "from metaomni.{filename.split('.py')[0]} import *" > IMPORT_STRUCTURE_PREFIX = "metaomni."
EVALUATION_DIRECTORY_PATH = GENERATION_DIRECTORY_PATH
DESCRIPTION_DIRECTORY_PATH = ""

# Batch 1: Linear & Statistical
# NUM_IDEAS = 4 
# RESEARCH_PRINCIPLES = [
#     "Simplicity vs. complexity",
#     "Bias - Variance Decomposition",
#     "Linearity and Additivity",
#     "Closed-form solutions",
#     "Statistical independence"
# ]

# MODELS = [
#     "Logistic Regression",
#     "Linear Discriminant Analysis",
#     "Quadratic Discriminant Analysis",
#     "Naive Bayes",
#     "Perceptron Classifier",
#     "Passive Aggressive Classifier",
#     "Ridge Classifier",
#     "Stochastic Gradient Descent Classifier",
#     "Nearest Centroid",
#     "Bernoulli Naive Bayes",
#     "Multinomial Naive Bayes"
# ]

# # Batch 2: Non-Linear & Ensemble
# NUM_IDEAS = 4
# RESEARCH_PRINCIPLES = [
#     "Abstraction - fine vs. coarse grain structure",
#     "Randomness and Bagging",
#     "Boosting and Residuals",
#     "Non-parametric density estimation",
#     "Feature subspace projection",
#     "Dimensionality - Interactions between features"
# ]

# MODELS = [
#     "Random Forests",
#     "Gradient Boosting Machine",
#     "K-Nearest Neighbors",
#     "Decision Trees for Classification",
#     "Bagged Decision Trees",
#     "Extra Trees Classifier",
#     "AdaBoost",
#     "Kernel Support Vector Machine",
#     "Histogram-based Gradient Boosting",
#     "Radius Neighbors Classifier",
#     "Multi-layer Perceptron (MLP)"
# ]

# # Batch 3: Abstract & Frontier
# NUM_IDEAS = 4
# RESEARCH_PRINCIPLES = [
#     "Framed as Compression",
#     "Framed as Compression",
#     "Degree of Compression",
#     "Information Geometry",
#     "Bio-inspired Metaheuristics",
#     "Discrete vs. Continuous structure mapping",
#     "Kolmogorov Complexity as an Inductive Bias"
# ]

# MODELS = [
#     "Kolmogorov Complexity Classifier",
#     "Kolmogorov Complexity Classifier",
#     "Minimum Description Length (MDL) Learner",
#     "Fisher Information Metric Classifier",
#     "Cellular Automata-based Ensemble",
#     "Quantum-Inspired Probabilistic Learner",
#     "Fractal Dimension Subspace Classifier",
#     "Reservoir Computing Classifier",
#     "Self-Organizing Map Classifier",
#     "Evolutionary Strategy Ensemble",
#     "Hyperdimensional Computing Classifier",
#     "Topological Data Analysis (TDA) Classifier"
# ]