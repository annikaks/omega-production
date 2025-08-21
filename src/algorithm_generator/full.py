import anthropic

import re
import os
from sklearn.metrics import accuracy_score
import sys
import multiprocessing
from multiprocessing import Process
import concurrent.futures
from tqdm.notebook import tqdm
# import metaomni as mo

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_california_housing, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator
from typing import List, Tuple, Dict, Union

import traceback
import os
import importlib.util
import time

import importlib
sys.path.append('/Users/jeremynixon/Dropbox/python_new/Misc/omega_research/metaomni')
