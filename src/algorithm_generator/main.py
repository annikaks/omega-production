import os
import time

import anthropic

from metaprompt import RESEARCH_PRINCIPLES, MODELS
from generate import AlgoGen

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

algo_gen = AlgoGen(anthropic.Anthropic(api_key=ANTHROPIC_API_KEY))
# benchmark_suite = BenchmarkSuite()

for m in MODELS[1:]:
    ideas_raw = algo_gen.gen(f"""Principles of Machine Learning: {RESEARCH_PRINCIPLES}
    Inspired by these principles, come up with 10 very creative ideas for how to vary the {m} classifier for better performance.""")
    ideas_list = [i.split('. ')[1] for i in ideas_raw.split('\n') if len(i) > 0 and i[0].isnumeric()]
    with open(m + 'txt', 'w') as f:
        for line in ideas_list:
            f.write(f"{line}\n")

    generated_files_result = algo_gen.parallel_genML(ideas_list)

    
