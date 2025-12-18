import os
import time

import anthropic

from metaprompt import RESEARCH_PRINCIPLES, MODELS, NUM_IDEAS
from generate import AlgoGen

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

algo_gen = AlgoGen(anthropic.Anthropic(api_key=ANTHROPIC_API_KEY))
# benchmark_suite = BenchmarkSuite()

for m in MODELS[1:]:
    print(f"Generating ideas for model {m}...")
    ideas_raw = algo_gen.gen(f"""Principles of Machine Learning: {RESEARCH_PRINCIPLES}
    Inspired by these principles, come up with 10 very creative ideas for how to vary the {m} classifier for better performance.
    
    IMPORTANT: Format your response as exactly one idea per line, with each line starting with "IDEA: " followed by the idea description.
    Example format:
    IDEA: Use adaptive compression based on local density
    IDEA: Implement multi-level abstraction layers
    IDEA: Apply entropy-guided feature selection
    
    Do not include any other text, just the 10 IDEA lines.""")
    
    ideas_list = []
    for line in ideas_raw.split('\n'):
        line = line.strip()
        if line.startswith('IDEA: '):
            idea = line[6:].strip()
            if idea:
                ideas_list.append(idea)
    
    print(f"Extracted {len(ideas_list)} ideas")
    
    if not ideas_list:
        print(f"Warning: No ideas extracted for {m}")
        continue
    
    ideas_list = ideas_list[:NUM_IDEAS]    

    print(f"Generating implementations for {len(ideas_list)} ideas (limited to {NUM_IDEAS})...")
    generated_files_result = algo_gen.parallel_genML(ideas_list)
    print(f"Results: {generated_files_result}")
