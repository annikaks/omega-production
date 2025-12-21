import concurrent.futures
import importlib
import os
import re
import traceback
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import anthropic
from tqdm import tqdm
from metaprompt import GENERATION_DIRECTORY_PATH, IMPORT_STRUCTURE_PREFIX
import metaomni
import threading
import time
import anthropic
from anthropic import (
    InternalServerError, 
    APIConnectionError, 
    RateLimitError, 
    APIStatusError,
)
X, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

class AlgoGen:

    def __init__(self, anthropic_client: anthropic.Anthropic, log_file: str = "experiment_log.csv"):
        self.anthropic_client = anthropic_client
        self.file_lock = threading.Lock()
        self.log_file = log_file
    
    def _get_metaomni_path(self, filename=None):
        """Get the path to metaomni directory, optionally with a filename."""
        current_dir = os.path.dirname(__file__)
        metaomni_dir = os.path.join(current_dir, 'metaomni')
        if filename:
            print(f"get_metaomni_path {os.path.join(metaomni_dir, filename)}")
            return os.path.join(metaomni_dir, filename)
        print(f"get_metaomni_path {metaomni_dir}")
        return metaomni_dir

    def gen(self, prompt: str) -> str:
        max_retries = 8 
        for attempt in range(max_retries):
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4000,
                    temperature=0,
                    system="You are a world-class research engineer.",
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                )
                return message.content[0].text
            # claude overload handle
            except (InternalServerError, APIConnectionError, RateLimitError, APIStatusError) as e:
                if attempt == max_retries - 1:
                    print(f"Final attempt failed. Error: {e}")
                    raise e
                
                wait_time = (2 ** attempt) + 2 
                print(f"Anthropic Overloaded (Attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)

    def extract_code_snippets(self, text: str) -> str:
        pattern = r'```(?:python)?\n(.*?)```'
        snippets = re.findall(pattern, text, re.DOTALL)
        return [snippet.strip() for snippet in snippets]

    def save_first_snippet(self, snippets, filename: str):
        if snippets:
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            with open(filename, 'w') as file:
                file.write(snippets[0])
            print(f"First code snippet saved to {filename}")
            return True
        else:
            print("No code snippets found")
            return False

    def extract_name(self, text: str) -> str:
        pattern = r'<name>(.*?)</name>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None
        
    def log_experiment(self, model_idea, class_name, status, history):
        import csv
        log_file = self.log_file
        
        with self.file_lock:
            file_exists = os.path.isfile(log_file)
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Model", "Class", "Status", "Retries", "Errors"])
                error_summary = " | ".join([f"{h['attempt']}:{h['error']}" for h in history])
                writer.writerow([model_idea, class_name, status, len(history), error_summary])

    def execute(self, filename, class_name, model, count=1, history=None):
        if history is None:
            history = []
        filepath = os.path.join(GENERATION_DIRECTORY_PATH, filename)
        if count > 2:
            self.log_experiment(model, class_name, "FAILED", history)
            try:
                os.remove(filepath)
                return False
            except:
                pass
            return False
        
        module_name = filename.split(".py")[0]

        EXECUTION_STRINGS = f"""
m = importlib.import_module("{IMPORT_STRUCTURE_PREFIX}{module_name}")
importlib.reload(m)
print("Module:", m)
print("Has class:", hasattr(m, "{class_name}"))

Cls = getattr(m, "{class_name}")
ml_model = Cls()

ml_model.fit(x_train, y_train)
preds = ml_model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print("{class_name}", accuracy)
        """
        
        try:
            exec_globals = {
                "importlib": importlib,
                "metaomni": metaomni,
                "accuracy_score": accuracy_score,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
            }

            exec(EXECUTION_STRINGS, exec_globals)
            self.log_experiment(model, class_name, "SUCCESS", history)
            return True
        except Exception as e:
            error_message = traceback.format_exc()
            error_type = type(e).__name__
            print("Hit error: ", error_message)
            history.append({"attempt": count, "error": error_type, "message": error_message})
            filepath = os.path.join(GENERATION_DIRECTORY_PATH, filename)
            prompt = f"""
            Existing code:
            {open(filepath, 'r').read()}
        
            Error message on original execution:
            {e}
        
            Full traceback:
            {error_message}
        
            Given the original code and this error, rewrite a {model} classifier in the style of SciKit learn, with a {class_name} class that implements the methods fit(self, X_train, y_train) and predict(self, X_test)"""
            implementation = self.gen(prompt)
            
            snippets = self.extract_code_snippets(implementation)
            self.save_first_snippet(snippets, filepath)
            # Recursively try to fix - return the result
            return self.execute(filename, class_name, model, count+1)

    def add_import_to_init(self, init_file_path, import_string):
        with self.file_lock: # parallel imports
            with open(init_file_path, 'r') as file:
                content = file.read()
            if import_string not in content:
                with open(init_file_path, 'a') as file:
                    file.write('\n' + import_string)

    def remove_import_from_init(self, init_file_path, import_string):
        with self.file_lock: # parallel removals
            if not os.path.exists(init_file_path): return
            with open(init_file_path, 'r') as file:
                lines = file.readlines()
            new_lines = [line for line in lines if line.strip() != import_string.strip()]
            with open(init_file_path, 'w') as file:
                file.writelines(new_lines)

    def genML(self, model: str):
        metaomni_dir = GENERATION_DIRECTORY_PATH
        os.makedirs(metaomni_dir, exist_ok=True)
        
        # SPEED OPTIMIZATION: Combine Naming and Coding into 1 prompt
        mega_prompt = f"""
        Design a {model} classifier in the style of SciKit learn.
        
        1. Provide a succinct pythonic class name between <class_name></class_name> tags.
        2. Provide a succinct pythonic filename (ending in .py) between <file_name></file_name> tags.
        3. Provide the complete implementation in a single markdown python code block.
        
        The class must implement fit(self, X_train, y_train) and predict(self, X_test).
        Only return the tags and the code block.
        """
        response = self.gen(mega_prompt)
        
        # Robust Extraction
        class_name = re.search(r'<class_name>(.*?)</class_name>', response, re.DOTALL).group(1).strip()
        filename = re.search(r'<file_name>(.*?)</file_name>', response, re.DOTALL).group(1).strip()
        snippets = self.extract_code_snippets(response)
        
        filepath = os.path.join(GENERATION_DIRECTORY_PATH, filename)
        if(IMPORT_STRUCTURE_PREFIX == ""): 
            Warning("IMPORT_STRUCTURE_PREFIX is empty, imports may fail.")
        import_string = f"from {IMPORT_STRUCTURE_PREFIX}{filename.split('.py')[0]} import *"
        
        if self.save_first_snippet(snippets, filepath):
            init_file_path = os.path.join(GENERATION_DIRECTORY_PATH, '__init__.py')

            self.add_import_to_init(init_file_path, import_string)
            
            if self.execute(filename, class_name, model, count=1):                
                return (filename, class_name, model)
            else:
                self.remove_import_from_init(init_file_path, import_string)
        return None

    def parallel_genML(self, prompt_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            return list(tqdm(executor.map(self.genML, prompt_list), total=len(prompt_list)))

# NOTE (V.S) : Not sure what to do with this
# The directory has been generated already which is why the call is commented
# out below, but should this functionality be call if directory does not exist?
def generate_init_file(directory):
    # Get all Python files in the directory
    python_files = [f for f in os.listdir(directory) if f.endswith('.py') and f != '__init__.py']
    
    # Generate import statements
    import_statements = []
    for file in python_files:
        module_name = file[:-3]  # Remove .py extension
        
        # Check if the module can be imported
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(directory, file))
        if spec is not None:
            import_statements.append(f"from {module_name} import *")
    
    # Write the __init__.py file
    init_path = os.path.join(directory, '__init__.py')
    with open(init_path, 'w') as init_file:
        init_file.write('\n'.join(import_statements))
    
    print(f"__init__.py file has been generated in {directory}")
# generate_init_file('metaomni')
