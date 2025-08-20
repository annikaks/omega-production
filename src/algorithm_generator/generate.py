import concurrent.futures
import importlib
import os
import re
import traceback

import anthropic
from tqdm import tqdm

import metaomni

class AlgoGen:

    def __init__(self, anthropic_client: anthropic.Anthropic):
        self.anthropic_client = anthropic_client

    def gen(self, prompt: str) -> str:
        message = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0,
            system="You are a world-class research engineer.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return message.content[0].text

    def extract_code_snippets(self, text: str) -> str:
        pattern = r'```(?:python)?\n(.*?)```'
        snippets = re.findall(pattern, text, re.DOTALL)
        return [snippet.strip() for snippet in snippets]

    def save_first_snippet(self, snippets, filename: str):
        if snippets:
            with open(filename, 'w') as file:
                file.write(snippets[0])
            print(f"First code snippet saved to {filename}")
        else:
            print("No code snippets found")

    def extract_name(self, text: str) -> str:
        pattern = r'<name>(.*?)</name>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def execute(self, filename, class_name, model, count=1):
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
            implementation = self.gen(prompt)
            
            snippets = self.extract_code_snippets(implementation)
            self.save_first_snippet(snippets, 'metaomni/' + filename)
            self.execute(filename, class_name, model, count+1)

    def add_import_to_init(self, init_file_path, import_string):
        # Read the current contents of the file
        with open(init_file_path, 'r') as file:
            content = file.read()

        # Check if the import statement already exists
        if import_string not in content:
            # If it doesn't exist, add it to the end of the file
            with open(init_file_path, 'a') as file:
                file.write('\n' + import_string)
            print(f"Added {import_string} to {init_file_path}")
        else:
            print(f"{import_string} already exists in {init_file_path}")

    def genML(self, model: str):
        class_name_prompt = f"""Write a succinct pythonic class name for a model with name {model}, putting the name between the XML tags <name>Insert Class Name of Model Here</name>"""
        class_name = self.extract_name(self.gen(class_name_prompt))
        
        filename_prompt = f"""Write a succinct pythonic file name for a model with name {model}, putting the file name between the XML tags <name>Insert Class Name of File Here</name>"""
        filename = self.extract_name(self.gen(filename_prompt))
        
        import_string = f"from metaomni.{filename.split('.py')[0]} import *"
        
        prompt = f"""Write a {model} classifier in the style of SciKit learn, with a {class_name} class that implements the methods fit(self, X_train, y_train) and predict(self, X_test)"""
        implementation = self.gen(prompt)
        
        snippets = self.extract_code_snippets(implementation)
        self.save_first_snippet(snippets, 'metaomni/' + filename)
        self.add_import_to_init('metaomni/__init__.py', import_string)
        #NOTE: Move this to the execute file
        # self.execute(filename, class_name, model, count=1)
        return (filename, class_name)

    def parallel_genML(self, prompt_list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
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
