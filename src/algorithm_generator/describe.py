import os
import anthropic
from metaprompt import DESCRIPTION_DIRECTORY_PATH

class ModelAnalyzer:
    def __init__(self, anthropic_client: anthropic.Anthropic):
        self.anthropic_client = anthropic_client

    def describe_code(self, code: str):
        if not code:
            return "Error: Empty code"
        try:
            return self.llm_api_call(code, batch=False)
        except Exception as e:
            return f"Error generating description ({e})"

    def analyze_repo(self, dir_path, output_file="model_summaries.txt"):
        all_files_content = []
        print(f"Analyzing repository at: {dir_path}")
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    file_path = os.path.join(root, file)
                    print(f"Reading file: {file_path}")
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Wrap in XML tags so Claude can separate them
                        all_files_content.append(f"<file name='{file}'>\n{content}\n</file>")

        batch_size = 20
        index = 0
        total_files = len(all_files_content)
        all_summaries = []

        while index < total_files:
            current_batch = all_files_content[index : min(index + batch_size, len(all_files_content))]
            bulk_code = "\n".join(current_batch)
            batch_summary = self.llm_api_call(bulk_code, batch=True)

            if batch_summary:
                all_summaries.append(batch_summary)
            
            index += batch_size

        if all_summaries:
            summary_text = "\n".join(all_summaries)
            logs_dir = os.path.join(DESCRIPTION_DIRECTORY_PATH, "evaluation_logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            with open(os.path.join(logs_dir, output_file), "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"Summaries written to {output_file}")
        else:
            print("Failed to generate any summaries.")
        
    
    def describe_single(self, dir_path, filename):
        """Generates a summary for a specific file using the exact filename provided by api.py."""
        file_path = os.path.join(dir_path, filename)
        
        if not os.path.exists(file_path):
            return f"Error: File not found"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
                # Use a small snippet for the log
                print(f"Generating description for {filename}...")

            return self.llm_api_call(code, batch=False)
        except Exception as e:
            return f"{filename} : Error generating description ({e})"
        
    def llm_api_call(self, code, batch) :
        prompt = (
                f"Analyze this synthesized machine learning algorithm: \n\n"
                f"CODE:\n{code}\n\n"
                f"Provide a 2-3 sentence technical summary of its mathematical intuition. "
                f"Return ONLY the format: <classname> : <description>\n"
            )
        if batch:
            prompt = (
                "You are a world-class ML research engineer. I have provided a collection of "
                "synthesized machine learning models below, each wrapped in <file> tags. "
                "\n\n"
                "### TASK:\n"
                "1. Read every <file> provided.\n"
                "2. For each file, identify the main class name.\n"
                "3. Write a 2-sentence technical description of the mathematical intuition.\n"
                "4. Return strictly in the format: <classname> : <description>\n"
                "\n"
                "### DATA:\n"
                f"{code}" 
            )

        try:
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4000,
                temperature=0,
                system="Always output the results in the requested <classname> : <description> format. Do not include conversational filler.",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
            
            summary_text = message.content[0].text
            
            return summary_text

        except Exception as e:
            print(f"Batch API Error: {e}")
            return None


if __name__ == "__main__":
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    anthopic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if(DESCRIPTION_DIRECTORY_PATH == ""):
        Warning("DESCRIPTION_DIRECTORY_PATH is not set.")
    
    analyzer = ModelAnalyzer(anthopic_client)
    analyzer.analyze_repo(DESCRIPTION_DIRECTORY_PATH, output_file="model_summaries.txt")
