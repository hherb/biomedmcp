import ollama
from dataclasses import dataclass

DEFAULT_MODEL = "phi4:latest"
DEFAULT_MODEL_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "num_ctx": 4096
}

@dataclass
class ContextItem:
    """Class to represent a context item."""
    id: str
    source: str
    content: str
    summary : str=None
    metadata: dict = None
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.summary is None:
            self.summary = self.content[:50]


class AgentHelper:
    """Class to assist with agent operations."""
    
    def __init__(self, model=DEFAULT_MODEL, params=DEFAULT_MODEL_PARAMS):
        self.model=model
        self.model_params=params
        self.fullresponse = None

    def answer(self, task):
        """Answer a task using the model."""
        response = ollama.generate(
            self.model,
            task,
            options=self.model_params
        )
        return response
    
    def timed_answer(self, task):
        """Answer a task with timing."""
        response = ollama.generate(
            self.model,
            task,
            options=self.model_params
        )
        return response
    
 
    
class Summarizer(AgentHelper):
    """Class to summarize content."""
    
    def __init__(self, model=DEFAULT_MODEL, params=DEFAULT_MODEL_PARAMS):
        super().__init__(model, params)
    
    def answer(self, task, max_sentences=3):
        """Summarize the provided content."""
        task = f"""Extract the core message of the text and summarize it in {max_sentences} sentences.
Use the most concise form without introduction or comments. Be brief! 
Consider every word whether it is necessary or not without missing any key concepts or information
Text to summarize:
<content> {task} </content>"""
        return super().answer(task)
    

if __name__== "__main__":
    # Example usage
    long_text="""
    AI is now a big part of software development because of tools like GitHub Copilot offering auto-complete features for writing code. 
    More recently, this has extended beyond modifying existing code and creating short snippets, towards generating whole applications 
    from scratch. In this article, we explore Cursor for AI-powered code development and use it to generate a new open-source user 
    interface for the Stanford Co-STORM AI Research algorithm. We show how it is now possible to develop code conversationally for 
    rapid prototypes using AI in a fraction of the time of traditional methods — the new Co-STORM user interface app for this article 
    was created in under an hour— and present some suggestions on how to get the best results with these new workflows.
    """
    # Dictionary to store performance metrics for each model
    model_performance = {}
    
    for model in ["qwen2.5:3b",
                  "qwen2.5:3b-instruct-q8_0",
                  "nemotron-mini:4b-instruct-q8_0", 
                  "gemma3:4b", 
                  "gemma3:4b-it-q8_0",
                  "granite3.2:8b-instruct-q8_0", 
                  "smollm2:1.7b-instruct-q8_0", 
                  "reader-lm:1.5b-fp16", 
                  "phi4:latest",
                  "hermes3:8b-llama3.1-q8_0"]:
        print(f"Using model {model}")
        agent = Summarizer(model=model)
        nanosec_to_msec = 1_000_000  # Correct conversion factor for ns to ms
        
        # Lists to store performance metrics across iterations
        prompt_tps_values = []
        eval_tps_values = []
        
        for i in range(3):
            print(f"Running {i+1}th iteration")
            print("-" * 80)
            response = agent.answer(task=long_text, max_sentences=1)
            prompt_eval_time = response.get("prompt_eval_duration", 0)
            eval_time = response.get("eval_duration", 0)
            prompt_tokens = response.get("prompt_eval_count", 0)
            eval_tokens = response.get("eval_count", 0)
            
            # Calculate metrics with error handling to avoid division by zero
            prompt_ms = prompt_eval_time / nanosec_to_msec if prompt_eval_time > 0 else 0
            prompt_tps = (prompt_tokens / (prompt_eval_time / 1e9)) if prompt_eval_time > 0 else 0
            eval_ms = eval_time / nanosec_to_msec if eval_time > 0 else 0
            eval_tps = (eval_tokens / (eval_time / 1e9)) if eval_time > 0 else 0
            
            # Store the TPS values
            if prompt_tps > 0:
                prompt_tps_values.append(prompt_tps)
            if eval_tps > 0:
                eval_tps_values.append(eval_tps)
            
            print(f"Prompt evaluation: {prompt_ms:.2f}ms, {prompt_tokens} tokens, {prompt_tps:.2f} tokens/sec")
            print(f"Generation: {eval_ms:.2f}ms, {eval_tokens} tokens, {eval_tps:.2f} tokens/sec")
            print(f"Response: {response['response']}")
        
        # Calculate average of the two best runs if available
        prompt_avg_tps = 0
        if prompt_tps_values:
            prompt_tps_values.sort(reverse=True)
            prompt_avg_tps = sum(prompt_tps_values[:min(2, len(prompt_tps_values))]) / min(2, len(prompt_tps_values))
        
        eval_avg_tps = 0
        if eval_tps_values:
            eval_tps_values.sort(reverse=True)
            eval_avg_tps = sum(eval_tps_values[:min(2, len(eval_tps_values))]) / min(2, len(eval_tps_values))
        
        # Store the average performance
        model_performance[model] = {
            'prompt_avg_tps': prompt_avg_tps,
            'eval_avg_tps': eval_avg_tps,
            'combined_avg': (prompt_avg_tps + eval_avg_tps) / 2
        }
            
        print("-" * 80)
    
    # Display performance table sorted by combined average
    print("\nPerformance Summary (Tokens per Second)")
    print("-" * 80)
    print(f"{'Model':<30} | {'Prompt TPS':>12} | {'Eval TPS':>12} | {'Combined':>12}")
    print("-" * 80)
    
    # Sort models by combined average performance (descending)
    sorted_models = sorted(model_performance.items(), key=lambda x: x[1]['combined_avg'], reverse=True)
    
    for model, perf in sorted_models:
        print(f"{model:<30} | {perf['prompt_avg_tps']:>12.2f} | {perf['eval_avg_tps']:>12.2f} | {perf['combined_avg']:>12.2f}")
    
    print("-" * 80)
