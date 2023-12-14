import json
import openai
import os
from pathlib import Path

# Fetch the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set!")

# Directory containing the problem JSON files
problem_dir = Path("problemsJSON")

# Directory to save the generated solutions
solutions_dir = Path("solutions")
solutions_dir.mkdir(exist_ok=True)

# The engine to use for generating completions
engine = "gpt-3.5-turbo-instruct-0914"

# Iterate over each problem JSON file
for problem_file in problem_dir.iterdir():
    # Load the problem data
    with open(problem_file, 'r') as f:
        problem_data = json.load(f)

    # Extract the prompt from the problem data
    prompt = problem_data['prompts'][0]['prompt']

    # Use OpenAI's API to generate a solution for the problem
    response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=150)

    # Extract the generated code from the response
    generated_code = response.choices[0].text.strip()

    # Save the generated solution to a separate file
    solution_file = solutions_dir / (problem_file.stem + "_solution.txt")
    with open(solution_file, 'w') as f:
        f.write(generated_code)

    print(f"Generated solution for {problem_file.name} and saved to {solution_file.name}")

print("All solutions generated successfully!")
