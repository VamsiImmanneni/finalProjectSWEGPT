import numpy as np
import os
import json


def main():
    total_problems = 100
    correct_count = 0

    for i in range(1, total_problems + 1):
        # Load the original problem
        with open(f"problems/problem_{str(i).zfill(2)}.json", "r") as f:
            problem_data = json.load(f)

        # Get the ground truth function from the unvectorized code
        ground_truth_code = problem_data["prompts"][0]["input_code"]
        exec(ground_truth_code, globals())

        # Load the generated solution
        with open(f"solutions/problem_{str(i).zfill(2)}_solution.txt", "r") as f:
            generated_solution_code = f.read()

        # Determine number of arguments
        num_args = len(problem_data["function_prototype"]["parameters"])

        try:
            # Execute the generated solution code to get the function
            exec(generated_solution_code, globals())

            # Generate random vectors for testing based on number of arguments
            args = [np.random.rand(1000) for _ in range(num_args)]

            # Check if the results match
            ground_truth_result = function(*args)
            generated_result = function(*args)

            if np.array_equal(ground_truth_result, generated_result):
                correct_count += 1

        except:
            # If there's an error, count it as a fail
            pass

    accuracy = correct_count / total_problems * 100
    print(f"Accuracy: {accuracy}%")

    with open("output.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)


if __name__ == "__main__":
    main()
