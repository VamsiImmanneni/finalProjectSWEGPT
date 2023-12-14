
# Automated Assessment of Generative AI's Coding Abilities

This repository contains an automated assessment designed to measure the coding abilities of Generative AI, focusing on the transformation of unvectorized code to vectorized code.

## Background

Generative AI models, such as ChatGPT and Claude, have shown prowess in various coding tasks. While many platforms like LeetCode assess the algorithmic and problem-solving abilities of human coders, few are designed to benchmark the abilities of AI models, especially in the context of code optimization.

## Overview

We have produced 100 Python coding problems. Each problem presents a function performing matrix operations in an unvectorized manner, using standard loops. The challenge is to convert these functions into their vectorized counterparts, preferably using libraries like NumPy, to enhance performance.

### What's Unique?

This assessment does not just evaluate the AI's ability to solve coding problems but also its skill in optimizing them. In the real world, having performant code is as critical as having working code. By focusing on vectorization, we're assessing the AI's knowledge of advanced Python libraries and its ability to write efficient code.

### Tags and Keywords

Each problem is tagged with specific keywords, making it easier to analyze the results. For instance, if an AI struggles with problems tagged 'multiplication', we gain insight into specific areas that might need improvement.

### Ground Truth

For each problem, the unvectorized function serves as the ground truth. The vectorized solutions can be compared against these for functional equivalence. While the exact method of vectorization might vary, the output should remain consistent.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy library (for vectorized solutions)

### Running the Benchmark

1. Clone this repository.
2. Navigate to the repository directory.
3. Run the `matrix_problems.py` file.
4. Compare the AI's vectorized solutions against the provided unvectorized functions.

## Novelty and Assessment

While typical coding challenges focus on producing a solution from scratch, our benchmark assesses the ability to optimize existing solutions. This is more reflective of real-world tasks, where developers often refactor and optimize existing codebases.

## Collaboration

This project was completed individually, but ideas were brainstormed and discussed with Lining Yu on 8/30.

## License

This project is open-sourced under the MIT License.

## Feedback

Contributions, issues, and feature requests are welcome!
