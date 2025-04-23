# SpecFix

This repository provides a Python-based tool for **Specification Fix**(SpecFix) using language model APIs (
e.g., OpenAI, Qwen, Deepseek, etc). It contains an end-to-end workflow for:

1. Generating multiple candidate programs for a given requirement.
2. Testing these programs in parallel to identify clusters of functionally equivalent implementations.
3. Automatically refining requirements based on clarifications and test outputs to guide subsequent code generation, in
   a loop until convergence or until a maximum number of iterations is reached.
4. Tracking and reporting on accuracy metrics.

This repository also contains experiments which use this framework to draw interesting conclusions.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
- [Prompt](#prompt)

## Dataset

We use two datasets in our experiments: Humaneval+ and MBPP+. You can find all necessary datasets and their variants (WOE, Pilot) in the `dataset` folder. More details can be found in the readme there. 

## Installation

1. Clone the repository (or copy the script) to your local environment:
   ```bash
   git clone https://github.com/robbiebmorris/FYP-SpecFix.git
   cd SpecFix
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have the following libraries installed (they might already be in the `requirements.txt`):
   - `openai`
   - `jsonlines`
   - `evalplus`

3. Setup .config file:

   Create a new file called .config in the root directory. Put your api keys in this file like so:
   ```
   [API_KEY]
   open_ai_key = YOUR_OPEN_AI_KEY (For GPT)
   huoshan_key = YOUR_HUOSHAN_KEY (For deepseek)
   aliyun_key = YOUR_ALIYUN_KEY (For qwen)
   ```

## Usage

After installing the requirements, experiment scripts can be run with the appropriate command-line arguments:

```bash
python main.py --dataset <dataset_name> \
               --model <model_name> \
               --program_number <N> \
               --without_example
```

Note that if you are using any IDE besides PyCharm, you will encounter issues using the SpecFix Module. To resolve these issues, simply move the specfix folder into the directory of the python script you are trying to run.

### Command-Line Arguments

- **`--dataset`** (str): Name of the dataset to load (e.g., `humaneval`, `mbpp`).
- **`--model`** (str): The model name to use (e.g., `gpt-3.5-turbo`, `qwen2.5-coder-7b-instruct`, etc.).
- **`--program_number`** (int): How many candidate programs to generate per iteration.
- **`--without_example`**: Switches the dataset to the without example variant.

### Prompt

All prompts used in the experiments are stored in the `prompts.py`, including prompt for code generation, test
generation, and requirement refinement, etc.
