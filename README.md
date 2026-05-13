# Explainable Autonomous Driving: Large Language Models as Semantic Validators for Autonomous Vehicles

The objective of this Master’s thesis is to investigate if open-weights Large Language Models (specifically Llama 3.1) can be effectively applied to autonomous driving decision making by interpreting structured environment and sensor data from the CARLA driving simulator. As foundational Large Language Models essentially struggle with strict structural formatting and spatial reasoning, the research specifically evaluates how efficient Parameter-Efficient Fine-Tuning can be in closing the gap between natural language comprehension and strict programmatic low-level vehicle control. The overall objective is to explore how LLMs can act as explainable, reasoning based validators, not to replace machine learning models, but rather improve them by validating their decisions with LLMs.

---

## Project Structure

llm_sensor_fusion/\
├── carla_setup/ # Link to CARLA simulator folder in shared storage\
├── data/ # Collected routes data and images for manual debugging, and curated data\
├── prompts/ # Prompts dataset generated from collected and curateddata\
├── results/ # Results from evaluation of the LLMs\
├── results/analysis # Analyzing results from oneshot baseline model, one-shot with semantic mapping and fine-tuned model againste the ground truth baseline \
├── scripts/ # All the scripts use for the pipeline, setup, collection, curation , formatting, utility scripts, evaluations scrips...\
├── README.md # This file





## System Architecture

The architecture of this research consists of eight main modules. An end-to-end pipeline where each module is responsible for distinct parts of the process, starting with data collection to training a fine-tuned Large Language Model and then evaluating it against the baseline CARLA expert. The data flow is structured as follows:


<img width="3070" height="1818" alt="pipeline_diagram-chapter3" src="https://github.com/user-attachments/assets/9706d4ba-51be-4af0-a4dc-c6955d1dfab5" />



1. **Data Collection. (log\_ticks)** The process starts with the log_ticks. In short, a script that runs the CARLA simulator with variable parameters, such as obstacles and towns, and on routes provided from the CARLA Leaderboard datasets in the form of waypoints. It collects necessary data from every tenth frame of the simulation.


2. **Data Curation. (data\_curation)** A filtration script for the raw data collected from the simulation by log\_ticks. It applies a strict set of programmatic rules responsible for cleaning up and ensuring as much poisoned data, like deadlocks and glitches,  are removed. 


3. **Prompt Generation. (jsonl\_to\_prompt\_dataset and prompt\_formatter)** A dual-script formatting mechanism responsible for the generation of three role based prompts using the curated low-level data, and given specific rules for each type of data. This mechanism converts the spatial physics from the simulator to the semantic input required by the Large Language Model.


4. **LLM Training Script. (training\_engine)** The core machine learning module. The script contains the parameters to perform Supervised Fine-Tuning (SFT) and Low-Rank Adaptation (LoRA) with the generated prompts, and inject the dataset into an open-weights Ollama based Large Language Model using the Unsloth framework.


5. **Trained LLM Evaluator. (trained\_evaluator)** An automated evaluator script that feeds an unseen test dataset of route prompts on the trained model. It then logs the textual predictions and semantic reasoning in csv format alongside the ground-truth from the BehaviorAgent in CARLA. 


6. **Baseline One-shot LLM Evaluator. (oneshot\_evaluator)** A second evaluator script that feeds an unseen test dataset of route prompts on a base, untrained Llama model using a set of  constructed one-shot prompts. It then logs the results in csv format along with the ground-truth from the BehaviorAgent in CARLA.


7. **Analysis and Result Visualisation. (result\_classifier)** The final analytical component. A script that creates a visual explanation of the data from the evaluations using graphs. It also mathematically compares the outputs from both evaluators against the CARLA ground truth and then represents the results with classification reports.


Along with all these components there is also a data\_harvester script. A helper script meant to run multiple instances of log\_ticks after each other on a list of routes to ensure we have a rich and varied collection of data. 




