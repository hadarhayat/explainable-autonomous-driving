import json
import csv
import requests
import re
import os

# default adress to run ollama locally
OLLAMA_URL = "http://localhost:11434/api/chat"
# name of the model i trained
MODEL = "carla_driver" 


def format_llm_response(response_text: str):
    """
    extractws the three different types of response from the models respons.
    """
    # error logging in case the response is not the expected format for each type of response
    thought = "ERROR: wrong format"
    high_level_action = "ERROR: wrong format"
    low_level_control= "ERROR: wrong format"

    try:
        # using regex to cut the text respons based on the exact keywords
        thought_response = re.search(r"Thought:\s*(.*?)\nHigh-Level Action:", response_text, re.DOTALL) # start looking at Thought: until High-Level Action:, ignoring any spaces
        high_level_action_response = re.search(r"High-Level Action:\s*(.*?)\nLow-Level Control:", response_text, re.DOTALL) #use dotall so regex doen't stop at paragraph end
        low_level_control_response = re.search(r"Low-Level Control:\s*(.*)", response_text, re.DOTALL)

        #if regex finds expected pattern, remove unneccessary space or lines and  map it to return variable  
        if thought_response: 
            thought = thought_response.group(1).strip()
        if high_level_action_response: 
            high_level_action = high_level_action_response.group(1).strip()
        if low_level_control_response: 
            low_level_control= low_level_control_response.group(1).strip()
            
    except Exception as e: #exception handling
        print(f"Error parsing response: {e}")

    return thought, high_level_action, low_level_control



def send_prompt_to_llm(system_prompt: str, user_prompt: str) -> str:
    """
    sends the prompt text to the local Ollama container and returns the llm's response.
    """
    payload = { #json format/structure used by Ollama api endpoint
        "model": MODEL, #the trained model to send the prompts to
        "messages": [  #role-based message to send to as the prompt 
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False, #forces model to generate output answeer before sending it back as once block of text 
        "options": {
            "temperature": 0.0 #controls randomness of the model, stops it from creating creative answers and instead give the same answer to the same question every time
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload) #post request with local ollama adress and the payload json 
        response.raise_for_status() # check for errors with connection to ollama
        return response.json()["message"]["content"]
    except Exception as e:  #exception handling
        print(f"Ollama connecton error: {e}")
        return ""



def run_evaluation(input_jsonl: str, output_csv: str):
    """
    loop through the test dataseton the llm and logs the results to a CSV file along with the CARLA agent responses.
    """
    print(f"Starting evaluation. Processing  carla logs...")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    #creating a new CSV file and writer writer
    with open(input_jsonl, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # setting up the headers for the cvs file columns
        writer.writerow(["Frame", "Light_Context", "Obstacle_Context", "Carla_Agent_Action", "LLM_Action", "Match", "LLM_Low-Level_Control", "LLM_Thought"])

        #loop stops if the number of lines exceed the number of frames
        for i, line in enumerate(infile):

            data = json.loads(line)
            messages = data.get("messages", [])

            # extract the prompts from the dataset devided in different roles 
            system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "") 
            user_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
            carla_agent_response = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            # extract the CARLA agents action, the expected expert answer
            carla_agent_action = "Unknown"
            action_response = re.search(r"High-Level Action:\s*(.*?)\nLow-Level Control:", carla_agent_response, re.DOTALL)
            #remove unneccessary space or lines and  map it to return variable
            if action_response:
                carla_agent_action = action_response.group(1).strip()

            #context for taxonomy (Light and Obstacle)
            light_match = re.search(r"There is a (\w+) traffic light", user_prompt, re.IGNORECASE)
            traffic_light = light_match.group(1) if light_match else "None"
            
            obs_match = re.search(r"A (\w+) is approaching", user_prompt, re.IGNORECASE)
            obstacle = obs_match.group(1) if obs_match else "None"

            # ask the LLM the same question without access to the assitant role (correct response)
            print(f"Processing Prompt {i+1}...")
            llm_raw_response = send_prompt_to_llm(system_prompt, user_prompt)
            
            # formar the llms answer to exctract the three differnt kinds of response
            thought, llm_high_level_action, low_level_control = format_llm_response(llm_raw_response)

            #grading responses automatically
            response_grading = (carla_agent_action.lower() == llm_high_level_action.lower())

            # write the answer side by side on cvs file for comparison
            writer.writerow([i+1, traffic_light, obstacle, carla_agent_action, llm_high_level_action, response_grading, low_level_control, thought])

    print(f"\nEvaluation done. Exported results: {output_csv}")




if __name__ == "__main__":
    run_evaluation("data/captures/testing/testing_dataset.jsonl", "results/thesis_evaluation.csv")