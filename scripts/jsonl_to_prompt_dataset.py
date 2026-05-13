import json
import sys
from pathlib import Path


from prompt_formatter import format_chat_template


def process_logs():
    """
    takes output from log_ticks as input, and feeds it into the prompt template in prompt_formatter to get a jsonl of prompts
    """

    #dev paths
    '''
    INPUT_CURATED_LOGS = Path("data/captures/dev/town01_ticks_log.jsonl")
    OUTPUT_PROMPTS = Path("data/captures/dev/dev_dataset.jsonl")
    '''

    #training paths
    
    INPUT_CURATED_LOGS = Path("data/captures/training/curated_data")
    OUTPUT_PROMPTS = Path("data/captures/training/training_dataset.jsonl")


    #testing paths
    '''
    INPUT_CURATED_LOGS = Path("data/captures/testing/town03_run1.jsonl")
    OUTPUT_PROMPTS = Path("data/captures/testing/testing_dataset.jsonl")
    '''


    if not INPUT_CURATED_LOGS.exists():
        print(f"Error. {INPUT_CURATED_LOGS} not found.")
        return

    # get all the cureated logs datasets from one foldder
    curated_logs_dataset = list(INPUT_CURATED_LOGS.glob("*.jsonl"))
    
    if not curated_logs_dataset:
        print("No curated .jsonl files found.")
        return

    total_prompts = 0

    #rund each line form the imported path in the prompt formmatter
    with open(OUTPUT_PROMPTS, 'w', encoding='utf-8') as outfile:
        
        for file_path in curated_logs_dataset:
            print(f"Current file: {file_path.name}.")
        
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if not line.strip(): continue
                    raw_data = json.loads(line)
                    
                    sensor_data = {
                        "speed": raw_data.get("ego", {}).get("speed_kmh", 0.0),
                        "location": raw_data.get("ego", {}).get("location"),
                        "rotation": raw_data.get("ego", {}).get("rotation"),
                        "imu": raw_data.get("imu"),
                        "traffic_light": raw_data.get("traffic_light"),
                        "obstacle": raw_data.get("obstacle"),
                        "control": raw_data.get("control"),
                        "navigation": raw_data.get("navigation", "Drive Straight"),
                        "previous_action": raw_data.get("previous_action")
                    }
                    
                    chat_format = format_chat_template(sensor_data)
                    
                    outfile.write(json.dumps(chat_format) + "\n")
                    total_prompts += 1
                
    print(f"Data formatted. Saved to {OUTPUT_PROMPTS}")

if __name__ == "__main__":
    process_logs()