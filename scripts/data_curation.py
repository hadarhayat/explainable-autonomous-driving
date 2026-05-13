import json
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#manual cutoff of data for routes that clitched or complelty got stuck during simulation, devided in training and testing sections
#used for training data
MANUAL_GLITCH_CUTOFF = {
    "route1": 9250,  # stuck at pedestrian crossing
    "route2": 3030
    }

'''
#used for testing data
MANUAL_GLITCH_CUTOFF = {
    }
    '''

def calculate_distance(loc1, loc2):
    #simple equation to calcualte distance between two poitns in euclidean space
    return math.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['y'] - loc2['y'])**2)

def plot_route_speed(jsonl_file):
    frames = []
    speeds = []

    #simple graph with frames oin x axis and speed on each frame on thhe y axis for visualisation of the cut off data
    #should show unneccesary idle spots cut off or shortened 
    with open(jsonl_file, 'r') as f: 
        
        #for echa line in the file append the fram and the speed at that frame
        for line in f:
            data = json.loads(line)
            frames.append(data["frame"])
            speeds.append(data["ego"]["speed_kmh"])
            
    plt.figure(figsize=(12, 4)) #figure size
    plt.plot(frames, speeds, label='Speed (km/h)', color='blue')
    
    file_path = Path(jsonl_file)
    plt.title(f"Selected route file: {file_path.name}")
    plt.xlabel("Frame")
    plt.ylabel("Speed")
    plt.grid(True)
    
    # save  png of plotted graph
    output_graph = file_path.with_suffix('.png')
    plt.savefig(output_graph, bbox_inches='tight')
    plt.close() # close background graph figure
    

def clean_dataset(input_file):
    INPUT_FILE_PATH = Path(input_file)
    OUTPUT_FILE_PATH = INPUT_FILE_PATH.parent / "curated_training_data" / f"curated_{INPUT_FILE_PATH.name}"
    #OUTPUT_FILE_PATH = INPUT_FILE_PATH.parent / "curated_testing_data" / f"curated_{INPUT_FILE_PATH.name}"
    
    #if input file not found excepetion
    if not INPUT_FILE_PATH.exists():
        print(f"Error: file not found.  {input_file}")
        return

    with open(INPUT_FILE_PATH, 'r') as f:
        
        original_logs = [json.loads(line) for line in f]

    frames_prior_cleanup = len(original_logs)
    route_id = INPUT_FILE_PATH.stem

    #cleanup 1 
    if route_id in MANUAL_GLITCH_CUTOFF:
        terminal_frame = MANUAL_GLITCH_CUTOFF[route_id]
        
        #the number of actual saved logs are 10% of the total carla frames, and therefor the tick in the terminal
        #divide the the terminal ticks number form MANUAL_GLITCH_CUTOFF in 10, to get the number of logs to cut at
        log_cutoff_number = terminal_frame // 10
        
        # Use Python list slicing to keep only the logs up to the cutoff index
        original_logs = original_logs[:log_cutoff_number]
        
        print(f"Cut glitched logs for {route_id}. Kept {log_cutoff_number} logs. ")


    cleaned_frames = []
    idle_logs = []

    #main loop to remove poisoned data
    for i, frame in enumerate(original_logs):
        #extract variables used to determinate whether data can/should be used
        speed = frame["ego"]["speed_kmh"]
        light = frame["traffic_light"]
        steer = abs(frame["control"]["steer"])
        throttle = frame["control"]["throttle"]

        #cleanup 2: remove teleport glitch logs... 
        ## a distance jump of more then 15m while throttle is coasting below 0.5 means teleportation
        if i > 0:
            previous_original_log = original_logs[i-1]
            #compare the frame previous to the current and calculate distance between the two
            jumped_distance = calculate_distance(frame["ego"]["location"], previous_original_log["ego"]["location"])

            #if distance more then 15 and throttle less then 0.5
            if jumped_distance > 15.0 and throttle < 0.5:
                continue #skip this frame without saving




        #cleanup 3: out of way pedetrian
        #carlas sensor logs a pedestrian  but the driving agent ignores it and drives fgaster then 15kmh
        # set obstacle to none because it is not in the line of ego vehicle
        if frame.get("obstacle") and frame["obstacle"]["type"] == "pedestrian":
            if speed > 15.0:
                frame["obstacle"] = None 



        #cleanup 4: idle trimming, trim down the number of logs the ego vehicle spends sittig, for example at red light (ususally a  lot in carla sims)
        #if cars speed is below 0.5 count is as idle
        if speed < 0.5:
            idle_logs.append(frame)

        # if idle frames exceed 40 it  means it has been idle for more than 20 seconds, it's a long red light
        #keep the first 10 frames for deceleration and last 10 accelleration,
        # delete identical frame sin the middle
        else:
            if len(idle_logs) > 40:
                cleaned_frames.extend(idle_logs[:10])
                cleaned_frames.extend(idle_logs[-10:])
                #if short stop, keep the idle frames
            else:
                cleaned_frames.extend(idle_logs)
                    
            idle_logs = []
            cleaned_frames.append(frame)



    #cleanup 5: simulation ends on idle frames like a red light
    #check if there is anything at all in the idle frame list, else ignore it completly
    if len(idle_logs) > 0:
        #if more than 40 idle frames at the end of the file, keep 10 of the start of the idle and 10 at the end
        if len(idle_logs) > 40:
            cleaned_frames.extend(idle_logs[:10])
            cleaned_frames.extend(idle_logs[-10:])
            
            #if less then 40, keep all
        else:
            cleaned_frames.extend(idle_logs)




    OUTPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    #write cleaned frames list to an output file
    with open(OUTPUT_FILE_PATH, 'w') as f:
        for frame in cleaned_frames:
            f.write(json.dumps(frame) + "\n")



    #print feedback report
    print(f"Data curation for: {INPUT_FILE_PATH.name}")
    print(f"Original file: {frames_prior_cleanup} frames")
    print(f"Cleaned file:  {len(cleaned_frames)} frames")
    print("Clened file exported.")

    # generate graph for current file
    plot_route_speed(OUTPUT_FILE_PATH)



if __name__ == "__main__":
    #path to each orginal data log file
    INPUT_LOG_DIRECTORY = Path.home() / "llm_sensor_fusion/data/captures/testing"
    town = "Town05"
    routes = ["route18.jsonl"]
    
    #run curator on each file listed in the route list
    for r in routes:
        clean_dataset(INPUT_LOG_DIRECTORY / town / r)