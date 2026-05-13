import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as seaborn
from pathlib import Path

# direcotries to data and exports
RESULTS_TRAINED_CSV_FILE = "results/thesis_evaluation.csv"
RESULTS_ONESHOT_CSV_FILE = "results/thesis_evaluation_oneshot_llama.csv"

OUTPUT_DIRECTORY = Path("results/analysis")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

def analyze_results():
    print(f"Load results from: {RESULTS_TRAINED_CSV_FILE} and {RESULTS_ONESHOT_CSV_FILE}")
    
    try:
        # load the csv file into pandas dataFrame
        dataframe_trained = pd.read_csv(RESULTS_TRAINED_CSV_FILE)
        dataframe_oneshot = pd.read_csv(RESULTS_ONESHOT_CSV_FILE)
    except FileNotFoundError: 
        print("ERROR: CSV file not found")
        return

    # force the data to be read as string, remove accidental spaces or capitalization issues and force everithing into title case
    expert_action = dataframe_trained["Carla_Agent_Action"].astype(str).str.strip().str.title()
    predicted_action = dataframe_trained["LLM_Action"].astype(str).str.strip().str.title()


    #map the various forms of high-level commands to my preset ones for the trained model....
    semantic_mapping = {
        #variations of decelerate
        "Yield / Slow Down": "Decelerate",
        "Reduce Speed": "Decelerate",
        "Gradual Brake": "Decelerate",
        "Gradual Brake / Speed Reduction": "Decelerate",
        "Soft Brake": "Decelerate",
        "Gradual Speed Reduction": "Decelerate",
        "Gradual Deceleration / Yield": "Decelerate",
        "Gradual Brake / Yield": "Decelerate",
        "Ease Off Throttle / Gradual Deceleration": "Decelerate",
        "Gradual Brake / Slow Down": "Decelerate",
        "Ease Off Throttle / Gradual Brake": "Decelerate",
        "Slow Down / Yield": "Decelerate",
        "Prepare for Potential Stop / Yield": "Decelerate",
        "Gradual Deceleration": "Decelerate",
        "Yield / Soft Brake": "Decelerate",
        "Maintain Speed / Slow Down": "Decelerate",
        "Adjust Steering / Maintain Safe Distance": "Decelerate",
        
        #variations of hard brake / emergency stop 
        "Maintain Current State / Yield": "Hard Brake / Emergency Stop",
        "Maintain Current State / Cautionary Braking": "Hard Brake / Emergency Stop",
        "Maintain Current State / Prepare for Potential Changes": "Hard Brake / Emergency Stop",
        "Maintain Current State": "Hard Brake / Emergency Stop",
        "Maintain Braking / Yield": "Hard Brake / Emergency Stop",
        "Maintain Current State / Wait": "Hard Brake / Emergency Stop",
        "Maintain Current State / Wait for Clear Intersection": "Hard Brake / Emergency Stop",
        "Stop": "Hard Brake / Emergency Stop",
        
        #variations of light acceleration 
        "Resume Normal Speed / Proceed with Caution": "Light Acceleration",
        "Continue Driving / Maintain Lane Keeping": "Light Acceleration",
        "Maintain Current Speed / Yield": "Light Acceleration",
        "Maintain Current Course": "Light Acceleration"
    }

    #replace llama high lavel values with predetermined for this test
    dataframe_oneshot["Mapped_Action"] = dataframe_oneshot["LLM_Action"].astype(str).str.strip().str.title().replace(semantic_mapping)


    expert_actions_oneshot = dataframe_oneshot["Carla_Agent_Action"].astype(str).str.strip().str.title()
    dataframe_oneshot["Semantic_Match"] = (expert_actions_oneshot == dataframe_oneshot["Mapped_Action"])


    # simple ovverall accuracy calculation with checking the boolan in the csv chekcing for match
    trained_accuracy = dataframe_trained["Match"].mean() * 100
    oneshot_accuracy = dataframe_oneshot["Match"].mean() * 100
    semantic_mapped_oneshot_accuracy = dataframe_oneshot["Semantic_Match"].mean() * 100
    
    print(f"Trained model accuracy: {trained_accuracy:.2f}%")
    print(f"Llama one-shot model accuracy: {oneshot_accuracy:.2f}%")
    print(f"Llama one-shot semantic mapped accuracy: {semantic_mapped_oneshot_accuracy:.2f}%")



    dataframe_trained["Light_Context"] = dataframe_trained["Light_Context"].astype(str).str.strip().str.title()
    dataframe_oneshot["Light_Context"] = dataframe_oneshot["Light_Context"].astype(str).str.strip().str.title()

    dataframe_trained["Light_Context"] = dataframe_trained["Light_Context"].replace({"Nan": "No Light", "None": "No Light"})
    dataframe_oneshot["Light_Context"] = dataframe_oneshot["Light_Context"].replace({"Nan": "No Light", "None": "No Light"})

    light_trained= dataframe_trained.groupby("Light_Context")["Match"].mean() * 100
    light_oneshot = dataframe_oneshot.groupby("Light_Context")["Match"].mean() * 100
    light_oneshot_semantic_mapped = dataframe_oneshot.groupby("Light_Context")["Semantic_Match"].mean() * 100

    dataframe_lights = pd.DataFrame({
        "CARLA Expert BehaviorAgent (Ground truth)": 100.0,
        "Trained model": light_trained,
        "Base llama model w/oneshot": light_oneshot,
        "Base llama model w/oneshot and semantic reasoning": light_oneshot_semantic_mapped
    }).fillna(0)

    # using matplotlib to plot a comparatibe bar chart for context/taxonomy
    plt.figure(figsize=(12, 6))
    dataframe_lights.plot(kind='bar', figsize=(10, 6), color=["#cceeff", "#66cc66", "#ffcc99", "#ff99ff"])
    plt.title("LLM driving accuracy by traffic light")
    plt.ylabel("Accuracy percentage")
    plt.xlabel("Traffic light state")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend(title="Model type", bbox_to_anchor=(1.05, 1), loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    bar_chart_dir = OUTPUT_DIRECTORY / "bar_chart.png"
    plt.savefig(bar_chart_dir, dpi=300)
    print(f"Bar chart exported: {bar_chart_dir}")



    # generate classification report from scikit-learn icluding parameters like precision, recall, F1-score 
    print("Classification report:")
    class_report = classification_report(expert_action, predicted_action, zero_division=0) 
    #zero_division=0 for error handling, if a situation occurs where due to a decision from the ai the math tries to divide by zero and crashes the script, python puts a 0 instaed of crashing
    print(class_report)
    
    # export text report to a file for thesis attachment
    with open(OUTPUT_DIRECTORY / "result_report.txt", "w") as report_file:
        report_file.write(f"Trained model accuracy: {trained_accuracy:.2f}%\n") #add accuracy score from trained model to report file
        report_file.write(f"Llama one-shot model accuracy: {oneshot_accuracy:.2f}%\n\n")  #add accuracy score from untrained llama model to report file
        report_file.write("Classification report:\n")
        report_file.write(class_report) #add classification report to report file

    # build a confusion matrix from the result using scikit-learn built in function
    conf_matrix = confusion_matrix(expert_action, predicted_action)
    labels = sorted(list(set(expert_action) | set(predicted_action))) # 

    #using matplotlib to plot a heatmap from the matrix
    plt.figure(figsize=(12, 8)) #generatea blank canvas with size 12x8

    # using seaborns heatmap function to display confiusion matrix numbers as colors in a heatmap
    seaborn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)  #print numbers inside boxes, "d" for decimal interger so the numbers are rounded, and color gradiaten from white to dark blue
    plt.title("LLM vs. Expert: action confusion matrix") #graph title
    plt.ylabel("Actual Carla expert action") #y axis label (expert)
    plt.xlabel("Predicted LLM action") #x axis label (prediction)
    plt.xticks(rotation=45, ha='right') #label rotation and spacing
    plt.tight_layout()

    # export graph image for your thesis 
    graph_save_dir = OUTPUT_DIRECTORY / "confusion_matrix.png"
    plt.savefig(graph_save_dir, dpi=300) #high res 300 dot per inch
    print(f"Heatmap exported: {graph_save_dir}")



if __name__ == "__main__":
    analyze_results()