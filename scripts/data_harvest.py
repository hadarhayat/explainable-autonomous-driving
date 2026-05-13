import os
import subprocess

target_town = "Town02"
routes_to_run = ["2"]

for route in routes_to_run:
    print(f"\nStarting logging    Town: {target_town}   Route: {route}.")
    
    # runs log_ticks.py script for each route
    # uses stuck_frames crash handling in log_ticks to check if route was completed or stopped
    try:
        subprocess.run([
            "python", "scripts/log_ticks.py", 
            "--route", route, 
            "--town", target_town
        ], check=True)
    except Exception as e:
        print(f"{target_town} route {route} failed or was killed. Moving to next route.")
        continue

print("Data harvest done.")