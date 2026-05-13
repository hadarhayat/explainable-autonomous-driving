from typing import Dict
import math

def get_high_level_action(throttle: float, steer: float, brake: float, speed: float) -> str:
    """
    maps carla control values for steering, brakes and throttle to high-level action in text 
    """
    
    # braking
    if brake > 0.6:
        return "Hard Brake / Emergency Stop"
    elif brake > 0.05:
        return "Decelerate"
        
    # steering 
    elif steer < -0.3:
        return "Sharp Left Turn"
    elif steer < -0.05:
        return "Slight Left / Follow Curve"
    elif steer > 0.3:
        return "Sharp Right Turn"
    elif steer > 0.05:
        return "Slight Right / Follow Curve"
        
    # acceleration
    elif throttle > 0.5:
        return "Strong Acceleration"
    elif throttle > 0.05:
        return "Light Acceleration"
        
    # idle/constanst state
    elif speed > 0.5:
        return "Maintain Speed / Coasting"
    else:
        return "Idle / Stopped"



def format_chat_template(sensor_data: Dict) -> Dict:
    """
    main template function to create a three roles based prompt. 
    integrates chain of thought into the generated prompt basedon obstacles, steering, speed, and braking
    Roles: System, User and Assitant. 
    """
    
    prev = sensor_data.get("previous_action")
    if prev is None:
        prev_str = "None (Standing still / First frame)"
    else:
        prev_str = f"Throttle: {prev.get('throttle', 0.0)}, Steer: {prev.get('steer', 0.0)}, Brake: {prev.get('brake', 0.0)}"

    speed = sensor_data.get("speed", 0.0) #get speed data, set to 0 if not accessable to avoif crash
    gnss = sensor_data.get("gnss", ["unknown", "unknown", "unknown"]) #get gps data, set to unknow if not accessable 
    loc = sensor_data.get("location", {"x": 0.0, "y": 0.0, "z": 0.0})
    rot = sensor_data.get("rotation", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    imu = sensor_data.get("imu", {}) #inertial measurement unit data, accelerometers, gyroscopes and magnetometers,sets to empty if not accessable
    compass = imu.get("compass", None) #gets compass from the previous imu data, set to none if not accessable
    direction = compass_to_direction(compass) if compass is not None else "unknown" #runs another function to convert compass degrees to readble directions 
    traffic_light = sensor_data.get("traffic_light", "none") #trafic light status
    obstacle = sensor_data.get("obstacle", None) # gets obstacle info, if any
    control = sensor_data.get("control", None)
    navigation_command = sensor_data.get("navigation", "Drive Straight")

    #role based data separation for llm training. ROles: System, User and Asssistant
    
    system_content = (
        "You are an autonomous driving safety agent. Based on the following vehicle state and sensor data, "
        "suggest the next safe driving action. \n"
        "CRITICAL INSTRUCTION: The 'GPS Route Instruction' represents the high-level strategic intent "
        "for the NEXT upcoming intersection. It does NOT mean you should turn immediately. You must "
        "maintain lane keeping based on immediate road geometry, and only execute the navigation command "
        "when a physical intersection is reached."
    )

    user_content = (
        f"The vehicle is currently at position ({loc['x']:.1f}, {loc['y']:.1f}, {loc['z']:.1f}), it is facing {direction} with a yaw of {rot['yaw']:.1f} degrees.\n"
        f"The orientation is level, with a pitch of {rot['pitch']:.1f} and roll of {rot['roll']:.1f} degrees.\n"
        f"The current speed is {speed:.1f} km/h heading {direction}.\n\n"
        f"- GPS Route Instruction: {navigation_command}\n"
        f"- Previous Action: {prev_str}\n\n"
    )

    if str(traffic_light).lower() != "none":
        user_content += f"There is a {traffic_light} traffic light coming up ahead.\n"
    else:
        user_content += "There are no traffic lights visible.\n"

    if obstacle:
        obstacle_speed = obstacle.get('speed_kmh', 0.0)
        user_content += f"A {obstacle['type']} is approaching from the {obstacle['relative_direction']}, " \
                        f"{obstacle['distance']} meters away, traveling at {obstacle_speed:.1f} km/h.\n"

    user_content += "\nWhat should the vehicle do next?"

    if control:
        throttle = control.get('throttle', 0.0)
        steer = control.get('steer', 0.0)
        brake = control.get('brake', 0.0)

        # if stopped or stopping at a red light force the brake
        if str(traffic_light).lower() == "red" and speed < 5.0:
            throttle = 0.0
            brake = 1.0

            #old logic without chain of thought
        


        
        #get high level action based on input value and action category
        action_type = get_high_level_action(throttle, steer, brake, speed)

        # adding chain-of-thought
        thoughts = []
        if str(traffic_light).lower() == "red":
            if brake > 0.0:
                thoughts.append("I am at or approaching a red traffic light, so I am braking to stop safely.")
            else:
                thoughts.append("I see a red traffic light ahead. I will prepare to stop soon, but am currently maintaining momentum.")

        if obstacle:
            thoughts.append(f"There is a {obstacle.get('type', 'hazard')} in my path, requiring me to brake or yield.")

        if str(traffic_light).lower() != "red" and not obstacle:
            thoughts.append(f"The path is clear. I will follow the route instruction to '{navigation_command}'.")
        
        if steer < -0.1:
            thoughts.append("I am steering left to follow the road curve or make a turn.")
        elif steer > 0.1:
            thoughts.append("I am steering right to follow the road curve or make a turn.")
            
        if speed < 1.0 and throttle == 0.0: 
            thoughts.append("I am keeping the vehicle stopped.")
            
        thought_string = " ".join(thoughts)

        # final assitant content for llm training with all 3 levels of input, chain of thought, high level command and low level values
        assistant_content = (
            f"Thought: {thought_string}\n"
            f"High-Level Action: {action_type}\n"
            f"Low-Level Control: Throttle: {throttle}, Steer: {steer}, Brake: {brake}"
        )
    else:
        # in case no control data
        assistant_content = (
            "Thought: I have lost connection to the control systems and must halt.\n"
            "High-Level Action: Hard Stop\n"
            "Low-Level Control: Throttle: 0.0, Steer: 0.0, Brake: 1.0"
        )
        
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    

#function to translate degrees from car compass to readable directions
def compass_to_direction(compass_radians: float) -> str:
    """
    maps the compass values extracted from the simulation to high level text directions
    """
    compass_degrees = math.degrees(compass_radians) % 360
    #set eight different direction dividing the compass degrees ever 45
    directions = [
        (0, "north"),
        (45, "northeast"),
        (90, "east"),
        (135, "southeast"),
        (180, "south"),
        (225, "southwest"),
        (270, "west"),
        (315, "northwest")
    ]

    for angle, name in directions:
        if abs(compass_degrees - angle) <= 15 or abs(compass_degrees - angle + 360) <= 15:
            return name
    closest = min(directions, key=lambda d: abs(compass_degrees - d[0]))

    #safety fallback if function runs without returning precise direction
    return f"approximately {closest[1]}"



