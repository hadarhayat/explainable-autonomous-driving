import carla, json, math, time, random
from pathlib import Path
import xml.etree.ElementTree as ElementTree
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption
import math
import argparse



#passing town and route from scipt
parser = argparse.ArgumentParser()
parser.add_argument('--route', type=str, default="25", help='Route ID to run')
parser.add_argument('--town', type=str, default="Town03", help='CARLA Town name (Town03)')
args = parser.parse_args()

chosen_route = args.route 
chosen_town = args.town


#define host and port for the simulation to run on
HOST, PORT = "127.0.0.1", 2040

# set this to a specific number
random.seed(42)

'''
#dev parameters
# path to leaderboar xml file
xml_path = "/itf-fi-ml/shared/users/hadarh/carla_setup/leaderboard/data/routes_devtest.xml"
#npc spawn number
VEHICLES, PEDESTRIANS = 20, 10
sim_lenght_frames = 3000
log_frame_every = 10

#file path for data to be saved on
OUT = Path.home() / "llm_sensor_fusion" / "data" / "captures" / "dev" / f"{chosen_town.lower()}_route{chosen_route}.jsonl"




#training parameters
xml_path = "/itf-fi-ml/shared/users/hadarh/carla_setup/leaderboard/data/routes_training.xml"
VEHICLES, PEDESTRIANS = 100, 0
sim_lenght_frames = 10000
log_frame_every = 10
OUT = Path.home() / "llm_sensor_fusion" / "data" / "captures" / "training" / f"{chosen_town.lower()}" / f"route{chosen_route}.jsonl"

'''


#testing parameters
xml_path = "/itf-fi-ml/shared/users/hadarh/carla_setup/leaderboard/data/routes_testing.xml"
VEHICLES, PEDESTRIANS = 100, 50
sim_lenght_frames = 10000
log_frame_every = 10
OUT = Path.home() / "llm_sensor_fusion" / "data" / "captures" / "testing" / f"{chosen_town.lower()}_route{chosen_route}.jsonl"







#helper function to get speed in km/h instead of m/s
def speed_kmh(v): return 3.6 * (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5

def route_from_xml(xml_path: str, route_id: str = "0"):
    tree = ElementTree.parse(xml_path) #open and load xml file
    root = tree.getroot()

    route = root.find(f"./route[@id='{route_id}']") #find the specific route with a given id
    if route is None:
        raise ValueError(f"ERROR: Route {route_id} not found.")

    waypoints = [] #empty list of waypoints 
    for current_waypoint in route.findall("waypoint"): #extract and append each waypoint from a given route
        waypoint_data = {   
            "x": float(current_waypoint.attrib["x"]),
            "y": float(current_waypoint.attrib["y"]),
            "z": float(current_waypoint.attrib["z"]),
            "yaw": float(current_waypoint.attrib["yaw"]),
            "pitch": float(current_waypoint.attrib["pitch"]),
            "roll": float(current_waypoint.attrib["roll"])
        }
        waypoints.append(waypoint_data)

    return waypoints

def insert_custom_route(agent, route_waypoints, world):
    
    carla_map = world.get_map()
    
    # initialize the DAO with the map and the 2.0m sampling resolution
    data_access_object = GlobalRoutePlannerDAO(carla_map, 2.0)
    
    #pass the DAO to the GlobalRoutePlanner and build the graph
    route_planner = GlobalRoutePlanner(data_access_object)
    route_planner.setup()
    
    custom_detailed_route = []
    
    print("Creating a topologiacal route.")
    for i in range(len(route_waypoints) - 1):
        current_location = carla.Location(x=route_waypoints[i]["x"], y=route_waypoints[i]["y"], z=route_waypoints[i]["z"])
        next_location = carla.Location(x=route_waypoints[i+1]["x"], y=route_waypoints[i+1]["y"], z=route_waypoints[i+1]["z"])
        
        # use carlas built in route tracing to builde a legal route between the current waypoint and the next
        sub_route = route_planner.trace_route(current_location, next_location)
        
        # avoid duplcated waypoint where two of the new routing point connect
        if i > 0 and len(sub_route) > 0:
            sub_route = sub_route[1:]
            
        custom_detailed_route.extend(sub_route)
        
    # Inject this perfectly mapped route into the agent
    agent._local_planner.set_global_plan(custom_detailed_route)
    print(f"Topological route insterted. {len(custom_detailed_route)} points loaded to local planner.")
    
    return custom_detailed_route

class CarlaEnvironmentAdapter:
    """  hack to bypass CARLA 0.9.10's hardcoded BehaviorAgent requirements"""
    #run the behavior agent in a parallel fake world
    def __init__(self, actual_world, ego_vehicle):
        self._world = actual_world
        self.player = ego_vehicle 
        
    def __getattr__(self, name):
        # If the agent asks for anything else (like traffic lights), pass it to the real world
        return getattr(self._world, name)

#connection to the carla client and loading the correct town
client = carla.Client(HOST, PORT); client.set_timeout(30.0)

world = client.load_world(chosen_town) 
time.sleep(2.0) # give the server a few second clear itself

# force controlled weather
world.set_weather(carla.WeatherParameters.ClearNoon)


# ego - spawning the agent, setting up a tesla model 3 as the acting car
bp = world.get_blueprint_library()
ego_bp = bp.find("vehicle.tesla.model3"); ego_bp.set_attribute("role_name", "hero")
spawns = world.get_map().get_spawn_points(); random.shuffle(spawns)

#gettin route waypoint for given route from xml file
route_waypoints = route_from_xml(xml_path, route_id=chosen_route)

#starting point
start_waypoint = route_waypoints[0]

start_transform_matrix = carla.Transform(
    carla.Location(x=start_waypoint["x"], y=start_waypoint["y"], z=start_waypoint["z"] + 0.5), 
    carla.Rotation(pitch=start_waypoint["pitch"], yaw=start_waypoint["yaw"], roll=start_waypoint["roll"])
)

print(f"Spawning ego vehicle at the chosen Leaderboard routes starting point.")
ego = world.try_spawn_actor(ego_bp, start_transform_matrix)

if not ego: 
    raise RuntimeError("Could not spawn car. Potential glitch or obstacle on strating point.")


print("Using BehaviorAgent for precision routing.")
# create the algorithmic expert driver an dset behavior to normal
driving_agent = BehaviorAgent(ego, behavior='normal')

# define the start location using our first waypoint
#start_location = carla.Location(x=start_waypoint["x"], y=start_waypoint["y"], z=start_waypoint["z"])
full_plan = insert_custom_route(driving_agent, route_waypoints, world)


fake_world = CarlaEnvironmentAdapter(world, ego)

# get the  last waypoint from xml route tobe the finish line
end_waypoint = route_waypoints[-1]
end_location = carla.Location(x=end_waypoint["x"], y=end_waypoint["y"], z=end_waypoint["z"])

print("Locked onto destination.")


#set "sync mode" to ensure no data is lost because of descrapency between simulation and my scrip
orig = world.get_settings()
world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=0.05, no_rendering_mode=False))
tm = client.get_trafficmanager(); tm.set_synchronous_mode(True)

# extract  carla.Location objects too calculate left/right commands
path_locations = []
for wp in route_waypoints:
    path_locations.append(carla.Location(x=wp["x"], y=wp["y"], z=wp["z"]))


#Vehicles
# _________________________________
#creating background traffic/actors
npc_list = []
vehicle_bps = bp.filter('vehicle.*')
# spawns list was already shuffled earlier
for spawn in spawns:
    if len(npc_list) >= VEHICLES:  
        break

    # make sure to spawn at least 20m from an obstacle
    if spawn.location.distance(start_transform_matrix.location) < 20.0:
        continue 
    
    # random car model that isn't the ego vehicle
    random_bp = random.choice(vehicle_bps)
    npc = world.try_spawn_actor(random_bp, spawn)
    if npc:
        npc.set_autopilot(True, tm.get_port())
        npc_list.append(npc)
print(f"Spawned {len(npc_list)} NPC vehicles.")


# Pedestrians
# _________________________________
walker_bps = bp.filter('walker.pedestrian.*')
pedestrian_list = []

#fix due to not all pedestrians being spawn
attempts = 0
max_attempts = PEDESTRIANS * 3 # failsafe to prevent infinite loops, it will only try 3 times to spwn the correct number of pedestrisns

while len(pedestrian_list) < PEDESTRIANS and attempts < max_attempts:
    attempts += 1
    spawn_point = carla.Transform()
    
    # CARLA has a built-in function to find valid points on sidewalks/crosswalks
    location = world.get_random_location_from_navigation()
    
    if location is not None:
        spawn_point.location = location
        walker = world.try_spawn_actor(random.choice(walker_bps), spawn_point)
        if walker:
            pedestrian_list.append(walker)
            
print(f"Spawned {len(pedestrian_list)} pedestrians  on the sidewalks.")
# _________________________________

# sensors
imu_bp  = bp.find("sensor.other.imu")
gnss_bp = bp.find("sensor.other.gnss")
imu  = world.spawn_actor(imu_bp,  carla.Transform(carla.Location(z=2.0)), attach_to=ego)
gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=2.0)), attach_to=ego)
latest = {"imu": None, "gnss": None, "camera": None}
imu.listen(lambda d: latest.__setitem__("imu", d))
gnss.listen(lambda d: latest.__setitem__("gnss", d))

#set up camera
# directory to save the images
IMG_DIRECTORY = OUT.parent / chosen_town / f"route_{chosen_route}_images"
IMG_DIRECTORY.mkdir(parents=True, exist_ok=True)

# find and set up camera blueprint
camera_bp = bp.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_bp.set_attribute("fov", "90")

# adding camera on top of ego vehicle 
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego)

# tell camera to save the image and record the file path
def process_image(image):

    if image.frame % log_frame_every == 0:
        filepath = IMG_DIRECTORY / f"{image.frame:08d}.jpg"
        image.save_to_disk(str(filepath))
        latest["camera"] = str(filepath)

camera.listen(process_image)
# ________________________________________


# let buffers fill
for _ in range(10): world.tick()

#filter out nearby obstacles 
def get_nearby_obstacle(world, ego_vehicle, max_distance=50.0):
    """
    Check for the nearest vehicle or pedestrian in front of our ego actor.
    """
    # ask the simulation for all vehicles and pedestrians
    vehicles = world.get_actors().filter('vehicle.*')
    walkers  = world.get_actors().filter('walker.pedestrian.*')
    
    # combine the into one actors list
    all_actors = list(vehicles) + list(walkers)
    
   #locatios of out vehicle to find closest obstcle
    ego_location = ego_vehicle.get_location()
    
    closest_distance = max_distance
    closest_actor = None

    ego_forward_vector = ego_vehicle.get_transform().get_forward_vector()

    for actor in all_actors:
        #ignore our own vehicle
        if actor.id == ego_vehicle.id:
            continue
            
        # get the obstacle distance from location
        actor_location = actor.get_location()
        distance = ego_location.distance(actor_location)
        
        if distance < closest_distance:
            # vector pointing from our vehicle to the other actor
            vector_to_actor = carla.Vector3D(
                actor_location.x - ego_location.x, 
                actor_location.y - ego_location.y, 
                actor_location.z - ego_location.z
            )
            
            # using the dot product with x and y to find actor directly in front of us 
            dot = (ego_forward_vector.x * vector_to_actor.x) + (ego_forward_vector.y * vector_to_actor.y)
            
            if dot > 0: 
                closest_distance = distance
                closest_actor = actor

    # format the data for our json format 
    if closest_actor:
        
        #check what kind of actor with have
        obj_type = "vehicle"
        if "pedestrian" in closest_actor.type_id: obj_type = "pedestrian"        
        return {
            "type": obj_type,
            "distance": round(closest_distance, 2),
            "relative_direction": "front",
            "speed_kmh": round(speed_kmh(closest_actor.get_velocity()), 1)
        }
    
    # if the loop found nothing in front of us return None
    return None


def get_traffic_light(world, ego_vehicle, max_distance=50.0):
    """
    Check for the nearest trafiic light color in front of our ego actor.
    """
    lights = world.get_actors().filter('traffic.traffic_light')
    
    ego_location = ego_vehicle.get_location()
    ego_forward_vector = ego_vehicle.get_transform().get_forward_vector()
    
    closest_distance = max_distance
    light_state = "None"
    
    for light in lights:
        light_location = light.get_location()
        distance = ego_location.distance(light_location)
        
        if distance < closest_distance:
            # 3d vectoor pointing from our ego cehicle to the light
            vector_to_light = carla.Vector3D(
                light_location.x - ego_location.x, 
                light_location.y - ego_location.y, 
                light_location.z - ego_location.z
            )
            
             # using the dot product with x and y to find light  in front of us
            dot = (ego_forward_vector.x * vector_to_light.x) + (ego_forward_vector.y * vector_to_light.y)
            
            if dot > 0:
                closest_distance = distance
                # getting only the specific color from the light state return phrase
                light_state = str(light.get_state()).split(".")[-1]
                
    return light_state



def get_navigation_command(ego_transform, path_locations, search_distance=8.0):
    """
    Calculating if the route ahead goes left, right or goes straight 
    using a 2D cross product.
    """
    if not path_locations:
        return "Follow the road"

    ego_location = ego_transform.location #x, y, z coordianates of the ego car
    ego_forward = ego_transform.get_forward_vector() #normalized vector pointing directly out of the car
    
    # find waypoint further down the route to see where it bends
    target_location = path_locations[-1] 
    for waypoints in path_locations: # iterate through  GPS dots until we find one that is at least search_distance away
        if ego_location.distance(waypoints) > search_distance:
            target_location = waypoints
            break
            
    # simple vector calculation to find the direction to the target
    direction_to_target = target_location - ego_location
    
    # turn it into a "unit vector", keeping it pointing in the same direction, change its length to 1
    length = math.sqrt(direction_to_target.x**2 + direction_to_target.y**2)
    if length == 0.0: return "Follow the road"
    direction_to_target.x /= length
    direction_to_target.y /= length #ignore z axis, we only need to know if to turn right or left
    
    # cross product to find the turn direction
    cross_z = (ego_forward.x * direction_to_target.y) - (ego_forward.y * direction_to_target.x)
    
    # sertting up thresholds for turning
    if cross_z > 0.15:
        return "Turn Right"
    elif cross_z < -0.15:
        return "Turn Left"
    else:
        return "Drive Straight"


#building a schema for each data log, to include all sensor, enviroment and position data
try:
    stuck_frames = 0 
    total_idle_frames = 0
    last_control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
    latest = {"camera": None, "imu": None, "gnss": None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:

        for frame_index in range(sim_lenght_frames):

            world.tick()

            if ego is None or not ego.is_alive:
                print("Ego vehicle lost. Attempting to recover...")
                continue


            snap = world.get_snapshot()
            transform   = ego.get_transform()
            velocity  = ego.get_velocity()
            speed  = speed_kmh(velocity)
            waypoint  = world.get_map().get_waypoint(transform.location)
            
            # Run Agent
            driving_agent.update_information(fake_world)
            control = driving_agent.run_step()
            ego.apply_control(control)

            # Success Checks
            # 1. Get current status
            dist_to_finish = transform.location.distance(end_location)
            buffer_size = len(driving_agent._local_planner._waypoint_buffer)



            #check for light and stop sign so it does not count them as stuck
            current_light_color = get_traffic_light(world, ego)
                

            if speed < 2.0:
                if current_light_color == "Red": #or driving_agent._incoming_waypoint is None: count red light as valiud stop and not stuck
                    stuck_frames = 0 
                else:
                    stuck_frames += 1
            else:
                stuck_frames = 0

            
            if stuck_frames > 400:
                print("Potential glitch. Ego vehicle stuck . Nudg forward and resetting agent.")
                
                # nudge  car forward just enough to break the stop sign trigger box, 4m, an then reset the topologial map to skip 2 points (2m each)
                yaw_rad = math.radians(transform.rotation.yaw)
                new_loc = carla.Location(
                    x = transform.location.x + 4.0 * math.cos(yaw_rad),
                    y = transform.location.y + 4.0 * math.sin(yaw_rad),
                    z = transform.location.z + 0.5 
                )
                ego.set_transform(carla.Transform(new_loc, transform.rotation))
                
                # give the sim some time to process the reset
                for _ in range(10): world.tick() 
                
                # reset the driving agent
                driving_agent = BehaviorAgent(ego, behavior='normal')
                
                # full_plan contains the high-density (every 2m) route generated at the start
                closest_index = 0
                closest_dist = float('inf')
                
                for i, wp_tuple in enumerate(full_plan):
                    wp_loc = wp_tuple[0].transform.location
                    dist = math.sqrt((new_loc.x - wp_loc.x)**2 + (new_loc.y - wp_loc.y)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_index = i
                
                # skip ahead by 2 indexes, which equals to 3m, to guarantee the next target is in front of the car
                safe_start_index = min(closest_index + 2, len(full_plan) - 1) #change to 3 if car keeps crashing.....
                remaining_dense_plan = full_plan[safe_start_index:]
                
                # Feed it directly into the local planner (bypassing the XML parser)
                driving_agent._local_planner.set_global_plan(remaining_dense_plan)
                
                print(f"Agent reset. Picked up dense route at index {safe_start_index}/{len(full_plan)}. Resuming.")
                
                stuck_frames = 0
                total_idle_frames = 0
                continue
                
            #abandon completelty in case of constant failure
            if stuck_frames > 1000: #if teleporting does not help after multiple tries
                print(f" Permanenetly stuck at frame {frame_index}. Abandoning route.")
                break
            
            #calculate distance to end of route without z axis to stop the logging close to the end and avoid poisoned data
            distance_2d = math.sqrt((transform.location.x - end_location.x)**2 + (transform.location.y - end_location.y)**2)
            

            ego_transform = ego.get_transform() #transformation matrix from ego vehicle
            nav_command = get_navigation_command(ego_transform, path_locations)

            if frame_index % 10 == 0:
                
                rec = {
                    "frame": int(snap.frame),
                    "navigation": nav_command,
                    "environment": {
                        "weather": "ClearNoon", 
                        "town": chosen_town,
                        "route": chosen_route
                    },
                    "timestamp": float(snap.timestamp.elapsed_seconds),
                    "ego": {
                        "location": {"x": transform.location.x, "y": transform.location.y, "z": transform.location.z},
                        "rotation": {"yaw": transform.rotation.yaw, "pitch": transform.rotation.pitch, "roll": transform.rotation.roll},
                        "speed_kmh": speed,
                    },
                    "waypoint": {
                        "x": waypoint.transform.location.x, "y": waypoint.transform.location.y, "z": waypoint.transform.location.z,
                        "yaw": waypoint.transform.rotation.yaw, "pitch": waypoint.transform.rotation.pitch, "roll": waypoint.transform.rotation.roll,
                    },
                    "imu": None if latest["imu"] is None else {
                        "accel": [latest["imu"].accelerometer.x, latest["imu"].accelerometer.y, latest["imu"].accelerometer.z],
                        "gyro":  [latest["imu"].gyroscope.x,    latest["imu"].gyroscope.y,    latest["imu"].gyroscope.z],
                        "compass": latest["imu"].compass,
                    },
                    "gnss": None if latest["gnss"] is None else {
                        "lat": latest["gnss"].latitude, "lon": latest["gnss"].longitude, "alt": latest["gnss"].altitude,
                    },
                    "traffic_light": get_traffic_light(world, ego),
                    "speed_limit_kmh": ego.get_speed_limit(),
                    "obstacle": get_nearby_obstacle(world, ego),
                    "camera_rgb_path": latest["camera"],
                    "control": {
                        "throttle": round(control.throttle, 3),
                        "steer": round(control.steer, 3),
                        "brake": round(control.brake, 3)
                    },
                    "previous_action": last_control,
                    
                }

                # save current control for the next frame
                last_control = {
                    "throttle": round(control.throttle, 3),
                    "steer": round(control.steer, 3),
                    "brake": round(control.brake, 3)
                }


                f.write(json.dumps(rec) + "\n")

                print(f"Logged frame {frame_index} | Dist: {distance_2d:.1f}m | Stuck: {stuck_frames}/400")



# cleanup and turn off carla properly to ensure it does not affect next route run
except Exception as e:
    import traceback
    print("ERROR:")
    traceback.print_exc()

finally:
    # "turn off sensor" to stop callbacks 
    try:
        if 'imu' in locals() and imu:
            try: imu.stop()
            except: pass
            try: imu.listen(None)   # detach callback
            except: pass
    except: pass

    try:
        if 'gnss' in locals() and gnss:
            try: gnss.stop()
            except: pass
            try: gnss.listen(None)  # detach callback
            except: pass
    except: pass

    try:
        if 'camera' in locals() and camera:
            try: camera.stop()
            except: pass
            try: camera.listen(None)
            except: pass
    except: pass

    #Let pending callbacks drain
    import time
    time.sleep(0.2)

    #Destroy sensors, then ego
    for actor in (locals().get('imu'), locals().get('gnss'), locals().get('ego')):
        try:
            if actor and actor.is_alive:
                actor.destroy()
        except:
            pass

    #Turn off TM sync *before* restoring world settings
    try:
        tm.set_synchronous_mode(False)
    except:
        pass

    #Restore world settings
    try:
        world.apply_settings(orig)
    except:
        pass

    #Drop references so Python GC doesn’t call destructors later
    try:
        imu = None; gnss = None; ego = None; world = None; client = None; tm = None
    except:
        pass

    #Hard-exit to avoid UE4/CARLA destructor crashes on shutdown
    import os, sys
    print(f"clean exit; wrote to {OUT}")
    os._exit(0)  # bypass atexit/destructors that touch destroyed actors


    
            
