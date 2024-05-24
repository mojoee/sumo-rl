import argparse
import os
import sys
import pickle
import sumolib
import pandas as pd
import json
import csv

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

import sumo_rl
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    alpha = 0.2
    gamma = 0.99
    decay = 1
    runs = 1


    for run in range(1, runs + 1):
        filePath = "finalOutput/B/75"
        seedNumber = 4444

        tripinfo_output_file = f"{filePath}/tripinfo_run{run}seed{seedNumber}.xml"
        summary_output_file = f"{filePath}/summary_run{run}seed{seedNumber}.xml"
        emission_output_file = f"{filePath}/emission_run{run}seed{seedNumber}.xml"
        statistics_output_file = f"{filePath}/statistics_run{run}seed{seedNumber}.csv"  # File for statistics
        total_vehicle_file = f"{filePath}/total_vehicle_run{run}seed{seedNumber}1.csv"

        env = sumo_rl.env(
        net_file="nets/SUMO3/MODIFIED.net.xml",
        route_file="nets/Scenario/Demand/B/B_75.rou.xml",
        use_gui=True,
        min_green=5,
        delta_time=5,
        begin_time=0,
        num_seconds=21600,
        time_to_teleport = 150,
        additional_sumo_cmd=f"-a nets/Scenario/Turning/B/router.add.xml --tripinfo-output={tripinfo_output_file} --summary={summary_output_file} --emission-output={emission_output_file}",
        sumo_seed = seedNumber,
    )
        env.reset()
       
        q_tables = {}
        for agent_id in env.agents:
            try:
                with open(f'q_tables/agent_{agent_id}_run_{run-1}.pkl', 'rb') as f:
                    q_tables[agent_id] = pickle.load(f)
            except FileNotFoundError:
                q_tables[agent_id] = None  # No Q-table found for this agent

        initial_states = {ts: env.observe(ts) for ts in env.agents}
        ql_agents = {
            ts: QLAgent(
                starting_state=env.unwrapped.env.encode(initial_states[ts], ts),
                state_space=env.observation_space(ts),
                action_space=env.action_space(ts),
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
                q_table = q_tables.get(ts)
            )
            for ts in env.agents
        }
        infos = []
        totalWT = 0
        totalVehs = 0
        agent_list = env.agents
        agent_incoming_info = {
            "data":{},
            "timestamp":0,
        }
        

        # Creating data structure for junction information
        net = sumolib.net.readNet('nets/SUMO3/MODIFIED.net.xml')

        for age in agent_list:
            agent_incoming_info["data"][age] = {}
            junction = net.getNode(age)
            incoming = junction.getIncoming()
            for inc in incoming:
                agent_incoming_info["data"][age][inc.getID()] = 0


        timestamp = 60 # s
        with open(statistics_output_file, 'w') as stats_file:
            stats_file.write('TotalWaitingTime,AverageWaitingTime\n')

            i = 1
            for agent in env.agent_iter():
                s, r, terminated, truncated, info = env.last()
                done = terminated or truncated
                if ql_agents[agent].action is not None:
                    ql_agents[agent].learn(next_state=env.unwrapped.env.encode(s, agent), reward=r)

                action = ql_agents[agent].act() if not done else None
                env.step(action)
                
                # Collect statistics
                current_time = traci.simulation.getTime() # second
                print("curr time: ", current_time)
                print("timestamp: ", timestamp)
                if current_time % timestamp == 0:
                    for junction in agent_incoming_info["data"]:
                        for edge in agent_incoming_info["data"][junction]:
                            getLastStep = traci.edge.getLastStepHaltingNumber(edge)
                            agent_incoming_info["data"][junction][edge] = getLastStep
                    # print("curr time: ", current_time)

                    if i % 11 == 0:
                        agent_incoming_info["timestamp"] = current_time
                        # print(agent_incoming_info)
                        with open(total_vehicle_file, 'a', newline='') as csv_file:
                            # Define CSV writer
                            csv_writer = csv.writer(csv_file)
                            
                            # If the file is empty, write the header row
                            if csv_file.tell() == 0:
                                csv_writer.writerow(['timestamp', 'junction', 'edge', 'total vehicle'])
                            
                            # Extract data from the dictionary and write data rows
                            timestamp1 = agent_incoming_info["timestamp"]
                            data = agent_incoming_info["data"]
                            for junction, edge_data in data.items():
                                for edge, total_vehicle in edge_data.items():
                                    csv_writer.writerow([timestamp1, junction, edge, total_vehicle])
                
                for id in traci.vehicle.getIDList():
                    if traci.vehicle.getSpeed(id) < 0.1:
                        totalWT += 1
                totalVehs += traci.simulation.getArrivedNumber()
                # print("Total Vehs: ", totalVehs)
                avgWT = totalWT / totalVehs if totalVehs > 0 else 0
                    # Write statistics to file
                stats_file.write(f'{totalWT},{avgWT}\n')
                i += 1
           
       
        env.unwrapped.env.save_csv("outputs/test/pz_test", run)
        env.close()


# {
#     "data":{
#         "J60":{
#             "E1": 50,
#             "E2": 79
#         },
#        "J2":{
#            "E2":50
#        }
#     }
#     "timestamp": 80
# }
        
