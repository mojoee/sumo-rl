import os
import sys

# Check if SUMO_HOME is correctly set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

def run_sumo_simulation(sumo_cmd, stop_time, statistics_file):
    # Start the SUMO simulation
    traci.start(sumo_cmd)

    totalWT = 0
    totalVehs = 0

    with open(statistics_file, 'w') as file:
        file.write('Time;TotalWaitingTime;TotalVehicles;AverageWaitingTime\n')  # Write the header

        # Run the simulation until the stop time is reached
        while traci.simulation.getTime() < stop_time:
            traci.simulationStep()

            # Count vehicles with speed less than 0.1 as waiting
            waiting_time_this_step = sum(traci.vehicle.getSpeed(id) < 0.1 for id in traci.vehicle.getIDList())
            totalWT += waiting_time_this_step

            # Update total number of vehicles that have arrived
            totalVehs += traci.simulation.getArrivedNumber()

            # Calculate average waiting time
            avgWT = totalWT / totalVehs if totalVehs > 0 else 0

            # Write the time, total waiting time, total number of vehicles, and average waiting time to the file
            current_time = traci.simulation.getTime()
            file.write(f'{current_time};{totalWT};{totalVehs};{avgWT}\n')

    traci.close()

if __name__ == "__main__":
    runs = 1
    use_gui = True  # Set to False if you don't want the GUI
    simulation_stop_time = 21600  # Set your desisred stop time in seconds
    seeds = [1111,2222,3333,4444]  # Your list of seeds

    for run in range(1, runs + 1):
        for seed in seeds:
            filePath = "finalOutputNormal/B/125"
            os.makedirs(filePath, exist_ok=True)  # Ensure the directory exists

            tripinfo_output_file = f"{filePath}/tripinfo_run{run}_seed{seed}.xml"
            summary_output_file = f"{filePath}/summary_run{run}_seed{seed}.xml"
            emission_output_file = f"{filePath}/emission_run{run}_seed{seed}.xml"
            statistics_file = f"{filePath}/statistics_run{run}_seed{seed}.xml"

            sumo_binary = "sumo-gui" if use_gui else "sumo"
            net_file = "nets/SUMO3/MODIFIED.net.xml"

            route_file = "nets/Scenario/Demand/B/B_125.rou.xml"
            additional_files = "nets/Scenario/Turning/B/router.add.xml"
            sumo_cmd = [
                sumo_binary,
                "-n", net_file,
                "-r", route_file,
                "-a", additional_files,
                "--tripinfo-output", tripinfo_output_file,
                "--summary-output", summary_output_file,
                "--emission-output", emission_output_file,
                "--seed", str(seed),
            ]

            # Run the simulation with the current seed
            run_sumo_simulation(sumo_cmd, simulation_stop_time, statistics_file)
