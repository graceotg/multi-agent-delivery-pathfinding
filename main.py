# --------------- References --------------- #
# A few foundational concepts such as linking the Brain and Bot classes and the design of the agents were based on the
# lab codes in the COMP4105 - Designing Intelligent Agents module at University of Nottingham, UK
#
# Lock mechanism: Python Tutorial. (n.d.). How to use the Python Threading Lock to Prevent Race Conditions. [online]
# Available at: https://www.pythontutorial.net/python-concurrency/python-threading-lock/ [Accessed 13 May 2025].
#
# A* Algorithm: Tech With Tim (2020). A* Pathfinding Visualization Tutorial - Python A* Path Finding Tutorial. YouTube.
# Available at: https://www.youtube.com/watch?v=JtiK0DOeI4A [Accessed 11 May 2025].
#
# Grouped Bar Plot: GeeksforGeeks. (2020). Create a grouped bar plot in Matplotlib. [online]
# Available at: https://www.geeksforgeeks.org/create-a-grouped-bar-plot-in-matplotlib/ [Accessed 13 May 2025].










# --------------- Libraries & modules ----------- #

import tkinter as tk
import random
import math
import time
from queue import PriorityQueue
import threading
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
import sys

# ----------- Global configurations and variables ------------- #

# Increasing recursion limit to handle deeper call stacks
sys.setrecursionlimit(3000)  # Default is 1000, increasing to 3000

# Global dictionary to track active experiments
active_experiments = {}


# ------------------ Code for visualising results ----------- #
def analyse_results(json_file_path="results.json"):
    """ Analyse results and generate visualisations """

    # Load results
    with open(json_file_path, 'r') as f:
        result_data = json.load(f)

    # ------- Time vs Agent Graph ------ #
    data = []

    for env in ["urban", "suburban", "rural"]:
        for count in [1, 3, 5, 8]:
            if env in result_data and str(count) in result_data[env]:
                trials = result_data[env][str(count)]
                avg_time = np.mean([trial["completion_time"] for trial in trials])
                std_dev = np.std([trial["completion_time"] for trial in trials])

                data.append({
                    "Environment": env.capitalize(),
                    "Agent Count": count,
                    "Average Time": avg_time,
                    "Standard Deviation": std_dev
                })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Creating the bar graph

    plt.figure(figsize=(14, 10))

    # pivot reshapes the data so it is in the right format to be plotted for a grouped bar graph
    pivot_df = df.pivot(index="Agent Count", columns="Environment", values="Average Time")
    ax = pivot_df.plot(kind="bar")

    # Customizing the chart

    plt.xlabel('Number of Agents', fontsize=14)
    plt.ylabel('Completion Time (seconds)', fontsize=14)
    plt.title('Average Delivery Completion Time \nby Environment and Agent Count', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Environment")
    plt.xticks(rotation=0)  # Make x-axis labels horizontal
    plt.subplots_adjust(top=0.85)  # Adding more vertical space between plot and title

    # plt.subplots_adjust(top=0.85)  # 15% padding on top of graph window
    plt.tight_layout(pad=4.0)  # Padding for the title

    # Save the graph
    plt.savefig("complete_time_agent_graph.png")
    plt.close()

    # ------------ Metrics Table ------------- #

    table_data = []
    if env == "urban":
        total_delivery_points = 10
    elif env == "suburban":
        total_delivery_points = 6
    elif env == "rural":
        total_delivery_points = 3

    for env in ["urban", "suburban", "rural"]:
        for count in [1, 3, 5, 8]:
            if env in result_data and str(count) in result_data[env]:
                trials = result_data[env][str(count)]
                avg_bot_failures = np.mean([trial["bots_failed"] for trial in trials])  # avg of bots that died
                perc_bot_failures = (avg_bot_failures / count) * 100  # percentage

                avg_delivery_point_failures = np.mean([trial["failed_delivery_points"] for trial in
                                                       trials])  # avg of delivery points that were unreachable
                perc_delivery_point_failures = (avg_delivery_point_failures / total_delivery_points) * 100  # percentage

                avg_success_delivery = np.mean([1 if trial["all_deliveries_completed"] else 0 for trial in
                                                trials])  # avg of trials that completed deliveries
                perc_success_delivery = avg_success_delivery * 100  # percentage

                avg_delivery_completion = np.mean(
                    [trial["deliveries_completed"] for trial in trials])  # avg of total deliveries completed
                perc_delivery_completion = (avg_delivery_completion / 20) * 100  # percentage

                table_data.append({
                    "Environment": env.capitalize(),
                    "Number of Agents": count,
                    "Bot Failure Rate (%)": round(perc_bot_failures, 2),
                    "Delivery Points Failure Rate (%)": round(perc_delivery_point_failures, 2),
                    "Successful Delivery Rate (%)": round(perc_success_delivery, 2),
                    "Deliveries Completed Rate (%)": round(perc_delivery_completion, 2)
                })

    # Create DataFrame
    metrics_df = pd.DataFrame(table_data)

    # Save as HTML
    with open("experiment_metrics.html", 'w') as f:
        f.write(metrics_df.to_html(index=False))


# --------------------- Code for classes ----------------- #


class Brain:
    def __init__(self, botp, occupied_cells, delivery_list, delivery_manager, cell_manager):
        self.bot = botp
        self.all_occupied_cells = occupied_cells
        self.depot = occupied_cells[0]
        self.charger = occupied_cells[1]
        self.delivery_points = occupied_cells[2]
        self.target_changed = True
        self.delivery_list = delivery_list
        self.current_path = []
        self.current_delivery = None
        self.waiting_threshold_counter = 0
        self.delivery_manager = delivery_manager
        self.cell_manager = cell_manager
        self.blocked_targets = []

    def get_delivery_target(self):
        return self.delivery_manager.get_delivery_target()

    def release_cell(self, xycoord):
        return self.cell_manager.release_cell(xycoord)

    def reserve_cell(self, xyxoord):
        return self.cell_manager.reserve_cell(xyxoord)

    # Calls A* algorithm and returns best path to target
    def find_path(self, current_x, current_y, target_x, target_y, occupied_cells, noOfRowsCols):
        return a_star(current_x, current_y, target_x, target_y, occupied_cells, noOfRowsCols)

    def determine_target(self, battery, hasPackage, current_x, current_y, noOfRowsCols):
        """Decides where the drone should go next based on current state"""

        # Battery check
        if battery <= 1000:
            print(f"Low battery - {self.bot.bot_name} is going to charge")
            # Finding free neighbours of the charger
            neighbours = finding_free_neighbours(self.charger[0], self.charger[1], noOfRowsCols,
                                                 self.all_occupied_cells)
            if len(neighbours) == 0:
                print(f"No spots free to charge - {self.bot.bot_name} is going to charge")
                return None, None
            else:
                # Randomisation by bot name to further help multiple bots picking the same cell
                random.seed(hash(self.bot.bot_name) + time.time())
                choice = random.choice(neighbours)
                self.bot.isCharging = True
                return choice

        # No package - go to depot
        if not hasPackage:
            # Check free neighbours for both cells that hold depot - separately
            neighbours1 = finding_free_neighbours(self.depot[0][0], self.depot[0][1], noOfRowsCols,
                                                  self.all_occupied_cells)
            neighbours2 = finding_free_neighbours(self.depot[1][0], self.depot[1][1], noOfRowsCols,
                                                  self.all_occupied_cells)
            total_neighbours = neighbours1 + neighbours2

            if len(total_neighbours) == 0:
                print(f"{self.bot.bot_name} is waiting - no depot spaces available")
                return None, None
            else:
                # Randomisation by bot name to further help multiple bots picking the same cell
                random.seed(hash(self.bot.bot_name) + time.time())
                choice = random.choice(total_neighbours)
                print(f"{self.bot.bot_name} is going to depot: {choice}")
                return choice

        # Has package - deliver
        if hasPackage:
            delivery_target = self.get_delivery_target()
            if delivery_target and delivery_target != (None, None):  # A valid delivery point
                # Checking if the delivery target has a free neighbour
                print(f"{self.bot.bot_name} is delivering to: {delivery_target}")
                neighbours = finding_free_neighbours(delivery_target[0], delivery_target[1], noOfRowsCols,
                                                     self.all_occupied_cells)

                if neighbours:
                    # Randomisation by bot name to further help multiple bots picking the same cell
                    random.seed(hash(self.bot.bot_name) + time.time())
                    choice = random.choice(neighbours)
                    self.current_delivery = choice
                    return choice
                else:
                    self.blocked_targets.append(delivery_target)
                    return None, 1  # must be blocked by 4 static obstacles

            else:
                # No valid delivery target
                print(f"{self.bot.bot_name} is going to starting point - no more packages to deliver")

                # The bot's starting position
                start_x = noOfRowsCols - 1
                start_y = 0

                # Make sure the colour is set back to pink
                self.bot.bot_colour = "pink"

                self.bot.finishedPackages = True

                return start_x, start_y

        return current_x, current_y  # Default fallback

    def get_next_move(self, current_x, current_y, battery, hasPackage, noOfRowsCols):
        """Handles Path Execution"""
        if not self.current_path or self.target_changed:
            # Deciding the target
            target_x, target_y = self.determine_target(battery, hasPackage, current_x, current_y, noOfRowsCols)


            if (target_x, target_y) == (None, None):  # next grid cell is blocked
                self.bot.waiting = True
                self.waiting_threshold_counter += 1
                if self.waiting_threshold_counter > 3:
                    self.target_changed = True
                    self.waiting_threshold_counter = 0
                    print(f"{self.bot.bot_name} waited too long, recalculating path")
            elif (target_x, target_y) == (None, 1):  # blocked by static obstacles
                self.delivery_list = [item for item in self.delivery_list if item not in self.blocked_targets]
                return current_x, current_y

            # Calculate the path
            if target_x is not None and target_y is not None:  # Make sure x and y have been set
                self.current_path = self.find_path(current_x, current_y, target_x, target_y, self.all_occupied_cells,
                                                   noOfRowsCols)
                # If no path is found
                if self.current_path is None:
                    print(f"{self.bot.bot_name} couldn't find path, staying still")
                    self.bot.waiting = True
                    self.waiting_threshold_counter += 1
                    return current_x, current_y
                self.target_changed = False
                # Reset waiting flag if we have a new path
                self.bot.waiting = False

        # If bot is waiting, stay still
        if self.bot.waiting:
            return current_x, current_y

        # Following the path
        if self.current_path and len(self.current_path) > 0:
            if len(self.current_path) > 1:  # Make sure there's at least 2 elements
                self.current_path.pop(0)  # Remove the cell bot is leaving
                next_step = self.current_path[0]

                # --------- Commented code below was part of my attempt at collision avoidance for the bots --------- #

                # # Check if next step is occupied by another bot
                # if [next_step[0],next_step[1]] in self.all_occupied_cells[4] and [next_step[0], next_step[1]] != [current_x, current_y]:
                #     # random_timer = random.randint(20,50)  # If random_timer is 20, then 20 cycles (20 × 50ms = 1 second of waiting)
                #     # 50ms is the time per cycle (canvas.after)
                #     self.bot.wait_counter = 10
                #     self.bot.waiting = True
                #
                #     # Force recalculation after a few waits to find alternative path
                #     self.waiting_threshold_counter += 1
                #     print(f"waiting_threshold_counter: {self.waiting_threshold_counter}")
                #     print(f"wait counter: {self.bot.wait_counter}")
                #     if self.waiting_threshold_counter > 3:
                #         # print(f"waiting_threshold_counter: {self.waiting_threshold_counter}")
                #         self.current_path = self.find_path(current_x,current_y,target_x,target_y,self.all_occupied_cells,noOfRowsCols)
                #         self.waiting_threshold_counter = 0
                #
                #     return current_x, current_y  # stay where you currently are

                return next_step[0], next_step[1]
            else:
                next_step = self.current_path[0]  # Use the last element without popping
                self.current_path = []  # Clear the path
                return next_step[0], next_step[1]

        if self.current_path is None:
            # No path found, stay where you are
            return current_x, current_y

        return current_x, current_y  # Default fallback


class Bot:
    def __init__(self, bot_name, grid_choice, cell_size, noOfRowsCols, bot_number):
        # launch initializations
        self.launch_delay = bot_number * 20  # 20 cycles delay per bot number
        self.launch_countdown = self.launch_delay
        self.has_launched = False

        self.bot_name = bot_name
        self.bot_colour = "pink"

        self.grid_choice = grid_choice
        self.cell_size = cell_size
        self.pixel_x = (noOfRowsCols - 1) * cell_size + (cell_size / 2)  # starting x-coordinate pixel of bot
        self.pixel_y = 0 * cell_size + (cell_size / 2)  # starting y-coordinate pixel of bot

        # target positions
        self.target_grid_x = 0
        self.target_grid_y = 0
        self.target_pixel_x = 0
        self.target_pixel_y = 0
        self.speed = 2  # bot speed

        # starting angle
        self.theta = math.radians(180)

        # stops the robots movement
        self.stopMoving = False

        # checks if the bot has made it to the next grid
        self.target_reached = True

        # used for reserved cells monitoring
        self.current_reserve = []
        self.next_reserve = []

        # used for when the bot has no neighbours and has to wait
        self.waiting = False
        self.wait_counter = 0

        # battery of the bot
        self.battery = 7000
        self.isCharging = False
        self.bot_previous_target = ""
        self.batteryRunOut = False

        # Packages
        self.hasPackage = False  # checks if the bot has a package
        self.finishedPackages = False  # checks if the packages are finished

    def thinkAndAct(self, noOfRowsCols):
        print(f"\n--- {self.bot_name} STATUS ---")
        current_grid_x, current_grid_y = pixel_to_grid(self.pixel_x, self.pixel_y, self.cell_size)
        target_grid_x, target_grid_y = self.brain.get_next_move(current_grid_x, current_grid_y, self.battery,
                                                                self.hasPackage, noOfRowsCols)
        return target_grid_x, target_grid_y

    # connects the bot to the brain
    def setBrain(self, brainp):
        self.brain = brainp

    # draws the agent at its current position
    def draw(self, canvas, noOfRowsCols):
        bot_x_center = self.pixel_x
        bot_y_center = self.pixel_y
        bot_size = self.cell_size * 0.3  # Making the bot 30% of the cell size
        battery_oval_size = self.cell_size * 0.2

        points = [(bot_x_center + bot_size * math.sin(self.theta)) - bot_size * math.sin((math.pi / 2.0) - self.theta), \
                  (bot_y_center - bot_size * math.cos(self.theta)) - bot_size * math.cos((math.pi / 2.0) - self.theta), \
                  (bot_x_center - bot_size * math.sin(self.theta)) - bot_size * math.sin((math.pi / 2.0) - self.theta), \
                  (bot_y_center + bot_size * math.cos(self.theta)) - bot_size * math.cos((math.pi / 2.0) - self.theta), \
                  (bot_x_center - bot_size * math.sin(self.theta)) + bot_size * math.sin((math.pi / 2.0) - self.theta), \
                  (bot_y_center + bot_size * math.cos(self.theta)) + bot_size * math.cos((math.pi / 2.0) - self.theta), \
                  (bot_x_center + bot_size * math.sin(self.theta)) + bot_size * math.sin((math.pi / 2.0) - self.theta), \
                  (bot_y_center - bot_size * math.cos(self.theta)) + bot_size * math.cos((math.pi / 2.0) - self.theta) \
                  ]
        canvas.create_polygon(points, fill=self.bot_colour, tags=self.bot_name)

        wheel1PosX = bot_x_center - bot_size * math.cos(self.theta)
        wheel1PosY = bot_y_center + bot_size * math.sin(self.theta)
        canvas.create_oval(wheel1PosX - 3, wheel1PosY - 3, \
                           wheel1PosX + 3, wheel1PosY + 3, \
                           fill="red", tags=self.bot_name)

        wheel2PosX = bot_x_center + bot_size * math.cos(self.theta)
        wheel2PosY = bot_y_center - bot_size * math.sin(self.theta)
        canvas.create_oval(wheel2PosX - 3, wheel2PosY - 3, \
                           wheel2PosX + 3, wheel2PosY + 3, \
                           fill="green", tags=self.bot_name)

        # Adding cameras to show front of bot
        camera_size = 3
        camera_distance = bot_size * 0.9  # How far forward from center
        camera_spacing = bot_size * 0.4  # How far apart from each other

        # Left front camera
        camera1PosX = bot_x_center + camera_distance * math.sin(self.theta) - camera_spacing * math.cos(self.theta)
        camera1PosY = bot_y_center - camera_distance * math.cos(self.theta) - camera_spacing * math.sin(self.theta)

        # Right front camera
        camera2PosX = bot_x_center + camera_distance * math.sin(self.theta) + camera_spacing * math.cos(self.theta)
        camera2PosY = bot_y_center - camera_distance * math.cos(self.theta) + camera_spacing * math.sin(self.theta)

        # Draw the cameras
        canvas.create_oval(camera1PosX - camera_size,
                           camera1PosY - camera_size,
                           camera1PosX + camera_size,
                           camera1PosY + camera_size,
                           fill="yellow", tags=self.bot_name)

        canvas.create_oval(camera2PosX - camera_size,
                           camera2PosY - camera_size,
                           camera2PosX + camera_size,
                           camera2PosY + camera_size,
                           fill="yellow", tags=self.bot_name)

        if self.grid_choice == "u":  # urban
            charger_text_size = 10
        elif self.grid_choice == "s":  # suburban
            charger_text_size = 8
        else:  # rural
            charger_text_size = 7

        chargerPosX = bot_x_center
        chargerPosY = bot_y_center
        canvas.create_oval(chargerPosX - battery_oval_size, chargerPosY - battery_oval_size, \
                           chargerPosX + battery_oval_size, chargerPosY + battery_oval_size, \
                           fill="gold", tags=self.bot_name)
        canvas.create_text(bot_x_center, bot_y_center, text=str(self.battery), font=("Arial", charger_text_size),
                           tags=self.bot_name)

    # what happens at each timestep
    def update(self, canvas, noOfRowsCols, occupied_cells):

        # Handling launch delay - so all bots don't leave starting point at the same time
        if not self.has_launched:  # Waiting to be launched
            if self.launch_countdown > 0:
                self.launch_countdown -= 1
                return
            else:  # Bot is being launched
                self.has_launched = True
                print(f"{self.bot_name} has started moving")

        # Starting positions of the bots
        start_x = noOfRowsCols - 1
        start_y = 0

        # Pixels of the starting point
        starting_pixel_x1, starting_pixel_y1 = grid_to_pixel(start_x, start_y, self.cell_size)

        # If the bot is back to the starting point - stop
        if [self.pixel_x, self.pixel_y] == [starting_pixel_x1, starting_pixel_y1] and self.finishedPackages:
            self.stopMoving = True
            self.waiting = True

        # Check if battery is completely depleted
        if self.battery <= 0:
            # Power down - stop moving and turn grey
            self.stopMoving = True
            self.waiting = True
            self.batteryRunOut = True
            self.bot_colour = "grey"
            print(f"{self.bot_name} has powered down due to battery depletion at position "
                  f"({int(self.pixel_x / self.cell_size)},{int(self.pixel_y / self.cell_size)})")

            # Stop at the center of the current grid cell
            current_grid_x, current_grid_y = pixel_to_grid(self.pixel_x, self.pixel_y, self.cell_size)
            target_pixel_x, target_pixel_y = grid_to_pixel(current_grid_x, current_grid_y, self.cell_size)
            self.pixel_x = target_pixel_x
            self.pixel_y = target_pixel_y

        # Pixels of the charger's neighbours
        charging_pixel_x1, charging_pixel_y1 = grid_to_pixel(occupied_cells[1][0], occupied_cells[1][1] + 1,
                                                             self.cell_size)
        charging_pixel_x2, charging_pixel_y2 = grid_to_pixel(occupied_cells[1][0] + 1, occupied_cells[1][1],
                                                             self.cell_size)

        # Charging
        if ([self.pixel_x, self.pixel_y] == [charging_pixel_x1, charging_pixel_y1] or \
            [self.pixel_x, self.pixel_y] == [charging_pixel_x2, charging_pixel_y2]) and self.isCharging:

            self.stopMoving = True
            self.bot_colour = "Purple"
            self.battery = min(7000, self.battery + 5)  # Increase battery when at charger
            if self.battery >= 7000:
                self.brain.target_changed = True
                self.brain.current_delivery = None
                self.isCharging = False
                self.stopMoving = False
                if self.bot_previous_target == "depot":
                    self.bot_colour = "blue"
                elif self.bot_previous_target == "delivery":
                    self.bot_colour = "pink"

        # Pixels of depot's neighbours
        depot_pixel_x1, depot_pixel_y1 = grid_to_pixel(self.brain.depot[0][0], self.brain.depot[0][1] + 1,
                                                       self.cell_size)
        depot_pixel_x2, depot_pixel_y2 = grid_to_pixel(self.brain.depot[1][0], self.brain.depot[1][1] + 1,
                                                       self.cell_size)
        depot_pixel_x3, depot_pixel_y3 = grid_to_pixel(self.brain.depot[0][0] - 1, self.brain.depot[1][1],
                                                       self.cell_size)
        depot_pixel_x4, depot_pixel_y4 = grid_to_pixel(self.brain.depot[1][0] + 1, self.brain.depot[1][1],
                                                       self.cell_size)

        # Depot

        if ([self.pixel_x, self.pixel_y] == [depot_pixel_x1, depot_pixel_y1] or \
            [self.pixel_x, self.pixel_y] == [depot_pixel_x2, depot_pixel_y2] or \
            [self.pixel_x, self.pixel_y] == [depot_pixel_x3, depot_pixel_y3] or \
            [self.pixel_x, self.pixel_y] == [depot_pixel_x4, depot_pixel_y4]) \
                and not self.hasPackage and not self.isCharging:
            self.bot_colour = "blue"
            self.hasPackage = True
            self.brain.target_changed = True
            self.brain.current_delivery = None
            self.bot_previous_target = "depot"

        # Delivery
        if self.hasPackage and self.brain.current_delivery:
            # Pixel of depot's delivery point
            delivery_pixel_x, delivery_pixel_y = grid_to_pixel(self.brain.current_delivery[0],
                                                               self.brain.current_delivery[1], self.cell_size)
            if [self.pixel_x, self.pixel_y] == [delivery_pixel_x, delivery_pixel_y]:
                self.bot_colour = "pink"
                self.hasPackage = False
                self.brain.target_changed = True
                self.brain.current_delivery = None
                self.bot_previous_target = "delivery"

        if not self.waiting and not self.stopMoving:
            actually_moved = self.move(canvas, noOfRowsCols, occupied_cells)
            # Only decrease battery if movement actually happened
            if actually_moved:
                self.battery -= 1

        elif self.waiting:
            if self.wait_counter > 0:
                self.wait_counter -= 1
            else:
                self.waiting = False

        # Redraw the bot
        canvas.delete(self.bot_name)
        self.draw(canvas, noOfRowsCols)

    def move(self, noOfRowsCols, occupied_cells):

        if self.target_reached:
            self.target_grid_x, self.target_grid_y = self.thinkAndAct(noOfRowsCols)
            current_grid_x, current_grid_y = pixel_to_grid(self.pixel_x, self.pixel_y, self.cell_size)

            # If no valid target is returned - wait
            if self.target_grid_x is None or self.target_grid_y is None:
                self.waiting = True
                return False

            # Translating the heading of the bot
            theta_direction = (self.target_grid_x - current_grid_x, self.target_grid_y - current_grid_y)

            if theta_direction != (0, 0):  # Only update heading if actually moving
                if theta_direction == (0, -1):  # Up
                    self.theta = math.radians(0)
                elif theta_direction == (1, 0):  # Right
                    self.theta = math.radians(90)
                elif theta_direction == (0, 1):  # Down
                    self.theta = math.radians(180)
                elif theta_direction == (-1, 0):  # Left
                    self.theta = math.radians(270)

            self.next_reserve = [self.target_grid_x, self.target_grid_y]
            self.brain.reserve_cell(self.next_reserve)
            self.target_reached = False
            return False  # No movement occurred - planning to move

        else:

            # Making incremental movement
            self.pixel_x += self.speed * math.sin(self.theta)
            self.pixel_y -= self.speed * math.cos(self.theta)

            # Calculate distance to target
            self.target_pixel_x, self.target_pixel_y = grid_to_pixel(self.target_grid_x, self.target_grid_y,
                                                                     self.cell_size)
            distance = math.sqrt((self.pixel_x - self.target_pixel_x) ** 2 + (self.pixel_y - self.target_pixel_y) ** 2)

            # If distance is close enough, bot will go to exact center and stop
            if distance < self.speed + 2:  # Threshold based on movement speed
                self.pixel_x = self.target_pixel_x
                self.pixel_y = self.target_pixel_y
                if self.current_reserve in occupied_cells[4]:
                    self.brain.release_cell(self.current_reserve)
                self.current_reserve = self.next_reserve
                self.target_reached = True
            return True  # Movement has occurred


class DeliveryManager:
    def __init__(self, delivery_list):
        self.lock = threading.Lock()
        self.delivery_list = delivery_list
        print(f"Initial delivery list contains {len(self.delivery_list)} targets")

    def get_delivery_target(self):
        with self.lock:
            if self.delivery_list:
                target = self.delivery_list[0]
                self.delivery_list.pop(0)
                print(f"Assigned delivery target: {target}, remaining: {len(self.delivery_list)}")
                return target[0], target[1]

            print("No delivery targets left!")
            return None, None


class CellManager:
    """
    CellManager class was intended to implement collision avoidance between bots.
    It tracks which cells are occupied by bots, but the complete collision avoidance
    algorithm isn't fully implemented in this version. With more time, this would
    have been used to make bots wait or reroute when their paths would cross.
    """

    def __init__(self, occupied_cells):
        self.lock = threading.Lock()
        self.occupied_cells = occupied_cells

    def reserve_cell(self, xycoord):
        with self.lock:
            xycoord_list = [int(xycoord[0]), int(xycoord[1])]
            if xycoord_list not in self.occupied_cells[4]:
                self.occupied_cells[4].append(xycoord_list)
                return True
            return False

    def release_cell(self, xycoord):
        with self.lock:
            xycoord_list = [int(xycoord[0]), int(xycoord[1])]
            if xycoord_list in self.occupied_cells[4]:
                self.occupied_cells[4].remove(xycoord_list)
                return True
            return False


# ------------- Window,Environment and Agent creation ------------- #

def initialise(window):
    window.title('Grid')
    window.resizable(False, False)
    canvas = tk.Canvas(window, width=700, height=700)
    canvas.pack()
    return canvas


def createEnvironment(canvas, grid_type):
    # Adding random seed so that randomness can be more effective
    random.seed(time.time())

    # Static variables
    delivery_points = 0
    obstacles = 0

    # Grid layout
    noOfRowsCols = 0
    cell_size = 0

    if grid_type.lower() == 'urban':
        noOfRowsCols = 10
        cell_size = 70
        delivery_points = 10
        obstacles = 12
    elif grid_type.lower() == 'suburban':
        noOfRowsCols = 12
        cell_size = 58.33
        delivery_points = 6
        obstacles = 8
    elif grid_type.lower() == 'rural':
        noOfRowsCols = 15
        cell_size = 46.67
        delivery_points = 3
        obstacles = 5

    # Draw grid lines
    for i in range(noOfRowsCols):
        canvas.create_line(0, i * cell_size, noOfRowsCols * cell_size, i * cell_size, fill='grey')
    for j in range(noOfRowsCols):
        canvas.create_line(j * cell_size, 0, j * cell_size, noOfRowsCols * cell_size, fill='black')

    # Placing depot in cell (4,0) and (5, 0)
    if grid_type.lower() == 'urban':
        x_scale = 4
    elif grid_type.lower() == 'suburban':
        x_scale = 5
    else:
        x_scale = 7

    x1_depot = x_scale * cell_size
    y1_depot = 0 * cell_size

    x2_depot = x1_depot + (cell_size * 2)
    y2_depot = y1_depot + cell_size

    canvas.create_rectangle(x1_depot + 10, y1_depot + 10, x2_depot - 10, y2_depot - 10, fill='blue')

    # Placing the charger in cell (0,0)
    x_charger = 0
    y_charger = 0
    canvas.create_oval(x_charger + 10, y_charger + 10, cell_size - 10, cell_size - 10, fill='purple')

    # Stored co-ordinates to avoid overlap
    coord_list = []
    for i in range(noOfRowsCols):
        for j in range(2, noOfRowsCols):  # leaving the first two rows free from houses and obstacles
            coord_list.append([i, j])

    # ----------- List of occupied cells -------------- #
    # occupied_cells[0] for depot locations
    # occupied_cells[1] for the charger location
    # occupied_cells[2] for delivery points
    # occupied_cells[3] for obstacles
    # occupied_cells[4] for agents occupying cells

    occupied_cells = [[[x_scale, 0], [x_scale + 1, 0]], [0, 0], [], [], []]  # Already including the depot and charger

    # Placing delivery points
    for i in range(delivery_points):
        chosen_coord_choice = random.choice(coord_list)

        random_x = chosen_coord_choice[0] * cell_size
        random_y = chosen_coord_choice[1] * cell_size

        occupied_cells[2].append(chosen_coord_choice)
        coord_list.remove(chosen_coord_choice)

        canvas.create_oval(random_x + 10, random_y + 10, random_x + cell_size - 10, random_y + cell_size - 10,
                           fill='red')

    # Placing obstacles
    for i in range(obstacles):
        chosen_coord_choice = random.choice(coord_list)

        random_x = chosen_coord_choice[0] * cell_size
        random_y = chosen_coord_choice[1] * cell_size

        occupied_cells[3].append(chosen_coord_choice)
        coord_list.remove(chosen_coord_choice)
        canvas.create_oval(random_x + 10, random_y + 10, random_x + cell_size - 10, random_y + cell_size - 10,
                           fill='dark grey')


    return cell_size, noOfRowsCols, occupied_cells


def createAgents(canvas, noOfBots, cell_size, noOfRowsCols, occupied_cells, grid_choice, delivery_list,
                 delivery_manager, cell_manager):
    agents = []
    for i in range(0, noOfBots):
        bot_number = i
        bot = Bot("Agent" + str(i), grid_choice, cell_size, noOfRowsCols, bot_number)
        brain = Brain(bot, occupied_cells, delivery_list, delivery_manager, cell_manager)
        bot.setBrain(brain)
        agents.append(bot)
        bot.draw(canvas, noOfRowsCols)

    return agents





# ------------ Core Simulation Function -------------- #
def moveAgents(canvas, agents, noOfRowsCols, occupied_cells, noOfBots, delivery_list, trial, callback_function,
               start_time, grid_type):

    currently_alive = 0
    all_finished = False
    failedDeliveryPoints = []
    all_bots_home = True  # checks if all the bots are at the starting point

    for ag in agents:
        ag.update(canvas, noOfRowsCols, occupied_cells)

        # Add only unique blocked targets
        for target in ag.brain.blocked_targets:
            if target not in failedDeliveryPoints:
                failedDeliveryPoints.append(target)

        # Check if this agent is still operational
        if ag.battery > 0 and not ag.batteryRunOut:
            currently_alive += 1

            # Check if there are no more packages to deliver
            if ag.finishedPackages:
                all_finished = True

    # Checking if all the bots are in the starting position
    for ag in agents:

        # Skip dead bots
        if ag.battery <= 0 or ag.batteryRunOut:
            continue

        # Get the starting position coordinates
        start_x = noOfRowsCols - 1
        start_y = 0
        current_grid_x, current_grid_y = pixel_to_grid(ag.pixel_x, ag.pixel_y, ag.cell_size)

        # Check if this bot is at the starting position
        if not (current_grid_x == start_x and current_grid_y == start_y):
            all_bots_home = False
            break  # No need to check more bots, we know not all are home

    # Termination conditions

    # Only end if all packages are delivered AND all bots are back home
    if len(delivery_list) == 0 and all_finished and all_bots_home:
        end_time = time.time()
        time_taken = end_time - start_time


        # Creating results dictionary
        results = {
            "grid_type": grid_type,
            "bot_count": noOfBots,
            "trial": trial + 1,
            "completion_time": time_taken,
            "bots_failed": noOfBots - currently_alive,
            "deliveries_completed": 20 - len(delivery_list) - len(failedDeliveryPoints),
            "deliveries_remaining": len(delivery_list),
            "failed_delivery_points": len(failedDeliveryPoints),
            "all_deliveries_completed": len(failedDeliveryPoints) == 0 and len(delivery_list) == 0  # returns boolean
        }

        print("Simulation complete - all packages delivered!!")
        print(f"{currently_alive}/{noOfBots} agents are still alive")
        if len(failedDeliveryPoints) > 0:
            print(
                f"These delivery points were obstructed by obstacles and were taken off the delivery list: {failedDeliveryPoints}")

        # Create a copy of results to pass to callback (to avoid reference issues)
        results_copy = results.copy()

        # Close the window first
        root = canvas.master
        print(f"Destroying window for {grid_type}, {noOfBots}, trial {trial}")
        try:
            root.quit()  # Stop the mainloop
            root.destroy()  # Destroy the window
            print("Window destroyed successfully")
        except Exception as e:
            print(f"Error destroying window: {e}")

        callback_function(results_copy)
        return

    if currently_alive == 0:
        end_time = time.time()
        time_taken = end_time - start_time

        # Creating results dictionary
        results = {
            "grid_type": grid_type,
            "bot_count": noOfBots,
            "trial": trial + 1,
            "completion_time": time_taken,
            "bots_failed": noOfBots,  # All bots failed
            "deliveries_completed": 20 - len(delivery_list) - len(failedDeliveryPoints),
            "deliveries_remaining": len(delivery_list),
            "failed_delivery_points": len(failedDeliveryPoints),
            "all_deliveries_completed": len(failedDeliveryPoints) == 0 and len(delivery_list) == 0  # returns boolean
        }

        print("All bots died!")
        print(f"{len(delivery_list)}/{20} packages were not delivered")
        if len(failedDeliveryPoints) > 0:
            print(
                f"These delivery points were obstructed by obstacles and were taken off the delivery list: {failedDeliveryPoints}")

        # Create a copy of results to pass to callback (to avoid reference issues)
        results_copy = results.copy()

        # Close the window first
        root = canvas.master
        print(f"Destroying window for {grid_type}, {noOfBots}, trial {trial}")
        try:
            root.quit()  # Stop the mainloop
            root.destroy()  # Destroy the window
            print("Window destroyed successfully")
        except Exception as e:
            print(f"Error destroying window: {e}")

        # Calling callback function
        callback_function(results_copy)
        return

    canvas.after(50, moveAgents, canvas, agents, noOfRowsCols, occupied_cells, noOfBots, delivery_list, trial,
                 callback_function, start_time, grid_type)




# ---------------- A* Algorithm ------------------ #

# Heuristic function calculates using manhattan distance
def h_score(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def a_star(start_x, start_y, target_x, target_y, occupied_cells, noOfRowsCols):
    count = 0  # to track when the f_score was added
    # Priority queue for open set
    open_set = PriorityQueue()
    open_set.put((0, 0, (start_x, start_y)))  # (f_score, count, position) for starting position

    # Tracking which nodes are in open set for faster lookup
    open_set_hash = {(start_x, start_y)}

    # Tracking route from start to end node
    came_from = {}

    # Initialises g_score and f_score dictionaries
    g_score = {(start_x, start_y): 0}
    f_score = {(start_x, start_y): h_score(start_x, start_y, target_x, target_y)}

    while not open_set.empty():
        # Get node with lowest f_score
        current = open_set.get()[2]
        open_set_hash.remove(current)

        # Goal reached - reconstruct the path
        if current == (target_x, target_y):
            path = []
            current_node = current
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append((start_x, start_y))  # Adding the start node
            return path[::-1]  # Returning the path in the right order

        current_g_score = g_score[current]  # initialising with starting g_score

        neighbours = finding_free_neighbours(current[0], current[1], noOfRowsCols, occupied_cells)

        for neighbour in neighbours:
            # Calculate tentative g_score
            temp_g_score = current_g_score + 1  # each step is 1 unit (battery depletes one at a time)

            # Checking for a better path to the neighbour
            if temp_g_score < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h_score(neighbour[0], neighbour[1], target_x, target_y)

                # Add to open set if not already there
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
    # No path found
    return None


# ------------- Helper Functions -------------- #

def finding_free_neighbours(x_coord, y_coord, noOfRowsCols, occupied_cells):
    # Checking neighbours
    neighbours = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # possible directions
        neighbour_x, neighbour_y = x_coord + dx, y_coord + dy

        # Check against edge cases
        if 0 <= neighbour_x < noOfRowsCols and 0 <= neighbour_y < noOfRowsCols:
            if [neighbour_x, neighbour_y] not in occupied_cells[0] and \
                    [neighbour_x, neighbour_y] not in occupied_cells[1] and \
                    [neighbour_x, neighbour_y] not in occupied_cells[2] and \
                    [neighbour_x, neighbour_y] not in occupied_cells[3]:
                neighbours.append((neighbour_x, neighbour_y))
    return neighbours


def populate_delivery_list(occupied_delivery_cells):
    delivery_list = []
    for i in range(20):
        coord_choice = random.choice(occupied_delivery_cells)
        delivery_list.append([coord_choice[0], coord_choice[1]])
    return delivery_list


# Grid-Pixel conversion
def grid_to_pixel(grid_x, grid_y, cell_size):
    return grid_x * cell_size + (cell_size / 2), grid_y * cell_size + (cell_size / 2)


def pixel_to_grid(pixel_x, pixel_y, cell_size):
    return int(pixel_x / cell_size), int(pixel_y / cell_size)


# ---------------- Running the experiments ---------------- #

def launch_experiment():
    experiment_queue = []

    # Generating all experiment combinations
    for grid_type in ["urban", "suburban", "rural"]:
        for bot_count in [1, 3, 5, 8]:
            for trial in range(10):
                experiment_queue.append((grid_type, bot_count, trial))

    # Start first experiment
    run_next_experiment(experiment_queue, {})


def run_next_experiment(queue, all_results):
    if len(queue) == 0:  # All experiments are complete
        print("All experiments are finished!")
        return

    # Get the experiment
    grid_type, bot_count, trial = queue.pop(0)

    # Show progress
    print(f"Running - {grid_type}, {bot_count} bot(s), trial: {trial + 1}/10")


    try:
        # After main is finished, the callback will run
        main(grid_type, bot_count, trial, lambda single_result: experiment_completed(single_result, queue, all_results))
    except Exception as e:
        print(f"Error running experiment {grid_type}, {bot_count}, trial {trial}: {e}")
        # Still try to run the next experiment
        experiment_completed({"grid_type": grid_type, "bot_count": bot_count,
                              "trial": trial, "error": str(e)}, queue, all_results)


# Storing the results
def experiment_completed(single_result, queue, all_results):
    grid_type = single_result["grid_type"]
    bot_count = single_result["bot_count"]
    if grid_type not in all_results:
        all_results[grid_type] = {}
    if bot_count not in all_results[grid_type]:
        all_results[grid_type][bot_count] = []

    all_results[grid_type][bot_count].append(single_result)

    # Save results to JSON file after each experiment
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Force garbage collection to clear out any lingering references
    gc.collect()

    # Check if this was the last experiment
    if len(queue) == 0:
        print("All experiments completed, analyzing results...")
        analyse_results()  # Call analyze directly if queue is empty
    else:
        # Start next experiment
        print("Starting next experiment...")
        # Create a small delay to ensure previous resources are released
        time.sleep(0.5)
        # Start the next experiment
        run_next_experiment(queue, all_results)


def main(grid_type, bot_count, trial, callback_function):
    window = tk.Tk()
    canvas = initialise(window)
    cell_size, noOfRowsCols, occupied_cells = createEnvironment(canvas, grid_type)
    delivery_list = populate_delivery_list(occupied_cells[2])

    # Create separate resource managers
    delivery_manager = DeliveryManager(delivery_list)
    cell_manager = CellManager(occupied_cells)

    # Create the agents
    agents = createAgents(canvas, noOfBots=bot_count, cell_size=cell_size, noOfRowsCols=noOfRowsCols,
                          occupied_cells=occupied_cells, grid_choice=grid_type, delivery_list=delivery_list,
                          delivery_manager=delivery_manager, cell_manager=cell_manager)

    # start the timer
    start_time = time.time()

    moveAgents(canvas, agents, noOfRowsCols, occupied_cells, bot_count, delivery_list, trial, callback_function,
               start_time, grid_type)
    window.mainloop()


# Start the experiment
launch_experiment()
