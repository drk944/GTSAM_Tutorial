import numpy as np 
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
# from tqdm import tqdm
import time

import gtsam
import gtsam.utils.plot as gtsam_plot

# robot association is an array where each row corresponds a bar code value to a landmark
robot_association = np.loadtxt('datasets/MRCLAM_Dataset1/Barcodes.dat', skiprows=4, dtype='int')
# create empy dictionary
robot_association_dict = {}
# iterate through each row
for row in robot_association:
    # add the key value pair to the dictionary
    robot_association_dict[row[1]] = int(row[0])

# Landmark ground truth (where are the landmarks in the world)
landmark_gt = np.loadtxt('datasets/MRCLAM_Dataset1/Landmark_Groundtruth.dat', skiprows=4, dtype='float')
landmark_dict = {}
for row in landmark_gt:
    landmark_dict[int(row[0])] = row[1:3]

# Robot 1 ground truth
robot_1_gt = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Groundtruth.dat', skiprows=4, dtype='float')
# Robot measurements. Each row has a timestep, a landmark ID, and a measurement in the form of range (m) and bearing (rad)
robot1_measurements = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Measurement.dat', skiprows=4, dtype='float')

# remove rows in robot1 measurement that correspond to other robots and not landmarks
# This data set was designed for cooperative SLAM but we only care about the single agent Robot 1
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 14]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 41]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 32]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 23]
# 2 of the landmarks are switched in the data set so I just removed them
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 61] # 17 is dumb
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 18] # 11 is also dumb

# get only the unique lines based on the first column
unique_timesteps, idx = np.unique(robot1_measurements[:, 0], return_index=True)
# extract rows with unique timesteps
robot1_measurements = robot1_measurements[idx]

# Robot odometry is recorded as: timestep, forward velocity (m/s), angular velocity (rad/s)
robot1_odometry = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Odometry.dat', skiprows=4, dtype='float')

# Function to convert odometry data to dead reckoning positional data
def dead_reckoning(robot1_odometry, initial_x, initial_y, initial_theta):
    x = initial_x # We need to know where the robot is starting from
    y = initial_y
    theta = initial_theta

    positions = np.empty((robot1_odometry.shape[0], 3))
    # Add the initial position to the array
    positions[0] = [x, y, theta]

    for i in range(1, robot1_odometry.shape[0]):
        time_step = robot1_odometry[i,0] - robot1_odometry[i-1,0]
        velocity = robot1_odometry[i,1]
        angular_velocity = robot1_odometry[i,2]
        x += velocity * math.cos(theta) * time_step
        y += velocity * math.sin(theta) * time_step
        theta += angular_velocity * time_step
        positions[i] = [x, y, theta]
    return positions

graph_keys = []
num_iterations_to_run = 5000

def run(robot_odometry, initial_x, initial_y, initial_theta, robot_measurements):    
    graph = gtsam.NonlinearFactorGraph()
    priorMean = gtsam.Pose2(initial_x, initial_y, initial_theta)
    
    # Noise parameters
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0025, 0.0025, 0.00675]))
    sigma_range = 0.9125 #5125 #5 # 0.125
    sigma_bearing = 0.9025 #525 #125 #0.025 # 1.5 degree sigma, converted to radians

    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0005]))
    graph.push_back(gtsam.PriorFactorPose2(1, priorMean, priorNoise))
    graph_keys.append(1)
    
    # We need to find when the robot measurements and odometry line up
    measurement_starting_idx = 0
    starting_timestep = robot_odometry[0,0]
    for i in range(robot_measurements.shape[0]):
        if robot_measurements[i+1,0] > starting_timestep:
            measurement_starting_idx = i
            break
    robot_measurements = robot_measurements[measurement_starting_idx:-1] # Robot measurements are now in sync with odometry data

    positions = np.empty((robot1_odometry.shape[0], 3))
    # Add the initial position to the array
    positions[0] = np.array([initial_x, initial_y, initial_theta])
    measurement_idx = 0
    time_stamp_sensor = []

    seen_landmarks = set()
    # Time to actually iterate through the data
    start_time = time.time()
    for i in range(1, robot_odometry.shape[0]):
        if i > num_iterations_to_run:
            break

        dt = robot_odometry[i,0] - robot_odometry[i-1,0]

        vt = robot_odometry[i][1]
        wt = robot_odometry[i][2]

        motion = gtsam.Pose2(vt*dt*math.cos(wt*dt/2), vt*dt*math.sin(wt*dt/2), wt*dt)

        graph.push_back(gtsam.BetweenFactorPose2(i, i+1, motion, ODOMETRY_NOISE)) 
        graph_keys.append(i+1)
        
        # Check to see if we have a sensor reading at this time step
        reading = None
        while(robot_odometry[i][0] >= robot_measurements[measurement_idx][0]):
            # Add measurement
            angle = robot_measurements[measurement_idx][2]
            m_range = robot_measurements[measurement_idx][3]
            
            measurement_noise = gtsam.noiseModel.Diagonal.Sigmas((sigma_bearing, sigma_range))
            
            landmark_id = int(robot_measurements[measurement_idx][1])
            landmark_id = robot_association_dict[landmark_id]
            seen_landmarks.add(landmark_id)
            gt_landmark = gtsam.symbol('l', landmark_id)

            graph.add(gtsam.BearingRangeFactor2D(i, gt_landmark, gtsam.Rot2(angle), m_range, measurement_noise))
            measurement_idx += 1
            reading = robot_measurements[measurement_idx][1:]
            if measurement_idx >= robot_measurements.shape[0]-5: # The end of the data set gets a little weird.
                print("Computation Time:", time.time() - start_time)
                return graph, graph_keys, seen_landmarks

    return graph, graph_keys, seen_landmarks

robot1_odometry = robot1_odometry[100:-1]

# find the starting GT position
gt_index = 0
while robot_1_gt[gt_index,0] < robot1_odometry[0,0]:
    gt_index += 1

index = num_iterations_to_run
gt_stop = gt_index
while robot_1_gt[gt_stop,0] < robot1_odometry[index,0]:
    gt_stop += 1

graph, poses, seen_landmarks = run(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3], robot1_measurements)

positions_dr = dead_reckoning(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3])

estimate = gtsam.Values()
for i in range(0, len(poses)):
    # Using dead reckoning as an initial estimate
    estimate.insert(i+1, gtsam.Pose2(positions_dr[i,0], positions_dr[i,1], positions_dr[i,2]))
    
    # Using the ground truth as an initial estimate
    # estimate.insert(i+1, gtsam.Pose2(robot_1_gt[gt_index+i+1,1], robot_1_gt[gt_index+i+1,2], robot_1_gt[gt_index+i+1,3]))


landmark_keys = []
for i in range(0, len(landmark_gt)):
    landmark_id = int(landmark_gt[i,0])
    if landmark_id not in seen_landmarks:
        continue
    gt_landmark = gtsam.symbol('l', landmark_id)
    landmark_keys.append(gt_landmark)
    estimate.insert(gt_landmark, gtsam.Point2(landmark_gt[i,1], landmark_gt[i,2]))

# Optimize using Levenberg-Marquardt optimization
result = gtsam.LevenbergMarquardtOptimizer(graph, estimate).optimize()

# Recover the Marginals
marginals = gtsam.Marginals(graph, result)

# GTsam plotting
fig = plt.figure(layout='tight')
ax1 = fig.add_subplot(111, aspect='equal')

i = 0
for key in graph_keys:
    i += 1
    if i % 50 == 0:
        gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(key), 0.05, marginals.marginalCovariance(key))

# Currently can't plot landmarks, don't know correct function call
# for key in landmark_keys:
#     gtsam_plot.plot_pose2_on_axes(ax1, result.atPoint2(key), 0.1, marginals.marginalCovariance(key))

## Plotting
# plot the ground truth robot trajectory
ax1.plot(robot_1_gt[gt_index:gt_stop,1], robot_1_gt[gt_index:gt_stop,2], 'b-', label='Ground Truth')

# # plot dead reckoning
positions_dr = dead_reckoning(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3])
ax1.plot(positions_dr[:index,0], positions_dr[:index,1], 'y-', label='Dead Reckoning')

ax1.legend(loc='upper right')
plt.show()