import gtsam
import gtsam.utils.plot as gtsam_plot
import numpy as np
import matplotlib.pyplot as plt

graph = gtsam.NonlinearFactorGraph()

# Add a Gaussian prior on pose x1
priorMean = gtsam.Pose2(0.0, 0.0, 0.0)
priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
graph.push_back(gtsam.PriorFactorPose2(1, priorMean, priorNoise))

# Add two odometry factors
odometry = gtsam.Pose2(2.0, 0.0, 0.0)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])
graph.push_back(gtsam.BetweenFactorPose2(1, 2, odometry, ODOMETRY_NOISE))
graph.push_back(gtsam.BetweenFactorPose2(2, 3, odometry, ODOMETRY_NOISE))

# Print graph
# print(graph)

# Create the initial estimate for each node variable
estimate = gtsam.Values()
estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
estimate.insert(3, gtsam.Pose2(4.1, 0.1, 0.1))

# Print estimate
# print("Initial Estimate:")
# print(estimate)

# Optimize using Levenberg-Marquardt optimization
result = gtsam.LevenbergMarquardtOptimizer(graph, estimate).optimize()

# Print the results
# print("Final Result:")
# print(result)

# parameters = gtsam.LevenbergMarquardtParams()
# parameters.setRelativeErrorTol(1e-5)
# parameters.setMaxIterations(100)
# gtsam.LevenbergMarquardtOptimizer(graph, estimate, parameters).optimize()

marginals = gtsam.Marginals(graph, result)

# Create a figure
fig = plt.figure(figsize=(13, 7), layout='tight')
ax1 = fig.add_subplot(111)

# Plot all three of the Factor Nodes
gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(1), 0.1, marginals.marginalCovariance(1))
gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(2), 0.1, marginals.marginalCovariance(2))
gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(3), 0.1, marginals.marginalCovariance(3))

# Show the plot
plt.show(block=True)