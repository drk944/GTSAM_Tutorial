from gtsam import *
import gtsam.utils.plot as gtsam_plot

import math
from matplotlib import pyplot as plt

# Create graph container and add factors to it
graph = NonlinearFactorGraph()

# Create keys for variables
i1 = symbol('x', 1)
i2 = symbol('x', 2)
i3 = symbol('x', 3)
j1 = symbol('l', 1)
j2 = symbol('l', 2)

# Add prior
priorMean = Pose2(0.0, 0.0, 0.0)  # prior at origin
priorNoise = noiseModel.Diagonal.Sigmas((0.3, 0.3, 0.1))
graph.push_back(PriorFactorPose2(i1, priorMean, priorNoise))

# Add odometry
odometry = Pose2(2.0, 0.0, 0.0)
ODOMETRY_NOISE = noiseModel.Diagonal.Sigmas((0.2, 0.2, 0.1))
graph.push_back(BetweenFactorPose2(i1, i2, odometry, ODOMETRY_NOISE))
graph.push_back(BetweenFactorPose2(i2, i3, odometry, ODOMETRY_NOISE))

# Add bearing/range measurement factors
degrees = math.pi / 180.0
brNoise = noiseModel.Diagonal.Sigmas((0.00001, 0.00002))
graph.push_back(BearingRangeFactor2D(i1, j1, Rot2(45 * degrees), math.sqrt(8), brNoise))
graph.push_back(BearingRangeFactor2D(i2, j1, Rot2(90 * degrees), 2, brNoise))
graph.push_back(BearingRangeFactor2D(i3, j2, Rot2(90 * degrees), 2, brNoise))

# Create the initial estimate to the solution
estimate = gtsam.Values()
estimate.insert(i1, Pose2(0.5, 0.0, 0.2))
estimate.insert(i2, Pose2(2.3, 0.1, -0.2))
estimate.insert(i3, Pose2(4.1, 0.1, 0.1))
estimate.insert(j1, Point2(2, 2))
estimate.insert(j2, Point2(4, 2))

# Optimize using Levenberg-Marquardt optimization
result = gtsam.LevenbergMarquardtOptimizer(graph, estimate).optimize()

# Calculate and print marginal covariances for all poses
marginals = Marginals(graph, result)
# Create a figure
fig = plt.figure(figsize=(13, 7), layout='tight')
ax1 = fig.add_subplot(111)

# Plot all three of the Factor Nodes
gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(i1), 0.1, marginals.marginalCovariance(i1))
gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(i2), 0.1, marginals.marginalCovariance(i2))
gtsam_plot.plot_pose2_on_axes(ax1, result.atPose2(i3), 0.1, marginals.marginalCovariance(i3))

# Show the plot
plt.show(block=True)