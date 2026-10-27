# Albus's Ros2Bridge image (built from the pinned third_party/ur5e-manip-hardware) plus what talking to the robot
# laptop needs, as his newer Dockerfile.ros2 adds: CycloneDDS (the laptop's ros2-essentials stack runs it),
# control_msgs (FollowJointTrajectory, GripperCommand) and, for offline dry runs on mock hardware, the UR driver.
FROM manip-ros2-bridge:latest
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-control-msgs \
    ros-humble-ur-robot-driver \
    && rm -rf /var/lib/apt/lists/*
