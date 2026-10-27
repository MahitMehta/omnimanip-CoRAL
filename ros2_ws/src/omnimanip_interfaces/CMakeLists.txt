cmake_minimum_required(VERSION 3.8)
project(omnimanip_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TaskObject.msg"
  "msg/Stage.msg"
  "msg/InteractionElement.msg"
  "msg/TaskPlan.msg"
  "msg/ObjectModel.msg"
  "msg/Constraint.msg"
  "srv/Capture.srv"
  "srv/Marks.srv"
  "srv/PartitionTask.srv"
  "srv/RegisterObject.srv"
  "srv/Canonicalize.srv"
  "srv/ProposePoints.srv"
  "srv/ProposeDirections.srv"
  "srv/BuildConstraints.srv"
  "srv/Rrc.srv"
  "srv/SelectGrasp.srv"
  "srv/TargetPose.srv"
  DEPENDENCIES sensor_msgs geometry_msgs
)

ament_package()
