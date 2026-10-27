# Spatial constraint C (paper Eq. 1, Fig. 2): active axis aligned with passive axis; active point d along the
# passive axis from the passive point. Points and axes are in each object's canonical frame.
string rank                     # "1", "2", ... ; refined candidates "1.3"
geometry_msgs/Point active_point
geometry_msgs/Vector3 active_axis
geometry_msgs/Point passive_point
geometry_msgs/Vector3 passive_axis
float64 distance_m              # d, 0 until RRC sets it
