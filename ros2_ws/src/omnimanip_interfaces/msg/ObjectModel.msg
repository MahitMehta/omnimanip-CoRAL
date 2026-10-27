# A registered, canonicalized object (stages 4-9).
string name
int32 mark_id
sensor_msgs/Image mask          # mono8, >0 = object
string mesh_path                # Any6D's textured final mesh (body frame) on the desktop's filesystem
geometry_msgs/Pose cam_t_obj    # body frame in the camera's optical frame
float64[9] r_body_from_canon    # row-major; columns: canonical x = front, y, z = up, in the body frame
geometry_msgs/Pose cam_t_can    # canonical frame in the camera's optical frame
