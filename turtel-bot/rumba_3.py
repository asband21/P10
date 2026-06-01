import math
import time
import rclpy
import random
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped


class LidarToCartesian(Node):
    def __init__(self): super().__init__("lidar_to_cartesian")
        self.subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data, )
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel' ,10)
        self.stait = "frem"
    
    def scan_callback(self, msg):
        points = []
        angle = msg.angle_min
        self.stait = "frem"
        for r in msg.ranges:
            if math.isfinite(r):
                if 0.1 < r < 0.2:
                    #print(r)
                    self.stait = "stop"
        if self.stait == "frem":
            print("frem")
            self.publish_cmd_vel_frem(0.1, 0.0)
        else:
            print("stop")
            self.publish_cmd_vel_frem(-0.1, 0.0)
            time.sleep(0.2)
            self.publish_cmd_vel_frem(-0.0, 1.0)
            time_tune = random.uniform(0.0, 1.1)
            print(time_tune) time.sleep(time_tune)
            print("---- LiDAR scan ----")
        
        def publish_cmd_vel_frem(self, speed, angel): msg = TwistStamped() msg.header.stamp = self.get_clock().now().to_msg() msg.header.frame_id = 'base_link' msg.twist.linear.x = speed msg.twist.linear.y = 0.0 msg.twist.linear.z = 0.0 msg.twist.angular.x = 0.0 msg.twist.angular.y = 0.0 msg.twist.angular.z = angel self.publisher.publish(msg) def main(args=None): rclpy.init(args=args) node = LidarToCartesian() try: rclpy.spin(node) except KeyboardInterrupt: pass finally: node.destroy_node() if rclpy.ok(): rclpy.shutdown() from geometry_msgs.msg import TwistStamped if __name__ == "__main__": main()
