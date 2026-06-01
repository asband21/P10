#!/usr/bin/env python3

import math
import time
import rclpy
import random
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from rclpy.qos import qos_profile_sensor_data


class LidarToCartesian(Node):
    def __init__(self):
        super().__init__("lidar_to_cartesian")

        self.subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data,)
        self.record_sub = self.create_subscription(String, "/record_command", self.record_command_callback, 10,)
        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.stait = "frem"
        self.id = ""
        self.spin = 0

    def record_command_callback(self, msg):

        command = msg.data
        if command.startswith("start:"):
            prefix  = command.split(":", 1)[1]
            self.id = command.split(":", 2)[1]
            self.stait = "recording"
            self.record_prefix = prefix

        elif command == "stop":
            self.stait = "frem"

    def write_lidare_to_file(self, msg):
        p = f"lidar_data/time_s_{msg.header.stamp.sec}_id_{self.id}.csv"
        print(f"\rsavet at {p}")
        f = open(p, "w", newline="")
        for r in msg.ranges:
            print(r,file=f)
        f.close()

    def scan_callback(self, msg):
        points = []

        angle = msg.angle_min

        num = 0
        for r in msg.ranges:
            if math.isfinite(r):
                if 0.1 < r and r < 0.20:
                    num = num + 1
                    #print(f"distfres 0.1 < {r} < 0.20")

        if num > 3:
            self.stait = "stop"

        if self.spin == 0:
            print(f"\r--- stait:{self.stait} ---", end="")
            self.spin = 1

        elif self.spin == 1:
            print(f"\r||| stait:{self.stait} |||", end="")
            self.spin = 0

        if self.stait == "waite":
            pass
        elif self.stait == "frem":
            self.publish_cmd_vel_frem(0.14, 0.0)

        elif self.stait == "recording":
            self.publish_cmd_vel_frem(0.0, 0.0)
            self.write_lidare_to_file(msg)
            #save_data(prefix, data)
            self.stait = "waite"

        else:
            self.publish_cmd_vel_frem(-0.1, 0.0)
            time.sleep(2.0)
            self.publish_cmd_vel_frem(-0.0, 1.0)
            time_tune = random.uniform(0.0, 3.5)
            time.sleep(time_tune)
            self.publish_cmd_vel_frem(0.1, 0.0)
            time.sleep(0.4)
            self.stait = "frem"

    def publish_cmd_vel_frem(self, speed, angel):
        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.twist.linear.x = speed
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = angel

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = LidarToCartesian()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


from geometry_msgs.msg import TwistStamped


if __name__ == "__main__":
    main()
