#!/usr/bin/env bash

TURTEL_BOT_IP="100.88.171.40"
ECHO_IP="100.99.202.82"

NUM_RUNS=1000
SLEEP_BETWEEN_RUNS=3
ROS_ENV="source /opt/ros/jazzy/setup.bash && source ~/turtlebot3_ws/install/setup.bash && export ROS_DOMAIN_ID=30 && export LDS_MODEL=LDS-01 && export TURTLEBOT3_MODEL=burger"

for i in $(seq 1 "$NUM_RUNS"); do
    id=$(printf "_id_%06d_num_%05d_" "$(( RANDOM % 1000000 ))" "$i" ) 

    echo "Starting recording: $id"
    ssh klyx@"$TURTEL_BOT_IP" "$ROS_ENV && ros2 topic pub --once /record_command std_msgs/msg/String \"{data: 'start:$id'}\""

    echo "Running data script on second machine"
    ssh root@"$ECHO_IP" "cd /home/klyx/git/P10/record_data_set && sudo /home/klyx/git/P10/.venv/bin/python3 /home/klyx/git/P10/record_data_set/kom_ny_skal_sudo.py $id"

    echo "Stopping recording: $id"
    ssh klyx@"$TURTEL_BOT_IP" "$ROS_ENV && ros2 topic pub --once /record_command std_msgs/msg/String \"{data: 'stop'}\""

    echo "Sleeping for $SLEEP_BETWEEN_RUNS seconds"
    sleep "$SLEEP_BETWEEN_RUNS"
done
