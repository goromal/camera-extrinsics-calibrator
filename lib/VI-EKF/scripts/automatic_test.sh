#!/bin/bash

rosparam load ../params/firefly.yaml vi_ekf_rosbag

BAG_FILE=/home/superjax/rosbag/EuRoC/V1_03_difficult_NED.bag
START=0
DURATION=100

run_test () {
	sed -i "s/^drag_term: .*$/drag_term: $1,/" ../params/gains.yaml
	sed -i "s/^partial_update: .*$/partial_update: $2,/" ../params/gains.yaml
	sed -i "s/^keyframe_reset: .*$/keyframe_reset: $3,/" ../params/gains.yaml

	rosparam load ../params/gains.yaml vi_ekf_rosbag

	rosrun vi_ekf vi_ekf_rosbag -f $BAG_FILE -s $START -u $DURATION
}


run_test false false false
run_test false false true
run_test false true false
run_test false true true
run_test true false false
run_test true false true
run_test true true false
run_test true true true
