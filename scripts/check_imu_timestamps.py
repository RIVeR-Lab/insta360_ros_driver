#!/usr/bin/env python3
"""
check_imu_timestamps.py

PURPOSE
-------
Checks the HEADER STAMPS of a sensor_msgs/Imu topic, which `ros2 topic hz`
cannot do (hz measures arrival times, and batched IMU delivery makes arrivals
bunch together regardless of how the messages are stamped).

A filter such as imu_filter_madgwick computes dt from consecutive header
stamps, so it needs stamps that are strictly increasing and evenly spaced.

WHAT IT REPORTS
---------------
  * stamp rate (from header stamps, not arrival times)
  * spacing between consecutive stamps: min / mean / max / std dev
  * count of non-increasing stamps (dt <= 0)  -> should be 0
  * stamp-vs-arrival lag: how far each stamp is behind the time it arrived.
    Should be small and steady; a lag that grows over a long run means the
    camera-to-ROS clock mapping is drifting. A negative lag means stamps are
    in the future, which should never happen.

USAGE
-----
  python3 check_imu_timestamps.py --topic /insta360/imu/data_raw --duration 10
  python3 check_imu_timestamps.py --topic /insta360/imu/data --duration 10

PASS CRITERIA
-------------
  * non-increasing stamps: 0
  * min and max spacing close to the mean (steady spacing, no bursts of 0)
  * lag positive, small, and stable between runs
"""

import argparse
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu


class ImuStampChecker(Node):
    def __init__(self, topic, depth, reliable):
        super().__init__("imu_timestamp_check")
        self.stamps = []
        self.lags = []
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(ReliabilityPolicy.RELIABLE if reliable
                         else ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(Imu, topic, self.on_msg, qos)
        self.get_logger().info(f"Listening on {topic}...")

    def on_msg(self, msg: Imu):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        now_ns = self.get_clock().now().nanoseconds
        self.stamps.append(stamp_ns)
        self.lags.append((now_ns - stamp_ns) * 1e-6)  # ms


def main():
    parser = argparse.ArgumentParser(description="Check IMU header-stamp spacing.")
    parser.add_argument("--topic", default="/insta360/imu/data_raw")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--best-effort", action="store_true")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = ImuStampChecker(args.topic, args.depth, reliable=not args.best_effort)
    start = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.stamps and start is None:
                start = time.monotonic()
            if start is not None and time.monotonic() - start >= args.duration:
                break
    except KeyboardInterrupt:
        pass

    stamps, lags = node.stamps, node.lags
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    print("\n============ IMU TIMESTAMP SUMMARY ============")
    if len(stamps) < 3:
        print(f"Only {len(stamps)} message(s) received; check the topic and driver.")
        return

    dts = [(b - a) * 1e-6 for a, b in zip(stamps, stamps[1:])]  # ms
    bad = sum(1 for d in dts if d <= 0)
    span_s = (stamps[-1] - stamps[0]) * 1e-9
    rate = (len(stamps) - 1) / span_s if span_s > 0 else 0.0

    print(f"Messages:             {len(stamps)}")
    print(f"Stamp rate:           {rate:.1f} Hz (from header stamps)")
    print(f"Spacing (ms):         min {min(dts):.3f}  mean {statistics.mean(dts):.3f}  "
          f"max {max(dts):.3f}  std {statistics.pstdev(dts):.3f}")
    print(f"Non-increasing:       {bad}" + ("  <-- FAIL" if bad else "  (OK)"))
    print(f"Lag stamp->arrival:   min {min(lags):.1f} ms  mean {statistics.mean(lags):.1f} ms  "
          f"max {max(lags):.1f} ms")
    if min(lags) < 0:
        print("                      Negative lag: some stamps are in the future  <-- FAIL")


if __name__ == "__main__":
    main()
