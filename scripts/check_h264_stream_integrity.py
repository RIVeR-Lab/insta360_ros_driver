#!/usr/bin/env python3
"""
check_h264_stream_integrity.py

PURPOSE
-------
Determines WHERE H.264 video corruption originates in a ROS 2 camera pipeline:

    camera -> driver -> [CompressedImage topic] -> decoder node -> [Image topic]

Typical symptoms this script helps diagnose:
  * blocky / smeared regions in the decoded image
  * regions that stop updating when something moves through the view
  * image quality that degrades over time

HOW IT WORKS
------------
H.264 sends occasional full keyframes (IDR/I-frames) followed by P-frames that
only encode changes. If ANY compressed message is lost before decoding, every
frame until the next keyframe is corrupted.

This script subscribes to the compressed topic and writes the raw H.264 bytes,
unmodified, to an Annex-B .h264 file. Recording starts at the first message
containing an SPS (sequence parameter set): a stream joined mid-way cannot be
decoded until the parameter sets arrive, and those start-up failures
("non-existing PPS referenced") are join artifacts, not stream damage. It does NO decoding, so it cannot fall
behind the way a decoder node can. It then (optionally) runs ffmpeg over the
recording to count decode errors.

INTERPRETING THE RESULT
-----------------------
  * Recording is CLEAN (no ffmpeg errors, smooth playback in ffplay):
        The stream arriving over ROS is intact. The fault is in the DECODER
        NODE (falling behind, queue overflow, decode errors).

  * Recording has ERRORS / smears in ffplay:
        The stream is already damaged before decoding. The fault is in the
        CAMERA, DRIVER (USB read path), or DDS TRANSPORT.

The summary also reports:
  * receive rate and message sizes
  * keyframe interval (how long corruption can persist after a loss)
  * large arrival gaps (possible dropped messages at this subscriber)

USAGE
-----
  python3 check_h264_stream_integrity.py \
      --topic /insta360/dual_fisheye/image/compressed \
      --duration 30 \
      --output /tmp/stream_check.h264

  # Have someone move through the field of view while it records.
  # Afterwards, view the recording manually with:
  ffplay /tmp/stream_check.h264

OPTIONS
-------
  --topic        CompressedImage topic carrying H.264 (default below)
  --duration     Seconds to record after the first message arrives (default 30)
  --output       Path for the .h264 recording (default /tmp/stream_check.h264)
  --depth        Subscription queue depth (default 100)
  --best-effort  Subscribe BEST_EFFORT instead of RELIABLE (match the publisher;
                 check with: ros2 topic info -v <topic>)
  --expected-fps Camera frame rate, used to flag arrival gaps (default 30)
  --no-ffmpeg    Skip the automatic ffmpeg decode check

REQUIREMENTS
------------
  ROS 2 (rclpy, sensor_msgs). ffmpeg/ffplay optional but recommended:
      sudo apt install ffmpeg
"""

import argparse
import shutil
import statistics
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage

DEFAULT_TOPIC = "/insta360/dual_fisheye/image/compressed"
START_CODE = b"\x00\x00\x01"
NAL_TYPE_IDR = 5
NAL_TYPE_SPS = 7
NAL_TYPE_PPS = 8


def nal_types(data: bytes) -> set:
    """Return the set of H.264 NAL unit types present in an Annex-B payload."""
    types = set()
    i = data.find(START_CODE)
    while i != -1 and i + 3 < len(data):
        types.add(data[i + 3] & 0x1F)
        i = data.find(START_CODE, i + 3)
    return types


class H264StreamRecorder(Node):
    def __init__(self, topic, output, duration, depth, reliable):
        super().__init__("h264_stream_integrity_check")
        self.topic = topic
        self.output_path = output
        self.duration = duration
        self.out = open(output, "wb")

        self.msg_count = 0
        self.byte_count = 0
        self.sizes = []
        self.arrivals = []
        self.idr_times = []
        self.non_h264 = 0
        self.skipped_before_sync = 0
        self.sps_count = 0
        self.pps_count = 0
        self.last_wait_warn = time.monotonic()
        self.start = None
        self.done = False

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(ReliabilityPolicy.RELIABLE if reliable
                         else ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(CompressedImage, topic, self.on_msg, qos)
        self.create_timer(0.5, self.check_done)

        self.get_logger().info(
            f"Waiting for messages on {topic} "
            f"({'RELIABLE' if reliable else 'BEST_EFFORT'}, depth {depth})...")

    def on_msg(self, msg: CompressedImage):
        now = time.monotonic()
        if msg.format.lower() != "h264":
            self.non_h264 += 1
            return

        data = bytes(msg.data)
        types = nal_types(data)

        if self.start is None:
            # Joined mid-stream: nothing is decodable until an SPS arrives.
            # Skip until then so join artifacts are not reported as damage.
            if NAL_TYPE_SPS not in types:
                self.skipped_before_sync += 1
                return
            self.start = now
            self.get_logger().info(
                f"SPS received after skipping {self.skipped_before_sync} message(s). "
                f"Recording for {self.duration:.0f} s -> {self.output_path}")

        self.out.write(data)

        self.msg_count += 1
        self.byte_count += len(data)
        self.sizes.append(len(data))
        self.arrivals.append(now)
        if NAL_TYPE_IDR in types:
            self.idr_times.append(now)
        if NAL_TYPE_SPS in types:
            self.sps_count += 1
        if NAL_TYPE_PPS in types:
            self.pps_count += 1

    def check_done(self):
        now = time.monotonic()
        if self.start is None and self.skipped_before_sync > 0 and now - self.last_wait_warn > 5.0:
            self.last_wait_warn = now
            self.get_logger().warn(
                f"{self.skipped_before_sync} messages received but no SPS yet. If this "
                "persists, the camera only sends parameter sets at stream start: "
                "restart the driver while this script is running.")
        if self.start is not None and now - self.start >= self.duration:
            self.done = True

    def close(self):
        self.out.flush()
        self.out.close()


def print_summary(rec: H264StreamRecorder, expected_fps: float):
    print("\n================ STREAM SUMMARY ================")
    if rec.msg_count == 0:
        print("No H.264 messages received.")
        if rec.non_h264:
            print(f"  {rec.non_h264} messages had a format other than 'h264'.")
        if rec.skipped_before_sync:
            print(f"  {rec.skipped_before_sync} H.264 messages arrived but none contained an SPS.")
            print("  The camera may only send parameter sets at stream start; restart the")
            print("  driver while this script is running.")
            return False
        print("Check the topic name, namespace, and that the driver is running:")
        print("  ros2 topic list | grep compressed")
        return False

    elapsed = rec.arrivals[-1] - rec.arrivals[0] if rec.msg_count > 1 else 0.0
    rate = (rec.msg_count - 1) / elapsed if elapsed > 0 else 0.0
    print(f"Messages:        {rec.msg_count} over {elapsed:.1f} s  ({rate:.1f} msg/s)")
    print(f"Data:            {rec.byte_count / 1e6:.2f} MB  "
          f"({rec.byte_count * 8 / 1e6 / max(elapsed, 1e-9):.2f} Mbps)")
    print(f"Message size:    mean {statistics.mean(rec.sizes) / 1e3:.1f} KB, "
          f"min {min(rec.sizes) / 1e3:.1f} KB, max {max(rec.sizes) / 1e3:.1f} KB")

    if len(rec.idr_times) >= 2:
        intervals = [b - a for a, b in zip(rec.idr_times, rec.idr_times[1:])]
        print(f"Keyframes:       {len(rec.idr_times)}  "
              f"(interval ~{statistics.mean(intervals):.2f} s -> a single loss can "
              f"corrupt the image for up to this long)")
    elif len(rec.idr_times) == 1:
        print("Keyframes:       1 (recording too short to measure interval)")
    else:
        print("Keyframes:       none detected (stream may use non-IDR I-frames, "
              "or keyframes are being lost)")

    print(f"Sync:            skipped {rec.skipped_before_sync} message(s) before the first SPS")
    if rec.sps_count > 1 and rec.sps_count >= len(rec.idr_times) - 1:
        note = "repeated with keyframes (a decoder can join mid-stream)"
    elif rec.sps_count <= 1:
        note = "sent once only (a decoder started after the driver may never sync)"
    else:
        note = "repeated, but less often than keyframes"
    print(f"Param sets:      SPS in {rec.sps_count} msg(s), PPS in {rec.pps_count} msg(s) -> {note}")

    if rec.msg_count > 2:
        gaps = [b - a for a, b in zip(rec.arrivals, rec.arrivals[1:])]
        frame_period = 1.0 / expected_fps
        big = [g for g in gaps if g > 3 * frame_period]
        print(f"Arrival gaps:    max {max(gaps) * 1e3:.0f} ms, "
              f"{len(big)} gaps > {3 * frame_period * 1e3:.0f} ms "
              f"(3x the expected {frame_period * 1e3:.0f} ms frame period)")
        if big:
            print("                 Large gaps can indicate dropped messages or a "
                  "stalled driver.")
    return True


def run_ffmpeg_check(path: str):
    print("\n=============== FFMPEG DECODE CHECK ===============")
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; skipping. Install with: sudo apt install ffmpeg")
        print(f"Manual check: ffmpeg -v error -i {path} -f null -")
        return

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-f", "h264", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True)
    errors = [line for line in result.stderr.splitlines() if line.strip()]

    if not errors:
        print("RESULT: CLEAN - no decode errors in the recorded stream.")
        print("  -> The compressed stream arriving over ROS is intact.")
        print("  -> If the decoded image still looks corrupted, the fault is in")
        print("     the DECODER NODE (falling behind, queue overflow, decode errors).")
    else:
        print(f"RESULT: {len(errors)} decode error line(s) in the recorded stream.")
        print("  -> The stream is damaged BEFORE decoding.")
        print("  -> Investigate the camera settings, driver USB read path, or DDS transport.")
        print("  First errors:")
        for line in errors[:10]:
            print(f"    {line}")

    print(f"\nVisual check: ffplay {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Record a ROS 2 H.264 CompressedImage stream and check its integrity.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", default="/tmp/stream_check.h264")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--best-effort", action="store_true")
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--no-ffmpeg", action="store_true")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    rec = H264StreamRecorder(args.topic, args.output, args.duration,
                             args.depth, reliable=not args.best_effort)
    try:
        while rclpy.ok() and not rec.done:
            rclpy.spin_once(rec, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\nInterrupted; summarizing what was recorded.")
    finally:
        rec.close()
        got_data = print_summary(rec, args.expected_fps)
        rec.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if got_data and not args.no_ffmpeg:
        run_ffmpeg_check(args.output)


if __name__ == "__main__":
    main()
