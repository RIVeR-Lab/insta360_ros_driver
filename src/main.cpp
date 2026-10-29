#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>

#include <camera/camera.h>
#include <camera/photography_settings.h>
#include <camera/device_discovery.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/imu.hpp"

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    std::string frame_id_;
    std::string imu_frame_id_;

    // ---------------- IMU clock mapping ----------------
    // The SDK delivers gyro/accel samples in batches. Stamping each sample with
    // now() gives a whole batch the same stamp (dt = 0 for the filter, then one
    // large jump). Instead, each sample is stamped with its own camera
    // timestamp mapped into ROS time:
    //     ros_ns = camera_ts * ns_per_tick_ + offset_ns_
    // The timestamp unit is not documented, so it is measured over the first
    // second by comparing camera-clock elapsed time against ROS-clock elapsed
    // time and rounding to the nearest of s / ms / us / ns.
    static constexpr double kCalibrationSeconds = 1.0;
    static constexpr double kDriftGain = 1e-4;  // slow tracking of clock drift
    std::mutex imu_mutex_;
    bool imu_clock_ready_ = false;
    bool imu_calibrating_ = false;
    int64_t calib_cam0_ = 0;
    int64_t calib_wall0_ns_ = 0;
    double ns_per_tick_ = 0.0;
    double offset_ns_ = 0.0;
    int64_t last_cam_ts_ = std::numeric_limits<int64_t>::min();
    uint64_t skipped_samples_ = 0;

    void ResetImuClock() {
        imu_clock_ready_ = false;
        imu_calibrating_ = false;
        last_cam_ts_ = std::numeric_limits<int64_t>::min();
    }

    // Returns false while the clock mapping is still being measured.
    bool UpdateImuClock(const std::vector<ins_camera::GyroData>& data, int64_t now_ns) {
        if (!imu_clock_ready_) {
            if (!imu_calibrating_) {
                calib_cam0_ = data.back().timestamp;
                calib_wall0_ns_ = now_ns;
                imu_calibrating_ = true;
                RCLCPP_INFO(node_->get_logger(), "Measuring IMU timestamp clock (%.0f s)...",
                            kCalibrationSeconds);
                return false;
            }
            const double wall_dt_s = (now_ns - calib_wall0_ns_) * 1e-9;
            if (wall_dt_s < kCalibrationSeconds) {
                return false;
            }
            const double cam_dt = static_cast<double>(data.back().timestamp - calib_cam0_);
            if (cam_dt <= 0.0) {
                RCLCPP_WARN(node_->get_logger(),
                            "IMU timestamps did not advance during calibration; retrying");
                imu_calibrating_ = false;
                return false;
            }

            const double ticks_per_s = cam_dt / wall_dt_s;
            double exponent = std::round(std::log10(ticks_per_s) / 3.0) * 3.0;
            exponent = std::min(9.0, std::max(0.0, exponent));
            const double nominal = std::pow(10.0, exponent);
            ns_per_tick_ = 1e9 / nominal;
            offset_ns_ = static_cast<double>(now_ns) - data.back().timestamp * ns_per_tick_;
            imu_clock_ready_ = true;

            const char* unit = exponent == 0.0 ? "seconds" : exponent == 3.0 ? "milliseconds"
                             : exponent == 6.0 ? "microseconds" : "nanoseconds";
            RCLCPP_INFO(node_->get_logger(),
                        "IMU clock synced: camera timestamps are in %s (measured %.1f ticks/s)",
                        unit, ticks_per_s);
            if (std::fabs(ticks_per_s / nominal - 1.0) > 0.05) {
                RCLCPP_WARN(node_->get_logger(),
                            "Measured IMU clock rate is more than 5%% off the nearest unit; "
                            "check the timestamps");
            }
            return true;
        }

        // candidate = true offset + transport latency (latency >= 0). The
        // smallest candidate is the best estimate, so adopt lower values at
        // once and follow higher ones only slowly (camera/host clock drift).
        // This also guarantees stamps never lie in the future.
        const double candidate = static_cast<double>(now_ns) - data.back().timestamp * ns_per_tick_;
        if (candidate < offset_ns_) {
            offset_ns_ = candidate;
        } else {
            offset_ns_ += kDriftGain * (candidate - offset_ns_);
        }
        return true;
    }

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {
        // TF frames stamped on outgoing messages. Defaults match the frames
        // published by the bringup static transforms.
        frame_id_ = node_->declare_parameter<std::string>("frame_id", "insta_sensor");
        imu_frame_id_ = node_->declare_parameter<std::string>("imu_frame_id", "insta_imu");

        // Publisher for the compressed H.264 video stream
        compressed_pub_ = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            "dual_fisheye/image/compressed",
            rclcpp::QoS(10)
        );

        // Publisher for IMU data. RELIABLE is compatible with both reliable and
        // best-effort subscribers; depth covers a few batches of samples.
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::QoS(200).reliable());
        RCLCPP_INFO(node_->get_logger(), "Publisher for compressed images and IMU created.");
        RCLCPP_INFO(node_->get_logger(), "Frame ID: %s | IMU frame ID: %s",
                    frame_id_.c_str(), imu_frame_id_.c_str());
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
        // We only care about the main video stream (index 0)
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();

            // Set the header
            msg->header.stamp = node_->get_clock()->now();
            msg->header.frame_id = frame_id_;

            // Set the format to H.264
            // The subscriber will need to know this to select the correct decoder.
            msg->format = "h264";

            // Copy the compressed video data directly into the message
            msg->data.assign(data, data + size);

            compressed_pub_->publish(std::move(msg));
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        if (data.empty() || !imu_pub_) {
            return;
        }
        std::lock_guard<std::mutex> lock(imu_mutex_);

        const auto clock_type = node_->get_clock()->get_clock_type();
        const int64_t now_ns = node_->get_clock()->now().nanoseconds();

        // Camera clock jumped backwards (e.g. camera reset): re-measure.
        if (imu_clock_ready_ && last_cam_ts_ != std::numeric_limits<int64_t>::min() &&
            (last_cam_ts_ - data.front().timestamp) * ns_per_tick_ > 1e9) {
            RCLCPP_WARN(node_->get_logger(), "IMU timestamps jumped backwards; re-syncing clock");
            ResetImuClock();
        }

        if (!UpdateImuClock(data, now_ns)) {
            return;  // still measuring; drop these samples
        }

        for (const auto& gyro : data) {
            // Samples must be strictly increasing for the filter's dt.
            if (gyro.timestamp <= last_cam_ts_) {
                ++skipped_samples_;
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                                     "Skipped %lu IMU samples with non-increasing timestamps",
                                     static_cast<unsigned long>(skipped_samples_));
                continue;
            }
            last_cam_ts_ = gyro.timestamp;

            const auto stamp_ns = static_cast<int64_t>(gyro.timestamp * ns_per_tick_ + offset_ns_);

            auto msg = std::make_unique<sensor_msgs::msg::Imu>();
            msg->header.stamp = rclcpp::Time(stamp_ns, clock_type);
            msg->header.frame_id = imu_frame_id_;
            msg->angular_velocity.x = gyro.gx;
            msg->angular_velocity.y = gyro.gy;
            msg->angular_velocity.z = gyro.gz;
            
            msg->linear_acceleration.x = gyro.ax * 9.80665;
            msg->linear_acceleration.y = gyro.ay * 9.80665;
            msg->linear_acceleration.z = gyro.az * 9.80665;

            msg->orientation.x = 0.0;
            msg->orientation.y = 0.0;
            msg->orientation.z = 0.0;
            msg->orientation.w = 1.0; // Neutral orientation
            msg->orientation_covariance[0] = -1.0; // No orientation data available

            for (int i = 0; i < 9; i++)
            {
                msg->angular_velocity_covariance[i] = 0;
                msg->linear_acceleration_covariance[i] = 0;
            }
            imu_pub_->publish(std::move(msg));
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<rclcpp::Node> node_;

public:
    CameraWrapper(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {}

    ~CameraWrapper() {
        if (cam) {
            cam->Close();
        }
    }

    int run_camera() {
        ins_camera::DeviceDiscovery discovery;
        auto list = discovery.GetAvailableDevices();
        if (list.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "No available camera devices found.");
            return -1;
        }

        cam = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam->Open()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open camera.");
            return -1;
        }
        RCLCPP_INFO(node_->get_logger(), "Camera opened successfully.");
        discovery.FreeDeviceDescriptors(list);

        std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>(node_);
        cam->SetStreamDelegate(delegate);

        auto start = time(NULL);

        uint64_t utc_time = static_cast<uint64_t>(start);
        uint32_t offset_time = 0; //no offset from UTC

        cam->SyncLocalTimeToCamera(utc_time,offset_time);       
        ins_camera::LiveStreamParam param;
        param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30; //Change this line to edit the resolution
        //Possible resolutions (results may vary per model) are:
        //RES_3840_1920P30
        //RES_2560_1280P30
        //RES_1152_1152P30 (this will give 2304 x 1152 at 30 FPS)
        //RES_1920_960P30  
        //RES_2880_1440
        param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
        param.video_bitrate = 1024 * 1024 / 2;
        param.enable_audio = false;
        param.using_lrv = false;

        if (!cam->StartLiveStreaming(param)) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to start live streaming.");
            return -1;
        }
        
        RCLCPP_INFO(node_->get_logger(), "Live streaming started.");
        return 0;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("insta_publisher");
    
    CameraWrapper camera(node);
    if (camera.run_camera() != 0) {
        rclcpp::shutdown();
        return -1;
    }
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}