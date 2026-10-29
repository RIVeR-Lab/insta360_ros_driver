#ifndef EQUIRECTANGULAR_HPP
#define EQUIRECTANGULAR_HPP

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class EquirectangularNode : public rclcpp::Node
{
public:
    EquirectangularNode();
    ~EquirectangularNode() override = default;

private:
    // Callbacks
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    rcl_interfaces::msg::SetParametersResult parametersCallback(
        const std::vector<rclcpp::Parameter>& parameters);

    // Setup
    void loadParameters();
    void logParameters();
    void updateCameraParameters();
    void initMapping(int src_width, int src_height, int crop, int x0, int y0);

    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr dual_fisheye_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr equirect_pub_;

    // Must stay alive for the lifetime of the node: when this handle is
    // destroyed, the parameter callback is unregistered and
    // `ros2 param set` silently stops having any effect.
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;

    // Parameters
    double cx_offset_ = 0.0;
    double cy_offset_ = 0.0;
    int crop_size_ = 960;
    double tx_ = 0.0, ty_ = 0.0, tz_ = 0.0;
    double roll_ = 0.0, pitch_ = 0.0, yaw_ = 0.0;  // radians
    bool gpu_enabled_ = false;
    int out_width_ = 1920;
    int out_height_ = 960;
    std::string frame_id_;

    // Back-lens to front-lens extrinsics
    cv::Matx33d back_to_front_rotation_ = cv::Matx33d::eye();
    cv::Vec3d back_to_front_translation_;

    // One combined map from the full dual-fisheye image to the output image,
    // stored in OpenCV's fixed-point format (fastest with INTER_LINEAR).
    cv::Mat map1_, map2_;

    // What the current maps were built for; any change triggers a rebuild.
    bool maps_initialized_ = false;
    int map_src_width_ = 0;
    int map_src_height_ = 0;
    int map_crop_ = 0;
};

#endif  // EQUIRECTANGULAR_HPP