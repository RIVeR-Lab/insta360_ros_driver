#include "equirectangular.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

#include <sensor_msgs/image_encodings.hpp>

namespace enc = sensor_msgs::image_encodings;

EquirectangularNode::EquirectangularNode()
: Node("equirectangular_node")
{
    declare_parameter("cx_offset", 0.0);
    declare_parameter("cy_offset", 0.0);
    declare_parameter("crop_size", 960);
    declare_parameter("translation", std::vector<double>{0.0, 0.0, -0.105});
    declare_parameter("rotation_deg", std::vector<double>{-0.5, 0.0, 1.1});
    declare_parameter("gpu", true);
    declare_parameter("out_width", 1920);
    declare_parameter("out_height", 960);
    // TF frame stamped on the published equirectangular image. Defaults to the
    // Insta360 optical frame published by the bringup static transforms.
    declare_parameter("frame_id", "insta_sensor");

    loadParameters();
    updateCameraParameters();
    logParameters();

    if (gpu_enabled_) {
        RCLCPP_WARN(get_logger(), "gpu=true requested, but this node currently runs on the CPU only");
    }

    // Stored as a member so the callback stays registered (see header).
    params_callback_handle_ = add_on_set_parameters_callback(
        std::bind(&EquirectangularNode::parametersCallback, this, std::placeholders::_1));

    // Subscription: best effort, depth 1. A live processing node should always
    // work on the newest frame and never make the decoder's publisher wait on it.
    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();

    dual_fisheye_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "dual_fisheye/image", sub_qos,
        std::bind(&EquirectangularNode::imageCallback, this, std::placeholders::_1));

    equirect_pub_ = create_publisher<sensor_msgs::msg::Image>("equirectangular/image", pub_qos);
}

void EquirectangularNode::loadParameters()
{
    cx_offset_ = get_parameter("cx_offset").as_double();
    cy_offset_ = get_parameter("cy_offset").as_double();
    crop_size_ = static_cast<int>(get_parameter("crop_size").as_int());
    out_width_ = static_cast<int>(get_parameter("out_width").as_int());
    out_height_ = static_cast<int>(get_parameter("out_height").as_int());
    gpu_enabled_ = get_parameter("gpu").as_bool();
    frame_id_ = get_parameter("frame_id").as_string();

    const auto translation = get_parameter("translation").as_double_array();
    const auto rotation_deg = get_parameter("rotation_deg").as_double_array();
    if (translation.size() != 3 || rotation_deg.size() != 3) {
        throw std::runtime_error("translation and rotation_deg must each have exactly 3 elements");
    }
    tx_ = translation[0];
    ty_ = translation[1];
    tz_ = translation[2];
    roll_ = rotation_deg[0] * M_PI / 180.0;
    pitch_ = rotation_deg[1] * M_PI / 180.0;
    yaw_ = rotation_deg[2] * M_PI / 180.0;
}

void EquirectangularNode::logParameters()
{
    RCLCPP_INFO(get_logger(), "Equirectangular parameters:");
    RCLCPP_INFO(get_logger(), "  Crop size: %d", crop_size_);
    RCLCPP_INFO(get_logger(), "  Center offset: (%.1f, %.1f)", cx_offset_, cy_offset_);
    RCLCPP_INFO(get_logger(), "  Translation: [%.3f, %.3f, %.3f]", tx_, ty_, tz_);
    RCLCPP_INFO(get_logger(), "  Rotation (deg): [%.2f, %.2f, %.2f]",
                roll_ * 180.0 / M_PI, pitch_ * 180.0 / M_PI, yaw_ * 180.0 / M_PI);
    RCLCPP_INFO(get_logger(), "  Output size: %dx%d", out_width_, out_height_);
    RCLCPP_INFO(get_logger(), "  Frame ID: %s", frame_id_.c_str());
}

void EquirectangularNode::updateCameraParameters()
{
    const double c_r = std::cos(roll_), s_r = std::sin(roll_);
    const double c_p = std::cos(pitch_), s_p = std::sin(pitch_);
    const double c_y = std::cos(yaw_), s_y = std::sin(yaw_);

    const cv::Matx33d Rx(1.0, 0.0, 0.0,
                         0.0, c_r, -s_r,
                         0.0, s_r, c_r);
    const cv::Matx33d Ry(c_p, 0.0, s_p,
                         0.0, 1.0, 0.0,
                         -s_p, 0.0, c_p);
    const cv::Matx33d Rz(c_y, -s_y, 0.0,
                         s_y, c_y, 0.0,
                         0.0, 0.0, 1.0);

    back_to_front_rotation_ = Rz * Ry * Rx;
    back_to_front_translation_ = cv::Vec3d(tx_, ty_, tz_);
    maps_initialized_ = false;  // rebuild on the next image
}

void EquirectangularNode::initMapping(int src_width, int src_height, int crop, int x0, int y0)
{
    const auto t0 = std::chrono::steady_clock::now();
    RCLCPP_INFO(get_logger(),
                "Building equirectangular map: %dx%d dual fisheye (crop %d) -> %dx%d",
                src_width, src_height, crop, out_width_, out_height_);

    const int half_w = src_width / 2;

    // Fisheye center inside each cropped lens square.
    const double cx = crop / 2.0 + cx_offset_;
    const double cy = crop / 2.0 + cy_offset_;
    const double radius = crop / 2.0;

    // Where each lens's crop square starts inside the full dual-fisheye image:
    // the back lens is the left half, the front lens is the right half.
    const double front_x0 = half_w + x0, front_y0 = y0;
    const double back_x0 = x0, back_y0 = y0;

    cv::Mat map_x(out_height_, out_width_, CV_32F);
    cv::Mat map_y(out_height_, out_width_, CV_32F);

    for (int y = 0; y < out_height_; ++y) {
        const double lat = (static_cast<double>(y) / out_height_) * M_PI - M_PI / 2.0;
        const double cos_lat = std::cos(lat), sin_lat = std::sin(lat);
        float* mx = map_x.ptr<float>(y);
        float* my = map_y.ptr<float>(y);

        for (int x = 0; x < out_width_; ++x) {
            const double lon = (static_cast<double>(x) / out_width_) * 2.0 * M_PI - M_PI;
            const double X = cos_lat * std::sin(lon);
            const double Y = sin_lat;
            const double Z = cos_lat * std::cos(lon);

            double px, py, pz, ox, oy;
            if (Z >= 0.0) {
                // Front hemisphere: front lens, no extrinsic correction.
                px = X; py = Y; pz = Z;
                ox = front_x0; oy = front_y0;
            } else {
                // Back hemisphere: apply back-to-front extrinsics, mirror X.
                const cv::Vec3d p = back_to_front_rotation_ * cv::Vec3d(X, Y, Z)
                                    + back_to_front_translation_;
                px = -p[0]; py = p[1]; pz = p[2];
                ox = back_x0; oy = back_y0;
            }

            const double r = std::max(std::sqrt(px * px + py * py), 1e-6);
            const double theta = std::atan2(r, std::fabs(pz));
            const double r_fisheye = 2.0 * theta / M_PI * radius;  // equidistant model

            mx[x] = static_cast<float>(ox + cx + px / r * r_fisheye);
            my[x] = static_cast<float>(oy + cy + py / r * r_fisheye);
        }
    }

    // Fixed-point maps make cv::remap noticeably faster.
    cv::convertMaps(map_x, map_y, map1_, map2_, CV_16SC2);

    map_src_width_ = src_width;
    map_src_height_ = src_height;
    map_crop_ = crop;
    maps_initialized_ = true;

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO(get_logger(), "Equirectangular map ready (%.0f ms)", ms);
}

void EquirectangularNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    try {
        const auto t0 = std::chrono::steady_clock::now();

        // The decoder publishes bgr8, so this shares the incoming buffer
        // instead of copying and color-converting every frame.
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
        const cv::Mat& dual = cv_ptr->image;

        const int full_w = dual.cols;
        const int full_h = dual.rows;
        const int half_w = full_w / 2;

        // Centered square crop inside each lens half.
        int crop = crop_size_;
        if (crop <= 0 || crop > half_w || crop > full_h) {
            const int fallback = std::min(half_w, full_h);
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "crop_size %d does not fit a %dx%d lens half; using %d. "
                                 "Set crop_size for the current stream resolution.",
                                 crop_size_, half_w, full_h, fallback);
            crop = fallback;
        }
        const int x0 = (half_w - crop) / 2;
        const int y0 = (full_h - crop) / 2;

        if (!maps_initialized_ || full_w != map_src_width_ ||
            full_h != map_src_height_ || crop != map_crop_) {
            initMapping(full_w, full_h, crop, x0, y0);
        }

        // One remap of the full dual-fisheye image replaces the previous two
        // full-size remaps plus masked copies.
        cv::Mat equirect;
        cv::remap(dual, equirect, map1_, map2_, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        std_msgs::msg::Header header;
        header.stamp = msg->header.stamp;  // line up in time with the source frame
        header.frame_id = frame_id_;

        auto out = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(header, enc::BGR8, equirect).toImageMsg(*out);
        equirect_pub_->publish(std::move(out));

        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        RCLCPP_DEBUG(get_logger(), "Processing time: %.1f ms", ms);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
    }
}

rcl_interfaces::msg::SetParametersResult EquirectangularNode::parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    // On-set callbacks run BEFORE the new values are stored, so
    // get_parameter() would still return the old ones. Stage the new values
    // from `parameters`, validate them, then commit.
    double cx_offset = cx_offset_;
    double cy_offset = cy_offset_;
    int crop_size = crop_size_;
    int out_width = out_width_;
    int out_height = out_height_;
    bool gpu = gpu_enabled_;
    std::string frame_id = frame_id_;
    std::vector<double> translation{tx_, ty_, tz_};
    std::vector<double> rotation_deg{roll_ * 180.0 / M_PI, pitch_ * 180.0 / M_PI, yaw_ * 180.0 / M_PI};
    bool geometry_changed = false;

    try {
        for (const auto& p : parameters) {
            const std::string& name = p.get_name();
            if (name == "cx_offset") { cx_offset = p.as_double(); geometry_changed = true; }
            else if (name == "cy_offset") { cy_offset = p.as_double(); geometry_changed = true; }
            else if (name == "crop_size") { crop_size = static_cast<int>(p.as_int()); geometry_changed = true; }
            else if (name == "out_width") { out_width = static_cast<int>(p.as_int()); geometry_changed = true; }
            else if (name == "out_height") { out_height = static_cast<int>(p.as_int()); geometry_changed = true; }
            else if (name == "translation") { translation = p.as_double_array(); geometry_changed = true; }
            else if (name == "rotation_deg") { rotation_deg = p.as_double_array(); geometry_changed = true; }
            else if (name == "frame_id") { frame_id = p.as_string(); }
            else if (name == "gpu") { gpu = p.as_bool(); }
        }
    } catch (const rclcpp::ParameterTypeException& e) {
        result.successful = false;
        result.reason = e.what();
        return result;
    }

    if (crop_size <= 0 || out_width <= 0 || out_height <= 0) {
        result.successful = false;
        result.reason = "crop_size, out_width and out_height must be positive";
        return result;
    }
    if (translation.size() != 3 || rotation_deg.size() != 3) {
        result.successful = false;
        result.reason = "translation and rotation_deg must each have exactly 3 elements";
        return result;
    }

    // Commit.
    cx_offset_ = cx_offset;
    cy_offset_ = cy_offset;
    crop_size_ = crop_size;
    out_width_ = out_width;
    out_height_ = out_height;
    gpu_enabled_ = gpu;
    frame_id_ = frame_id;
    tx_ = translation[0];
    ty_ = translation[1];
    tz_ = translation[2];
    roll_ = rotation_deg[0] * M_PI / 180.0;
    pitch_ = rotation_deg[1] * M_PI / 180.0;
    yaw_ = rotation_deg[2] * M_PI / 180.0;

    if (geometry_changed) {
        updateCameraParameters();
        RCLCPP_INFO(get_logger(), "Parameters updated; map will be rebuilt on the next image");
        logParameters();
    }
    return result;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EquirectangularNode>();
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception during spin: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}