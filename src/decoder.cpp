#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
    #include <libavutil/log.h>
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    (void)ctx;
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

class H264DecoderNode : public rclcpp::Node {
private:
    // ---------------- FFmpeg state ----------------
    const AVCodec* codec_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVCodecParserContext* parser_ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    AVFrame* sw_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    cv::Mat bgr_frame_;
    AVBufferRef* hw_device_ctx_ = nullptr;
    enum AVHWDeviceType hw_type_ = AV_HWDEVICE_TYPE_NONE;

    // ---------------- ROS interfaces ----------------
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    // ---------------- Publisher thread ----------------
    std::thread publisher_thread_;
    std::queue<cv::Mat> frame_publish_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_publisher_thread_{false};
    size_t max_queue_size_ = 2;//10;

    // ---------------- Parameters ----------------
    std::string frame_id_;
    int skip_frame_ = 0;
    int frame_counter_ = 0;
    bool i_frame_only_ = false;

    // ---------------- Diagnostics ----------------
    // rx_msgs_:     compressed messages received per second
    // decoded_:     frames successfully decoded per second
    // queue_drops_: decoded frames dropped because the publish queue was full
    // send_errors_: packets the decoder rejected
    // max_cb_ms_:   slowest compressed callback in the last second
    //               (> ~33 ms at 30 fps means the decoder is falling behind)
    std::atomic<int> rx_msgs_{0};
    std::atomic<int> decoded_{0};
    std::atomic<int> queue_drops_{0};
    std::atomic<int> send_errors_{0};
    double max_cb_ms_ = 0.0;  // only touched on the executor thread
    rclcpp::TimerBase::SharedPtr stats_timer_;

    void InitFFmpegDecoder() {
        // Print FFmpeg warnings/errors (e.g. "concealing errors", missing refs).
        av_log_set_level(AV_LOG_WARNING);

        hw_type_ = AV_HWDEVICE_TYPE_CUDA;
        codec_ = avcodec_find_decoder_by_name("h264_cuvid");
        if (!codec_) {
            RCLCPP_WARN(get_logger(), "Hardware decoder not available, falling back to software");
            hw_type_ = AV_HWDEVICE_TYPE_NONE;
            codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
            if (!codec_) {
                RCLCPP_ERROR(get_logger(), "No H.264 decoder available");
                return;
            }
        } else {
            RCLCPP_INFO(get_logger(), "Using hardware H.264 decoder (NVDEC)");
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            int err = av_hwdevice_ctx_create(&hw_device_ctx_, hw_type_, nullptr, nullptr, 0);
            if (err < 0) {
                RCLCPP_WARN(get_logger(), "Failed to create hardware device context, falling back to software");
                hw_type_ = AV_HWDEVICE_TYPE_NONE;
                codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
                if (!codec_) {
                    RCLCPP_ERROR(get_logger(), "No H.264 decoder available");
                    return;
                }
            }
        }

        parser_ctx_ = av_parser_init(codec_->id);
        if (!parser_ctx_) {
            CleanupFFmpegDecoder();
            return;
        }

        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_) {
            CleanupFFmpegDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE && hw_device_ctx_) {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            codec_ctx_->get_format = get_hw_format;
        }

        if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
            RCLCPP_ERROR(get_logger(), "Failed to open codec");
            CleanupFFmpegDecoder();
            return;
        }

        pkt_ = av_packet_alloc();
        hw_frame_ = av_frame_alloc();
        if (!pkt_ || !hw_frame_) {
            CleanupFFmpegDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            sw_frame_ = av_frame_alloc();
            if (!sw_frame_) {
                CleanupFFmpegDecoder();
                return;
            }
        }
    }

    void PublisherThreadLoop() {
        while (!stop_publisher_thread_) {
            cv::Mat frame_to_publish;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_publish_queue_.empty() || stop_publisher_thread_;
                });

                if (stop_publisher_thread_ && frame_publish_queue_.empty()) {
                    break;
                }
                if (frame_publish_queue_.empty()) {
                    continue;
                }
                frame_to_publish = frame_publish_queue_.front();
                frame_publish_queue_.pop();
            }

            if (!frame_to_publish.empty() && publisher_) {
                auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
                std_msgs::msg::Header header;
                header.stamp = get_clock()->now();
                header.frame_id = frame_id_;
                cv_bridge::CvImage cv_image(header, sensor_msgs::image_encodings::BGR8, frame_to_publish);
                cv_image.toImageMsg(*img_msg);
                publisher_->publish(std::move(img_msg));
            }
        }
    }

    void DecodeAndDisplayPacket(AVPacket* packet) {
        int ret = avcodec_send_packet(codec_ctx_, packet);
        if (ret < 0) {
            send_errors_++;
            char err[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, err, sizeof(err));
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "avcodec_send_packet failed: %s", err);
            return;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return;  // no more frames available for this packet
            } else if (ret < 0) {
                return;
            }

            // A frame was successfully decoded.
            decoded_++;

            AVFrame* frame_to_display = hw_frame_;

            if (hw_frame_->format == AV_PIX_FMT_CUDA || hw_frame_->format == AV_PIX_FMT_VAAPI) {
                if (av_hwframe_transfer_data(sw_frame_, hw_frame_, 0) < 0) {
                    av_frame_unref(hw_frame_);
                    continue;
                }
                frame_to_display = sw_frame_;
            }

            if (!sws_ctx_ && frame_to_display->width > 0 && frame_to_display->height > 0) {
                sws_ctx_ = sws_getContext(
                    frame_to_display->width, frame_to_display->height,
                    static_cast<AVPixelFormat>(frame_to_display->format),
                    frame_to_display->width, frame_to_display->height, AV_PIX_FMT_BGR24,
                    SWS_POINT, nullptr, nullptr, nullptr);

                if (!sws_ctx_) {
                    av_frame_unref(hw_frame_);
                    if (frame_to_display == sw_frame_) av_frame_unref(sw_frame_);
                    return;
                }
                bgr_frame_.create(frame_to_display->height, frame_to_display->width, CV_8UC3);
            }

            if (sws_ctx_ && !bgr_frame_.empty()) {
                uint8_t* dst_data[4] = { bgr_frame_.data, nullptr, nullptr, nullptr };
                int dst_linesize[4] = { static_cast<int>(bgr_frame_.step[0]), 0, 0, 0 };

                sws_scale(sws_ctx_,
                          (const uint8_t* const*)frame_to_display->data, frame_to_display->linesize,
                          0, frame_to_display->height,
                          dst_data, dst_linesize);

                bool should_publish = true;
                if (skip_frame_ > 0 && !i_frame_only_) {
                    should_publish = (frame_counter_++ % (skip_frame_ + 1) == 0);
                }

                if (should_publish) {
                    cv::Mat frame_copy = bgr_frame_.clone();
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (frame_publish_queue_.size() >= max_queue_size_) {
                            frame_publish_queue_.pop();   // discard the oldest; always keep the newest
                            queue_drops_++;
                        }
                        frame_publish_queue_.push(std::move(frame_copy));
                    }
                    queue_cv_.notify_one();
                }
            }

            av_frame_unref(hw_frame_);
            if (frame_to_display == sw_frame_) {
                av_frame_unref(sw_frame_);
            }
        }
    }

    void CleanupFFmpegDecoder() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
        }
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
        }
        if (pkt_) {
            av_packet_free(&pkt_);
        }
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
        }
        if (parser_ctx_) {
            av_parser_close(parser_ctx_);
            parser_ctx_ = nullptr;
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
        }
        codec_ = nullptr;
    }

    void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        auto t0 = std::chrono::steady_clock::now();
        rx_msgs_++;

        if (msg->format != "h264") {
            return;
        }
        if (!codec_ctx_ || !parser_ctx_ || !pkt_ || !hw_frame_) {
            return;
        }

        const uint8_t* cur_data = msg->data.data();
        size_t remaining_size = msg->data.size();

        while (remaining_size > 0) {
            int bytes_parsed = av_parser_parse2(parser_ctx_, codec_ctx_,
                                                &pkt_->data, &pkt_->size,
                                                cur_data, static_cast<int>(remaining_size),
                                                AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (bytes_parsed < 0) {
                break;
            }
            cur_data += bytes_parsed;
            remaining_size -= bytes_parsed;

            if (pkt_->size > 0) {
                if (i_frame_only_) {
                    // The parser sets key_frame for I-frames.
                    if (parser_ctx_->key_frame == 1) {
                        DecodeAndDisplayPacket(pkt_);
                    }
                } else {
                    DecodeAndDisplayPacket(pkt_);
                }
            }
        }

        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        max_cb_ms_ = std::max(max_cb_ms_, ms);
    }

public:
    H264DecoderNode() : Node("h264_decoder_node") {
        declare_parameter("compressed_topic", "dual_fisheye/image/compressed");
        declare_parameter("uncompressed_topic", "dual_fisheye/image");
        declare_parameter("skip_frame", 0);
        declare_parameter("i_frame_only", false);
        declare_parameter("frame_id", "insta_sensor");
        declare_parameter("subscription_depth", 10);

        std::string subscribe_topic = get_parameter("compressed_topic").as_string();
        std::string publish_topic = get_parameter("uncompressed_topic").as_string();
        skip_frame_ = get_parameter("skip_frame").as_int();
        i_frame_only_ = get_parameter("i_frame_only").as_bool();
        frame_id_ = get_parameter("frame_id").as_string();
        int sub_depth = get_parameter("subscription_depth").as_int();

        // Initialize the decoder before the subscription exists so no
        // messages arrive while the codec is still being set up.
        InitFFmpegDecoder();

        publisher_ = create_publisher<sensor_msgs::msg::Image>(publish_topic, 10);
        publisher_thread_ = std::thread(&H264DecoderNode::PublisherThreadLoop, this);

        subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            subscribe_topic, sub_depth,
            std::bind(&H264DecoderNode::compressed_image_callback, this, std::placeholders::_1));

        // Once-per-second diagnostics. Runs on the same single-threaded
        // executor as the subscription callback, so max_cb_ms_ is race-free.
        stats_timer_ = create_wall_timer(std::chrono::seconds(1), [this] {
            RCLCPP_INFO(get_logger(),
                        "rx %d msg/s | decoded %d fps | queue drops %d | send errors %d | max callback %.1f ms",
                        rx_msgs_.exchange(0), decoded_.exchange(0), queue_drops_.exchange(0),
                        send_errors_.exchange(0), max_cb_ms_);
            max_cb_ms_ = 0.0;
        });

        RCLCPP_INFO(get_logger(), "H.264 Decoder Node initialized");
        RCLCPP_INFO(get_logger(), "Subscribing to: %s (depth %d)", subscribe_topic.c_str(), sub_depth);
        RCLCPP_INFO(get_logger(), "Publishing to: %s", publish_topic.c_str());
        RCLCPP_INFO(get_logger(), "Skip frame: %d, I-frame only: %s",
                    skip_frame_, i_frame_only_ ? "true" : "false");
        RCLCPP_INFO(get_logger(), "Frame ID: %s", frame_id_.c_str());
    }

    ~H264DecoderNode() override {
        stop_publisher_thread_ = true;
        queue_cv_.notify_one();
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
        CleanupFFmpegDecoder();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<H264DecoderNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}