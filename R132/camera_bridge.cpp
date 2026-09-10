// camera_bridge.cpp
// 把 R132 相机的 RGB 帧取出来，发布成 /camera/image (sensor_msgs/Image, bgr8)
// = InuDev 取帧 (原 grab_rgb.cpp) + ROS2 发布 (原 test_image_publisher.py) 合体
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "InuSensor.h"
#include "HwInformation.h"
#include "ImageStream.h"
#include "ImageFrame.h"

#include <memory>
#include <vector>
#include <map>
#include <cstdint>

using namespace InuDev;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("camera_bridge");
    auto pub = node->create_publisher<sensor_msgs::msg::Image>("/camera/image", 10);

    // ---- 1) 连接相机并枚举通道 ----
    std::shared_ptr<CInuSensor> sensor = CInuSensor::Create();
    CHwInformation hwInfo;
    CInuError rc = sensor->Init(hwInfo);
    if (rc != eOK) {
        RCLCPP_ERROR(node->get_logger(), "camera Init failed: 0x%x", (int)rc);
        rclcpp::shutdown();
        return 1;
    }
    for (const auto& kv : hwInfo.GetChannels()) {
        RCLCPP_INFO(node->get_logger(), "channel %d type=%d res=%d fps=%d",
                    kv.first, (int)kv.second.ChannelType,
                    (int)kv.second.ChannelControlParams.SensorRes,
                    kv.second.ChannelControlParams.FPS);
    }

    // ---- 2) 找 RGB 通道（遍历通道，挑出类型为 GeneralCamera 的那个；旧版 SDK 没有 GetChannelsPerType）----
    uint32_t rgbChannel = 4;  // 找不到时回退到 4
    for (const auto& kv : hwInfo.GetChannels()) {
        if (kv.second.ChannelType == eGeneralCameraChannel) {
            rgbChannel = kv.first;
            break;
        }
    }
    RCLCPP_INFO(node->get_logger(), "using RGB channel = %d", rgbChannel);

    // ---- 3) 启动 RGB 通道 (binning, 15 fps) ----
    std::map<uint32_t, CChannelControlParams> params;
    params[rgbChannel] = CChannelControlParams(eBinning, 15);
    std::map<uint32_t, CChannelSize> sizes;
    rc = sensor->Start(sizes, params);
    if (rc != eOK) {
        RCLCPP_ERROR(node->get_logger(), "Start failed: 0x%x", (int)rc);
        rclcpp::shutdown();
        return 1;
    }

    // ---- 4) 创建 BGR 图像流 ----
    std::shared_ptr<CImageStream> stream = sensor->CreateImageStream(rgbChannel);
    if (!stream) {
        RCLCPP_ERROR(node->get_logger(), "CreateImageStream failed");
        rclcpp::shutdown();
        return 1;
    }
    rc = stream->Init(CImageStream::EOutputFormat::eBGR);
    if (rc != eOK) {
        RCLCPP_ERROR(node->get_logger(), "Image Init failed: 0x%x", (int)rc);
        rclcpp::shutdown();
        return 1;
    }
    rc = stream->Start();
    if (rc != eOK) {
        RCLCPP_ERROR(node->get_logger(), "Image Start failed: 0x%x", (int)rc);
        rclcpp::shutdown();
        return 1;
    }

    // ---- 5) 取帧 + 发布循环 ----
    RCLCPP_INFO(node->get_logger(), "camera_bridge running, publishing to /camera/image");
    while (rclcpp::ok()) {
        std::shared_ptr<const CImageFrame> frame;
        rc = stream->GetFrame(frame, 1000);
        if (rc != eOK || !frame) {
            continue;  // 超时或失败就重试
        }

        auto msg = std::make_unique<sensor_msgs::msg::Image>();
        msg->header.stamp = node->now();
        msg->header.frame_id = "camera";
        msg->height = (uint32_t)frame->Height();
        msg->width  = (uint32_t)frame->Width();
        msg->encoding = "bgr8";
        msg->is_bigendian = 0;
        msg->step = frame->Width() * frame->BytesPerPixel();

        size_t nbytes = (size_t)frame->Width() * frame->Height() * frame->BytesPerPixel();
        const uint8_t* p = reinterpret_cast<const uint8_t*>(frame->GetData());
        msg->data.assign(p, p + nbytes);

        pub->publish(std::move(msg));

        rclcpp::spin_some(node);
    }

    // ---- 6) 清理 ----
    stream->Stop();
    stream->Terminate();
    sensor->Stop();
    sensor->Terminate();
    rclcpp::shutdown();
    return 0;
}
