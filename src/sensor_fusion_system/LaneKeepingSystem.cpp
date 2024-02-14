#include "sensor_fusion_system/LaneKeepingSystem.hpp"

struct Point2D {
    float x;
    float y;
};

namespace Xycar {
template <typename PREC>
LaneKeepingSystem<PREC>::LaneKeepingSystem()
{
    std::string configPath;
    mNodeHandler.getParam("config_path", configPath);
    YAML::Node config = YAML::LoadFile(configPath);

    mPID = new PIDController<PREC>(config["PID"]["P_GAIN"].as<PREC>(), config["PID"]["I_GAIN"].as<PREC>(), config["PID"]["D_GAIN"].as<PREC>());
    mMovingAverage = new MovingAverageFilter<PREC>(config["MOVING_AVERAGE_FILTER"]["SAMPLE_SIZE"].as<uint32_t>());
    mCameraDetector = new CameraDetector<PREC>(config);
    /*
        create your lane detector.
    */
    setParams(config);

    mPublisher = mNodeHandler.advertise<xycar_msgs::xycar_motor>(mPublishingTopicName, mQueueSize);
    mSubscriber = mNodeHandler.subscribe(mSubscribedTopicName, mQueueSize, &LaneKeepingSystem::imageCallback, this);
    mSubLidar = mNodeHandler.subscribe(mSubscribedLidarName, mQueueSize, &LaneKeepingSystem::scanCallback, this);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::setParams(const YAML::Node& config)
{
    mPublishingTopicName = config["TOPIC"]["PUB_NAME"].as<std::string>();
    mSubscribedTopicName = config["TOPIC"]["SUB_NAME"].as<std::string>();
    mSubscribedLidarName = config["TOPIC"]["LIDAR_NAME"].as<std::string>();
    mQueueSize = config["TOPIC"]["QUEUE_SIZE"].as<uint32_t>();
    mXycarSpeed = config["XYCAR"]["START_SPEED"].as<PREC>();
    mXycarMaxSpeed = config["XYCAR"]["MAX_SPEED"].as<PREC>();
    mXycarMinSpeed = config["XYCAR"]["MIN_SPEED"].as<PREC>();
    mXycarSpeedControlThreshold = config["XYCAR"]["SPEED_CONTROL_THRESHOLD"].as<PREC>();
    mAccelerationStep = config["XYCAR"]["ACCELERATION_STEP"].as<PREC>();
    mDecelerationStep = config["XYCAR"]["DECELERATION_STEP"].as<PREC>();
    mDebugging = config["DEBUG"].as<bool>();
}

template <typename PREC>
LaneKeepingSystem<PREC>::~LaneKeepingSystem()
{
    delete mPID;
    delete mMovingAverage;
    // delete your CameraDetector if you add your CameraDetector.
}

template <typename PREC>
void LaneKeepingSystem<PREC>::run()
{
    ros::Rate rate(kFrameRate);
    mCameraDetector->undistortAndDNNConfig();
    while (ros::ok())
    {
        ros::spinOnce();
        mCameraDetector->boundingBox(mFrame);
        
        // Lidar
        // std::vector<cv::Point2f> image2D= mCameraDetector->Generate2DPoints();
        // std::vector<cv::Point3f> lidar3D = mCameraDetector->Generate3DLidarPoints();
        // std::vector<cv::Point3f> vcs3D = mCameraDetector->Generate3DVCSPoints();

        // mCameraDetector->solvePnP(image2D, lidar3D);
        // mCameraDetector->solvePnP(image2D, vcs3D);
    }
}

template <typename PREC>
void LaneKeepingSystem<PREC>::imageCallback(const sensor_msgs::Image& message)
{
    cv::Mat src = cv::Mat(message.height, message.width, CV_8UC3, const_cast<uint8_t*>(&message.data[0]), message.step);
    cv::cvtColor(src, mFrame, cv::COLOR_RGB2BGR);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
{
    // Process the incoming laser scan data
    // For example, print the range at some point in the scan
    int midpoint = scan->ranges.size() / 2;  // Assuming the LIDAR has 360 degrees field of view
    int idx = 504;
    // std::cout << "Ranges size: " << scan->ranges.size() << std::endl;
    // std::cout << "midpoint: " << midpoint << ", " << scan->ranges[midpoint] << std::endl;
    // std::cout << "0 range: " << scan->ranges[0] << std::endl;
    // std::cout << "126 range: " << scan->ranges[126] << std::endl;
    // std::cout << "252 range: " << scan->ranges[252] << std::endl;
    // std::cout << "378 range: " << scan->ranges[378] << std::endl;
    // std::cout << "504 range: " << scan->ranges[504] << std::endl;

    // float theta_126 = scan->angle_min + 126 * scan->angle_increment;
    // float x_126 = scan->ranges[126] * cos(theta_126);
    // float y_126 = scan->ranges[126] * sin(theta_126);

    // float theta_504 = scan->angle_min + 504 * scan->angle_increment;
    // float x_504 = scan->ranges[504] * cos(theta_504);
    // float y_504 = scan->ranges[504] * sin(theta_504);

    // float theta_378 = scan->angle_min + 378 * scan->angle_increment;
    // float x_378 = scan->ranges[378] * cos(theta_378);
    // float y_378 = scan->ranges[378] * sin(theta_378);

    // float theta_252 = scan->angle_min + 252 * scan->angle_increment;
    // float x_252 = scan->ranges[252] * cos(theta_252);
    // float y_252 = scan->ranges[252] * sin(theta_252);

    // std::cout << "126 x,y: " << x_126 << ", " << y_126 << std::endl;
    // std::cout << "504 x,y: " << x_504 << ", " << y_504 << std::endl;
    // std::cout << "378 x,y: " << x_378 << ", " << y_378 << std::endl;
    // std::cout << "252 x,y: " << x_252 << ", " << y_252 << std::endl;
    // std::cout << "Angle Min: " << scan->angle_min << ", Angle Max: " << scan->angle_max << std::endl;
    // std::cout << "Angle Increment: " << scan->angle_increment << ", Time Increment: " << scan->time_increment << std::endl;
    // std::cout << "Scan Time: " << scan->scan_time << ", Range Min: " << scan->range_min << ", Range Max: " << scan->range_max << std::endl;

    // int lStart = 0;
    // int lEnd = 126 + 1;
    // int rStart = 378;
    // int rEnd = 504 + 1;
    // float xDepth = -1.35;  // foward length, meter
    // float margin = 0.02;

    // int lStart = 0;
    // int lEnd = 126 + 1;
    // int rStart = 378;
    // int rEnd = 504 + 1;
    // float xDepth = -0.9;  // foward length, meter
    // float margin = 0.02;

    // std::vector<Point2D> paperbox;

    // for (int i = lStart; i < lEnd; ++i)
    // {
    //     float r = scan->ranges[i]; // 거리
    //     float theta = scan->angle_min + i * scan->angle_increment; // 각도

    //     float x = r * cos(theta);
    //     float y = r * sin(theta);

    //     if (x > xDepth-margin and x < xDepth+margin)
    //     {
    //         Point2D point;
    //         point.x = x;
    //         point.y = y;
    //         paperbox.push_back(point);
    //     }
    // }

    // for (int i = rStart; i < rEnd; ++i)
    // {
    //     float r = scan->ranges[i]; // 거리
    //     float theta = scan->angle_min + i * scan->angle_increment; // 각도

    //     float x = r * cos(theta);
    //     float y = r * sin(theta);

    //     if (x > xDepth-margin and x < xDepth+margin)
    //     {
    //         Point2D point;
    //         point.x = x;
    //         point.y = y;
    //         paperbox.push_back(point);
    //     }

    // }

    // for (int i = 0; i < paperbox.size(); ++i)
    // {
    //     float x = paperbox[i].x;
    //     float y = paperbox[i].y;

    //     std::cout << "x, y : " << x << ", " << y << std::endl;
    // }
}

template <typename PREC>
void LaneKeepingSystem<PREC>::speedControl(PREC steeringAngle)
{
    if (std::abs(steeringAngle) > mXycarSpeedControlThreshold)
    {
        mXycarSpeed -= mDecelerationStep;
        mXycarSpeed = std::max(mXycarSpeed, mXycarMinSpeed);
        return;
    }

    mXycarSpeed += mAccelerationStep;
    mXycarSpeed = std::min(mXycarSpeed, mXycarMaxSpeed);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::drive(PREC steeringAngle)
{
    xycar_msgs::xycar_motor motorMessage;
    motorMessage.angle = std::round(steeringAngle);
    motorMessage.speed = std::round(mXycarSpeed);

    mPublisher.publish(motorMessage);
}

template class LaneKeepingSystem<float>;
template class LaneKeepingSystem<double>;
} // namespace Xycar
