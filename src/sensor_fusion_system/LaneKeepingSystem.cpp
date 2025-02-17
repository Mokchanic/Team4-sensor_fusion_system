#include "sensor_fusion_system/LaneKeepingSystem.hpp"

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

    // intrinsic setting & model setting
    mCameraDetector->undistortAndDNNConfig();

    // extrinsic matrix
    std::vector<cv::Point2f> image2D= mCameraDetector->Generate2DPoints();
    std::vector<cv::Point3f> lidar3D = mCameraDetector->Generate3DLidarPoints();
    std::vector<cv::Point3f> vcs3D = mCameraDetector->Generate3DVCSPoints();

    mCameraDetector->getLidarExtrinsicMatrix(image2D, lidar3D);
    mCameraDetector->getVCSExtrinsicMatrix(image2D, vcs3D);

    while (ros::ok())
    {
        ros::spinOnce();

        // Lidar
        std::vector<cv::Point3f> objectPoints;

        if (mLidarCoord.size() == 0)
            continue;

        std::cout << "mLidarCoord size: " << mLidarCoord.size() << std::endl;

        for (int i=0; i < mLidarCoord.size(); ++i){
            // convert lidar coord to camera coord
            objectPoints.push_back(cv::Point3f(mLidarCoord[i].y, -0.058, -mLidarCoord[i].x));
        }

        // get (u,v) 2d images from projectPoints
        std::vector<cv::Point2f> lidarImagePoints = mCameraDetector->getProjectPoints(objectPoints);

        // for (int i=0; i<lidarImagePoints.size(); ++i) {
        //     std::cout << "lidar image point x, y : " << lidarImagePoints[i].x << lidarImagePoints[i].y << std::endl;
        // }
        // visualize
        std::vector<int> bboxIdx = mCameraDetector->boundingBox(mFrame, lidarImagePoints);

        std::vector<cv::Point3f> vcsCoords;
        // convert lidar coord points to VCS coord
        for (int idx = 0; idx < bboxIdx.size(); ++idx) {
            cv::Point3f vcs = mCameraDetector->getVCSCoordPointsFromLidar(objectPoints[bboxIdx[idx]]);
            vcsCoords.push_back(vcs);
            std::cout << "vcs coordinate: " << vcs << std::endl;
        }
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
    int lStart = 0;
    int lEnd = 126 + 1;
    int rStart = 378;
    int rEnd = 504 + 1;

    mLidarCoord.clear();

    for (int i = lStart; i < lEnd; ++i)
    {
        float r = scan->ranges[i]; // 거리
        float theta = scan->angle_min + i * scan->angle_increment; // 각도

        float x = r * cos(theta);
        float y = r * sin(theta);

        cv::Point2f point;
        point.x = x;
        point.y = y;
        mLidarCoord.push_back(point);
    }

    for (int i = rStart; i < rEnd; ++i)
    {
        float r = scan->ranges[i]; // 거리
        float theta = scan->angle_min + i * scan->angle_increment; // 각도

        float x = r * cos(theta);
        float y = r * sin(theta);

        cv::Point2f point;
        point.x = x;
        point.y = y;
        mLidarCoord.push_back(point);
    }

    // for (int i = 0; i < mLidarCoord.size(); ++i)
    // {
    //     float x = mLidarCoord[i].x;
    //     float y = mLidarCoord[i].y;

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
