#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <pointCloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace pcl;
using namespace cv;

void getPointCloud(Mat& depthMap, Mat& imgLeft, int filter)
{
  PointCloud<PointXYZRGB>::Ptr pointCloud(new PointCloud<PointXYZRGB>());
  pointCloud->width = depthMap.cols * depthMap.rows;
  pointCloud->height = 1;
  pointCloud->is_dense = false;
  PointXYZRGB point;
  cout << "Point cloud creation.." << endl;
  clock_t start = clock();
  for (int i = 0; i < depthMap.rows; ++i)
  {
    uchar* leftRow_ptr = imgLeft.ptr<uchar>(i);
    float* depthRow_ptr = depthMap.ptr<float>(i);
    for (int j = 0; j < depthMap.cols; ++j)
    {
      if (depthRow_ptr[3*j+2]>= 10000)
      {
        point.x = numeric_limits<float>::quiet_NaN();
        point.y = numeric_limits<float>::quiet_NaN();
        point.z = numeric_limits<float>::quiet_NaN();
        point.b = numeric_limits<float>::quiet_NaN();
        point.g = numeric_limits<float>::quiet_NaN();
        point.r = numeric_limits<float>::quiet_NaN();
      }
      else
      {
        point.x = depthRow_ptr[3*j];
        point.y = depthRow_ptr[3*j+1];
        point.z = depthRow_ptr[3*j+2];
        point.b = leftRow_ptr[3 * j];
        point.g = leftRow_ptr[3 * j + 1];
        point.r = leftRow_ptr[3 * j + 2];
      }
      pointCloud->points.push_back(point);
    }
  }
  cout << "Point cloud created in " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
  // Rotate the point cloud to have the straight view
  Eigen::Affine3f straightView = Eigen::Affine3f::Identity();
  straightView.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitX()));
  transformPointCloud (*pointCloud, *pointCloud, straightView);
  // Filter the point cloud
  if(filter)
  {
    StatisticalOutlierRemoval<PointXYZRGB> outRemover;
    outRemover.setInputCloud (pointCloud);
    outRemover.setMeanK (50);
    outRemover.setStddevMulThresh (1.0);
    outRemover.filter (*pointCloud);
  }
  // Save the point cloud in a file
  String outputFile = "../results/pointCloud.pcd";
  io::savePCDFileASCII(outputFile, *pointCloud);
  // Visualization of the point cloud
  visualization::PCLVisualizer viewer("PCL Viewer");
  viewer.setBackgroundColor (0, 0, 0);
  viewer.addCoordinateSystem (0.1, "Cloud");
  visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(pointCloud);
  viewer.addPointCloud<pcl::PointXYZRGB> (pointCloud, rgb, "Cloud");
  viewer.spin ();
  cout << "Visualization point cloud.." << endl;
}
