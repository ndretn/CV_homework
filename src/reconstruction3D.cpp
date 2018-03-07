#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <reconstruction3D.h>

using namespace std;
using namespace cv;

reconstruction3D::reconstruction3D(string& calibFile)
{
  FileStorage fileReader(calibFile, cv::FileStorage::READ);
  fileReader["K1"] >> K1;
  fileReader["K2"] >> K2;
  fileReader["D1"] >> D1;
  fileReader["D2"] >> D2;
  fileReader["R1"] >> R1;
  fileReader["R2"] >> R2;
  fileReader["P1"] >> P1;
  fileReader["P2"] >> P2;
  fileReader["Q"] >> Q;
  fileReader.release();
}

void reconstruction3D::retify(Mat& imgLeft, Mat& imgRight, Mat& outputLeft,Mat& outputRight)
{
  Mat map1Left, map2Left, map1Right, map2Right;
  cout << "Images retification.." << endl;
  clock_t start = clock();
  initUndistortRectifyMap(K1, D1, R1, P1, imgLeft.size(), CV_32F, map1Left, map2Left);
  initUndistortRectifyMap(K2, D2, R2, P2, imgRight.size(), CV_32F, map1Right, map2Right);
  remap(imgLeft, outputLeft, map1Left, map2Left, cv::INTER_LINEAR);
  remap(imgRight, outputRight, map1Right, map2Right, cv::INTER_LINEAR);
  cout << "Images rectification done in " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;

}

void reconstruction3D::getDisparity(Mat& imgLeft, Mat& imgRight, Mat& disparity, FileNode& sgbmParam)
{
  int sadWindowSize, numberOfDisparities, preFilterCap, minDisparity, uniquenessRatio;
  int speckleWindowSize, speckleRange, disp12MaxDiff, fullDP, p1, p2;
  sgbmParam["SADWindowSize"] >> sadWindowSize;
  sgbmParam["NumberOfDisparities"] >> numberOfDisparities;
  sgbmParam["PreFilterCap"] >> preFilterCap;
  sgbmParam["MinDisparity"] >> minDisparity;
  sgbmParam["UniquenessRatio"] >> uniquenessRatio;
  sgbmParam["SpeckleWindowSize"] >> speckleWindowSize;
  sgbmParam["SpeckleRange"] >> speckleRange;
  sgbmParam["Disp12MaxDiff"] >> disp12MaxDiff;
  sgbmParam["FullDP"] >> fullDP;
  sgbmParam["P1"] >> p1;
  sgbmParam["P2"] >> p2;
  Mat imgLeftGray, imgRightGray;
  cvtColor(imgLeft, imgLeftGray, CV_BGR2GRAY);
  cvtColor(imgRight, imgRightGray, CV_BGR2GRAY);
  StereoSGBM sbm;
  sbm.SADWindowSize = sadWindowSize;
  sbm.numberOfDisparities = numberOfDisparities;
  sbm.preFilterCap = preFilterCap;
  sbm.minDisparity = minDisparity;
  sbm.uniquenessRatio = uniquenessRatio;
  sbm.speckleWindowSize = speckleWindowSize;
  sbm.speckleRange = speckleRange;
  sbm.disp12MaxDiff = disp12MaxDiff;
  sbm.fullDP = fullDP;
  sbm.P1 = p1;
  sbm.P2 = p2;
  cout << "Calculating disparity map.." << endl;
  clock_t start = clock();
  sbm(imgLeftGray, imgRightGray, disparity);
  disparity.convertTo( disparity, CV_32F, 1./16);
  medianBlur(disparity, disparity, 5);
  cout << "Disparity map done in " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
}

void reconstruction3D::getDepth(Mat& disparity, Mat& depth)
{
  cout << "Calculating depth map.." << endl;
  clock_t start = clock();
  reprojectImageTo3D(disparity, depth, Q, true,CV_32F);
  cout << "Depth map done in " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
}
