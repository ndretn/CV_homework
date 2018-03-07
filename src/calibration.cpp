#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <calibration.h>

using namespace std;
using namespace cv;

calibration::calibration(FileNode& calibConf)
{
  calibConf["BoardSize_Width"] >> width;
  calibConf["BoardSize_Height"] >> height;
  calibConf["Frame_Number"] >> nFrame;
  calibConf["Square_Size"] >> squareSize;
  calibConf["Input_Image"] >> imgListFile;
  calibConf["Show_CalibImg"] >> showImg;
  imgList.clear();
  FileStorage fileReader(imgListFile, FileStorage::READ);
  FileNode  imgSeq = fileReader.getFirstTopLevelNode();
  FileNodeIterator imgIter = imgSeq.begin(), imgIterEnd = imgSeq.end();
  for( ; imgIter != imgIterEnd; ++imgIter ) imgList.push_back((string)*imgIter);
  fileReader.release();
}

void calibration::stereoCalibration(string& outputFile)
{
  Size boardSize = Size(width,height);
  Mat imgLeft, imgLeftGray;
  Mat imgRight, imgRightGray;
  vector< vector< Point3f > > objectPoints;
  vector< vector< Point2f > > imgPointsLeft;
  vector< vector< Point2f > > imgPointsRight;
  vector< Point2f > cornersLeft;
  vector< Point2f > cornersRight;
  for (int i = 0; i < nFrame; ++i)
  {
    imgLeft = imread(imgList[2*i]);
    imgRight = imread(imgList[2*i+1]);
    cvtColor(imgLeft, imgLeftGray, CV_BGR2GRAY);
    cvtColor(imgRight, imgRightGray, CV_BGR2GRAY);
    bool foundLeft = false, foundRight = false;
    foundLeft = findChessboardCorners(imgLeftGray, boardSize, cornersLeft,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    foundRight = findChessboardCorners(imgRightGray, boardSize, cornersRight,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    vector< Point3f > object;
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        object.push_back(Point3f((float)j * squareSize, (float)i * squareSize, 0));
      }
    }
    if (foundLeft && foundRight)
    {
      cornerSubPix(imgLeftGray, cornersLeft, Size(5, 5), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      cornerSubPix(imgRightGray, cornersRight, Size(5, 5), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      imgPointsLeft.push_back(cornersLeft);
      imgPointsRight.push_back(cornersRight);
      objectPoints.push_back(object);
      if(showImg)
      {
        cvtColor(imgLeftGray, imgLeftGray, CV_GRAY2BGR);
        cvtColor(imgRightGray, imgRightGray, CV_GRAY2BGR);
        drawChessboardCorners(imgLeftGray, boardSize, cornersLeft, foundLeft);
        drawChessboardCorners(imgRightGray, boardSize, cornersRight, foundRight);
        imshow("Left Image", imgLeftGray);
        imshow("Right Image", imgRightGray);
        waitKey(0);
        destroyWindow("Left Image");
        destroyWindow("Right Image");
      }
    }
  }
  Mat D1, D2;
  Vec3d T;
  Mat K1, K2, R, F, E;
  vector< Mat > rvecs, tvecs;
  cout << "Calibrating left camera.." << endl;
  clock_t start = clock();
  double errorLeft = calibrateCamera(objectPoints, imgPointsLeft, imgLeft.size(), K1, D1, rvecs, tvecs);
  cout << "Left calibration done with " << errorLeft << " re-projection error in " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
  cout << "Calibrating right camera.." << endl;
  clock_t left = clock();
  double errorRight = calibrateCamera(objectPoints, imgPointsRight, imgRight.size(), K2, D2, rvecs, tvecs);
  cout << "Right calibration done with " << errorRight << " re-projection error in " << (clock() - left) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
  FileStorage fileWriter_left("../config/Left.xml", FileStorage::WRITE);
  fileWriter_left << "Error" << errorLeft;
  fileWriter_left << "K1" << K1;
  fileWriter_left << "D1" << D1;
  fileWriter_left.release();
  FileStorage fileWriter_right("../config/Right.xml", FileStorage::WRITE);
  fileWriter_right << "Error" << errorRight;
  fileWriter_right << "K2" << K2;
  fileWriter_right << "D2" << D2;
  fileWriter_right.release();
  int flag = 0;
  flag |= CV_CALIB_USE_INTRINSIC_GUESS;
  flag |= CV_CALIB_RATIONAL_MODEL;
  cout << "Stereo calibration.." << endl;
  clock_t right = clock();
  double error =  stereoCalibrate(objectPoints, imgPointsLeft, imgPointsRight, K1, D1, K2, D2, imgLeft.size(), R, T, E, F,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6),flag);
  cout << "Stereo calibration done with " << error << " re-projection error in " << (clock() - right) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
  Mat R1, R2, P1, P2, Q;
  cout << "Stereo rectification.." << endl;
  clock_t stereo = clock();
  stereoRectify(K1, D1, K2, D2, imgLeft.size(), R, T, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY);
  cout << "Stereo rectification done in " << (clock() - stereo) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
  cout << "Calibration finished in " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms!" << endl;
  FileStorage fileWriter(outputFile, FileStorage::WRITE);
  fileWriter << "Error" << error;
  fileWriter << "K1" << K1;
  fileWriter << "K2" << K2;
  fileWriter << "D1" << D1;
  fileWriter << "D2" << D2;
  fileWriter << "R" << R;
  fileWriter << "T" << T;
  fileWriter << "E" << E;
  fileWriter << "F" << F;
  fileWriter << "R1" << R1;
  fileWriter << "R2" << R2;
  fileWriter << "P1" << P1;
  fileWriter << "P2" << P2;
  fileWriter << "Q" << Q;
  fileWriter.release();
}
