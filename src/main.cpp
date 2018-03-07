#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <pointCloud.h>
#include <calibration.h>
#include <reconstruction3D.h>

using namespace std;
using namespace cv;

int main(int argc, char const **argv)
{
  const string inputConfig = "../config/config.xml";
  FileStorage fileReader(inputConfig, FileStorage::READ);
  if (!fileReader.isOpened())
  {
        cout << "Could not open the configuration file: \"" << inputConfig << "\"!" << endl;
        return -1;
   }
   int calib = fileReader["Calibration"];
   string stereoFile= fileReader["Calibration_Data"];
   int showRect = fileReader["Show_Rectified"];
   int showDisparity = fileReader["Show_Disparity"];
   int showDepth = fileReader["Show_Depth"];
   int filterPC = fileReader["Filter_PC"];
   if(calib)
   {  
     FileNode calibConf =  fileReader["Calibration_Parameters"];
     calibration c(calibConf);
     c.stereoCalibration(stereoFile);
   }
   Mat imgLeft = imread(argv[1]);
   Mat imgRight = imread(argv[2]);
   Mat imgLeftRect,imgRightRect, disparity, depth;
   reconstruction3D rec(stereoFile);
   rec.retify(imgLeft,imgRight,imgLeftRect,imgRightRect);
   imwrite("../results/rectLeft.png",imgLeftRect);
   imwrite("../results/rectRight.png",imgRightRect);
   if(showRect){
     Mat showLeft,showRight;
     cvtColor(imgLeftRect, showLeft, CV_BGR2GRAY);
     cvtColor(imgRightRect, showRight, CV_BGR2GRAY);
     cvtColor(showLeft, showLeft, CV_GRAY2BGR);
     cvtColor(showRight, showRight, CV_GRAY2BGR);
     for( int j = 0; j < imgLeftRect.rows; j += 64 ){
       line( showLeft, cvPoint(0,j), cvPoint(imgLeftRect.cols*2,j),CV_RGB(255,0,0),2);
       line( showRight, cvPoint(0,j), cvPoint(imgLeftRect.cols*2,j),CV_RGB(255,0,0),2);
     }
     imwrite("../results/rectLeftLine.png",showLeft);
     imwrite("../results/rectRightLine.png",showRight);
     imshow("Left Image Rectified", showLeft);
     imshow("Right Image Rectified", showRight);
     waitKey(0);
     destroyWindow("Left Image Rectified");
     destroyWindow("Right Image Rectified");

   }
   FileNode sgbmParam = fileReader["SGBM_Parameters"];
   rec.getDisparity(imgLeftRect,imgRightRect,disparity,sgbmParam);
   Mat disp8;
   normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8U);
   imwrite("../results/disparity.png",disp8);
   if(showDisparity){
     imshow("Disparity", disp8);
     waitKey(0);
     destroyWindow("Disparity");
   }
   rec.getDepth(disparity,depth);
   Mat depth8;
   normalize(depth, depth8, 0, 255, CV_MINMAX, CV_8U);
   imwrite("../results/depth.png",depth8);
   if(showDepth){
     imshow("DepthMap", depth8);
     waitKey(0);
     destroyWindow("DepthMap");
   }
   getPointCloud(depth,imgLeft,filterPC);
   return 0;
}
