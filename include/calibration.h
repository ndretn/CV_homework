#ifndef CALIBRATION_H  
#define CALIBRATION_H

class calibration
{
public:
  void stereoCalibration(std::string&);
  calibration(cv::FileNode&);
private:
  int width,height,nFrame;
  float squareSize;
  bool showImg;
  std::vector<std::string> imgList;
  std::string leftCalibFile, rightCalibFile, stereoCalibFile, imgListFile;
};

#endif
