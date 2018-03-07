#ifndef RECONSTRUCTION3D_H
#define RECONSTRUCTION3D_H

class reconstruction3D
{
public:
	void retify(cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&);
        void getDisparity(cv::Mat&, cv::Mat&, cv::Mat&, cv::FileNode&);
        void getDepth(cv::Mat&, cv::Mat&);
        reconstruction3D(std::string&);
private:
        cv::Mat R1, R2, P1, P2, Q;
        cv::Mat K1, K2, D1, D2;
};

#endif
