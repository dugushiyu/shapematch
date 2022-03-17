// copyrights  :  SHENHAO inc. all rights reserved.
// date        :  2022.03.10
// description :  subway detection testbed

#include <corecrt_io.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "include/shape_match.h"

#define CALC_RECALL (1)
using algorithm::shapematch::Matcher;
using namespace std;
using namespace cv;
int main()
{
 Mat temp_Img = imread("C:\\Users\\DELL\\ZED\\2\\DepthViewer_Right_22182765_720_15-03-2022-15-35-26.png",-1);
 flip(temp_Img, temp_Img, -1);
 Mat src_Img = imread("C:\\Users\\DELL\\ZED\\1\\DepthViewer_Right_22182765_720_15-03-2022-15-34-41.png", -1);
 Point matchCenter;
 string folder = "C:\\Users\\DELL\\ZED\\";
 string imageName = "result";

 Matcher detect;
 detect.match_Feedback(temp_Img, src_Img, matchCenter, folder, imageName);
 

 std::vector<cv::String> filenames; 
 cv::String folder2 = "E:/git/1214/datacrack";
 cv::glob(folder, filenames); 
 for (size_t i = 0; i < filenames.size(); ++i)
 {
  std::cout << filenames[i] << std::endl;
  cv::Mat img = cv::imread(filenames[i],-1);
  
 }
 return 1;
}
