#pragma once
#include "line2Dup.h"
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/types_c.h>
#include "glog/logging.h"

namespace algorithm {
 namespace shapematch {
#define DLLEXPORT __declspec(dllexport)

  using namespace std;
  using namespace cv;
  using namespace cv::ml;
  using namespace cv::xfeatures2d;
  using namespace cv::dnn;

#define minHessian 20

  class MatchInfo {
  public:
   MatchInfo();
   ~MatchInfo();
   float x;
   float y;
   float angle;
   float score;
   vector<Point> points;
   RotatedRect rectangle2();
  };

  typedef struct MatchStruct {
   bool  Successflag; //是否匹配成功
   Mat match_H; //匹配单应矩阵
   Mat drawImg; //绘结果图
   vector<Point2f> objPoints;//目标图上的识别点集
  } MatcheResult;


  class DLLEXPORT Matcher {
  public:
   /**************************************************************************
   *  函数名称: Matcher 
   *  函数说明: 创建模板，缩放为1 带旋转的模板
   *  返 回 值: 无
   *  参    数:  path模板文件夹根目录，type 文件夹下子文件夹名字；num特征提取数目
   **************************************************************************/
   //Matcher(string file_path, int num, const vector<std::string>& class_ids, string class_id);
   Matcher();
   ~Matcher();
   /**************************************************************************
   *  函数名称: CreateShapeModel
   *  函数说明: 创建模板，缩放为1 带旋转的模板
   *  返 回 值: 无
   *  参    数: name 图像名称 
   **************************************************************************/
   void CreateShapeModel(const Mat& in_img, const string& class_name, const Rect& in_roi, 
    const float& angle_start, const float&  angle_end, const float&  angle_step);
   bool ReadShapeModel(const vector<std::string>& ids, string type);
   bool FindShapeModel(const Mat& test_img, const int& matchscore, const vector<std::string>& class_ids,
    MatchInfo& result);
   void ConvertImg(const Mat&src, Mat&dst);

   /**************************************************************************
   *  函数名称: drawCross
   *  函数说明: 依据点信息画出十字架line
   *  返 回 值: 无
   *  参    数: RGB图,点信息,颜色,宽度大小，线宽
   **************************************************************************/
   void drawCross(Mat img, const Point& point, const Scalar& color, const int& size, 
    const int& thickness);
   /**************************************************************************
   *  函数名称: get_roiImage
   *  函数说明: 截取识别区域图片
   *  返 回 值: 是否截取成功flag
   *  参    数: 原图,模板图,输出图像,中心点
   **************************************************************************/
   int get_roiImage(const Mat& inImg, const Mat& tempImg, Mat& outImg, const Point& in_center);
   /**************************************************************************
   *  函数名称: match_Feedback
   *  函数说明：获取匹配中心
   *  返 回 值: 匹配成功返回1,否则返回-1
   *  参    数: 目标图,场景图,匹配中心，文件目录
   **************************************************************************/
   int match_Feedback(Mat &temp_Img, Mat &in_Img, Point &matchCenter, const string& folder, const string& imageName);

   Mat roiImage_correct(Mat tempImage, Mat roiIamge, string folder, Mat &H_matrix, string imageName);
   /**************************************************************************
   *  函数名称: ucharToMat
   *  函数说明: 依据unsigned char* 数组还原Mat类型图片
   *  返 回 值: 无
   *  参    数: unsigned char* 数组,图像高,图像宽,图像通道数，还原Mat类型图片
   **************************************************************************/
   void ucharToMat(unsigned char* uchar_image, int image_height, int image_width, int channels, cv::Mat &dst_image);

   void coordinate_conversion(cv::Mat srcImg, cv::Mat H_inverse_matrix, cv::Rect& roi_rect);
   MatcheResult featureMatch_Bak2021(Mat objectImg, Mat sceneImg, int Hessian, MatcheResult matches);
   /**************************************************************************
   *  函数名称: featureMatch
   *  函数说明：match_Feedback子函数 图像匹配 获取单应矩阵
   *  返 回 值: H矩阵,显示图像,是否匹配成功flag
   *  参    数: 目标图,场景图,Hessian值,H矩阵,显示图像,是否匹配成功flag
   **************************************************************************/
   MatcheResult featureMatch(Mat objectImg, Mat sceneImg, int Hessian, MatcheResult matches);
   Mat roiImage_correct_Bak2021(Mat tempImage, Mat roiIamge, string folder, Mat &H_matrix, string imageName);
  private:
   vector<shape_based_matching::shapeInfo_producer::Info> infos_shape_;
   line2Dup::Detector detector_shape_;
   string filename_path_;
   int num_features;
   vector<std::string> ids;
   float width_scale;
   float height_scale;

   string type_name;
   string file_path;
   int num;
   vector<std::string> class_ids;
   string class_id;
  };
 }
}