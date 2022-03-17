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
   bool  Successflag; //�Ƿ�ƥ��ɹ�
   Mat match_H; //ƥ�䵥Ӧ����
   Mat drawImg; //����ͼ
   vector<Point2f> objPoints;//Ŀ��ͼ�ϵ�ʶ��㼯
  } MatcheResult;


  class DLLEXPORT Matcher {
  public:
   /**************************************************************************
   *  ��������: Matcher 
   *  ����˵��: ����ģ�壬����Ϊ1 ����ת��ģ��
   *  �� �� ֵ: ��
   *  ��    ��:  pathģ���ļ��и�Ŀ¼��type �ļ��������ļ������֣�num������ȡ��Ŀ
   **************************************************************************/
   //Matcher(string file_path, int num, const vector<std::string>& class_ids, string class_id);
   Matcher();
   ~Matcher();
   /**************************************************************************
   *  ��������: CreateShapeModel
   *  ����˵��: ����ģ�壬����Ϊ1 ����ת��ģ��
   *  �� �� ֵ: ��
   *  ��    ��: name ͼ������ 
   **************************************************************************/
   void CreateShapeModel(const Mat& in_img, const string& class_name, const Rect& in_roi, 
    const float& angle_start, const float&  angle_end, const float&  angle_step);
   bool ReadShapeModel(const vector<std::string>& ids, string type);
   bool FindShapeModel(const Mat& test_img, const int& matchscore, const vector<std::string>& class_ids,
    MatchInfo& result);
   void ConvertImg(const Mat&src, Mat&dst);

   /**************************************************************************
   *  ��������: drawCross
   *  ����˵��: ���ݵ���Ϣ����ʮ�ּ�line
   *  �� �� ֵ: ��
   *  ��    ��: RGBͼ,����Ϣ,��ɫ,��ȴ�С���߿�
   **************************************************************************/
   void drawCross(Mat img, const Point& point, const Scalar& color, const int& size, 
    const int& thickness);
   /**************************************************************************
   *  ��������: get_roiImage
   *  ����˵��: ��ȡʶ������ͼƬ
   *  �� �� ֵ: �Ƿ��ȡ�ɹ�flag
   *  ��    ��: ԭͼ,ģ��ͼ,���ͼ��,���ĵ�
   **************************************************************************/
   int get_roiImage(const Mat& inImg, const Mat& tempImg, Mat& outImg, const Point& in_center);
   /**************************************************************************
   *  ��������: match_Feedback
   *  ����˵������ȡƥ������
   *  �� �� ֵ: ƥ��ɹ�����1,���򷵻�-1
   *  ��    ��: Ŀ��ͼ,����ͼ,ƥ�����ģ��ļ�Ŀ¼
   **************************************************************************/
   int match_Feedback(Mat &temp_Img, Mat &in_Img, Point &matchCenter, const string& folder, const string& imageName);

   Mat roiImage_correct(Mat tempImage, Mat roiIamge, string folder, Mat &H_matrix, string imageName);
   /**************************************************************************
   *  ��������: ucharToMat
   *  ����˵��: ����unsigned char* ���黹ԭMat����ͼƬ
   *  �� �� ֵ: ��
   *  ��    ��: unsigned char* ����,ͼ���,ͼ���,ͼ��ͨ��������ԭMat����ͼƬ
   **************************************************************************/
   void ucharToMat(unsigned char* uchar_image, int image_height, int image_width, int channels, cv::Mat &dst_image);

   void coordinate_conversion(cv::Mat srcImg, cv::Mat H_inverse_matrix, cv::Rect& roi_rect);
   MatcheResult featureMatch_Bak2021(Mat objectImg, Mat sceneImg, int Hessian, MatcheResult matches);
   /**************************************************************************
   *  ��������: featureMatch
   *  ����˵����match_Feedback�Ӻ��� ͼ��ƥ�� ��ȡ��Ӧ����
   *  �� �� ֵ: H����,��ʾͼ��,�Ƿ�ƥ��ɹ�flag
   *  ��    ��: Ŀ��ͼ,����ͼ,Hessianֵ,H����,��ʾͼ��,�Ƿ�ƥ��ɹ�flag
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