#include "shape_match.h"

namespace algorithm {
namespace shapematch {

Matcher::Matcher() {
 string file_path ="";
 int num =1;
 vector<std::string> class_ids;
 string class_id ="";

  filename_path_ = file_path;
  num_features = num;
  detector_shape_ = line2Dup::Detector(num_features, { 4, 8 });
  width_scale = 0.215 * 2;
  height_scale = 1.41 * 2;
  ReadShapeModel(class_ids, class_id);
}

Matcher::~Matcher() {
}

void Matcher::CreateShapeModel(const Mat& in_img,const string& class_name, const Rect& in_roi,
 const float& angle_start, const float&  angle_end, const float&  angle_step) {
 Mat img = in_img.clone();
  //assert(img.data() && "check your img path");
  if (!img.data) {
    std::cout << "!!! 2d template load error !!!" << std::endl;
    return;
  }
  Rect roi = in_roi;
  roi &= Rect(0, 0, img.cols, img.rows);
  img = img(roi).clone();
  Mat mask = Mat(img.size(), CV_8UC1, { 255 });
  // padding to avoid rotating out
  int padding = (roi.height - roi.width) / 2 + 5;
  cv::Mat padded_img = cv::Mat(img.rows + 2 * padding, img.cols + 2 * padding, img.type(), cv::Scalar::all(0));
  img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));
  cv::Mat padded_mask = cv::Mat(mask.rows + 2 * padding, mask.cols + 2 * padding, mask.type(), cv::Scalar::all(0));
  mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));
  shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
  shapes.angle_range = { angle_start, angle_end };
  shapes.angle_step = angle_step;
  shapes.scale_range = { 1 }; // support just one
  shapes.produce_infos();
  std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
  string class_id = class_name;
  bool is_first = true;
  // for other scales you want to re-extract points:
  // set shapes.scale_range then produce_infos; set is_first = false;
  int first_id = 0;
  float first_angle = 0;
  for (auto& info : shapes.infos) {
    Mat to_show = shapes.src_of(info);
    int templ_id;
    if (is_first) {
      templ_id = detector_shape_.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
      first_id = templ_id;
      first_angle = info.angle;
      if (true) is_first = false;
    } else {
      templ_id = detector_shape_.addTemplate_rotate(class_id, first_id,
                 info.angle - first_angle,
      { shapes.src.cols / 2.0f, shapes.src.rows / 2.0f });
    }
    // 查看提取到的特征点
    auto templ = detector_shape_.getTemplates(class_name, templ_id);
    for (int i = 0; i < templ[0].features.size(); i++) {
      auto feat = templ[0].features[i];
      cv::circle(to_show, { feat.x + templ[0].tl_x, feat.y + templ[0].tl_y }, 3, { 0, 0, 255 }, -1);
    }
    if (templ_id != -1) {
      infos_have_templ.push_back(info);
    }
  }
  detector_shape_.writeClasses(filename_path_ +  "/" + class_id + "%s_templ.yaml");
  shapes.save_infos(infos_have_templ, filename_path_ +  "/" + class_id + "_info.yaml");
  std::cout << "train end" << std::endl << std::endl;
}

bool Matcher::FindShapeModel(const Mat& test_img, const int& matchscore,const vector<std::string>& class_ids, MatchInfo& result) {
  if (!test_img.data) {
    std::cout << "!!! 2d template Image load error !!!" << std::endl;
    return false;
  }
  line2Dup::Detector detector;
  std::vector<shape_based_matching::shapeInfo_producer::Info> infos;
 
  detector = detector_shape_;
  infos = infos_shape_;
  assert(!test_img.empty() && "check your img path");
  int padding = 40;
  cv::Mat padded_img = cv::Mat(test_img.rows + 2 * padding,
                               test_img.cols + 2 * padding, test_img.type(), cv::Scalar::all(0));
  test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));
  int stride = 16;
  int n = padded_img.rows / stride;
  int m = padded_img.cols / stride;
  Rect roi(0, 0, stride * m, stride * n);
  Mat img = padded_img(roi).clone();
  assert(img.isContinuous());
  auto matches = detector.match(img, matchscore, class_ids);
  if (img.channels() == 1) {
   cvtColor(img, img, CV_GRAY2BGR);
  }
  size_t top5 = 3;//最多找10个
  if (top5 > matches.size()) {
   top5 = matches.size();
  }
  result.points.clear();
  vector<Rect> boxes;
  vector<float> scores;
  vector<int> idxs;
  for (size_t i = 0; i < top5; i++) {
   for (int j = 0; i < class_ids.size(); j++) {
    string class_name = class_ids[j];
    auto match = matches[i];
    auto templ = detector.getTemplates(class_name, match.template_id);
    result.x = match.x - padding;
    result.y = match.y - padding;
    result.x /= width_scale;
    result.y /= height_scale;
    result.score = match.similarity;
    result.angle = -infos[match.template_id].angle;

    Rect box;
    box.x = match.x;
    box.y = match.y;
    box.width = templ[0].width;
    box.height = templ[0].height;

    boxes.push_back(box);
    scores.push_back(match.similarity);

    result.points.clear();
    for (int index = 0; index < templ[0].features.size(); index++) {
      result.points.push_back(Point(templ[0].features[index].x / width_scale + result.x,
                                    templ[0].features[index].y / height_scale + result.y));
    }
   }
  }

  cv::dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);
  for (auto idex : idxs) {
   auto match = matches[idex];
   auto templ = detector.getTemplates("test", match.template_id);

   int x = templ[0].width + match.x;
   int y = templ[0].height + match.y;
   int r = templ[0].width / 2;
   cv::Vec3b randColor;
   randColor[0] = rand() % 155 + 100;
   randColor[1] = rand() % 155 + 100;
   randColor[2] = rand() % 155 + 100;

   for (int i = 0; i < templ[0].features.size(); i++) {
    auto feat = templ[0].features[i];
    cv::circle(test_img, { feat.x + match.x, feat.y + match.y }, 2, randColor, -1);
   }

   cv::putText(test_img, to_string(int(round(match.similarity))),
    Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 2, randColor);
   cv::rectangle(test_img, { match.x, match.y }, { x, y }, randColor, 2);

   std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
  }

  if (result.points.size()<1) {
   return false;
  }
  return true;
}

bool Matcher::ReadShapeModel(const vector<std::string>& class_ids, string class_id) {
  detector_shape_ = line2Dup::Detector(num_features, { 4, 8 });
  detector_shape_.readClasses(class_ids, filename_path_ + "/" + class_id + "%s_templ.yaml");
  infos_shape_ = shape_based_matching::shapeInfo_producer::load_infos(filename_path_ + "/" + class_id + "_info.yaml");
  if (infos_shape_.size() < 1) {
   return false;
  }
  return true;
}

void Matcher::ConvertImg(const Mat&src, Mat&dst) {
  if (!src.data) {
    return;
  }
  resize(src, dst, Size(0, 0), width_scale, height_scale);
  dst = dst * 0.05;
  dst.convertTo(dst, CV_8UC1);
}

RotatedRect MatchInfo::rectangle2() {
  return minAreaRect(points);
}
MatchInfo::MatchInfo() {
  x = 0;
  y = 0;
  angle = 0;
  score = 0;
  points.clear();
}
MatchInfo::~MatchInfo() {
}


void Matcher::drawCross(Mat img,const Point& point, const Scalar& color,const int& size, const int& thickness) {
 //绘制横线
 line(img, Point(point.x - size / 2, point.y), Point(point.x + size / 2, point.y), color, thickness, 8, 0);
 //绘制竖线
 line(img, Point(point.x, point.y - size / 2), Point(point.x, point.y + size / 2), color, thickness, 8, 0);
 return;
}


int Matcher::get_roiImage(const Mat& inImg, const Mat& tempImg, Mat& outImg, const Point& in_center) {
 Point center = in_center;
 Rect roiRect;
 roiRect.x = center.x - tempImg.cols / 2;
 if (roiRect.x < 0)	roiRect.x = 0;
 roiRect.y = center.y - tempImg.rows / 2;
 if (roiRect.y < 0)	roiRect.y = 0;
 roiRect.width = tempImg.cols;
 roiRect.height = tempImg.rows;
 Mat src_Img = inImg.clone();
 Mat border_Img;
 copyMakeBorder(src_Img, border_Img, tempImg.rows, tempImg.rows, tempImg.cols, tempImg.cols, BORDER_CONSTANT, Scalar(0));
 if (center.x < 0 || center.y < 0) {
  center.x = int(tempImg.cols / 2);
  center.y = int(tempImg.rows / 2);
 }
 Mat dst_Img;
 bool  centerPoint_within_boundary_flag = NULL;
 Rect rect_border = Rect(center.x - tempImg.cols / 2 + roiRect.width, center.y - tempImg.rows / 2 + roiRect.height, roiRect.width, roiRect.height);
 //rectangle(border_Img, rect_border, Scalar(0, 0, 255), 5, 8);
 if (center.x > 0 && center.x < src_Img.cols && center.y > 0 && center.y < src_Img.rows) {
  dst_Img = border_Img(rect_border);
  outImg = dst_Img.clone();
  return true;
 }
 else {
  return false;
 }
}


MatcheResult Matcher::featureMatch_Bak2021(Mat objectImg, Mat sceneImg, int Hessian, MatcheResult matches) {
 MatcheResult Match = matches;
 //关键点检测
 Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(50);
 vector<KeyPoint> keypoints_object, keypoints_scene;
 detector->detect(objectImg, keypoints_object);
 detector->detect(sceneImg, keypoints_scene);
 if (keypoints_scene.size() < 10 || keypoints_object.size() < 10) {
  Match.Successflag = false;
  return Match;
 }
 //描述子生成（DLCO）
 //Ptr<VGG> vgg_descriptor = VGG::create();
 //Mat descriptors_object, descriptors_scene;
 //vgg_descriptor->compute(objectImg, keypoints_object, descriptors_object);
 //vgg_descriptor->compute(sceneImg, keypoints_scene, descriptors_scene);
 //调用detect函数检测出SIFT特征关键点，保存在vector容器中
 Mat descriptors_object, descriptors_scene;
 detector->detectAndCompute(objectImg, Mat(), keypoints_object, descriptors_object);
 detector->detectAndCompute(sceneImg, Mat(), keypoints_scene, descriptors_scene);
 /*cout << "keypoint_object size:" << keypoints_object.size() << endl;
 cout << "keypoints_scene size:" << keypoints_scene.size() << endl;*/
 //使用FLANN匹配算子进行匹配
 FlannBasedMatcher matcher;
 vector<DMatch> mach;
 matcher.match(descriptors_object, descriptors_scene, mach);;
 //计算出关键点之间距离的最大值和最小值
 double Max_dist = 0.0;
 double Min_dist = 100000.0;
 for (int i = 0; i < descriptors_object.rows; i++) {
  double dist = mach[i].distance;
  if (dist < Min_dist) Min_dist = dist;
  if (dist > Max_dist) Max_dist = dist;
 }
 int Min_dist_times = 2;         //最小距离倍数（初始化为2）
 int Angle_filter_number = 0;    //记录距离过滤后点对数目
 int Distance_filter_number = 0; //记录距离过滤后点对数目
 while (Min_dist_times <= 5) {
  //计算出关键点之间距离的最大值和最小值
  vector<DMatch>good_matches;
  for (int i = 0; i < descriptors_object.rows; i++) {
   if (mach[i].distance < Min_dist_times * Min_dist)
    good_matches.push_back(mach[i]);
  }
  Mat img_maches;
  drawMatches(objectImg, keypoints_object, sceneImg, keypoints_scene, good_matches, img_maches);
  /*cout << " good_matches :" << good_matches.size() <<endl;
  imshow("img_maches",img_maches);*/
  //waitKey(0);
  //计算匹配点对的角度和距离
  vector<int> Keypoints_angle;
  vector<int> Keypoints_distance;
  for (unsigned int i = 0; i < good_matches.size(); i++) {
   Keypoints_angle.push_back(atan((keypoints_object[good_matches[i].queryIdx].pt.y - keypoints_scene[good_matches[i].trainIdx].pt.y)
    / (keypoints_object[good_matches[i].queryIdx].pt.x - keypoints_scene[good_matches[i].trainIdx].pt.x)) * 180 / CV_PI + 0.5);
   Keypoints_distance.push_back(sqrt(pow(abs(keypoints_object[good_matches[i].queryIdx].pt.y - keypoints_scene[good_matches[i].trainIdx].pt.y), 2)
    + pow(abs(keypoints_object[good_matches[i].queryIdx].pt.x - keypoints_scene[good_matches[i].trainIdx].pt.x), 2)) + 0.5);
  }
  //imshow("img_maches", img_maches);
  //备份（防止重新排序之后index值会打乱）
  vector<int> Keypoints_angle_colne = Keypoints_angle;
  vector<int> Keypoints_distance_colne = Keypoints_distance;
  //找出元素相等最多的角度值和数目
  sort(Keypoints_angle.begin(), Keypoints_angle.end());
  int value_angle = 0;
  int value_angle_pre = 0;
  int angle_value = 0;
  int count_angle = 0;
  int min_value = 0;
  for (int i = 0; i < Keypoints_angle.size(); i++) {
   value_angle = Keypoints_angle[i];
   //cout << "value_angle = " << value_angle << endl;
   if (value_angle == value_angle_pre) {
    count_angle++;
    if (count_angle > min_value) {
     min_value = count_angle;
     angle_value = Keypoints_angle[i];
    }
   }
   else {
    count_angle = 0;
   }
   value_angle_pre = value_angle;
  }
  //角度过滤后比配点
  Angle_filter_number = 0;
  int angle_thresh = 2;  //角度范围值
  vector<int>good_Keypoints_distance;
  vector<int>good_Keypoints_index;
  for (unsigned int i = 0; i < good_matches.size(); i++) {
   if ((Keypoints_angle_colne[i] >(angle_value - angle_thresh)) &&
    (Keypoints_angle_colne[i] < (angle_value + angle_thresh))) {      //选出角度相等对应的距离
    good_Keypoints_distance.push_back(Keypoints_distance_colne[i]);
    Angle_filter_number++;
    good_Keypoints_index.push_back(i);
    //cout << "it_angle"<<i<<":"<< Keypoints_angle_colne[i] <<endl;
    //cout << "Keypoints_distance_colne:" << Keypoints_distance_colne[i] << endl;
   }
  }
  if (Angle_filter_number < 10) {
   Min_dist_times++;
   Match.Successflag = false;
   continue;
  }
  //cout << "Angle_filter_number: " << Angle_filter_number <<endl;
  //求余取整
  sort(good_Keypoints_distance.begin(), good_Keypoints_distance.end());
  vector<int>good_Keypoints_distance_int;
  for (int i = 0; i < good_Keypoints_distance.size(); i++) {
   //cout << "good_Keypoints_distance : " << good_Keypoints_distance[i] << endl;
   good_Keypoints_distance_int.push_back(int(good_Keypoints_distance[i] / 10) * 10);
  }
  //筛选的距离进行大小排序
  sort(good_Keypoints_distance_int.begin(), good_Keypoints_distance_int.end());
  //筛选出出现最多的距离值
  int value_distance = 0;
  int value_distance_pre = 0;
  int distance_value = 0;
  int count_distance = 0;
  int min_distance_value = 0;
  int distanceIndex = 0;
  for (int i = 0; i < good_Keypoints_distance_int.size(); i++) {
   value_distance = good_Keypoints_distance_int[i];
   //cout << "value_distance = " << value_distance << endl;
   if (value_distance == value_distance_pre) {
    count_distance++;
    if (count_distance > min_distance_value) {
     min_distance_value = count_distance;
     distance_value = good_Keypoints_distance_int[i];
     distanceIndex = i;
    }
   }
   else {
    count_distance = 0;
   }
   value_distance_pre = value_distance;
  }
  //选出距离相等数目最多的中间值
  int middle_distance = good_Keypoints_distance[distanceIndex - (min_distance_value / 2.0)];
  //vector<int>::iterator distance_it;
  //for(distance_it=good_Keypoints_distance.begin();distance_it!=good_Keypoints_distance.end();distance_it++)
  //	cout<<"it_distance:"<< *distance_it <<endl;
  ////过滤距离
  //int middle_distance = good_Keypoints_distance[(int)(good_Keypoints_distance.size() / 2)]; //获取排序后中间值
  //cout << "middle_distance : " << middle_distance << endl;
  /*middle_distance = 460;*/
  //waitKey(0);
  int distance_thresh = 18;   //距离范围值
  vector<Point2f> obj;
  vector<Point2f> scene;
  vector<DMatch> best_matches;
  Distance_filter_number = 0;
  for (unsigned int i = 0; i < good_Keypoints_index.size(); i++) {
   if ((Keypoints_distance_colne[good_Keypoints_index[i]] >(middle_distance - distance_thresh)) &&
    (Keypoints_distance_colne[good_Keypoints_index[i]] < (middle_distance + distance_thresh))) {
    //cout << "it_diatance"<<Distance_filter_number<<":"<< Keypoints_distance_colne[good_Keypoints_index[i]] << endl;
    obj.push_back(keypoints_object[good_matches[good_Keypoints_index[i]].queryIdx].pt);
    scene.push_back(keypoints_scene[good_matches[good_Keypoints_index[i]].trainIdx].pt);
    best_matches.push_back(good_matches[good_Keypoints_index[i]]);
    Distance_filter_number++;
    /*cout << "Keypoints_distance_colne:" << Keypoints_distance_colne[i] << endl;*/
   }
  }
  Match.objPoints = obj;
  //cout << "Distance_filter_number: " << Distance_filter_number <<endl;
  //小于10对特征点返回
  if (Distance_filter_number < 10) {
   Min_dist_times++;
   Match.Successflag = false;
   continue;
  }
  else {
   Match.Successflag = true;
   Match.match_H = findHomography(obj, scene, RANSAC);
   drawMatches(objectImg, keypoints_object, sceneImg, keypoints_scene, best_matches, Match.drawImg,
    Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
   break;
  }
 }
 return Match;
}
MatcheResult Matcher::featureMatch(Mat objectImg, Mat sceneImg, int Hessian, MatcheResult matches) {
 MatcheResult Match = matches;
 //1.关键点检测
 Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(50);
 vector<KeyPoint> keypoints_object, keypoints_scene;
 detector->detect(objectImg, keypoints_object);
 detector->detect(sceneImg, keypoints_scene);
 if (keypoints_scene.size() < 10 || keypoints_object.size() < 10) {
  cout << "********  keypoints_scene.size()<10 || keypoints_object.size()<10 *******" << endl;
  Match.Successflag = false;
  return Match;
 }
#if 1    // SURF
 //2.调用surf  detect函数检测出SIFT特征关键点，保存在vector容器中
 Mat descriptors_object, descriptors_scene;
 detector->detectAndCompute(objectImg, Mat(), keypoints_object, descriptors_object);
 detector->detectAndCompute(sceneImg, Mat(), keypoints_scene, descriptors_scene);
 if (descriptors_object.cols == 0 || descriptors_scene.cols == 0) {
  cout << "********  descriptors_object.cols == 0||descriptors_scene.cols == 0 *******" << endl;
  Match.Successflag = false;
  return Match;
 }
 //3.使用FLANN匹配算子进行匹配
 FlannBasedMatcher matcher;
 vector<DMatch> match;
 matcher.match(descriptors_object, descriptors_scene, match);
#endif
#if 0           // GMS
 // 采用ORB的方式进行特征关键点的提取；
 Ptr<Feature2D> orb = ORB::create(10000);
 orb.dynamicCast<cv::ORB>()->setFastThreshold(10);
 Mat descriptors_object, descriptors_scene;
 orb->detectAndCompute(objectImg, Mat(), keypoints_object, descriptors_object);
 orb->detectAndCompute(sceneImg, Mat(), keypoints_scene, descriptors_scene);
 if (descriptors_object.cols == 0 || descriptors_scene.cols == 0) {
  cout << "********  descriptors_object.cols == 0||descriptors_scene.cols == 0 *******" << endl;
  Match.Successflag = false;
  return Match;
 }
 //暴力匹配
 Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
 std::vector<DMatch> match;
 matcher->match(descriptors_object, descriptors_scene, match);
#endif
 //4.计算出关键点之间距离的最大值和最小值
 double Max_dist = 0.0;
 double Min_dist = 100000.0;
 for (int i = 0; i < descriptors_object.rows; i++) {
  double dist = match[i].distance;
  if (dist < Min_dist) Min_dist = dist;
  if (dist > Max_dist) Max_dist = dist;
 }
 int Min_dist_times = 2;         //最小距离倍数（初始化为2）
 int Angle_filter_number = 0;    //记录角度过滤后点对数目
 int Distance_filter_number = 0; //记录距离过滤后点对数目
 while (Min_dist_times <= 5) {
  //5.计算出关键点之间距离的最大值和最小值
  vector<DMatch>good_matches;
  for (int i = 0; i < descriptors_object.rows; i++) {
   if (match[i].distance < Min_dist_times * Min_dist)
    good_matches.push_back(match[i]);
  }
  Mat img_matches;
  drawMatches(objectImg, keypoints_object, sceneImg, keypoints_scene, good_matches, img_matches);
  //  GMS
  //matchGMS(objectImg.size(), sceneImg.size(), keypoints_object, keypoints_scene, match, good_matches, false, false);
  //计算匹配点对的角度和距离
  vector<int> Keypoints_angle;
  vector<int> Keypoints_distance;
  for (unsigned int i = 0; i < good_matches.size(); i++) {
   Keypoints_angle.push_back(atan((keypoints_object[good_matches[i].queryIdx].pt.y - keypoints_scene[good_matches[i].trainIdx].pt.y)
    / (keypoints_object[good_matches[i].queryIdx].pt.x - keypoints_scene[good_matches[i].trainIdx].pt.x)) * 180 / CV_PI + 0.5);
   Keypoints_distance.push_back(sqrt(pow(abs(keypoints_object[good_matches[i].queryIdx].pt.y - keypoints_scene[good_matches[i].trainIdx].pt.y), 2)
    + pow(abs(keypoints_object[good_matches[i].queryIdx].pt.x - keypoints_scene[good_matches[i].trainIdx].pt.x), 2)) + 0.5);
  }
  //备份（防止重新排序之后index值会打乱）
  vector<int> Keypoints_angle_colne = Keypoints_angle;
  vector<int> Keypoints_distance_colne = Keypoints_distance;
  //6.找出元素相等最多的角度值和数目
  sort(Keypoints_angle.begin(), Keypoints_angle.end());
  int value_angle = 0;
  int value_angle_pre = 0;
  int angle_value = 0;
  int count_angle = 0;
  int min_value = 0;
  for (int i = 0; i < Keypoints_angle.size(); i++) {
   value_angle = Keypoints_angle[i];
   //cout << "value_angle = " << value_angle << endl;
   if (value_angle == value_angle_pre) {
    count_angle++;
    if (count_angle > min_value) {
     min_value = count_angle;
     angle_value = Keypoints_angle[i];
    }
   }
   else {
    count_angle = 0;
   }
   value_angle_pre = value_angle;
  }
  //角度过滤后比配点
  Angle_filter_number = 0;
  int angle_thresh = 2;  //角度范围值
  vector<int>good_Keypoints_distance;
  vector<int>good_Keypoints_index;
  for (unsigned int i = 0; i < good_matches.size(); i++) {
   if ((Keypoints_angle_colne[i] >(angle_value - angle_thresh)) &&
    (Keypoints_angle_colne[i] < (angle_value + angle_thresh))) {      //选出角度相等对应的距离
    good_Keypoints_distance.push_back(Keypoints_distance_colne[i]);
    Angle_filter_number++;
    good_Keypoints_index.push_back(i);
    //cout << "it_angle"<<i<<":"<< Keypoints_angle_colne[i] <<endl;
    //cout << "Keypoints_distance_colne:" << Keypoints_distance_colne[i] << endl;
   }
  }
  if (Angle_filter_number < 10) {
   Min_dist_times++;
   Match.Successflag = false;
   continue;
  }
  //cout << "Angle_filter_number: " << Angle_filter_number <<endl;
  //求余取整
  sort(good_Keypoints_distance.begin(), good_Keypoints_distance.end());
  vector<int>good_Keypoints_distance_int;
  for (int i = 0; i < good_Keypoints_distance.size(); i++) {
   //cout << "good_Keypoints_distance : " << good_Keypoints_distance[i] << endl;
   good_Keypoints_distance_int.push_back(int(good_Keypoints_distance[i] / 10) * 10);
  }
  //筛选的距离进行大小排序
  sort(good_Keypoints_distance_int.begin(), good_Keypoints_distance_int.end());
  /*for (int i = 0; i < good_Keypoints_distance_int.size(); i++)
  {
  cout << "good_Keypoints_distance_int " <<good_Keypoints_distance_int[i]<<endl;
  }*/
  //筛选出出现最多的距离值
  int value_distance = 0;
  int value_distance_pre = 0;
  int distance_value = 0;
  int count_distance = 0;
  int min_distance_value = 0;
  int distanceIndex = 0;
  for (int i = 0; i < good_Keypoints_distance_int.size(); i++) {
   value_distance = good_Keypoints_distance_int[i];
   //cout << "value_distance = " << value_distance << endl;
   if (value_distance == value_distance_pre) {
    count_distance++;
    if (count_distance > min_distance_value) {
     min_distance_value = count_distance;
     distance_value = good_Keypoints_distance_int[i];
     distanceIndex = i;
    }
   }
   else {
    count_distance = 0;
   }
   value_distance_pre = value_distance;
  }
  //选出距离相等数目最多的中间值
  int middle_distance = good_Keypoints_distance[distanceIndex - (min_distance_value / 2.0)];
  //cout<<"middle_distance : " << good_Keypoints_distance[distanceIndex - (min_distance_value/2.0)] << endl;
  /*cout<<"min_distance_value = " << min_distance_value << endl;
  cout << "value_distance = " << distance_value << endl;*/
  //waitKey(0);
  //vector<int>::iterator distance_it;
  //for(distance_it=good_Keypoints_distance.begin();distance_it!=good_Keypoints_distance.end();distance_it++)
  //	cout<<"it_distance:"<< *distance_it <<endl;
  ////过滤距离
  //int middle_distance = good_Keypoints_distance[(int)(good_Keypoints_distance.size() / 2)]; //获取排序后中间值
  //cout << "middle_distance : " << middle_distance << endl;
  /*middle_distance = 460;*/
  //waitKey(0);
  int distance_thresh = 18;   //距离范围值
  vector<Point2f> obj;
  vector<Point2f> scene;
  vector<DMatch> best_matches;
  Distance_filter_number = 0;
  for (unsigned int i = 0; i < good_Keypoints_index.size(); i++) {
   if ((Keypoints_distance_colne[good_Keypoints_index[i]] >(middle_distance - distance_thresh)) &&
    (Keypoints_distance_colne[good_Keypoints_index[i]] < (middle_distance + distance_thresh))) {
    //cout << "it_diatance"<<Distance_filter_number<<":"<< Keypoints_distance_colne[good_Keypoints_index[i]] << endl;
    obj.push_back(keypoints_object[good_matches[good_Keypoints_index[i]].queryIdx].pt);
    scene.push_back(keypoints_scene[good_matches[good_Keypoints_index[i]].trainIdx].pt);
    best_matches.push_back(good_matches[good_Keypoints_index[i]]);
    Distance_filter_number++;
    /*cout << "Keypoints_distance_colne:" << Keypoints_distance_colne[i] << endl;*/
   }
  }
  Match.objPoints = obj;
  //cout << "Distance_filter_number: " << Distance_filter_number <<endl;
  //小于10对特征点返回
  if (Distance_filter_number < 10) {
   Min_dist_times++;
   Match.Successflag = false;
   continue;
  }
  else {
   Match.Successflag = true;
   Match.match_H = findHomography(obj, scene, RANSAC);
   drawMatches(objectImg, keypoints_object, sceneImg, keypoints_scene, best_matches, Match.drawImg,
    Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
   break;
  }
 }
 /*cout << "Min_dist_times=" << Min_dist_times << endl;
 cout << "Angle_filter_number:" << Angle_filter_number << endl;
 cout << "Distance_filter_number:" << Distance_filter_number << endl;*/
 /*imshow("Match.drawImg", Match.drawImg);
 waitKey(0);*/
 return Match;
}

int Matcher::match_Feedback(Mat &temp_Img, Mat &src_Img, Point &matchCenter,const string& folder, const string& imageName) {
 /*********************特征点匹配***************************/
 Mat temp_Img_resize;
 Mat src_Img_resize;
 MatcheResult temp_src_match;      //初始化
 Mat temp_Img_clone = temp_Img.clone();
 Mat src_Img_clone = src_Img.clone();
 //图像降采样
 resize(temp_Img, temp_Img_resize, Size(temp_Img.cols / 2, temp_Img.rows / 2));
 resize(src_Img, src_Img_resize, Size(src_Img.cols / 2, src_Img.rows / 2));
 temp_src_match = featureMatch(temp_Img_resize, src_Img_resize, minHessian, temp_src_match);
 //cout << "Match flag:" << temp_src_match.Successflag << endl;
 if (temp_src_match.Successflag) {   //特征匹配成功
  LOG(INFO) << "->%s\n", "GMS match successfully.";
  //模板图中心
  int src_center_x = temp_Img_resize.cols / 2;
  int src_center_y = temp_Img_resize.rows / 2;
  //计算obj图特征点中心
  long int objPoint_total_X = 0;
  long int objPoint_total_Y = 0;
  for (int i = 0; i < temp_src_match.objPoints.size(); i++) {
   objPoint_total_X += temp_src_match.objPoints[i].x;
   objPoint_total_Y += temp_src_match.objPoints[i].y;
  }
  Point2f objCenter_Keypoint;
  objCenter_Keypoint.x = objPoint_total_X / temp_src_match.objPoints.size();
  objCenter_Keypoint.y = objPoint_total_Y / temp_src_match.objPoints.size();
  //模板图中心与obj图特征点中心距离
  Point2f srcCenter_objCenter_distance;
  srcCenter_objCenter_distance.x = src_center_x - objCenter_Keypoint.x;
  srcCenter_objCenter_distance.y = src_center_y - objCenter_Keypoint.y;
  //校正点位
  vector<Point2f> obj_corners(5);
  obj_corners[0] = cvPoint(0, 0);
  obj_corners[1] = cvPoint(temp_Img_resize.cols, 0);
  obj_corners[2] = cvPoint(temp_Img_resize.cols, temp_Img_resize.rows);
  obj_corners[3] = cvPoint(0, temp_Img_resize.rows);
  obj_corners[4] = objCenter_Keypoint;
  vector<Point2f> scene_corners(5);
  // 判断计算的单应性矩阵的正确性；
  if (temp_src_match.match_H.cols) {
   //进行透视变换
   LOG(INFO) << "->enter perspectiveTransform().";
   perspectiveTransform(obj_corners, scene_corners, temp_src_match.match_H);        //进行透视变换时容易出错的;
  }
  else {
   LOG(INFO) << "->%s\n", " ***** get the homography match_H is empty  ... ***** ";
  }
  //进行透视变换
  //perspectiveTransform(obj_corners, scene_corners, temp_src_match.match_H);        //进行透视变换时容易出错的；
  //绘制出角点之间的直线
  Mat last_maches = temp_src_match.drawImg;
  Mat Keypoints_maches = last_maches.clone();
  line(Keypoints_maches, scene_corners[0] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), scene_corners[1] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), Scalar(255, 0, 0), 2);
  line(Keypoints_maches, scene_corners[1] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), scene_corners[2] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), Scalar(255, 0, 0), 2);
  line(Keypoints_maches, scene_corners[2] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), scene_corners[3] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), Scalar(255, 0, 0), 2);
  line(Keypoints_maches, scene_corners[3] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), scene_corners[0] + Point2f(static_cast<float>(temp_Img_resize.cols), 0), Scalar(255, 0, 0), 2);
  //circle(last_maches, (scene_corners[4] + Point2f(static_cast<float>(temp_Img_resize.cols), 0)), 3, Scalar(255, 255, 0), -1);
  circle(last_maches, (scene_corners[4] + Point2f(static_cast<float>(temp_Img_resize.cols), 0)), 3, Scalar(0, 0, 255), -1);
  circle(last_maches, (scene_corners[4] + Point2f(static_cast<float>(temp_Img_resize.cols), 0) + srcCenter_objCenter_distance), 3, Scalar(255, 0, 0), -1);
  //circle(last_maches, obj_corners[4], 3, Scalar(255, 255, 0), -1);
  circle(last_maches, objCenter_Keypoint, 3, Scalar(0, 0, 255), -1);
  circle(last_maches, objCenter_Keypoint + srcCenter_objCenter_distance, 3, Scalar(255, 0, 0), -1);
  // 为了解决处理时间，改图暂不保存
  //string keypoints_match = folder;
  //keypoints_match.append(imageName);
  //keypoints_match.append("_keypoints_maches.jpg");
  //if(!keypoints_match.empty())
  //	imwrite(keypoints_match, Keypoints_maches);
  //匹配中心(由于前面降采样一倍，需要还原回来中心*2)
  matchCenter.x = (scene_corners[4].x + srcCenter_objCenter_distance.x) * 2;
  matchCenter.y = (scene_corners[4].y + srcCenter_objCenter_distance.y) * 2;
  if (matchCenter.x > 1920 || matchCenter.y > 1080)      //  匹配异常;
   return  0;
  drawCross(last_maches, (scene_corners[4] + Point2f(static_cast<float>(temp_Img_resize.cols), 0) + srcCenter_objCenter_distance), Scalar(0, 255, 0), 30, 1);
  //透视后中心
  rectangle(last_maches, Point(scene_corners[4].x + srcCenter_objCenter_distance.x + src_center_x, scene_corners[4].y + srcCenter_objCenter_distance.y - src_center_y),
   Point(scene_corners[4].x + srcCenter_objCenter_distance.x + 3 * src_center_x, scene_corners[4].y + srcCenter_objCenter_distance.y + src_center_y), Scalar(255, 0, 0), 2, 8);
  string last_macheImg = folder;
  last_macheImg.append(imageName);
  //last_macheImg.append("_GMS_matchImg.jpg");
  last_macheImg.append("_SURF_matchImg.jpg");
  if (!last_maches.empty())
   imwrite(last_macheImg, last_maches);
  //LOG(INFO) <<"->%s\n", "GMS match end.");
  LOG(INFO) << "SURF match end.";
 }
 else { //模板匹配
  LOG(INFO) << "SURF match failed.";
  LOG(INFO) << "entry match_Template().";
  //fflush(fp);
  int resultImg_cols = src_Img.cols - temp_Img.cols + 1;
  int resultImg_rows = src_Img.rows - temp_Img.rows + 1;
  if (resultImg_cols < 0 || resultImg_rows < 0) {
   LOG(INFO) << "->match_Template match failed resultImg.\n";
   LOG(INFO) << "->resultImg_cols <0 || resultImg_rows < 0.\n";
   return 0;
  }
  Mat resultImg;
  double minValue, maxValue;
  Point minLoc, maxLoc;
  resultImg = Mat::zeros(resultImg_cols, resultImg_rows, CV_32FC1);    //初始化；
  matchTemplate(src_Img, temp_Img, resultImg, CV_TM_CCOEFF_NORMED);
  minMaxLoc(resultImg, &minValue, &maxValue, &minLoc, &maxLoc);
  LOG(INFO) << "->match value: %f\n" << maxValue;
  //fflush(fp);
  if (maxValue < 0.3) {
   LOG(INFO) << "->match template failed!\n";
   LOG(INFO) << "->please check up Result image correctly ....\n";
   return 0;
  }
  //识别区域坐标
  matchCenter.x = maxLoc.x + temp_Img.cols / 2;
  matchCenter.y = maxLoc.y + temp_Img.rows / 2;
  LOG(INFO) << "->match_Template matchCenter: [%d, %d]\n" << matchCenter.x << " y:" << matchCenter.y;
  //fflush(fp);
  drawCross(src_Img_clone, matchCenter, Scalar(0, 0, 255), 30, 2);
  rectangle(src_Img_clone, Point(maxLoc.x, maxLoc.y), Point(maxLoc.x + temp_Img.cols, maxLoc.y + temp_Img.rows), Scalar(0, 255, 0), 2);
  string matchImage_str = folder;
  matchImage_str.append(imageName);
  matchImage_str.append("_matchImage.jpg");
  if (!src_Img_clone.empty())
   imwrite(matchImage_str, src_Img_clone);
  LOG(INFO) << "->%s\n", "match_Template match successfully.";
  LOG(INFO) << "->%s\n", "match_Template end.";
  //fflush(fp);
  resultImg.release();
 }
 return true;
}

Mat Matcher::roiImage_correct_Bak2021(Mat tempImage, Mat roiIamge, string folder, Mat &H_matrix, string imageName) {
 /********************识别区域矫正**************************/
 Ptr<SurfFeatureDetector> Roi_detector = SurfFeatureDetector::create(30);
 vector<KeyPoint> Roi_keypoints_object, Roi_keypoints_scene;
 Mat Roi_descriptors_object, Roi_descriptors_scene;
 Mat descriptors_object, descriptors_scene;
 Roi_detector->detectAndCompute(tempImage, Mat(), Roi_keypoints_object, Roi_descriptors_object);
 Roi_detector->detectAndCompute(roiIamge, Mat(), Roi_keypoints_scene, Roi_descriptors_scene);
 if (Roi_descriptors_scene.cols == 0 && Roi_descriptors_scene.rows == 0) {
  LOG(INFO) << "Cant find the featurePoints,Please check the image...\n";
  return roiIamge;
 }
 //使用FLANN匹配算子进行匹配
 FlannBasedMatcher Roi_matcher;
 vector<DMatch> Roi_mach;
 Roi_matcher.match(Roi_descriptors_object, Roi_descriptors_scene, Roi_mach);
 //计算出关键点之间距离的最大值和最小值
 double Roi_Max_dist = 0;
 double Roi_Min_dist = 1000;
 for (int i = 0; i < Roi_descriptors_object.rows; i++) {
  double Roi_dist = Roi_mach[i].distance;
  if (Roi_dist < Roi_Min_dist) Roi_Min_dist = Roi_dist;
  if (Roi_dist > Roi_Max_dist) Roi_Max_dist = Roi_dist;
 }
 //存下匹配小距离的点对
 vector<DMatch>Roi_good_matches;
 for (int i = 0; i < Roi_descriptors_object.rows; i++) {
  if (Roi_mach[i].distance < 4 * Roi_Min_dist)
   Roi_good_matches.push_back(Roi_mach[i]);
 }
 Mat img_maches;
 drawMatches(tempImage, Roi_keypoints_object, roiIamge, Roi_keypoints_scene, Roi_good_matches, img_maches);
 //从匹配成功的匹配对中获取关键点
 vector<Point2f> Roi_obj;
 vector<Point2f> Roi_scene;
 for (int i = 0; i < Roi_good_matches.size(); i++) {
  Roi_obj.push_back(Roi_keypoints_object[Roi_good_matches[i].queryIdx].pt);
  Roi_scene.push_back(Roi_keypoints_scene[Roi_good_matches[i].trainIdx].pt);
 }
 vector<unsigned char> listpoints;
 Mat H_test = findHomography(Roi_obj, Roi_scene, RANSAC, 3, listpoints);    //计算透视变换
 vector< DMatch > goodgood_matches;
 for (int i = 0; i < listpoints.size(); i++) {
  if ((int)listpoints[i]) {
   goodgood_matches.push_back(Roi_good_matches[i]);
  }
 }
 Mat Homgimg_matches;
 drawMatches(tempImage, Roi_keypoints_object, roiIamge, Roi_keypoints_scene,
  goodgood_matches, Homgimg_matches, Scalar::all(-1), Scalar::all(-1),
  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 if (H_test.empty() || Roi_good_matches.size() < 12) {
  return roiIamge;
 }
 vector<Point2f> Roi_obj_corners(4);
 Roi_obj_corners[0] = cvPoint(0, 0);
 Roi_obj_corners[1] = cvPoint(tempImage.cols, 0);
 Roi_obj_corners[2] = cvPoint(tempImage.cols, tempImage.rows);
 Roi_obj_corners[3] = cvPoint(0, tempImage.rows);
 vector<Point2f> Roi_scene_corners(4);
 //进行透视变换
 perspectiveTransform(Roi_obj_corners, Roi_scene_corners, H_test);
 // 求Hisee矩阵的逆变换；
 H_matrix = H_test;
 //绘制出角点之间的直线
 line(Homgimg_matches, Roi_scene_corners[0] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[1] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 line(Homgimg_matches, Roi_scene_corners[1] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[2] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 line(Homgimg_matches, Roi_scene_corners[2] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[3] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 line(Homgimg_matches, Roi_scene_corners[3] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[0] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 string Homgimg_matches_str = folder;
 Homgimg_matches_str.append(imageName);
 Homgimg_matches_str.append("_Rhomgimg_matches.jpg");
 if (!Homgimg_matches.empty())
  imwrite(Homgimg_matches_str, Homgimg_matches);
 //获取两个图4个角点对应的单应矩阵
 Mat temp_roi_H = findHomography(Roi_scene_corners, Roi_obj_corners, RANSAC);
 //return roiIamge;
 if (temp_roi_H.cols > 1) {
  if (
   // 限制缩放
   fabs(temp_roi_H.at<double>(0, 0) - 1) < 0.35
   && fabs(temp_roi_H.at<double>(1, 1) - 1) < 0.35
   //// 限制斜切
   &&	fabs(temp_roi_H.at<double>(2, 0)) < 5e-4
   &&	fabs(temp_roi_H.at<double>(2, 1)) < 5e-4
   //// 限制
   //&&	fabs(temp_roi_H.at<double>(0, 1)) < 5e-1
   //&&	fabs(temp_roi_H.at<double>(1, 0)) < 5e-1
   ) {
   Mat resultImage;
   warpPerspective(roiIamge, resultImage, temp_roi_H, roiIamge.size());
   return resultImage;
  }
  else {
   return roiIamge;
  }
 }
 else {
  return roiIamge;
 }
}

Mat Matcher::roiImage_correct(Mat tempImage, Mat roiIamge, string folder, Mat &H_matrix, string imageName) {
 /********************识别区域矫正**************************/
#if 1
 /*SURF方法*/
 Ptr<SurfFeatureDetector> Roi_detector = SurfFeatureDetector::create(30);
 vector<KeyPoint> Roi_keypoints_object, Roi_keypoints_scene;
 Mat Roi_descriptors_object, Roi_descriptors_scene;
 Mat descriptors_object, descriptors_scene;
 Roi_detector->detectAndCompute(tempImage, Mat(), Roi_keypoints_object, Roi_descriptors_object);
 Roi_detector->detectAndCompute(roiIamge, Mat(), Roi_keypoints_scene, Roi_descriptors_scene);
 if (Roi_descriptors_scene.cols == 0 && Roi_descriptors_scene.rows == 0) {
  LOG(INFO) << "Cant find the featurePoints,Please check the image...\n";
  //return roiIamge;
 }
 //使用FLANN匹配算子进行匹配
 FlannBasedMatcher Roi_matcher;
 vector<DMatch> Roi_match;
 Roi_matcher.match(Roi_descriptors_object, Roi_descriptors_scene, Roi_match);
#endif
#if 0
 // 采用ORB的方式进行特征关键点的提取；
 Ptr<Feature2D> orb = ORB::create(10000);
 orb.dynamicCast<cv::ORB>()->setFastThreshold(20);
 vector<KeyPoint> Roi_keypoints_object, Roi_keypoints_scene;
 Mat Roi_descriptors_object, Roi_descriptors_scene;
 orb->detectAndCompute(tempImage, Mat(), Roi_keypoints_object, Roi_descriptors_object);
 orb->detectAndCompute(roiIamge, Mat(), Roi_keypoints_scene, Roi_descriptors_scene);
 if (Roi_descriptors_scene.cols == 0 && Roi_descriptors_scene.rows == 0) {
  LOG(INFO) << "Cant find the featurePoints,Please check the image...\n");
  return roiIamge;
 }
 //暴力匹配
 Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
 std::vector<DMatch> Roi_match;
 matcher->match(Roi_descriptors_object, Roi_descriptors_scene, Roi_match);
#endif
 //计算出关键点之间距离的最大值和最小值
 double Roi_Max_dist = 0;
 double Roi_Min_dist = 1000;
 for (int i = 0; i < Roi_descriptors_object.rows; i++) {
  double Roi_dist = Roi_match[i].distance;
  if (Roi_dist < Roi_Min_dist) Roi_Min_dist = Roi_dist;
  if (Roi_dist > Roi_Max_dist) Roi_Max_dist = Roi_dist;
 }
 //存下匹配小距离的点对
 vector<DMatch>Roi_good_matches;
 for (int i = 0; i < Roi_descriptors_object.rows; i++) {
  if (Roi_match[i].distance < 4 * Roi_Min_dist)
   Roi_good_matches.push_back(Roi_match[i]);
 }
#if 0
 matchGMS(tempImage.size(), roiIamge.size(), Roi_keypoints_object, Roi_keypoints_scene, Roi_match, Roi_good_matches, false, false);
 //std::cout << "matchesGMS: " << Roi_good_matches.size() << std::endl;
#endif
#if 1
 Mat img_matches;
 drawMatches(tempImage, Roi_keypoints_object, roiIamge, Roi_keypoints_scene, Roi_good_matches, img_matches);
#endif
 //从匹配成功的匹配对中获取关键点
 vector<Point2f> Roi_obj;
 vector<Point2f> Roi_scene;
 for (int i = 0; i < Roi_good_matches.size(); i++) {
  Roi_obj.push_back(Roi_keypoints_object[Roi_good_matches[i].queryIdx].pt);
  Roi_scene.push_back(Roi_keypoints_scene[Roi_good_matches[i].trainIdx].pt);
 }
 vector<unsigned char> listpoints;
 Mat H_test;
 try {
  H_test = findHomography(Roi_obj, Roi_scene, RANSAC, 3, listpoints);    //计算透视变换
  if (H_test.empty() || Roi_good_matches.size() < 12) {
   return roiIamge;
  }
 }
 catch (cv::Exception) {
  return roiIamge;
 }
 vector< DMatch > goodgood_matches;
 for (int i = 0; i < listpoints.size(); i++) {
  if ((int)listpoints[i]) {
   goodgood_matches.push_back(Roi_good_matches[i]);
  }
 }
 Mat Homgimg_matches;
 drawMatches(tempImage, Roi_keypoints_object, roiIamge, Roi_keypoints_scene,
  goodgood_matches, Homgimg_matches, Scalar::all(-1), Scalar::all(-1),
  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 vector<Point2f> Roi_obj_corners(4);
 Roi_obj_corners[0] = cvPoint(0, 0);
 Roi_obj_corners[1] = cvPoint(tempImage.cols, 0);
 Roi_obj_corners[2] = cvPoint(tempImage.cols, tempImage.rows);
 Roi_obj_corners[3] = cvPoint(0, tempImage.rows);
 vector<Point2f> Roi_scene_corners(4);
 //进行透视变换
 perspectiveTransform(Roi_obj_corners, Roi_scene_corners, H_test);
 // 求Hisee矩阵的逆变换；
 //H_matrix = H_test;
 //绘制出角点之间的直线
 line(Homgimg_matches, Roi_scene_corners[0] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[1] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 line(Homgimg_matches, Roi_scene_corners[1] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[2] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 line(Homgimg_matches, Roi_scene_corners[2] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[3] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 line(Homgimg_matches, Roi_scene_corners[3] + Point2f(static_cast<float>(tempImage.cols), 0), Roi_scene_corners[0] + Point2f(static_cast<float>(tempImage.cols), 0), Scalar(255, 0, 123), 2);
 string Homgimg_matches_str = folder;
 Homgimg_matches_str.append(imageName);
 Homgimg_matches_str.append("_correct_matches.jpg");
 //if(!Homgimg_matches.empty())
 //    imwrite(Homgimg_matches_str, Homgimg_matches);
 //获取两个图4个角点对应的单应矩阵
 Mat temp_roi_H = findHomography(Roi_scene_corners, Roi_obj_corners, RANSAC);
 H_matrix = temp_roi_H;
 //double temp = invert(temp_roi_H,H_matrix,DECOMP_LU);
 //return roiIamge;
 if (temp_roi_H.cols > 1) {
  if (
   // 限制缩放
   fabs(temp_roi_H.at<double>(0, 0) - 1) < 0.35
   && fabs(temp_roi_H.at<double>(1, 1) - 1) < 0.35
   //// 限制斜切
   &&	fabs(temp_roi_H.at<double>(2, 0)) < 5e-4
   &&	fabs(temp_roi_H.at<double>(2, 1)) < 5e-4
   //// 限制
   //&&	fabs(temp_roi_H.at<double>(0, 1)) < 5e-1
   //&&	fabs(temp_roi_H.at<double>(1, 0)) < 5e-1
   ) {
   Mat resultImage;
   warpPerspective(roiIamge, resultImage, temp_roi_H, roiIamge.size());
   return resultImage;
  }
  else {
   return roiIamge;
  }
 }
 else {
  return roiIamge;
 }
}

void Matcher::ucharToMat(unsigned char* uchar_image, int image_height, int image_width, int channels, cv::Mat &dst_image) {
 cv::Mat cv_image;
 if (channels == 1) {
  cv_image = cv::Mat::zeros(image_height, image_width, CV_8UC1);
 }
 if (channels == 3) {
  cv_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
 }
 else {}
 cv_image.data = uchar_image;
 dst_image = cv_image;
}
}
}
