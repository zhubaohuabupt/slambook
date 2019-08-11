#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void bundleAdjustmentG2o(const vector<Point3f> points_3d,
                         const vector<Point2f> points_2d, const Mat &K, Mat &R,
                         Mat &t);

void bundleAdjustmentCeres(vector<Point3f> points_3d,
                           const vector<Point2f> points_2d, const Mat &K,
                           Mat &R, Mat &t, bool use_auto_diff);

int main(int argc, char **argv) {
  if (argc != 5) {
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  Mat d1 = imread(argv[3],
                  CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (DMatch m : matches) {
    ushort d = d1.ptr<unsigned short>(
        int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0) // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t,
           false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  cout << "calling bundle adjustment" << endl;
  
  bundleAdjustmentG2o(pts_3d, pts_2d, K, R, t);
  bundleAdjustmentCeres(pts_3d, pts_2d, K, R, t, true);
  bundleAdjustmentCeres(pts_3d, pts_2d, K, R, t, false);
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB"
  // );
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离,
  //即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                 (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void bundleAdjustmentG2o(const vector<Point3f> points_3d,
                         const vector<Point2f> points_2d, const Mat &K, Mat &R,
                         Mat &t) {
  // 初始化g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>
      Block; // pose 维度为 6, landmark 维度为 3
  Block::LinearSolverType *linearSolver =
      new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
  Block *solver_ptr = new Block(linearSolver); // 矩阵块求解器
  g2o::OptimizationAlgorithmLevenberg *solver =
      new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  // vertex
  g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap(); // camera pose
  Eigen::Matrix3d R_mat;
  R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
      R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0, 0),
                                                        t.at<double>(1, 0),
                                                        t.at<double>(2, 0))));
  optimizer.addVertex(pose);

  int index = 1;
  for (const Point3f p : points_3d) // landmarks
  {
    g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();
    point->setId(index++);
    point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
    point->setMarginalized(true); // g2o 中必须设置 marg 参见第十讲内容
    optimizer.addVertex(point);
  }

  // parameter: camera intrinsics
  g2o::CameraParameters *camera = new g2o::CameraParameters(
      K.at<double>(0, 0),
      Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
  camera->setId(0);
  optimizer.addParameter(camera);

  // edges
  index = 1;
  for (const Point2f p : points_2d) {
    g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId(index);
    edge->setVertex(
        0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(index)));
    edge->setVertex(1, pose);
    edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
    edge->setParameterId(0, 0);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }
  cout << "\n<============= G2o optimization =============>\n" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "Costs time: " << time_used.count() << " seconds." << endl;

  cout << "T=" << endl
       << Eigen::Isometry3d(pose->estimate()).matrix() << "\n\n"
       << endl;
}

// Reprojected cost to used numeric diff of ceres.
class CeresNumericDiffReprojectedCost {
public:
  CeresNumericDiffReprojectedCost(const Eigen::Vector2d &p2d) : p2d_(p2d) {}

  bool operator()(const double *const pose, const double *const point,
                  double *residual) const /* must be const */ {
    Eigen::Vector3d r(pose[0], pose[1], pose[2]);
    Eigen::AngleAxisd rvec(r.norm(), r.normalized());

    // Project point to camera coordinate from worldcoordinate.
    Eigen::Vector3d p3d_in_cam =
        rvec.toRotationMatrix() * Eigen::Vector3d(point[0], point[1], point[2]);
    p3d_in_cam += Eigen::Vector3d(pose[3], pose[4], pose[5]);

    double x_project = p3d_in_cam[0] / p3d_in_cam[2] * f_ + cx_;
    double y_project = p3d_in_cam[1] / p3d_in_cam[2] * f_ + cy_;

    residual[0] = x_project - p2d_[0];
    residual[1] = y_project - p2d_[1];
    return true;
  }

  static ceres::CostFunction *
  CreateNumericDiffCost(const Eigen::Vector2d &p2d) {
    return (new ceres::NumericDiffCostFunction<CeresNumericDiffReprojectedCost,
                                               ceres::CENTRAL, 2, 6, 3>(
        new CeresNumericDiffReprojectedCost(p2d)));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const Eigen::Vector2d p2d_;
  double f_ = 520.9;
  double cx_ = 325.1;
  double cy_ = 249.7;
};

// Reprojected cost to used autodiff of ceres.
class CeresAutoDiffReprojectedCost {
public:
  CeresAutoDiffReprojectedCost(const Eigen::Vector2d &p2d) : p2d_(p2d) {}

  template <typename T>
  bool operator()(const T *const pose, const T *const point, T *residual) const
  /* must be const */ {
    T p3d_in_cam[3];
    // Template function to project point to camera coordinate from world
    // coordinate.
    ceres::AngleAxisRotatePoint(pose, point, p3d_in_cam);

    p3d_in_cam[0] += pose[3];
    p3d_in_cam[1] += pose[4];
    p3d_in_cam[2] += pose[5];

    T x_project = p3d_in_cam[0] / p3d_in_cam[2] * f_ + cx_;
    T y_project = p3d_in_cam[1] / p3d_in_cam[2] * f_ + cy_;

    residual[0] = x_project - T(p2d_[0]);
    residual[1] = y_project - T(p2d_[1]);
    return true;
  }

  static ceres::CostFunction *CreateAutoDiffCost(const Eigen::Vector2d &p2d) {
    return (
        new ceres::AutoDiffCostFunction<CeresAutoDiffReprojectedCost, 2, 6, 3>(
            new CeresAutoDiffReprojectedCost(p2d)));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const Eigen::Vector2d p2d_;
  double f_ = 520.9;
  double cx_ = 325.1;
  double cy_ = 249.7;
};

void bundleAdjustmentCeres(vector<Point3f> points_3d,
                           const vector<Point2f> points_2d, const Mat &K,
                           Mat &R, Mat &t, bool use_auto_diff) {
  cv::Mat rvec_cv;
  cv:Rodrigues(R, rvec_cv);
  double pose[6] = {rvec_cv.at<double>(0), rvec_cv.at<double>(1),
                    rvec_cv.at<double>(2), t.at<double>(0),
                    t.at<double>(1),       t.at<double>(2)};
  double(*points3d)[3] = new double[points_3d.size()][3];
  ceres::Problem problem;
  for (int i = 0; i < points_3d.size(); ++i) {
    points3d[i][0] = points_3d[i].x;
    points3d[i][1] = points_3d[i].y;
    points3d[i][2] = points_3d[i].z;
    Eigen::Vector2d obs_2d(points_2d[i].x, points_2d[i].y);
    problem.AddResidualBlock(
        use_auto_diff
            ? CeresAutoDiffReprojectedCost::CreateAutoDiffCost(obs_2d)
            : CeresNumericDiffReprojectedCost::CreateNumericDiffCost(obs_2d),
        nullptr, pose, points3d[i]);
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

  std::string diff_method = use_auto_diff ? "<Autodiff>" : "<NumericDiff>";
  cout << "\n<============= Ceres optimization " << diff_method
       << " =============>\n"
       << endl;

  ceres::Solver::Options options; // 这里有很多配置项可以填
  options.linear_solver_type = ceres::DENSE_QR; // 增量方程如何求解
  options.minimizer_progress_to_stdout = true;  // 输出到cout
  ceres::Solver::Summary summary;               // 优化信息
  ceres::Solve(options, &problem, &summary);    // 开始优化
  Eigen::Vector3d r(pose[0], pose[1], pose[2]);

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  std::cout << "\nCosts time: " << time_used.count() << " seconds." << endl;

  Eigen::AngleAxisd rvec(r.norm(), r.normalized());
  std::cout << "\nR :\n" << rvec.toRotationMatrix() << std::endl;
  std::cout << "\nt :\n"
            << pose[3] << " " << pose[4] << " " << pose[5] << "\n\n "
            << std::endl;
}
