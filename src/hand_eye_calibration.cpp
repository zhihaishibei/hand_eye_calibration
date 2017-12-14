/*************************************************************************
	> File Name: hand_eye_calibration.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年09月26日 星期二 14时57分40秒
 ************************************************************************/

#include<iostream>
#include <opencv2/opencv.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "PoseProcess.h"
#include <fstream>

using namespace std;

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "VERTEX_SE3:QUAT";
}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class ReprojectionError
{
public:
    ReprojectionError(double observed_x,
                                    double observed_y,
                                    const cv::Point3d& point,
                                    const double* intrinc_param,
                                    const Pose3d& T_e2b
                                    ):observed_x(observed_x),observed_y(observed_y),
                                       intrinc_param(intrinc_param),point(point),
                                        T_e2b_(T_e2b){}
    template <typename T>
    bool operator() (const T* const p_c2e_ptr,
                                const T* const q_c2e_ptr,
                                const T* const p_w2b_ptr,
                                const T* const q_w2b_ptr,
                                T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T,3,1>> p_c2e(p_c2e_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_c2e(q_c2e_ptr);
        Eigen::Map<const Eigen::Matrix<T,3,1>> p_w2b(p_w2b_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_w2b(q_w2b_ptr);

        EigenCV::Pose<T> T_c2e(p_c2e,q_c2e);
        EigenCV::Pose<T> T_w2b(p_w2b,q_w2b);

        Eigen::Matrix<T,3,1> T_e2b_p;
        T_e2b_p[0] = T(T_e2b_.p[0]);
        T_e2b_p[1] = T(T_e2b_.p[1]);
        T_e2b_p[2] = T(T_e2b_.p[2]);
        Eigen::Quaternion<T> T_e2b_q = T_e2b_.q.template cast<T>();

        EigenCV::Pose<T> T_e2b(T_e2b_p,T_e2b_q);
        EigenCV::Pose<T> T_w2c_estimated = T_c2e.invert().MultiplyPose(T_e2b.invert()).MultiplyPose(T_w2b);

        Eigen::Matrix<T,3,1> p;
        p[0] = T(point.x);
        p[1] = T(point.y);
        p[2] = T(point.z);
        Eigen::Matrix<T,3,1> p_rotated;
        p_rotated = T_w2c_estimated.get_quat()*p;
        Eigen::Matrix<T,3,1> p_in_cam;
        p_in_cam=p_rotated + T_w2c_estimated.get_trans();

        T predicted_x = (T(intrinc_param[0])*p[0]+T(intrinc_param[2])*p[2])/p[2];
        T predicted_y = (T(intrinc_param[4])*p[1]+T(intrinc_param[5])*p[2])/p[2];
        residuals_ptr[0] = T(observed_x) - predicted_x;
        residuals_ptr[1] = T(observed_y) - predicted_y;
        return true;
    }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(const double observed_x,
                                                                    const double observed_y,
                                                                    const  cv::Point3d point,
                                                                    const double* intric_param,
                                                                    const Pose3d& T_e2b)
        {
          return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 4, 3, 4>(
                      new ReprojectionError(observed_x, observed_y, point, intric_param, T_e2b)));
        }

private:
    double observed_x;
    double observed_y;
    const double* intrinc_param;
    cv::Point3d point;
    Pose3d T_e2b_;
};



class ReprojectionErrorTc2e
{
public:
    ReprojectionErrorTc2e(double observed_x,
                                    double observed_y,
                                    const cv::Point3d& point,
                                    const double* intrinc_param,
                                    const Pose3d& T_e2b,
                                    const Pose3d& T_w2b
                                    ):observed_x(observed_x),observed_y(observed_y),
                                       intrinc_param(intrinc_param),point(point),
                                        T_e2b_(T_e2b), T_w2b_(T_w2b){}
    template <typename T>
    bool operator() (const T* const p_c2e_ptr,
                                const T* const q_c2e_ptr,
                                T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T,3,1>> p_c2e(p_c2e_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_c2e(q_c2e_ptr);

        EigenCV::Pose<T> T_c2e(p_c2e,q_c2e);

        Eigen::Matrix<T,3,1> T_e2b_p;
        T_e2b_p[0] = T(T_e2b_.p[0]);
        T_e2b_p[1] = T(T_e2b_.p[1]);
        T_e2b_p[2] = T(T_e2b_.p[2]);
        Eigen::Quaternion<T> T_e2b_q = T_e2b_.q.template cast<T>();
        EigenCV::Pose<T> T_e2b(T_e2b_p,T_e2b_q);

        Eigen::Matrix<T,3,1> T_w2b_p;
        T_w2b_p[0] = T(T_w2b_.p[0]);
        T_w2b_p[1] = T(T_w2b_.p[1]);
        T_w2b_p[2] = T(T_w2b_.p[2]);
        Eigen::Quaternion<T> T_w2b_q = T_w2b_.q.template cast<T>();
        EigenCV::Pose<T> T_w2b(T_w2b_p,T_w2b_q);

        EigenCV::Pose<T> T_w2c_estimated = T_c2e.invert().MultiplyPose(T_e2b.invert()).MultiplyPose(T_w2b);

        Eigen::Matrix<T,3,1> p;
        p[0] = T(point.x);
        p[1] = T(point.y);
        p[2] = T(point.z);
        Eigen::Matrix<T,3,1> p_rotated;
        p_rotated = T_w2c_estimated.get_quat()*p;
        Eigen::Matrix<T,3,1> p_in_cam;
        p_in_cam=p_rotated + T_w2c_estimated.get_trans();

        T predicted_x = (T(intrinc_param[0])*p[0]+T(intrinc_param[2])*p[2])/p[2];
        T predicted_y = (T(intrinc_param[4])*p[1]+T(intrinc_param[5])*p[2])/p[2];
        residuals_ptr[0] = T(observed_x) - predicted_x;
        residuals_ptr[1] = T(observed_y) - predicted_y;
        return true;
    }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(const double observed_x,
                                                                    const double observed_y,
                                                                    const  cv::Point3d point,
                                                                    const double* intric_param,
                                                                    const Pose3d& T_e2b,
                                                                    const Pose3d& T_w2b)
        {
          return (new ceres::AutoDiffCostFunction<ReprojectionErrorTc2e, 2, 3, 4>(
                      new ReprojectionErrorTc2e(observed_x, observed_y, point, intric_param, T_e2b, T_w2b)));
        }

private:
    double observed_x;
    double observed_y;
    const double* intrinc_param;
    cv::Point3d point;
    Pose3d T_e2b_;
    Pose3d T_w2b_;
};



class ReprojectionErrorTw2b
{
public:
    ReprojectionErrorTw2b(double observed_x,
                                    double observed_y,
                                    const cv::Point3d& point,
                                    const double* intrinc_param,
                                    const Pose3d& T_e2b,
                                    const Pose3d& T_c2e
                                    ):observed_x(observed_x),observed_y(observed_y),
                                       intrinc_param(intrinc_param),point(point),
                                        T_e2b_(T_e2b), T_c2e_(T_c2e){}
    template <typename T>
    bool operator() (const T* const p_w2b_ptr,
                                const T* const q_w2b_ptr,
                                T* residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T,3,1>> p_w2b(p_w2b_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_w2b(q_w2b_ptr);

        EigenCV::Pose<T> T_w2b(p_w2b,q_w2b);

        Eigen::Matrix<T,3,1> T_e2b_p;
        T_e2b_p[0] = T(T_e2b_.p[0]);
        T_e2b_p[1] = T(T_e2b_.p[1]);
        T_e2b_p[2] = T(T_e2b_.p[2]);
        Eigen::Quaternion<T> T_e2b_q = T_e2b_.q.template cast<T>();
        EigenCV::Pose<T> T_e2b(T_e2b_p,T_e2b_q);

        Eigen::Matrix<T,3,1> T_c2e_p;
        T_c2e_p[0] = T(T_c2e_.p[0]);
        T_c2e_p[1] = T(T_c2e_.p[1]);
        T_c2e_p[2] = T(T_c2e_.p[2]);
        Eigen::Quaternion<T> T_c2e_q = T_c2e_.q.template cast<T>();
        EigenCV::Pose<T> T_c2e(T_c2e_p,T_c2e_q);

        EigenCV::Pose<T> T_w2c_estimated = T_c2e.invert().MultiplyPose(T_e2b.invert()).MultiplyPose(T_w2b);

        Eigen::Matrix<T,3,1> p;
        p[0] = T(point.x);
        p[1] = T(point.y);
        p[2] = T(point.z);
        Eigen::Matrix<T,3,1> p_rotated;
        p_rotated = T_w2c_estimated.get_quat()*p;
        Eigen::Matrix<T,3,1> p_in_cam;
        p_in_cam=p_rotated + T_w2c_estimated.get_trans();

        T predicted_x = (T(intrinc_param[0])*p[0]+T(intrinc_param[2])*p[2])/p[2];
        T predicted_y = (T(intrinc_param[4])*p[1]+T(intrinc_param[5])*p[2])/p[2];
        residuals_ptr[0] = T(observed_x) - predicted_x;
        residuals_ptr[1] = T(observed_y) - predicted_y;
        return true;
    }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(const double observed_x,
                                                                    const double observed_y,
                                                                    const  cv::Point3d point,
                                                                    const double* intric_param,
                                                                    const Pose3d& T_e2b,
                                                                    const Pose3d& T_c2e)
        {
          return (new ceres::AutoDiffCostFunction<ReprojectionErrorTw2b, 2, 3, 4>(
                      new ReprojectionErrorTw2b(observed_x, observed_y, point, intric_param, T_e2b, T_c2e)));
        }

private:
    double observed_x;
    double observed_y;
    const double* intrinc_param;
    cv::Point3d point;
    Pose3d T_e2b_;
    Pose3d T_c2e_;
};



void BuildOptimizationProblem(std::vector<std::vector<cv::Point2d> >& point_in_img,
                                                       std::vector<std::vector<cv::Point3d> >& point_in_world,
                                                       double* intr_param,
                                                       std::vector<Pose3d>& T_e2b,
                                                       //std::vector<Pose3d>& T_w2c,
                                                       Pose3d& T_c2e,
                                                       Pose3d& T_w2b,
                                                       ceres::Problem* problem)
{
    ceres::LossFunction *loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
    for(int index=0;index!=T_e2b.size();index++)
    {
        for(int i=0;i!=point_in_img[index].size();i++)
        {
            ceres::CostFunction* cost_function = ReprojectionError::Create(point_in_img[index][i].x,
                                              point_in_img[index][i].y, point_in_world[index][i],intr_param, T_e2b[index]);
            problem->AddResidualBlock(cost_function, loss_function, T_c2e.p.data(), T_c2e.q.coeffs().data(),
                                                                                                               T_w2b.p.data(), T_w2b.q.coeffs().data());
        }
    }
    problem->SetParameterization(T_c2e.q.coeffs().data(),quaternion_local_parameterization);
    problem->SetParameterization(T_w2b.q.coeffs().data(),quaternion_local_parameterization);
}


void BuildOptimizationProblemTc2e(std::vector<std::vector<cv::Point2d> >& point_in_img,
                                                       std::vector<std::vector<cv::Point3d> >& point_in_world,
                                                       double* intr_param,
                                                       std::vector<Pose3d>& T_e2b,
                                                       Pose3d& T_c2e,
                                                       Pose3d& T_w2b,
                                                       ceres::Problem* problem)
{
    //optimize T_c2e
    ceres::LossFunction *loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
    for(int index=0;index!=T_e2b.size();index++)
    {
        for(int i=0;i!=point_in_img[index].size();i++)
        {
            ceres::CostFunction* cost_function = ReprojectionErrorTc2e::Create(point_in_img[index][i].x,
                                              point_in_img[index][i].y, point_in_world[index][i],intr_param, T_e2b[index], T_w2b);
            problem->AddResidualBlock(cost_function, loss_function, T_c2e.p.data(), T_c2e.q.coeffs().data());
        }
    }
    problem->SetParameterization(T_c2e.q.coeffs().data(),quaternion_local_parameterization);
}


void BuildOptimizationProblemTw2b(std::vector<std::vector<cv::Point2d> >& point_in_img,
                                                       std::vector<std::vector<cv::Point3d> >& point_in_world,
                                                       double* intr_param,
                                                       std::vector<Pose3d>& T_e2b,
                                                       Pose3d& T_c2e,
                                                       Pose3d& T_w2b,
                                                       ceres::Problem* problem)
{
    //optimize T_w2b
    ceres::LossFunction *loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
    for(int index=0;index!=T_e2b.size();index++)
    {
        for(int i=0;i!=point_in_img[index].size();i++)
        {
            ceres::CostFunction* cost_function = ReprojectionErrorTw2b::Create(point_in_img[index][i].x,
                                              point_in_img[index][i].y, point_in_world[index][i],intr_param, T_e2b[index], T_c2e);
            problem->AddResidualBlock(cost_function, loss_function, T_w2b.p.data(), T_w2b.q.coeffs().data());
        }
    }
    problem->SetParameterization(T_w2b.q.coeffs().data(),quaternion_local_parameterization);
}


// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem)
{
  CHECK(problem != NULL);

  ceres::Solver::Options options;
  options.max_num_iterations = 5000;
  options.max_num_consecutive_invalid_steps = 20;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  //options.m
  //options.linear_solver_type = ceres::DENSE_SCHUR;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout <<  "summary:\n" <<summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename, const Pose3d& pose) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    LOG(ERROR) << "Error opening the file: " << filename;
    return false;
  }

    outfile << pose.p[0] << " "<<pose.p[1]<<" "<<pose.p[2]<<" "
            << pose.q.x() << " " << pose.q.y() << " "
            << pose.q.z() << " " << pose.q.w() << '\n';

  return true;
}


int main()
{
    vector<vector<cv::Point2d>> point_in_img;
    vector<vector<cv::Point3d> > point_in_world;
    double* intr_param;
    vector<Pose3d> T_e2b;
    Pose3d T_c2e; //initial value
    Pose3d T_w2b; //initial value

    int iretation_index = 0;
    while(iretation_index<5)
    {
        ceres::Problem problem;
        BuildOptimizationProblemTc2e(point_in_img, point_in_world, intr_param, T_e2b, T_c2e, T_w2b, &problem);
        CHECK(SolveOptimizationProblem(&problem))
          << "The solve was not successful, exiting.";

        BuildOptimizationProblemTc2e(point_in_img, point_in_world, intr_param, T_e2b, T_c2e, T_w2b, &problem);
        CHECK(SolveOptimizationProblem(&problem))
          << "The solve was not successful, exiting.";

        iretation_index++;
    }

    OutputPoses("Tc2e.txt", T_c2e);

    return 0;
}
