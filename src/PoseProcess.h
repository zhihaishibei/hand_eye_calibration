/*************************************************************************
	> File Name: PoseProcess.h
	> Author: 
	> Mail: 
	> Created Time: 2017年09月21日 星期四 10时11分11秒
 ************************************************************************/

#ifndef _POSEPROCESS_H
#define _POSEPROCESS_H
#include<iostream>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include<vector>
#include <opencv2/opencv.hpp>

namespace EigenCV {


enum EulerType
{
    XYZ,
    ZYX,
    ZYZ,
};


template <class T>
class Pose
{
public:
    //default: (trans,quat)=(0, 0, 0, 0, 0, 0, 1)
    Pose();
    //copy constructor
    Pose(const Pose<T>& pose);
    Pose(const Eigen::Matrix<T,3,1>& trans, const Eigen::Quaternion<T>& quat );
    Pose(const Eigen::Matrix<T,3,1>& trans, const Eigen::Matrix<T,3,3>& rot);
    Pose(const Eigen::Matrix<T,3,1>& trans, const Eigen::AngleAxis<T>& angleaxis);
    //the euler angle is degree default.. sequnce is always rotx, roty, rotz, euler_type is XYZ default
    Pose(const Eigen::Matrix<T,3,1>& trans, const std::vector<T>& euler, EulerType euler_type=XYZ, bool isdegree=true);
    Pose(const cv::Mat& trans, const cv::Mat& rot);
    //trans is 0 default
    Pose(const Eigen::Quaternion<T>& quat );
    //trans is 0 default
    Pose(const Eigen::Matrix<T,3,3>& rot);
    //trans is 0 default
    Pose(const Eigen::AngleAxis<T>& angleaxis);

    Pose(const std::vector<T>& euler, EulerType euler_type=XYZ, bool isdegree=true);
    //trans is 0 default
    Pose(const cv::Mat& rot);
    Pose& operator=(const Pose& p);
    ~Pose();
    //    this * pose
    Pose MultiplyPose(const Pose& pose) const;
    //    pose * this
    Pose MultiplyLeft(const Pose& pose) const;
    //   pose invert
    Pose invert() const;
    /*return Matrix4d pose type[ rotation, trans
                                                            0,  0,  0,  1]*/
    template<class T_new>
    Pose<T_new> cast()  const;
    Eigen::Matrix<T,4,4> to_matpose() const;
    /*return trans*/
    Eigen::Matrix<T,3,1> get_trans() const;
    void set_trans(T x, T y, T z );
    void set_quat(T x, T y, T z, T w);
    void set_trans(const Eigen::Matrix<T,3,1>& trans);
    void set_quat(const Eigen::Quaternion<T>& quat);
    /*retrun quaternion*/
    Eigen::Quaternion<T> get_quat() const ;
    //eigen pose ------->(cv::Mat trans, cv::Mat rot)
    void pose2cvmat(cv::Mat& trans,cv::Mat& rot) const;
    //retrun (transx, transy, transz, x, y, z, w)
    std::vector<T> vec() const;
    /*show pose type[ rotation, trans
                                          0,  0,  0,  1]*/
    void show_matpose() const;
    /*(trans[3], euler[3])*/
    std::vector<T> to_eulerpose(EulerType euler_type) const;
    /*trans , angle, axis*/
    std::vector<T> to_angleaxispose() const;

    /* output transx,transy,transz, rotx, roty, rotz, rotw   */
    friend std::ostream& operator<<(std::ostream& out, const Pose<T>& pose)
    {
        out<<pose.trans[0]<<" "
              <<pose.trans[1]<<" "
              <<pose.trans[2]<<" "
              <<pose.quat.x()<<" "
              <<pose.quat.y()<<" "
              <<pose.quat.z()<<" "
              <<pose.quat.w()<<'\n';
    }

private:
    Eigen::Matrix<T,3,1> trans;
    Eigen::Quaternion<T> quat;
    Eigen::Matrix<T,4,4> compute_matpose() const;
    Eigen::Matrix<T,3,1> quat2euler(EulerType euler_type) const;
    Eigen::AngleAxis<T> quat2angleaxis() const;
};


template <class T>
Pose<T>::Pose()
{
    this->trans=Eigen::Matrix<T,3,1>(T(0),T(0),T(0));
    this->quat=Eigen::Quaternion<T>(T(1),T(0),T(0),T(0));
}

template <class T>
Pose<T>::Pose(const Pose& pose)
{
    this->trans=pose.trans;
    this->quat=pose.quat;
}

template <class T>
Pose<T>& Pose<T>::operator=(const Pose<T>& p)
{
    this->trans=p.trans;
    this->quat=p.quat;
    return *this;
}

template <class T>
Pose<T>::Pose(const Eigen::Matrix<T,3,1>& trans, const Eigen::Quaternion<T>& quat)
{
    this->trans=trans;
    this->quat=quat;
}

template <class T>
Pose<T>::Pose(const Eigen::Matrix<T,3,1>& trans, const Eigen::Matrix<T,3,3>& rot)
{
    this->trans=trans;
    this->quat=rot;
}

template <class T>
Pose<T>::Pose(const Eigen::Matrix<T,3,1>& trans, const Eigen::AngleAxis<T>& angleaxis)
{
    this->trans=trans;
    this->quat=angleaxis;
}

template <class T>
Pose<T>::Pose(const Eigen::Matrix<T,3,1>& trans, const std::vector<T>& euler, EulerType euler_type, bool isdegree)
{
    std::vector<T> euler_rad;
    if(isdegree)
    {
        euler_rad.push_back(euler[0]/180*M_PI);
        euler_rad.push_back(euler[1]/180*M_PI);
        euler_rad.push_back(euler[2]/180*M_PI);
    }
    else
        euler_rad = euler;
    switch (euler_type) {
    case XYZ:
    {
        Eigen::AngleAxis<T> aax_1(euler_rad[0], Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> aay_1(euler_rad[1], Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> aaz_1(euler_rad[2], Eigen::Matrix<T,3,1>::UnitZ());
        this->trans=trans;
        this->quat=aax_1 * aay_1 * aaz_1;;
        break;
    }
    case ZYX:
    {
        Eigen::AngleAxis<T> aax_2(euler_rad[0], Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> aay_2(euler_rad[1], Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> aaz_2(euler_rad[2], Eigen::Matrix<T,3,1>::UnitZ());
        this->trans=trans;
        this->quat=aaz_2 * aay_2 * aax_2;;
        break;
    }
    case ZYZ:
    {
        Eigen::AngleAxis<T> aaz1_3(euler_rad[0], Eigen::Matrix<T,3,1>::UnitZ());
        Eigen::AngleAxis<T> aay_3(euler_rad[1], Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> aaz2_3(euler_rad[2], Eigen::Matrix<T,3,1>::UnitZ());
        this->trans=trans;
        this->quat=aaz1_3 * aay_3 * aaz2_3;;
        break;
    }
    default:
        break;
    }
}

template <class T>
Pose<T>::Pose(const std::vector<T>& euler, EulerType euler_type, bool isdegree)
{
    std::vector<T> euler_rad;
    if(isdegree)
    {
        euler_rad.push_back(euler[0]/180*M_PI);
        euler_rad.push_back(euler[1]/180*M_PI);
        euler_rad.push_back(euler[2]/180*M_PI);
    }
    else
        euler_rad = euler;
    switch (euler_type) {
    case XYZ:
    {
        Eigen::AngleAxis<T> aax_1(euler_rad[0], Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> aay_1(euler_rad[1], Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> aaz_1(euler_rad[2], Eigen::Matrix<T,3,1>::UnitZ());
        this->trans=Eigen::Matrix<T,3,1>(T(0),T(0),T(0));
        this->quat=aax_1 * aay_1 * aaz_1;
        break;
    }
    case ZYX:
    {
        Eigen::AngleAxis<T> aax_2(euler_rad[0], Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> aay_2(euler_rad[1], Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> aaz_2(euler_rad[2], Eigen::Matrix<T,3,1>::UnitZ());
        this->trans=Eigen::Matrix<T,3,1>(T(0),T(0),T(0));
        this->quat=aaz_2 * aay_2 * aax_2;
        break;
    }

    case ZYZ:
    {
        Eigen::AngleAxis<T> aaz1_3(euler_rad[0], Eigen::Matrix<T,3,1>::UnitZ());
        Eigen::AngleAxis<T> aay_3(euler_rad[1], Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> aaz2_3(euler_rad[2], Eigen::Matrix<T,3,1>::UnitZ());
        this->trans=Eigen::Matrix<T,3,1>(T(0),(0),(0));
        this->quat=aaz1_3 * aay_3 * aaz2_3;
        break;
    }
    default:
        break;
    }
}

template <class T>
Pose<T>::Pose(const cv::Mat& trans, const cv::Mat& rot)
{
    cv::Mat temp_trans=trans.clone();
    Eigen::Map<Eigen::Matrix<T,3,1>> temp_trans1(temp_trans.ptr<T>(0));
    cv::Mat rot_transpose = rot.t();
    Eigen::Map<Eigen::Matrix<T,3,3>> temp_rot(rot_transpose.ptr<T>(0));
    this->trans=temp_trans1;
    this->quat=temp_rot;
}

template <class T>
Pose<T>::Pose(const Eigen::Quaternion<T>& quat )
{
    T T_0 = static_cast<T>(0);
    this->trans=Eigen::Matrix<T,3,1>(T_0,T_0,T_0);
    this->quat=quat;
}

template <class T>
Pose<T>::Pose(const Eigen::Matrix<T,3,3>& rot)
{
    T T_0 = static_cast<T>(0);
    this->trans=Eigen::Matrix<T,3,1>(T_0,T_0,T_0);
    this->quat=rot;
}

template <class T>
Pose<T>::Pose(const Eigen::AngleAxis<T>& angleaxis)
{\
    T T_0 = static_cast<T>(0);
    this->trans=Eigen::Matrix<T,3,1>(T_0,T_0,T_0);
    this->quat=angleaxis;
}

template <class T>
Pose<T>::Pose(const cv::Mat& rot)
{
    cv::Mat rot_transpose = rot.t();
    Eigen::Map<Eigen::Matrix<T,3,3>> temp_rot(rot_transpose.ptr<T>(0));
    this->trans=Eigen::Matrix<T,3,1>(T(0),T(0),T(0));
    this->quat=temp_rot;
}


template <class T>
Pose<T>::~Pose()
{

}

template <class T>
Pose<T> Pose<T>::MultiplyPose(const Pose<T>& pose) const
{
    Pose<T> ans;
    /*ans.trans=this->quat*pose.trans + this->trans;
    ans.quat=this->quat*pose.quat*/;
    ans.set_trans(this->quat*pose.trans + this->trans);
    ans.set_quat(this->quat*pose.quat);
    return ans;
}

template <class T>
Pose<T> Pose<T>::MultiplyLeft(const Pose<T>& pose) const
{
    Pose<T> ans;
    /*ans.trans=pose.quat*this->trans + pose.trans;
    ans.quat=pose.quat*this->quat;*/
    ans.set_trans(pose.quat*this->trans + pose.trans);
    ans.set_quat(pose.quat*this->quat);
    return ans;
}

template <class T>
Pose<T> Pose<T>::invert() const
{
    Pose<T> ans;
    /*ans.trans=(this->quat.conjugate()) * (this->trans) *T(-1);
    ans.quat=this->quat.conjugate();*/
    ans.set_trans((this->quat.conjugate()) * (this->trans) *T(-1));
    ans.set_quat(this->quat.conjugate());
    return ans;
}

template<class T>
template<class T_new>
Pose<T_new> Pose<T>::cast() const
{
    Pose<T_new> pose;
    pose.set_trans(static_cast<T_new>(this->trans[0]),
                              static_cast<T_new>(this->trans[1]),
                              static_cast<T_new>(this->trans[2]));
    pose.set_quat(static_cast<T_new>(this->quat.x()),
                             static_cast<T_new>(this->quat.y()),
                             static_cast<T_new>(this->quat.z()),
                             static_cast<T_new>(this->quat.w()));

    return pose;
}

template <class T>
Eigen::Matrix<T,4,4> Pose<T>::to_matpose() const
 {
    return compute_matpose();
 }

template <class T>
void Pose<T>::pose2cvmat(cv::Mat& trans,cv::Mat& rot) const
{
    trans=(cv::Mat_<T>(3,1)<<this->trans[0],this->trans[1], this->trans[2]);
    Eigen::Matrix<T,3,3> rot_temp;
    rot_temp=this->quat.matrix();
    rot=(cv::Mat_<T>(3,3)<<rot_temp(0,0),rot_temp(0,1),rot_temp(0,2),rot_temp(1,0),rot_temp(1,1),rot_temp(1,2),rot_temp(2,0),rot_temp(2,1),rot_temp(2,2));
}

template <class T>
Eigen::Matrix<T,3,1>  Pose<T>::get_trans() const
{
    return this->trans;
}

template <class T>
void Pose<T>::set_trans(T x, T y, T z )
{
    this->trans[0]=x;
    this->trans[1]=y;
    this->trans[2]=z;
}

template <class T>
void Pose<T>::set_quat(T x, T y, T z, T w)
{
    this->quat.x()=x;
    this->quat.y()=y;
    this->quat.z()=z;
    this->quat.w()=w;
}

template <class T>
void Pose<T>::set_trans(const Eigen::Matrix<T,3,1>& trans)
{
    this->trans=trans;
}

template <class T>
void Pose<T>::set_quat(const Eigen::Quaternion<T>& quat)
{
    this->quat=quat;
}

template <class T>
Eigen::Quaternion<T> Pose<T>::get_quat() const
{
    return this->quat;
}

template <class T>
std::vector<T> Pose<T>::to_eulerpose(EulerType euler_type) const
{
    Eigen::Matrix<T,3,1> euler=quat2euler(euler_type);
    std::vector<T> euler_pose;
    euler_pose.push_back(this->trans[0]);
    euler_pose.push_back(this->trans[1]);
    euler_pose.push_back(this->trans[2]);
    euler_pose.push_back(euler[0]/M_PI * 180);
    euler_pose.push_back(euler[1]/M_PI * 180);
    euler_pose.push_back(euler[2]/M_PI * 180);
    return euler_pose;
}

/*angle, axis*/
template <class T>
std::vector<T> Pose<T>::to_angleaxispose() const
{
    Eigen::AngleAxis<T> angleaxis = quat2angleaxis();
    std::vector<T> angleaxis_pose;
    angleaxis_pose.push_back(this->trans[0]);
    angleaxis_pose.push_back(this->trans[1]);
    angleaxis_pose.push_back(this->trans[2]);
    angleaxis_pose.push_back(angleaxis.angle());
    angleaxis_pose.push_back(angleaxis.axis()[0]);
    angleaxis_pose.push_back(angleaxis.axis()[1]);
    angleaxis_pose.push_back(angleaxis.axis()[2]);
    return angleaxis_pose;
}

template <class T>
std::vector<T> Pose<T>::vec() const
{
    std::vector<T> res;
    res.push_back(this->trans[0]);
    res.push_back(this->trans[1]);
    res.push_back(this->trans[2]);
    res.push_back(this->quat.x());
    res.push_back(this->quat.y());
    res.push_back(this->quat.z());
    res.push_back(this->quat.w());
    return res;
}

template <class T>
void Pose<T>::show_matpose() const
{
    std::cout<<compute_matpose()<<std::endl;
}

template <class T>
Eigen::Matrix<T,4,4> Pose<T>::compute_matpose() const
{
    Eigen::Matrix<T,4,4> matpose;
    Eigen::Matrix<T,3,3> rot;
    rot=this->quat.matrix();
    matpose(0,0) = rot(0,0);
    matpose(0,1) = rot(0,1);
    matpose(0,2) = rot(0,2);
    matpose(1,0) = rot(1,0);
    matpose(1,1) = rot(1,1);
    matpose(1,2) = rot(1,2);
    matpose(2,0) = rot(2,0);
    matpose(2,1) = rot(2,1);
    matpose(2,2) = rot(2,2);
    matpose(0,3) = this->trans[0];
    matpose(1,3) = this->trans[1];
    matpose(2,3) = this->trans[2];
    matpose(3,0) = T(0);
    matpose(3,1) = T(1);
    matpose(3,2) = T(2);
    matpose(3,3) = T(3);

    /*matpose.block<3, 3>(0,0) = rot;
    matpose.block<3, 1>(0,3) = this->trans;
    matpose.block<1, 4>(3,0)<<0,0,0,1;*/
    return matpose;
}




//quat----->
template <class T>
Eigen::Matrix<T,3,1> Pose<T>::quat2euler(EulerType euler_type) const
{
    Eigen::Matrix<T,3,1> euler;
    switch(euler_type)
    {
    case XYZ:
        euler=this->quat.matrix().eulerAngles(0,1,2);
        break;
    case ZYX:
        euler=this->quat.matrix().eulerAngles(2,1,0);
        break;
    case ZYZ:
        euler=this->quat.matrix().eulerAngles(2,1,2);
        break;
    }
    return euler;
}

template <class T>
Eigen::AngleAxis<T> Pose<T>::quat2angleaxis() const
{
    Eigen::AngleAxis<T> angleaxis;
    angleaxis=quat;
    return angleaxis;
}



}

#endif
