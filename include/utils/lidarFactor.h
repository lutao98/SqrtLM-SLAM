#ifndef LIDAR_FACTOR_H_
#define LIDAR_FACTOR_H_
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include "utils/math_utils.h"
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

class SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        Eigen::Map<const Eigen::Matrix<double, 3, 1>> lie(x);
        Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_lie(delta);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> x_plus_delta_lie(x_plus_delta);

        Sophus::SO3<double> R = Sophus::SO3<double>::exp(lie);
        Sophus::SO3<double> delta_R = Sophus::SO3<double>::exp(delta_lie);

        // 李代数右乘更新
		x_plus_delta_lie = (R * delta_R).log();

        return true;
    }
    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        ceres::MatrixRef(jacobian, 3, 3) = ceres::Matrix::Identity(3, 3);
        return true;
    }
    virtual int GlobalSize() const { return 3; }
    virtual int LocalSize() const { return 3; }
};



struct LidarDistanceFactor {
	LidarDistanceFactor(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_) 
						: curr_point(curr_point_), last_point(last_point_), so3_vec(so3_vec_), t_vec(t_vec_) {}

	template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_last_curr(so3);
		Sophus::SO3<T> SO3_last_curr = Sophus::SO3<T>::exp(so3_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr(t);

    Eigen::Matrix<T, 3, 1> curr_to_last = SO3_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();

		residual[0] = curr_to_last(0, 0);
		residual[1] = curr_to_last(1, 0);
		residual[2] = curr_to_last(2, 0);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_) {
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 3, 3>(
			new LidarDistanceFactor(curr_point_, last_point_, so3_vec_, t_vec_)));
	}

	Eigen::Vector3d curr_point, last_point, so3_vec, t_vec;
};


struct LidarDistanceFactorTest {
	LidarDistanceFactorTest(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_) 
						: curr_point(curr_point_), last_point(last_point_) {}

	template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_last_curr(so3);
		Sophus::SO3<T> SO3_last_curr = Sophus::SO3<T>::exp(so3_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr(t);

    Eigen::Matrix<T, 3, 1> curr_to_last = SO3_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();

		residual[0] = curr_to_last(0, 0);
		residual[1] = curr_to_last(1, 0);
		residual[2] = curr_to_last(2, 0);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_) {
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactorTest, 3, 3, 3>(
			new LidarDistanceFactorTest(curr_point_, last_point_)));
	}

	Eigen::Vector3d curr_point, last_point;
};



struct LidarEdgeVectorFactorZRotXYTrans {
	LidarEdgeVectorFactorZRotXYTrans(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_vect_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_)
		: curr_point(curr_point_), last_point(last_point_), curr_point_vect(curr_point_vect_), so3_vec(so3_vec_), t_vec(t_vec_) {}

	template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_last_curr(T(so3_vec(0)), T(so3_vec(1)), so3[2]);
		Sophus::SO3<T> SO3_last_curr = Sophus::SO3<T>::exp(so3_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr(t[0], t[1], T(t_vec(2)));

    Eigen::Matrix<T, 3, 1> curr_to_last = SO3_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();
		residual[0] = (curr_to_last.cross(curr_point_vect.cast<T>())).norm();
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_vect_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_) {
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeVectorFactorZRotXYTrans, 1, 3, 3>(
			new LidarEdgeVectorFactorZRotXYTrans(curr_point_, last_point_, curr_point_vect_, so3_vec_, t_vec_)));
	}

	Eigen::Vector3d curr_point, last_point, curr_point_vect, so3_vec, t_vec;
};



struct LidarEdgeVectorFactor {
	LidarEdgeVectorFactor(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_vect_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_)
		: curr_point(curr_point_), last_point(last_point_), curr_point_vect(curr_point_vect_), so3_vec(so3_vec_), t_vec(t_vec_) {}

	template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_last_curr(so3);
		Sophus::SO3<T> SO3_last_curr = Sophus::SO3<T>::exp(so3_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr(t);

        Eigen::Matrix<T, 3, 1> curr_to_last = SO3_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();
		residual[0] = (curr_to_last.cross(curr_point_vect.cast<T>())).norm();
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_vect_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_) {
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeVectorFactor, 1, 3, 3>(
			new LidarEdgeVectorFactor(curr_point_, last_point_, curr_point_vect_, so3_vec_, t_vec_)));
	}

	Eigen::Vector3d curr_point, last_point, curr_point_vect, so3_vec, t_vec;
};



struct LidarPlaneNormFactor {
	LidarPlaneNormFactor(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_norm_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_)
		: curr_point(curr_point_), last_point(last_point_), curr_point_norm(curr_point_norm_), so3_vec(so3_vec_), t_vec(t_vec_) {}

	template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_last_curr(so3);
		Sophus::SO3<T> SO3_last_curr = Sophus::SO3<T>::exp(so3_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr(t);

    Eigen::Matrix<T, 3, 1> curr_to_last = SO3_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();
		residual[0] =  curr_to_last.dot(curr_point_norm.cast<T>());

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_norm_, const Eigen::Vector3d &so3_vec_, const Eigen::Vector3d &t_vec_) {
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 3, 3>(
			new LidarPlaneNormFactor(curr_point_, last_point_, curr_point_norm_, so3_vec_, t_vec_)));
	}

	Eigen::Vector3d curr_point, last_point, curr_point_norm, so3_vec, t_vec;
};



struct LidarPlaneNormFactorTest {
	LidarPlaneNormFactorTest(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_norm_)
		: curr_point(curr_point_), last_point(last_point_), curr_point_norm(curr_point_norm_) {}

	template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_last_curr(so3);
		Sophus::SO3<T> SO3_last_curr = Sophus::SO3<T>::exp(so3_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr(t);

    Eigen::Matrix<T, 3, 1> curr_to_last = SO3_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();
		residual[0] =  curr_to_last.dot(curr_point_norm.cast<T>());

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_, const Eigen::Vector3d &curr_point_norm_) {
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactorTest, 1, 3, 3>(
			new LidarPlaneNormFactorTest(curr_point_, last_point_, curr_point_norm_)));
	}

	Eigen::Vector3d curr_point, last_point, curr_point_norm;
};



struct EndBackFactor {
	EndBackFactor(const Eigen::Vector3d &pretonextso3_, const Eigen::Vector3d &pretonextt_)
		          : pretonextso3(pretonextso3_), pretonextt(pretonextt_){}

	template <typename T>
	bool operator()(const T *so3_0, const T *t_0, const T *so3_1, const T *t_1, T *residual) const {
		Eigen::Matrix<T, 3, 1> so3_pre(so3_0);
		Eigen::Matrix<T, 3, 1> so3_next(so3_1);

		Eigen::Matrix<T, 3, 1> t_pre(t_0);
		Eigen::Matrix<T, 3, 1> t_next(t_1);

		Eigen::Matrix<T, 3, 1> pretonext_so3 = (Sophus::SO3<T>::exp(so3_next).inverse() * Sophus::SO3<T>::exp(so3_pre)).log();
		Eigen::Matrix<T, 3, 1> pretonext_t = Sophus::SO3<T>::exp(so3_next).inverse() * (t_pre - t_next);

    residual[0] = pretonext_so3(0,0) - T(pretonextso3(0));
		residual[1] = pretonext_so3(1,0) - T(pretonextso3(1));
		residual[2] = pretonext_so3(2,0) - T(pretonextso3(2));
		residual[3] = pretonext_t(0,0) - T(pretonextt(0));
		residual[4] = pretonext_t(1,0) - T(pretonextt(1));
		residual[5] = pretonext_t(2,0) - T(pretonextt(2));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &pretonextso3_, const Eigen::Vector3d &pretonextt_) {
		return (new ceres::AutoDiffCostFunction<
				EndBackFactor, 6, 3, 3, 3, 3>(
			new EndBackFactor(pretonextso3_, pretonextt_)));
	}

	Eigen::Vector3d pretonextso3, pretonextt;
};



struct EndBackFirstFrameFactor {
	EndBackFirstFrameFactor(const Eigen::Vector3d &pretonextso3_, const Eigen::Vector3d &pretonextt_, const Eigen::Vector3d &firstfreamso3_, const Eigen::Vector3d &firstfreamt_)
		          : pretonextso3(pretonextso3_), pretonextt(pretonextt_), firstfreamso3(firstfreamso3_), firstfreamt(firstfreamt_) {}
    template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {

		Eigen::Matrix<T, 3, 1> so3_next(so3);
		Eigen::Matrix<T, 3, 1> t_next(t);

		Eigen::Matrix<T, 3, 1> pretonext_so3 = (Sophus::SO3<T>::exp(so3_next).inverse() * Sophus::SO3<T>::exp(firstfreamso3.cast<T>())).log();
		Eigen::Matrix<T, 3, 1> pretonext_t = Sophus::SO3<T>::exp(so3_next).inverse() * (firstfreamt.cast<T>() - t_next);

    residual[0] = pretonext_so3(0,0) - T(pretonextso3(0));
		residual[1] = pretonext_so3(1,0) - T(pretonextso3(1));
		residual[2] = pretonext_so3(2,0) - T(pretonextso3(2));
		residual[3] = pretonext_t(0,0) - T(pretonextt(0));
		residual[4] = pretonext_t(1,0) - T(pretonextt(1));
		residual[5] = pretonext_t(2,0) - T(pretonextt(2));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &pretonextso3_, const Eigen::Vector3d &pretonextt_, const Eigen::Vector3d &firstfreamso3_, const Eigen::Vector3d &firstfreamt_) {
		return (new ceres::AutoDiffCostFunction<
				EndBackFirstFrameFactor, 6, 3, 3>(
			new EndBackFirstFrameFactor(pretonextso3_, pretonextt_, firstfreamso3_, firstfreamt_)));
	}

	Eigen::Vector3d pretonextso3, pretonextt, firstfreamso3, firstfreamt;
};



struct EndBackLastFrameFactor {
	EndBackLastFrameFactor(const Eigen::Vector3d &pretonextso3_, const Eigen::Vector3d &pretonextt_, const Eigen::Vector3d &lastfreamso3_, const Eigen::Vector3d &lastfreamt_)
		          : pretonextso3(pretonextso3_), pretonextt(pretonextt_), lastfreamso3(lastfreamso3_), lastfreamt(lastfreamt_) {}
    template <typename T>
	bool operator()(const T *so3, const T *t, T *residual) const {

		Eigen::Matrix<T, 3, 1> so3_pre(so3);
		Eigen::Matrix<T, 3, 1> t_pre(t);

		Eigen::Matrix<T, 3, 1> pretonext_so3 = (Sophus::SO3<T>::exp(lastfreamso3.cast<T>()).inverse() * Sophus::SO3<T>::exp(so3_pre)).log();
		Eigen::Matrix<T, 3, 1> pretonext_t = Sophus::SO3<T>::exp(lastfreamso3.cast<T>()).inverse() * (t_pre - lastfreamt.cast<T>());

    residual[0] = pretonext_so3(0,0) - T(pretonextso3(0));
		residual[1] = pretonext_so3(1,0) - T(pretonextso3(1));
		residual[2] = pretonext_so3(2,0) - T(pretonextso3(2));
		residual[3] = pretonext_t(0,0) - T(pretonextt(0));
		residual[4] = pretonext_t(1,0) - T(pretonextt(1));
		residual[5] = pretonext_t(2,0) - T(pretonextt(2));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &pretonextso3_, const Eigen::Vector3d &pretonextt_, const Eigen::Vector3d &lastfreamso3_, const Eigen::Vector3d &lastfreamt_) {
		return (new ceres::AutoDiffCostFunction<
				EndBackLastFrameFactor, 6, 3, 3>(
			new EndBackLastFrameFactor(pretonextso3_, pretonextt_, lastfreamso3_, lastfreamt_)));
	}

	Eigen::Vector3d pretonextso3, pretonextt, lastfreamso3, lastfreamt;
};























// class SE3Parameterization : public ceres::LocalParameterization
// {
// public:
//     SE3Parameterization() {}
//     virtual ~SE3Parameterization() {}

//     virtual bool Plus(const double* x,
//                       const double* delta,
//                       double* x_plus_delta) const
//     {
//         Eigen::Map<const Eigen::Matrix<double, 3, 1>> lie(x);
//         Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_lie(delta);
//         Eigen::Map<Eigen::Matrix<double, 3, 1>> x_plus_delta_lie(x_plus_delta);

//         Sophus::SO3<double> R = Sophus::SO3<double>::exp(lie);
//         Sophus::SO3<double> delta_R = Sophus::SO3<double>::exp(delta_lie);

//         // 李代数右乘更新
// 		x_plus_delta_lie = (R * delta_R).log();

//         return true;
//     }
//     virtual bool ComputeJacobian(const double* x,
//                                  double* jacobian) const
//     {
//         ceres::MatrixRef(jacobian, 3, 3) = ceres::Matrix::Identity(3, 3);
//         return true;
//     }
//     virtual int GlobalSize() const { return 3; }
//     virtual int LocalSize() const { return 3; }
// };



struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;



		// Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		// Eigen::Matrix<T, 3, 1> de = lpa - lpb;
		// residual[0] = nu.x() / de.norm();
		// residual[1] = nu.y() / de.norm();
		// residual[2] = nu.z() / de.norm();




		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> nor = nu.cross(lpa - lpb);
		nor.normalize();
		residual[0] = nor.dot(lp - lpa);


		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 1, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};



struct LidarEdgeVectorFactorOld
{
	LidarEdgeVectorFactorOld(Eigen::Vector3d curr_point_, Eigen::Vector3d curr_point_vect_, Eigen::Vector3d last_point_, double s_)
		: curr_point(curr_point_), curr_point_vect(curr_point_vect_), last_point(last_point_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

		Eigen::Matrix<T, 3, 1> curr_to_last = q_last_curr.inverse() * (last_point.cast<T>() - t_last_curr) - curr_point.cast<T>();
		residual[0] = (curr_to_last.cross(curr_point_vect.cast<T>())).norm();
		return true;
	}

	static ceres::CostFunction *Create(Eigen::Vector3d curr_point_, Eigen::Vector3d curr_point_vect_, Eigen::Vector3d last_point_, double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeVectorFactorOld, 1, 4, 3>(
			new LidarEdgeVectorFactorOld(curr_point_, curr_point_vect_, last_point_, s_)));
	}

	Eigen::Vector3d curr_point, curr_point_vect, last_point;
	double s;
};



struct LidarEdgeFactorLast
{
	LidarEdgeFactorLast(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		//Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;
        Eigen::Matrix<T, 3, 1> lp = q_last_curr.inverse() * (cp - t_last_curr);

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> nor = nu.cross(lpa - lpb);
		nor.normalize();
		residual[0] = nor.dot(lp - lpa);


		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactorLast, 1, 4, 3>(
			new LidarEdgeFactorLast(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};




class LidarPlaneFactor_z_rot_xy_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_z_rot_xy_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = 0;
				// jacobians[0][2] = -ljm(0) * (q_last_curr * cp)(1) + ljm(1) * (q_last_curr * cp)(0);
				// jacobians[0][3] = 0;



				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = 0;
				jacobians[0][2] = result(0, 2);
				jacobians[0][3] = 0;
			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = ljm(0);
				jacobians[1][1] = ljm(1);
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};


class LidarPlaneFactor_z_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_z_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = 0;
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = 0;
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;

			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = ljm(2);
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};


class LidarPlaneFactor_x_rot : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_x_rot(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = -ljm(1) * (q_last_curr * cp)(2) + ljm(2) * (q_last_curr * cp)(1);
				// jacobians[0][1] = 0;
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;



				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = result(0, 0);
				jacobians[0][1] = 0;
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;
			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};


class LidarPlaneFactor_y_rot : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_y_rot(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = ljm(0) * (q_last_curr * cp)(2) - ljm(2) * (q_last_curr * cp)(0);
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = result(0, 1);
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;

			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};



class LidarPlaneFactor_xy_rot_z_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_xy_rot_z_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = ljm(0) * (q_last_curr * cp)(2) - ljm(2) * (q_last_curr * cp)(0);
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;

				jacobians[0][0] = result(0, 0);
				jacobians[0][1] = result(0, 1);
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;

			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = ljm(2);
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};



class LidarPlaneFactor_xyz_rot_xyz_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_xyz_rot_xyz_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = -ljm(1) * (q_last_curr * cp)(2) + ljm(2) * (q_last_curr * cp)(1);
				// jacobians[0][1] = ljm(0) * (q_last_curr * cp)(2) - ljm(2) * (q_last_curr * cp)(0);
				// jacobians[0][2] = -ljm(0) * (q_last_curr * cp)(1) + ljm(1) * (q_last_curr * cp)(0);
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = result(0, 0);
				jacobians[0][1] = result(0, 1);
				jacobians[0][2] = result(0, 2);
				jacobians[0][3] = 0;				
			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = ljm(0);
				jacobians[1][1] = ljm(1);
				jacobians[1][2] = ljm(2);
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};




struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};


struct LidarPlaneFactorLast
{
	LidarPlaneFactorLast(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr.inverse() * (cp - t_last_curr);

		residual[0] = (lp - lpj).dot(ljm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactorLast, 1, 4, 3>(
			new LidarPlaneFactorLast(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};



struct LidarPlaneNormFactorOld
{
	LidarPlaneNormFactorOld(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d curr_point_norm_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), curr_point_norm(curr_point_norm_), s(s_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		Eigen::Matrix<T, 3, 1> norm{T(curr_point_norm.x()), T(curr_point_norm.y()), T(curr_point_norm.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp = q_last_curr.inverse() * (lpj - t_last_curr);
		residual[0] =  (lp - cp).dot(norm);

		// Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;
		// residual[0] =  (lp - lpj).dot(norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d curr_point_norm_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactorOld, 1, 4, 3>(
			new LidarPlaneNormFactorOld(curr_point_, last_point_j_, curr_point_norm_, s_)));
	}


	double s;
	Eigen::Vector3d curr_point, last_point_j, curr_point_norm;
};



struct LidarPlaneNormFactor_1
{
	LidarPlaneNormFactor_1(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d curr_point_norm_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), curr_point_norm(curr_point_norm_), s(s_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		Eigen::Matrix<T, 3, 1> norm{T(curr_point_norm.x()), T(curr_point_norm.y()), T(curr_point_norm.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		// Eigen::Matrix<T, 3, 1> lp = q_last_curr.inverse() * (lpj - t_last_curr);
		// residual[0] =  (lp - cp).dot(norm);

		Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;
		residual[0] =  (lp - lpj).dot(norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d curr_point_norm_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor_1, 1, 4, 3>(
			new LidarPlaneNormFactor_1(curr_point_, last_point_j_, curr_point_norm_, s_)));
	}
	double s;
	Eigen::Vector3d curr_point, last_point_j, curr_point_norm;
};



struct LidarPlaneNormFactorCDX
{
	LidarPlaneNormFactorCDX(Eigen::Vector3d curr_point_, double last_point_j_, Eigen::Vector3d curr_point_norm_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), curr_point_norm(curr_point_norm_), s(s_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> norm{T(curr_point_norm.x()), T(curr_point_norm.y()), T(curr_point_norm.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		// Eigen::Matrix<T, 3, 1> lp = q_last_curr.inverse() * (lpj - t_last_curr);
		// residual[0] =  (lp - cp).dot(curr_point_norm);

		Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;
		residual[0] =  lp.dot(norm) + T(last_point_j);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const double last_point_j_,
									   const Eigen::Vector3d curr_point_norm_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactorCDX, 1, 4, 3>(
			new LidarPlaneNormFactorCDX(curr_point_, last_point_j_, curr_point_norm_, s_)));
	}
	double s , last_point_j;
	Eigen::Vector3d curr_point, curr_point_norm;
};



class LidarPlaneNormFactor_xy_rot_z_trans : public ceres::SizedCostFunction<1, 4, 3>
{
	public:
	LidarPlaneNormFactor_xy_rot_z_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d curr_point_norm_, const double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), curr_point_norm(curr_point_norm_), s(s_){}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {
		Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
		Eigen::Quaterniond q_identity{1, 0, 0, 0};
		q_last_curr = q_identity.slerp(s, q_last_curr);
		Eigen::Matrix<double, 3, 1> t_last_curr{s * parameters[1][0], s * parameters[1][1], s * parameters[1][2]};

		Eigen::Vector3d lp = q_last_curr.inverse() * (last_point_j - t_last_curr);
		residuals[0] = curr_point_norm.dot(lp - curr_point);

		if (jacobians)
		{
			Eigen::Matrix3d ror_mat = (q_last_curr.inverse()).toRotationMatrix();
            Eigen::Matrix3d jaco_mat = mathutils::SkewSymmetric(ror_mat * last_point_j);
			if (jacobians[0])
			{
				jacobians[0][0] = (curr_point_norm.transpose() * jaco_mat).x();//(curr_point_norm.transpose() * jaco_mat).x();
				jacobians[0][1] = (curr_point_norm.transpose() * jaco_mat).y();
				jacobians[0][2] = (curr_point_norm.transpose() * jaco_mat).z();
				jacobians[0][3] = 0;
			}
			if (jacobians[1])
			{
				jacobians[1][0] = (-curr_point_norm.transpose() * ror_mat).x();//(-curr_point_norm.transpose() * ror_mat).x();
				jacobians[1][1] = (-curr_point_norm.transpose() * ror_mat).y();//(-curr_point_norm.transpose() * ror_mat).y();
				jacobians[1][2] = (-curr_point_norm.transpose() * ror_mat).z();
			}
		}
		return true;
	}
	double s;
	Eigen::Vector3d curr_point, last_point_j, curr_point_norm;
};



struct LidarPlaneNormFactorLast
{
	LidarPlaneNormFactorLast(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d curr_point_norm_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), curr_point_norm(curr_point_norm_), s(s_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp = q_last_curr * lpj + t_last_curr;
		residual[0] =  (lp - cp).dot(curr_point_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d curr_point_norm_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactorLast, 1, 4, 3>(
			new LidarPlaneNormFactorLast(curr_point_, last_point_j_, curr_point_norm_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, curr_point_norm;
	double s;
};



struct LidarDistanceFactorOld
{
	LidarDistanceFactorOld(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_, double s_) 
						: curr_point(curr_point_), closed_point(closed_point_), s(s_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residual);
		residual[0] = T(1.0) * (lp.x() - T(closed_point.x()));
		residual[1] = T(1.0) * (lp.y() - T(closed_point.y()));
		residual[2] = T(1.0) * (lp.z() - T(closed_point.z()));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_, double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactorOld, 3, 4, 3>(
			new LidarDistanceFactorOld(curr_point_, closed_point_, s_)));
	}

	static ceres::CostFunction *Create_1(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_, double s_)
	{
		return (new ceres::NumericDiffCostFunction<
				LidarDistanceFactorOld, ceres::CENTRAL, 3, 4, 3>(
			new LidarDistanceFactorOld(curr_point_, closed_point_, s_)));
	}

	double s;
	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};



struct LidarDistanceFactorWithCov
{
	LidarDistanceFactorWithCov(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_, Eigen::Matrix3d curr_point_cov_, Eigen::Matrix3d closed_point_cov_) 
						: curr_point(curr_point_), closed_point(closed_point_), curr_point_cov(curr_point_cov_), closed_point_cov(closed_point_cov_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

		Eigen::Matrix<T, 3, 1> lp = q_last_curr * curr_point.cast<T>() + t_last_curr;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residual);
		residuals = (closed_point_cov.cast<T>() + q_last_curr * curr_point_cov.cast<T>() * q_last_curr.inverse()).inverse() * (closed_point.cast<T>() - lp);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_, Eigen::Matrix3d currt_point_cov_, Eigen::Matrix3d closed_point_cov_) 
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactorWithCov, 3, 4, 3>(
			new LidarDistanceFactorWithCov(curr_point_, closed_point_, currt_point_cov_, closed_point_cov_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
	Eigen::Matrix3d curr_point_cov;
	Eigen::Matrix3d closed_point_cov;	

};



struct LidarDistanceFactorLast
{
	LidarDistanceFactorLast(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_, double s_) 
						: curr_point(curr_point_), closed_point(closed_point_), s(s_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lp = q_last_curr.inverse() * (cp - t_last_curr);

		residual[0] = lp.x() - T(closed_point.x());
		residual[1] = lp.y() - T(closed_point.y());
		residual[2] = lp.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_, double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactorLast, 3, 4, 3>(
			new LidarDistanceFactorLast(curr_point_, closed_point_, s_)));
	}
    double s;
	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};



class LidarDistanceFactorCDX : public ceres::SizedCostFunction<3, 4, 3>
{
    public:

	LidarDistanceFactorCDX(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_)
		: curr_point(curr_point_), closed_point(closed_point_){}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {

		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};
		Eigen::Matrix<double, 3, 1> point_w = q_last_curr * cp + t_last_curr;

		residuals[0] = point_w.x() - closed_point.x();
		residuals[1] = point_w.y() - closed_point.y();
		residuals[2] = point_w.z() - closed_point.z();

		if (jacobians)
		{
			if (jacobians[0])
			{
				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
				Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > jaco_rot(jacobians[0]);
				jaco_rot.setZero();
				jaco_rot.topLeftCorner<3, 3>() = -1.0 * mathutils::SkewSymmetric(q_last_curr * cp);
			}
			if (jacobians[1])
			{
				Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > jaco_transf(jacobians[1]);
                jaco_transf.setIdentity(3, 3);
				jaco_transf.rightCols<1>().setZero();
			}
		}
	return true;
	}

	Eigen::Vector3d curr_point, closed_point;
};




// struct EndBackFactor
// {
// 	EndBackFactor(Eigen::Quaterniond pretonextq_, Eigen::Matrix<double, 3, 1> pretonextt_)
// 		          : pretonextq(pretonextq_), pretonextt(pretonextt_){}
// template <typename T>
// 	bool operator()(const T *q_0, const T *t_0, const T *q_1, const T *t_1, T *residual) const
// 	{
// 		Eigen::Quaternion<T> q_pre{q_0[3], q_0[0], q_0[1], q_0[2]};
// 		Eigen::Quaternion<T> q_next{q_1[3], q_1[0], q_1[1], q_1[2]};

// 		Eigen::Matrix<T, 3, 1> t_pre{t_0[0],t_0[1], t_0[2]};
// 		Eigen::Matrix<T, 3, 1> t_next{t_1[0], t_1[1], t_1[2]};

// 		Eigen::Quaternion<T> pretonext_q = q_next.inverse() * q_pre;
// 		Eigen::Matrix<T, 3, 1> pretonext_t = q_next.inverse() * (t_pre - t_next);

//         residual[0] = pretonext_q.x() - T(pretonextq.x());
// 		residual[1] = pretonext_q.y() - T(pretonextq.y());
// 		residual[2] = pretonext_q.z() - T(pretonextq.z());
// 		residual[3] = pretonext_t(0,0) - T(pretonextt(0,0));
// 		residual[4] = pretonext_t(1,0) - T(pretonextt(1,0));
// 		residual[5] = pretonext_t(2,0) - T(pretonextt(2,0));
// 		return true;
// 	}

// 	static ceres::CostFunction *Create(const Eigen::Quaterniond pretonextq_, const Eigen::Matrix<double, 3, 1> pretonextt_)
// 	{
// 		return (new ceres::AutoDiffCostFunction<
// 				EndBackFactor, 6, 4, 3, 4, 3>(
// 			new EndBackFactor(pretonextq_, pretonextt_)));
// 	}
// 	Eigen::Quaterniond pretonextq;
// 	Eigen::Matrix<double, 3, 1> pretonextt;


// };


// struct EndBackFirstFrameFactor
// {
// 	EndBackFirstFrameFactor(Eigen::Quaterniond pretonextq_, Eigen::Matrix<double, 3, 1> pretonextt_, Eigen::Quaterniond firstfreamq_, Eigen::Matrix<double, 3, 1> firstfreamt_)
// 		          : pretonextq(pretonextq_), pretonextt(pretonextt_), firstfreamq(firstfreamq_), firstfreamt(firstfreamt_){}
//     template <typename T>
// 	bool operator()(const T *q, const T *t, T *residual) const
// 	{
// 		Eigen::Quaternion <T> q_next{q[3], q[0], q[1], q[2]};
// 		Eigen::Matrix<T, 3, 1> t_next{t[0],t[1], t[2]};

// 		Eigen::Quaternion<T> pretonext_q = q_next.inverse() * firstfreamq.cast<T>();
// 		Eigen::Matrix<T, 3, 1> pretonext_t = q_next.inverse() * (firstfreamt.cast<T>() - t_next);

//         residual[0] = pretonext_q.x() - T(pretonextq.x());
// 		residual[1] = pretonext_q.y() - T(pretonextq.y());
// 		residual[2] = pretonext_q.z() - T(pretonextq.z());
// 		residual[3] = pretonext_t(0,0) - T(pretonextt(0,0));
// 		residual[4] = pretonext_t(1,0) - T(pretonextt(1,0));
// 		residual[5] = pretonext_t(2,0) - T(pretonextt(2,0));
// 		return true;
// 	}

// 	static ceres::CostFunction *Create(const Eigen::Quaterniond pretonextq_, const Eigen::Matrix<double, 3, 1> pretonextt_, const Eigen::Quaterniond fistfreamq_, const Eigen::Matrix<double, 3, 1> firstfreamt_)
// 	{
// 		return (new ceres::AutoDiffCostFunction<
// 				EndBackFirstFrameFactor, 6, 4, 3>(
// 			new EndBackFirstFrameFactor(pretonextq_, pretonextt_, fistfreamq_, firstfreamt_)));
// 	}
// 	Eigen::Quaterniond pretonextq;
// 	Eigen::Matrix<double, 3, 1> pretonextt;

// 	Eigen::Quaterniond firstfreamq;
//     Eigen::Matrix<double, 3, 1> firstfreamt;
// };


// struct EndBackLastFrameFactor
// {
// 	EndBackLastFrameFactor(Eigen::Quaterniond pretonextq_, Eigen::Matrix<double, 3, 1> pretonextt_, Eigen::Quaterniond lastfreamq_, Eigen::Matrix<double, 3, 1> lastfreamt_)
// 		          : pretonextq(pretonextq_), pretonextt(pretonextt_), lastfreamq(lastfreamq_), lastfreamt(lastfreamt_){}
//     template <typename T>
// 	bool operator()(const T *q, const T *t, T *residual) const
// 	{
// 		Eigen::Quaternion <T> q_pre{q[3], q[0], q[1], q[2]};
// 		Eigen::Matrix<T, 3, 1> t_pre{t[0],t[1], t[2]};

// 		Eigen::Quaternion<T> pretonext_q = lastfreamq.inverse().cast<T>() * q_pre;
// 		Eigen::Matrix<T, 3, 1> pretonext_t = lastfreamq.inverse().cast<T>() * (t_pre - lastfreamt.cast<T>());

//         residual[0] = pretonext_q.x() - T(pretonextq.x());
// 		residual[1] = pretonext_q.y() - T(pretonextq.y());
// 		residual[2] = pretonext_q.z() - T(pretonextq.z());
// 		residual[3] = pretonext_t(0,0) - T(pretonextt(0,0));
// 		residual[4] = pretonext_t(1,0) - T(pretonextt(1,0));
// 		residual[5] = pretonext_t(2,0) - T(pretonextt(2,0));
// 		return true;
// 	}

// 	static ceres::CostFunction *Create(const Eigen::Quaterniond pretonextq_, const Eigen::Matrix<double, 3, 1> pretonextt_, const Eigen::Quaterniond lastfreamq_, const Eigen::Matrix<double, 3, 1> lastfreamt_)
// 	{
// 		return (new ceres::AutoDiffCostFunction<
// 				EndBackLastFrameFactor, 6, 4, 3>(
// 			new EndBackLastFrameFactor(pretonextq_, pretonextt_, lastfreamq_, lastfreamt_)));
// 	}
// 	Eigen::Quaterniond pretonextq;
// 	Eigen::Matrix<double, 3, 1> pretonextt;

// 	Eigen::Quaterniond lastfreamq;
//     Eigen::Matrix<double, 3, 1> lastfreamt;
// };


struct GNSSFactor
{
	GNSSFactor(Eigen::Quaterniond gnssq_, Eigen::Vector3d gnsst_)
		          : gnssq(gnssq_), gnsst(gnsst_){}
template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
        residual[0] = T(gnssq.x()) - T(q[0]);
		residual[1] = T(gnssq.y()) - T(q[1]);
		residual[2] = T(gnssq.z()) - T(q[2]);
		residual[3] = T(gnsst.x()) - T(t[0]);
		residual[4] = T(gnsst.y()) - T(t[1]);
		residual[5] = T(gnsst.z()) - T(t[2]);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Quaterniond gnssq_, const Eigen::Vector3d gnsst_)
	{
		return (new ceres::AutoDiffCostFunction<
				GNSSFactor, 6, 4, 3>(
			new GNSSFactor(gnssq_, gnsst_)));
	}
	Eigen::Quaterniond gnssq;
	Eigen::Vector3d gnsst;
};


struct GNSSTFactor
{
	GNSSTFactor(Eigen::Vector3d gnsst_)
		          : gnsst(gnsst_){}
template <typename T>
	bool operator()(const T *t, T *residual) const
	{
		residual[0] = T(gnsst.x()) - T(t[0]);
		residual[1] = T(gnsst.y()) - T(t[1]);
		residual[2] = T(gnsst.z()) - T(t[2]);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d gnsst_)
	{
		return (new ceres::AutoDiffCostFunction<
				GNSSTFactor, 3, 3>(
			new GNSSTFactor(gnsst_)));
	}
	Eigen::Vector3d gnsst;
};



class TLocalParameterization : public ceres::LocalParameterization {

  	virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
		Eigen::Map<const Eigen::Vector3d> p(x);
  		Eigen::Map<const Eigen::Vector3d> dp(delta);
		Eigen::Map<Eigen::Vector3d> p_plus(x_plus_delta);
		p_plus = p + dp;
 	}
	virtual bool ComputeJacobian(const double *x, double *jacobian) const {
		Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> j(jacobian);
		j.topRows<2>().setIdentity();
		j.bottomRows<1>().setZero();
		return true;
	}
	virtual int GlobalSize() const { return 3; };
	virtual int LocalSize() const { return 2; };

};


class QLocalParameterization : public ceres::LocalParameterization {

  	virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
  		Eigen::Map<const Eigen::Quaterniond> q(x);
  		Eigen::Quaterniond dq = mathutils::DeltaQ(Eigen::Map<const Eigen::Vector3d>(delta));
  		Eigen::Map<Eigen::Quaterniond> q_plus(x_plus_delta);
		q_plus = (dq * q).normalized();
 	}
	virtual bool ComputeJacobian(const double *x, double *jacobian) const {
		Eigen::Map<Eigen::Matrix<double, 4, 3>> j(jacobian);

		j.topRows<3>().setIdentity();
		// j.row(2).setIdentity();
		j.bottomRows<1>().setZero();

		return true;
	}
	virtual int GlobalSize() const { return 4; };
	virtual int LocalSize() const { return 3; };

};



struct CalibrationFactor {

	CalibrationFactor(Eigen::Vector3d gnss_t_, Eigen::Vector3d slam_t_, Eigen::Quaterniond gnss_q_, Eigen::Quaterniond slam_q_) 
						: gnss_t(gnss_t_), slam_t(slam_t_), gnss_q(gnss_q_), slam_q(slam_q_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const {
		Eigen::Matrix<T, 3, 1> t_slam_gnss{t[0], t[1], T(0.0)};
		// Eigen::Matrix<T, 3, 1> t_slam_gnss{t[0], t[1], t[2]};	
		Eigen::Quaternion<T> q_slam_gnss{q[3], q[0], q[1], q[2]};

		Eigen::Transform<T, 3, 2> transf_slam_gnss;
		transf_slam_gnss.translation() = t_slam_gnss;
		transf_slam_gnss.linear() = q_slam_gnss.toRotationMatrix();

		Eigen::Transform<T, 3, 2> gnss_pose;
		gnss_pose.translation() = gnss_t.cast<T>();
		gnss_pose.linear() = gnss_q.toRotationMatrix().cast<T>();

		Eigen::Transform<T, 3, 2> slam_pose;
		slam_pose.translation() = slam_t.cast<T>();
		slam_pose.linear() = slam_q.toRotationMatrix().cast<T>();

		Eigen::Transform<T, 3, 2> delt = slam_pose.inverse() * transf_slam_gnss * gnss_pose * transf_slam_gnss.inverse();

		residual[0] = delt.translation().x();
		residual[1] = delt.translation().y();
		residual[2] = delt.translation().z();

		residual[3] = Eigen::Quaternion<T>(delt.linear()).x();
		residual[4] = Eigen::Quaternion<T>(delt.linear()).y();
		residual[5] = Eigen::Quaternion<T>(delt.linear()).z();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_, Eigen::Quaterniond curr_q_, Eigen::Quaterniond closed_q_) {
		return (new ceres::AutoDiffCostFunction<
				CalibrationFactor, 6, 4, 2>(
			new CalibrationFactor(curr_point_, closed_point_, curr_q_, closed_q_)));
	}

	Eigen::Vector3d gnss_t;
	Eigen::Vector3d slam_t;
	Eigen::Quaterniond gnss_q;
	Eigen::Quaterniond slam_q;
};

#endif
