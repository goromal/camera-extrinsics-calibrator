#ifndef NANOFLANN_EIGEN_H
#define NANOFLANN_EIGEN_H

#include <cstdlib>
#include <iostream>
#include <Eigen/Core>
#include <vector>

#include "nanoflann.hpp"

using namespace Eigen;
using namespace std;
using namespace nanoflann;

template<typename T>
struct PointCloud
{
	typedef Matrix<T,3,1> Vec3;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	std::vector<Vec3, aligned_allocator<Vector3d>>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	inline T kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return pts[idx][dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	const bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef PointCloud<double> PointCloudd;
typedef PointCloud<double> PointCloudf;

typedef KDTreeSingleIndexDynamicAdaptor<
	L2_Simple_Adaptor<double, PointCloud<double> > ,
	PointCloud<double>,
	3 /* dim */
	> KDTree3d;
typedef KDTreeSingleIndexDynamicAdaptor<
	L2_Simple_Adaptor<float, PointCloud<float> > ,
	PointCloud<float>,
	3 /* dim */
	> KDTree3f;

template<typename T>
void generateRandomPointCloud(PointCloud<T> &pt_cld, const size_t N, const double max_range = 10)
{
	// Generating Random Point Cloud
	pt_cld.pts.resize(N);
	for (size_t i = 0; i < N; i++)
	{
		pt_cld.pts[i].x() = max_range * (rand() % 1000) / double(1000);
		pt_cld.pts[i].y() = max_range * (rand() % 1000) / double(1000);
		pt_cld.pts[i].z() = max_range * (rand() % 1000) / double(1000);
	}
}

#endif
