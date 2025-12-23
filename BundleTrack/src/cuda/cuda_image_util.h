#ifndef CUDA_IMAGE_UTIL_H
#define CUDA_IMAGE_UTIL_H

#include <cuda_runtime.h>
#include "cuda_SimpleMatrixUtil.h"
#include <Eigen/Dense>


#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)



namespace cuda_image_util
{
	void convert_depth_to_camera_space_float4(float4* d_output, const float* d_input, const float4x4& intrinsicsInv, unsigned int width, unsigned int height);
	void computeNormals(float4* d_output, const float4* d_input, unsigned int width, unsigned int height);

	void erodeDepthMap(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq, float zfar);

	void gaussFilterDepthMap(float* d_output, const float* d_input, int radius, float sigmaD, float sigmaR, unsigned int width, unsigned int height, const float zfar);

	void filterDepthSmoothedEdges(float* d_output, const float* d_input, const float4* d_normal, unsigned int width, unsigned int height, const float angle_thres, const float fx, const float fy, const float cx, const float cy);

	float computeCovisibility(const int H, const int W, int umin, int vmin, int umax, int vmax, const Eigen::Matrix3f &K, const Eigen::Matrix4f &cur_in_kfcam, const float visible_angle_thres, const float4 *normalA, const float *depthA);
};

#endif //CUDA_IMAGE_UTIL_H
