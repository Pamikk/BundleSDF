#include "cuda_image_util.h"
#include "cudaUtil.h"
#include "common.h"

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)


namespace cuda_image_util
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Depth to Camera Space Positions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthFloatToCameraSpaceFloat4_Kernel(float4* d_output, const float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		d_output[y*width + x] = make_float4(0,0,0,0);

		float depth = d_input[y*width + x];

		if (depth >= 0.1)
		{
			float4 cameraSpace(intrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth));
			d_output[y*width + x] = make_float4(cameraSpace.x, cameraSpace.y, cameraSpace.w, 1.0f);
		}
	}
}

void convertDepthFloatToCameraSpaceFloat4(float4* d_output, const float* d_input, const float4x4& intrinsicsInv, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertDepthFloatToCameraSpaceFloat4_Kernel << <gridSize, blockSize >> >(d_output, d_input, intrinsicsInv, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeNormals_Kernel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = make_float4(0,0,0,0);

	const float z_diff_thres = 0.02;

	if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
	{
		const float4 CC = d_input[(y + 0)*width + (x + 0)];
		const float4 PC = d_input[(y + 1)*width + (x + 0)];
		const float4 CP = d_input[(y + 0)*width + (x + 1)];
		const float4 MC = d_input[(y - 1)*width + (x + 0)];
		const float4 CM = d_input[(y + 0)*width + (x - 1)];

		if (CC.z<0.1) return;

		float3 x_dir = make_float3(0,0,0);
		float3 y_dir = make_float3(0,0,0);

		if (PC.z>=0.1 && MC.z>=0.1 && abs(PC.z-CC.z)<=z_diff_thres && abs(MC.z-CC.z)<=z_diff_thres)
		{
			x_dir = make_float3(PC)-make_float3(MC);
		}
		else if (PC.z>=0.1 && abs(PC.z-CC.z)<=z_diff_thres)
		{
			x_dir = make_float3(PC)-make_float3(CC);
		}
		else if (MC.z>=0.1 && abs(MC.z-CC.z)<=z_diff_thres)
		{
			x_dir = make_float3(MC)-make_float3(CC);
		}
		else
		{
			return;
		}

		if (CP.z>=0.1 && CM.z>=0.1 && abs(CP.z-CC.z)<=z_diff_thres && abs(CM.z-CC.z)<=z_diff_thres)
		{
			y_dir = make_float3(CP-CM);
		}
		else if (CP.z>=0.1 && abs(CP.z-CC.z)<=z_diff_thres)
		{
			y_dir = make_float3(CP-CC);
		}
		else if (CM.z>=0.1 && abs(CM.z-CC.z)<=z_diff_thres)
		{
			y_dir = make_float3(CM-CC);
		}
		else
		{
			return;
		}

		float3 n = cross(x_dir, y_dir);
		const float  l = length(n);
		n = n/l;
		if (dot(n, make_float3(-CC.x, -CC.y, -CC.z))<0)
		{
			n = -n;
		}

		if (l > 0.0f)
		{
			d_output[y*width + x] = make_float4(n, 0.0f);
		}
	}
}

void computeNormals(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormals_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Erode Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void erodeDepthMapDevice(float* d_output, float* d_input, int structureSize, int width, int height, float dThresh, float fracReq, float zfar)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;


	if (x >= 0 && x < width && y >= 0 && y < height)
	{


		unsigned int count = 0;

		float oldDepth = d_input[y*width + x];
		if (oldDepth<=0.1f || oldDepth>zfar)
		{
			d_output[y*width + x] = 0;
			return;
		}
		for (int i = -structureSize; i <= structureSize; i++)
		{
			for (int j = -structureSize; j <= structureSize; j++)
			{
				if (x + j >= 0 && x + j < width && y + i >= 0 && y + i < height)
				{
					float depth = d_input[(y + i)*width + (x + j)];
					if (depth == MINF || depth < 0.1f || fabs(depth - oldDepth) > dThresh)
					{
						count++;
					}
				}
			}
		}

		unsigned int sum = (2 * structureSize + 1)*(2 * structureSize + 1);
		if ((float)count / (float)sum >= fracReq) {
			d_output[y*width + x] = 0;
		}
		else {
			d_output[y*width + x] = d_input[y*width + x];
		}
	}
}

void erodeDepthMap(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq, float zfar)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	erodeDepthMapDevice << <gridSize, blockSize >> >(d_output, d_input, structureSize, width, height, dThresh, fracReq, zfar);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gauss Filter Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gaussFilterDepthMapDevice(float* d_output, const float* d_input, int radius, float sigmaD, float sigmaR, unsigned int width, unsigned int height, const float zfar)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = radius;

	d_output[y*width + x] = 0;

	const float depthCenter = d_input[y*width + x];

	float mean_depth = 0;
	int num_valid = 0;
	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentDepth = d_input[n*width + m];

				if (currentDepth>=0.1f && currentDepth<=zfar)
				{
					num_valid++;
					mean_depth += currentDepth;
				}
			}
		}
	}
	if (num_valid==0) return;

	mean_depth /= num_valid;

	float sum = 0.0f;
	float sumWeight = 0.0f;
	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentDepth = d_input[n*width + m];

				if (currentDepth>=0.1f && currentDepth<=zfar && abs(currentDepth-mean_depth)<0.01)
				{
					const float weight = exp( -((m-x)*(m-x) + (y-n)*(y-n)) / (2.0f*sigmaD*sigmaD) - (depthCenter-currentDepth)*(depthCenter-currentDepth)/(2*sigmaR*sigmaR) );

					sumWeight += weight;
					sum += weight*currentDepth;
				}
			}
		}
	}

	float num_total = (2*kernelRadius+1)*(2*kernelRadius+1);
	if (sumWeight > 0.0f && num_valid/num_total>0)
	{
		d_output[y*width + x] = sum / sumWeight;
	}
}

void gaussFilterDepthMap(float* d_output, const float* d_input, int radius, float sigmaD, float sigmaR, unsigned int width, unsigned int height, const float zfar)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussFilterDepthMapDevice <<<gridSize, blockSize >>>(d_output, d_input, radius, sigmaD, sigmaR, width, height, zfar);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter Depth Smoothed Edges
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void filterDepthSmoothedEdgesDevice(float* d_output, const float* d_input, const float4* d_normal, unsigned int width, unsigned int height, const float angle_thres, const float fx, const float fy, const float cx, const float cy)
{
	const int u = blockIdx.x*blockDim.x + threadIdx.x;
	const int v = blockIdx.y*blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	const int pos = v*width+u;
	float Z = d_input[pos];
	if (Z<0.1) return;

	float X = (u-cx)*Z/fx;
	float Y = (v-cy)*Z/fy;
	float3 view_dir = make_float3(X,Y,Z);
	view_dir = normalize(view_dir);

	float3 normal_dir = make_float3(d_normal[pos].x,d_normal[pos].y,d_normal[pos].z);
	normal_dir = normalize(normal_dir);
	float dot = normal_dir.x*view_dir.x + normal_dir.y*view_dir.y + normal_dir.z*view_dir.z;
	float angle = acos(dot);    // [0,pi]
	if (abs(angle-M_PI/2)<angle_thres)
	{
		d_output[pos] = 0;
	}
	else
	{
		d_output[pos] = d_input[pos];
	}

}

void filterDepthSmoothedEdges(float* d_output, const float* d_input, const float4* d_normal, unsigned int width, unsigned int height, const float angle_thres, const float fx, const float fy, const float cx, const float cy)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	filterDepthSmoothedEdgesDevice << <gridSize, blockSize >> >(d_output, d_input, d_normal, width, height, angle_thres, fx,fy,cx,cy);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Covisibility
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeCovisibilityKernel(const int H, const int W, const int stride, Eigen::Matrix4f *cur_in_kfcam, const float visible_angle_thres, const float4 *xyz_mapA, const float4 *normalA, int *n_visible, int *n_total_gpu)
{
	const int w = (blockIdx.x*blockDim.x + threadIdx.x) * stride;
	const int h = (blockIdx.y*blockDim.y + threadIdx.y) * stride;
	if (w >= W || h >= H) return;

	const int i_pix = h*W+w;

	float4 ptA = xyz_mapA[i_pix];
	if (ptA.z<0.1) return;
	float4 normalA_tmp = normalA[i_pix];
	if (normalA_tmp.x==0 && normalA_tmp.y==0 && normalA_tmp.z==0) return;

	Eigen::Vector3f ptA_ = (*cur_in_kfcam * Eigen::Vector4f(ptA.x, ptA.y, ptA.z, 1)).head(3);
	Eigen::Vector3f normalA_ = (*cur_in_kfcam).block(0,0,3,3) * Eigen::Vector3f(normalA_tmp.x, normalA_tmp.y, normalA_tmp.z);
	Eigen::Vector3f pt_to_eye = -ptA_;
	float dot_prod = pt_to_eye.normalized().dot(normalA_.normalized());

	atomicAdd(n_total_gpu, 1);

	if (dot_prod>visible_angle_thres)
	{
		atomicAdd(n_visible, 1);
	}

}


float computeCovisibility(const int H, const int W, int umin, int vmin, int umax, int vmax, const Eigen::Matrix3f &K, const Eigen::Matrix4f &cur_in_kfcam, const float visible_angle_thres, const float4 *normalA, const float *depthA)
{
  const int n_pixels = H*W;

  float4 *xyz_map_gpu;
  cudaMalloc(&xyz_map_gpu, n_pixels*sizeof(float4));
	cudaMemset(xyz_map_gpu, 0, n_pixels*sizeof(float4));
  float4x4 K_inv_data;
  K_inv_data.setIdentity();
  Eigen::Matrix3f K_inv = K.inverse();
  for (int row=0;row<3;row++)
  {
    for (int col=0;col<3;col++)
    {
      K_inv_data(row,col) = K_inv(row,col);
    }
  }
  cuda_image_util::convertDepthFloatToCameraSpaceFloat4(xyz_map_gpu, depthA, K_inv_data, W, H);

	Eigen::Matrix4f *cur_in_kfcam_gpu;
	cudaMalloc(&cur_in_kfcam_gpu, sizeof(Eigen::Matrix4f));
	cudaMemcpy(cur_in_kfcam_gpu, &cur_in_kfcam, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

  int *n_visible_gpu, *n_total_gpu;
  cudaMalloc(&n_visible_gpu, sizeof(int));
	cudaMemset(n_visible_gpu, 0, sizeof(int));
	cudaMalloc(&n_total_gpu, sizeof(int));
	cudaMemset(n_total_gpu, 0, sizeof(int));
	const int stride = 2;
  dim3 threads = {32, 32};
  dim3 blocks = {divCeil(int(W/stride), threads.x), divCeil(int(H/stride), threads.y)};
  cuda_image_util::computeCovisibilityKernel<<<blocks, threads>>>(H, W, stride, cur_in_kfcam_gpu, visible_angle_thres, xyz_map_gpu, normalA, n_visible_gpu, n_total_gpu);
  int n_visible = 0, n_total = 0;
  cutilSafeCall(cudaMemcpy(&n_visible, n_visible_gpu, sizeof(int), cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(&n_total, n_total_gpu, sizeof(int), cudaMemcpyDeviceToHost));

  float visible = float(n_visible)/n_total;

	cutilSafeCall(cudaFree(xyz_map_gpu));
	cutilSafeCall(cudaFree(n_visible_gpu));
	cutilSafeCall(cudaFree(n_total_gpu));
	cutilSafeCall(cudaFree(cur_in_kfcam_gpu));

  return visible;
}


};
