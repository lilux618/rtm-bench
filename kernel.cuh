#pragma once

#include <cuda_runtime.h>

/**
 * 8th-order coefficients for 2nd derivative: d2/dx2.
 */
__device__ __constant__ float c2[5] = {
    -2.8472222222f,
    1.6000000000f,
    -0.2000000000f,
    0.0253968254f,
    -0.0017857143f
};

/**
 * 8th-order coefficients for 1st derivative: d/dx.
 */
__device__ __constant__ float c1[4] = {
    0.8f,
    -0.2f,
    0.0380952381f,
    -0.0035714286f
};

__device__ inline int didx3d(int x, int y, int z, int nx, int ny) {
    return z * nx * ny + y * nx + x;
}

/**
 * Forward one-step propagation of simplified pseudo-acoustic 3D TTI.
 * Includes rotated directional operator and cross derivatives.
 */
__global__ void forward_step_kernel(
    const float* __restrict__ prev,
    const float* __restrict__ curr,
    float* __restrict__ next,
    const float* __restrict__ vel,
    const float* __restrict__ eps,
    const float* __restrict__ delta,
    int nx,
    int ny,
    int nz,
    float dt,
    float dx,
    float dy,
    float dz,
    float nx_dir,
    float ny_dir,
    float nz_dir,
    float mx_dir,
    float my_dir,
    float mz_dir,
    float lx_dir,
    float ly_dir,
    float lz_dir) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < 4 || y < 4 || z < 4 || x >= nx - 4 || y >= ny - 4 || z >= nz - 4) return;

    const int id = didx3d(x, y, z, nx, ny);

    float dxx = c2[0] * curr[id];
    float dyy = c2[0] * curr[id];
    float dzz = c2[0] * curr[id];

    #pragma unroll
    for (int k = 1; k <= 4; ++k) {
        dxx += c2[k] * (curr[didx3d(x + k, y, z, nx, ny)] + curr[didx3d(x - k, y, z, nx, ny)]);
        dyy += c2[k] * (curr[didx3d(x, y + k, z, nx, ny)] + curr[didx3d(x, y - k, z, nx, ny)]);
        dzz += c2[k] * (curr[didx3d(x, y, z + k, nx, ny)] + curr[didx3d(x, y, z - k, nx, ny)]);
    }
    dxx /= (dx * dx);
    dyy /= (dy * dy);
    dzz /= (dz * dz);

    // 8th-order first derivatives for mixed derivative construction.
    float ux_yplus[4], ux_yminus[4], uy_xplus[4], uy_xminus[4];
    float uz_xplus[4], uz_xminus[4], uz_yplus[4], uz_yminus[4];

    #pragma unroll
    for (int k = 1; k <= 4; ++k) {
        ux_yplus[k - 1] = 0.0f;
        ux_yminus[k - 1] = 0.0f;
        uy_xplus[k - 1] = 0.0f;
        uy_xminus[k - 1] = 0.0f;
        uz_xplus[k - 1] = 0.0f;
        uz_xminus[k - 1] = 0.0f;
        uz_yplus[k - 1] = 0.0f;
        uz_yminus[k - 1] = 0.0f;

        #pragma unroll
        for (int j = 1; j <= 4; ++j) {
            const float c = c1[j - 1];
            ux_yplus[k - 1] += c * (curr[didx3d(x + j, y + k, z, nx, ny)] - curr[didx3d(x - j, y + k, z, nx, ny)]);
            ux_yminus[k - 1] += c * (curr[didx3d(x + j, y - k, z, nx, ny)] - curr[didx3d(x - j, y - k, z, nx, ny)]);

            uy_xplus[k - 1] += c * (curr[didx3d(x + k, y + j, z, nx, ny)] - curr[didx3d(x + k, y - j, z, nx, ny)]);
            uy_xminus[k - 1] += c * (curr[didx3d(x - k, y + j, z, nx, ny)] - curr[didx3d(x - k, y - j, z, nx, ny)]);

            uz_xplus[k - 1] += c * (curr[didx3d(x + k, y, z + j, nx, ny)] - curr[didx3d(x + k, y, z - j, nx, ny)]);
            uz_xminus[k - 1] += c * (curr[didx3d(x - k, y, z + j, nx, ny)] - curr[didx3d(x - k, y, z - j, nx, ny)]);

            uz_yplus[k - 1] += c * (curr[didx3d(x, y + k, z + j, nx, ny)] - curr[didx3d(x, y + k, z - j, nx, ny)]);
            uz_yminus[k - 1] += c * (curr[didx3d(x, y - k, z + j, nx, ny)] - curr[didx3d(x, y - k, z - j, nx, ny)]);
        }
        ux_yplus[k - 1] /= dx;
        ux_yminus[k - 1] /= dx;
        uy_xplus[k - 1] /= dy;
        uy_xminus[k - 1] /= dy;
        uz_xplus[k - 1] /= dz;
        uz_xminus[k - 1] /= dz;
        uz_yplus[k - 1] /= dz;
        uz_yminus[k - 1] /= dz;
    }

    float dxy = 0.0f;
    float dxz = 0.0f;
    float dyz = 0.0f;

    #pragma unroll
    for (int k = 1; k <= 4; ++k) {
        const float c = c1[k - 1];
        dxy += c * (ux_yplus[k - 1] - ux_yminus[k - 1]);
        dxy += c * (uy_xplus[k - 1] - uy_xminus[k - 1]);

        dxz += c * (uz_xplus[k - 1] - uz_xminus[k - 1]);
        dyz += c * (uz_yplus[k - 1] - uz_yminus[k - 1]);
    }
    dxy = 0.5f * dxy / dy;
    dxz = dxz / dx;
    dyz = dyz / dy;

    const float nxx = nx_dir * nx_dir;
    const float nyy = ny_dir * ny_dir;
    const float nzz = nz_dir * nz_dir;
    const float mxx = mx_dir * mx_dir;
    const float myy = my_dir * my_dir;
    const float mzz = mz_dir * mz_dir;
    const float lxx = lx_dir * lx_dir;
    const float lyy = ly_dir * ly_dir;
    const float lzz = lz_dir * lz_dir;

    const float nxy = nx_dir * ny_dir;
    const float nxz = nx_dir * nz_dir;
    const float nyz = ny_dir * nz_dir;
    const float mxy = mx_dir * my_dir;
    const float mxz = mx_dir * mz_dir;
    const float myz = my_dir * mz_dir;
    const float lxy = lx_dir * ly_dir;
    const float lxz = lx_dir * lz_dir;
    const float lyz = ly_dir * lz_dir;

    const float dn2 = nxx * dxx + nyy * dyy + nzz * dzz + 2.0f * (nxy * dxy + nxz * dxz + nyz * dyz);
    const float dm2 = mxx * dxx + myy * dyy + mzz * dzz + 2.0f * (mxy * dxy + mxz * dxz + myz * dyz);
    const float dl2 = lxx * dxx + lyy * dyy + lzz * dzz + 2.0f * (lxy * dxy + lxz * dxz + lyz * dyz);

    const float e = eps[id];
    const float d = delta[id];

    const float tti_op = (1.0f + 2.0f * e) * dn2 + dm2 + dl2 + 2.0f * d * (dn2 - dm2);
    const float v2 = vel[id] * vel[id];

    next[id] = 2.0f * curr[id] - prev[id] + dt * dt * v2 * tti_op;
}

/**
 * Sponge absorbing boundary taper.
 */
__global__ void absorb_boundary_kernel(float* wave, int nx, int ny, int nz, int pml, float strength) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;
    const int id = didx3d(x, y, z, nx, ny);

    int dx = min(x, nx - 1 - x);
    int dy = min(y, ny - 1 - y);
    int dz = min(z, nz - 1 - z);
    int dmin = min(dx, min(dy, dz));

    if (dmin < pml) {
        float s = (float)(pml - dmin) / (float)pml;
        float damp = expf(-strength * s * s);
        wave[id] *= damp;
    }
}

/**
 * Inject point source sample at (sx, sy, sz).
 */
__global__ void inject_source_kernel(float* wave, int nx, int ny, int sx, int sy, int sz, float amp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int id = didx3d(sx, sy, sz, nx, ny);
        wave[id] += amp;
    }
}

/**
 * Record pressure at z=rz plane.
 */
__global__ void record_surface_kernel(const float* wave, float* rec, int nx, int ny, int rz, int it) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    int rec_id = it * nx * ny + y * nx + x;
    rec[rec_id] = wave[didx3d(x, y, rz, nx, ny)];
}

/**
 * Inject recorded data at receiver plane during backward propagation.
 */
__global__ void backward_inject_kernel(float* wave, const float* rec, int nx, int ny, int rz, int it) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    int rec_id = it * nx * ny + y * nx + x;
    wave[didx3d(x, y, rz, nx, ny)] += rec[rec_id];
}

/**
 * Zero-lag imaging condition: image += forward(t) * backward(t).
 */
__global__ void imaging_kernel(const float* fw, const float* bw, float* image, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        image[i] += fw[i] * bw[i];
    }
}
