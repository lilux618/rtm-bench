#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "kernel.cuh"
#include "model.hpp"

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t err = (call);                                                                 \
        if (err != cudaSuccess) {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "            \
                      << cudaGetErrorString(err) << std::endl;                                    \
            std::exit(EXIT_FAILURE);                                                              \
        }                                                                                         \
    } while (0)

struct DeviceModel {
    float* vel = nullptr;
    float* eps = nullptr;
    float* delta = nullptr;
};

struct PerfStats {
    float forward_ms = 0.0f;
    float backward_ms = 0.0f;
    float reconstruct_ms = 0.0f;
};

inline void parse_args(Config& cfg, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto next_int = [&](int& v) {
            if (i + 1 < argc) v = std::atoi(argv[++i]);
        };
        auto next_float = [&](float& v) {
            if (i + 1 < argc) v = std::atof(argv[++i]);
        };

        if (a == "--nx") next_int(cfg.nx);
        else if (a == "--ny") next_int(cfg.ny);
        else if (a == "--nz") next_int(cfg.nz);
        else if (a == "--nt") next_int(cfg.nt);
        else if (a == "--dx") next_float(cfg.dx);
        else if (a == "--dy") next_float(cfg.dy);
        else if (a == "--dz") next_float(cfg.dz);
        else if (a == "--dt") next_float(cfg.dt);
        else if (a == "--f0") next_float(cfg.f0);
        else if (a == "--checkpoint") next_int(cfg.checkpoint_interval);
    }

    cfg.sx = cfg.nx / 2;
    cfg.sy = cfg.ny / 2;
    cfg.sz = std::max(6, cfg.pml / 2);
}

inline void calc_tti_basis(const Config& cfg, float& nx, float& ny, float& nz, float& mx, float& my,
                           float& mz, float& lx, float& ly, float& lz) {
    const float deg2rad = 3.14159265358979323846f / 180.0f;
    float th = cfg.tilt_deg * deg2rad;
    float ph = cfg.azimuth_deg * deg2rad;

    nx = std::sin(th) * std::cos(ph);
    ny = std::sin(th) * std::sin(ph);
    nz = std::cos(th);

    mx = -std::sin(ph);
    my = std::cos(ph);
    mz = 0.0f;

    lx = ny * mz - nz * my;
    ly = nz * mx - nx * mz;
    lz = nx * my - ny * mx;
}

inline void allocate_device_model(const Model& m, DeviceModel& d, int n) {
    CUDA_CHECK(cudaMalloc(&d.vel, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.eps, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.delta, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d.vel, m.vel.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.eps, m.eps.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.delta, m.delta.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

inline void free_device_model(DeviceModel& d) {
    cudaFree(d.vel);
    cudaFree(d.eps);
    cudaFree(d.delta);
}

inline void forward_step_driver(float* d_prev, float* d_curr, float* d_next, const DeviceModel& dm,
                                const Config& cfg, dim3 grid3d, dim3 blk3d, float nx, float ny, float nz,
                                float mx, float my, float mz, float lx, float ly, float lz) {
    forward_step_kernel<<<grid3d, blk3d>>>(d_prev, d_curr, d_next, dm.vel, dm.eps, dm.delta, cfg.nx, cfg.ny,
                                           cfg.nz, cfg.dt, cfg.dx, cfg.dy, cfg.dz, nx, ny, nz, mx, my, mz,
                                           lx, ly, lz);
}

/**
 * Reconstruct forward field at specific time index by replaying from nearest checkpoint.
 */
inline void reconstruct_forward_at_t(float* d_fw_out, int t_target, const Config& cfg, const DeviceModel& dm,
                                     const float* d_checkpoints, int checkpoint_stride, dim3 grid3d,
                                     dim3 blk3d, dim3 grid2d, dim3 blk2d, float nx, float ny, float nz,
                                     float mx, float my, float mz, float lx, float ly, float lz,
                                     PerfStats& perf) {
    int cp_id = t_target / cfg.checkpoint_interval;
    int t0 = cp_id * cfg.checkpoint_interval;

    float* d_prev;
    float* d_curr;
    float* d_next;
    CUDA_CHECK(cudaMalloc(&d_prev, checkpoint_stride * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curr, checkpoint_stride * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, checkpoint_stride * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_prev, d_checkpoints + cp_id * checkpoint_stride, checkpoint_stride * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_curr, d_prev, checkpoint_stride * sizeof(float), cudaMemcpyDeviceToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    for (int t = t0; t < t_target; ++t) {
        forward_step_driver(d_prev, d_curr, d_next, dm, cfg, grid3d, blk3d, nx, ny, nz, mx, my, mz, lx, ly, lz);
        absorb_boundary_kernel<<<grid3d, blk3d>>>(d_next, cfg.nx, cfg.ny, cfg.nz, cfg.pml, cfg.sponge_strength);

        std::swap(d_prev, d_curr);
        std::swap(d_curr, d_next);
    }

    CUDA_CHECK(cudaMemcpy(d_fw_out, d_curr, checkpoint_stride * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    perf.reconstruct_ms += ms;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_next);
}

int main(int argc, char** argv) {
    Config cfg;
    parse_args(cfg, argc, argv);

    const int n = cfg.nx * cfg.ny * cfg.nz;
    const int rec_n = cfg.nt * cfg.nx * cfg.ny;
    const int checkpoint_interval = std::max(1, cfg.checkpoint_interval);
    const int ncheck = (cfg.nt + checkpoint_interval - 1) / checkpoint_interval + 1;

    std::cout << "3D TTI RTM demo (internal validation)\n"
              << "Grid=" << cfg.nx << "x" << cfg.ny << "x" << cfg.nz << ", nt=" << cfg.nt
              << ", checkpoint interval=" << checkpoint_interval << std::endl;

    float vmax = 2800.0f;
    float cfl = vmax * cfg.dt * std::sqrt(1.0f / (cfg.dx * cfg.dx) + 1.0f / (cfg.dy * cfg.dy) + 1.0f / (cfg.dz * cfg.dz));
    std::cout << "Approx CFL indicator: " << cfl << " (target < ~0.6 for this stencil)" << std::endl;

    Model m = build_model(cfg);
    DeviceModel dm;
    allocate_device_model(m, dm, n);

    float nx_dir, ny_dir, nz_dir, mx_dir, my_dir, mz_dir, lx_dir, ly_dir, lz_dir;
    calc_tti_basis(cfg, nx_dir, ny_dir, nz_dir, mx_dir, my_dir, mz_dir, lx_dir, ly_dir, lz_dir);

    float *d_prev, *d_curr, *d_next, *d_bw_prev, *d_bw_curr, *d_bw_next;
    float *d_rec, *d_image, *d_fw_tmp, *d_checkpoints;

    CUDA_CHECK(cudaMalloc(&d_prev, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curr, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bw_prev, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bw_curr, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bw_next, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rec, rec_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_image, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fw_tmp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_checkpoints, ncheck * n * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_prev, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_curr, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_next, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bw_prev, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bw_curr, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bw_next, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_image, 0, n * sizeof(float)));

    dim3 blk3d(8, 8, 4);
    dim3 grid3d((cfg.nx + blk3d.x - 1) / blk3d.x, (cfg.ny + blk3d.y - 1) / blk3d.y,
               (cfg.nz + blk3d.z - 1) / blk3d.z);
    dim3 blk2d(16, 16);
    dim3 grid2d((cfg.nx + blk2d.x - 1) / blk2d.x, (cfg.ny + blk2d.y - 1) / blk2d.y);
    dim3 blk1d(256);
    dim3 grid1d((n + blk1d.x - 1) / blk1d.x);

    PerfStats perf;
    cudaEvent_t f0, f1, b0, b1;
    CUDA_CHECK(cudaEventCreate(&f0));
    CUDA_CHECK(cudaEventCreate(&f1));
    CUDA_CHECK(cudaEventCreate(&b0));
    CUDA_CHECK(cudaEventCreate(&b1));

    // Forward modeling + receiver recording + checkpoint saves.
    CUDA_CHECK(cudaEventRecord(f0));
    CUDA_CHECK(cudaMemcpy(d_checkpoints, d_curr, n * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int it = 0; it < cfg.nt; ++it) {
        forward_step_driver(d_prev, d_curr, d_next, dm, cfg, grid3d, blk3d, nx_dir, ny_dir, nz_dir, mx_dir,
                            my_dir, mz_dir, lx_dir, ly_dir, lz_dir);

        float src = ricker(it * cfg.dt, cfg.f0);
        inject_source_kernel<<<1, 1>>>(d_next, cfg.nx, cfg.ny, cfg.sx, cfg.sy, cfg.sz, src);
        absorb_boundary_kernel<<<grid3d, blk3d>>>(d_next, cfg.nx, cfg.ny, cfg.nz, cfg.pml, cfg.sponge_strength);
        record_surface_kernel<<<grid2d, blk2d>>>(d_next, d_rec, cfg.nx, cfg.ny, cfg.sz, it);

        if ((it + 1) % checkpoint_interval == 0) {
            int cp = (it + 1) / checkpoint_interval;
            CUDA_CHECK(cudaMemcpy(d_checkpoints + cp * n, d_next, n * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        std::swap(d_prev, d_curr);
        std::swap(d_curr, d_next);
    }

    // Save last checkpoint if nt not aligned.
    if (cfg.nt % checkpoint_interval != 0) {
        int cp = cfg.nt / checkpoint_interval;
        CUDA_CHECK(cudaMemcpy(d_checkpoints + cp * n, d_curr, n * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaEventRecord(f1));
    CUDA_CHECK(cudaEventSynchronize(f1));
    CUDA_CHECK(cudaEventElapsedTime(&perf.forward_ms, f0, f1));

    // Backward propagation + reconstruction + imaging.
    CUDA_CHECK(cudaEventRecord(b0));
    for (int it = cfg.nt - 1; it >= 0; --it) {
        forward_step_driver(d_bw_prev, d_bw_curr, d_bw_next, dm, cfg, grid3d, blk3d, nx_dir, ny_dir, nz_dir,
                            mx_dir, my_dir, mz_dir, lx_dir, ly_dir, lz_dir);
        backward_inject_kernel<<<grid2d, blk2d>>>(d_bw_next, d_rec, cfg.nx, cfg.ny, cfg.sz, it);
        absorb_boundary_kernel<<<grid3d, blk3d>>>(d_bw_next, cfg.nx, cfg.ny, cfg.nz, cfg.pml, cfg.sponge_strength);

        reconstruct_forward_at_t(d_fw_tmp, it, cfg, dm, d_checkpoints, n, grid3d, blk3d, grid2d, blk2d,
                                 nx_dir, ny_dir, nz_dir, mx_dir, my_dir, mz_dir, lx_dir, ly_dir, lz_dir,
                                 perf);

        imaging_kernel<<<grid1d, blk1d>>>(d_fw_tmp, d_bw_next, d_image, n);

        std::swap(d_bw_prev, d_bw_curr);
        std::swap(d_bw_curr, d_bw_next);
    }
    CUDA_CHECK(cudaEventRecord(b1));
    CUDA_CHECK(cudaEventSynchronize(b1));
    CUDA_CHECK(cudaEventElapsedTime(&perf.backward_ms, b0, b1));

    std::vector<float> image(n);
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream fout("image.bin", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(image.data()), n * sizeof(float));
    fout.close();

    // Very rough performance estimates for stencil kernel.
    const double updates = static_cast<double>(cfg.nt) * static_cast<double>(n);
    const double flops_per_update = 320.0; // approximate for derivatives + combination.
    const double bytes_per_update = 4.0 * 24.0; // approximate global read/write traffic in bytes.

    double fw_s = perf.forward_ms * 1e-3;
    double bw_s = perf.backward_ms * 1e-3;
    double gflops_fw = (updates * flops_per_update) / fw_s / 1e9;
    double bw_fw = (updates * bytes_per_update) / fw_s / 1e9;

    std::cout << "\n=== Performance (demo-level estimates) ===\n";
    std::cout << "Forward phase: " << perf.forward_ms << " ms\n";
    std::cout << "Backward phase: " << perf.backward_ms << " ms\n";
    std::cout << "Reconstruction cumulative: " << perf.reconstruct_ms << " ms\n";
    std::cout << "Forward estimated GFLOPS: " << gflops_fw << "\n";
    std::cout << "Forward estimated memory BW (GB/s): " << bw_fw << "\n";
    std::cout << "Output written to image.bin\n";

    cudaEventDestroy(f0);
    cudaEventDestroy(f1);
    cudaEventDestroy(b0);
    cudaEventDestroy(b1);

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_next);
    cudaFree(d_bw_prev);
    cudaFree(d_bw_curr);
    cudaFree(d_bw_next);
    cudaFree(d_rec);
    cudaFree(d_image);
    cudaFree(d_fw_tmp);
    cudaFree(d_checkpoints);
    free_device_model(dm);

    return 0;
}
