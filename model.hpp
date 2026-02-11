#pragma once

#include <cmath>
#include <vector>

#include "config.hpp"

/**
 * Host-side model storage and synthetic model builder.
 */
struct Model {
    std::vector<float> vel;
    std::vector<float> eps;
    std::vector<float> delta;

    explicit Model(size_t n) : vel(n), eps(n), delta(n) {}
};

inline int idx3d(int x, int y, int z, int nx, int ny) {
    return z * nx * ny + y * nx + x;
}

/**
 * Build a layered model with gentle lateral perturbation.
 */
inline Model build_model(const Config& cfg) {
    const int n = cfg.nx * cfg.ny * cfg.nz;
    Model m(n);

    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                const int id = idx3d(x, y, z, cfg.nx, cfg.ny);
                const float zn = static_cast<float>(z) / static_cast<float>(cfg.nz - 1);
                const float yn = static_cast<float>(y) / static_cast<float>(cfg.ny - 1);
                const float xn = static_cast<float>(x) / static_cast<float>(cfg.nx - 1);

                float v = 1800.0f + 900.0f * zn;
                v += 80.0f * std::sin(6.283185f * xn) * std::sin(6.283185f * yn);
                m.vel[id] = v;

                m.eps[id] = 0.08f + 0.04f * zn;
                m.delta[id] = 0.03f + 0.02f * yn;
            }
        }
    }
    return m;
}

/**
 * Ricker wavelet evaluated at time t.
 */
inline float ricker(float t, float f0) {
    const float pi = 3.14159265358979323846f;
    const float t0 = 1.0f / f0;
    const float a = pi * f0 * (t - t0);
    const float a2 = a * a;
    return (1.0f - 2.0f * a2) * std::exp(-a2);
}
