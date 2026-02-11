#pragma once

/**
 * Runtime configuration for the 3D TTI RTM demo.
 * All dimensions include physical domain only (without padding).
 */
struct Config {
    int nx = 96;
    int ny = 96;
    int nz = 96;
    int nt = 300;

    float dx = 10.0f;
    float dy = 10.0f;
    float dz = 10.0f;
    float dt = 0.001f;

    float f0 = 12.0f;                 // Ricker dominant frequency (Hz)
    int sx = nx / 2;                  // source x index
    int sy = ny / 2;                  // source y index
    int sz = 8;                       // source depth index

    int pml = 12;                     // sponge thickness
    float sponge_strength = 0.018f;   // sponge damping intensity

    int checkpoint_interval = 20;     // save forward state every k time steps

    // Simplified TTI angles (global for this demo).
    float tilt_deg = 25.0f;           // tilt angle theta
    float azimuth_deg = 35.0f;        // azimuth phi
};
