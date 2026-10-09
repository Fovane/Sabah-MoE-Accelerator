#include "sabah-bridge.h"

#include <windows.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <algorithm>
#include <vector>
#include <climits>

typedef int (__cdecl *sabah_rt_init_fn)(int);
typedef int (__cdecl *sabah_rt_shutdown_fn)(void);
typedef const char * (__cdecl *sabah_rt_last_error_fn)(void);

typedef int (__cdecl *sabah_rt_mul_mat_id_v2_fn)(
    const void *, size_t, size_t,
    const void *, size_t, int, size_t,
    const void *, size_t, size_t,
    int, int,
    float *, size_t, size_t,
    int, int, int, int,
    void *);

static HMODULE g_sabah = nullptr;
static sabah_rt_init_fn g_init = nullptr;
static sabah_rt_shutdown_fn g_shutdown = nullptr;
static sabah_rt_last_error_fn g_last_error = nullptr;
static sabah_rt_mul_mat_id_v2_fn g_mul_mat_id = nullptr;

static std::once_flag g_once;
static bool g_enabled = false;
static bool g_failed = false;
static std::string g_error;

static std::mutex g_init_mutex;
static int g_initialized_device = -1;
static bool g_debug_logs = false;

struct SabahCompareState {
    std::vector<float> sabah_result;
    size_t elements = 0;
    bool pending = false;
};

static SabahCompareState g_sabah_compare_state;
static std::mutex g_sabah_compare_mutex;

static void sabah_load_once() {
    std::call_once(g_once, [] {
        const char * debug_env = std::getenv("SABAH_DEBUG");
        g_debug_logs = (debug_env != nullptr && std::strcmp(debug_env, "0") != 0);

        const char * path = std::getenv("SABAH_RT_LIB");

        if (!path || !*path) {
            g_failed = true;
            g_error = "SABAH_RT_LIB is not set. Falling back to native llama.cpp CUDA kernels.";
            std::fprintf(stderr, "\n[SABAH] INFO: %s\n\n", g_error.c_str());
            std::fflush(stderr);
            return;
        }

        std::fprintf(stderr, "[SABAH] Loading DLL: %s\n", path);
        std::fflush(stderr);

        g_sabah = LoadLibraryA(path);
        if (!g_sabah) {
            DWORD err = GetLastError();
            char msg[512] = {};
            FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr, err, 0, msg, sizeof(msg), nullptr);
            g_failed = true;
            g_error = std::string("LoadLibraryA failed: ") + path + " | Win32=" + std::to_string(err) + " | " + msg;
            std::fprintf(stderr, "[SABAH] DLL LOAD FAILED: %s\n", g_error.c_str());
            std::fflush(stderr);
            return;
        }

        g_init = reinterpret_cast<sabah_rt_init_fn>(GetProcAddress(g_sabah, "sabah_rt_init"));
        g_shutdown = reinterpret_cast<sabah_rt_shutdown_fn>(GetProcAddress(g_sabah, "sabah_rt_shutdown"));
        g_last_error = reinterpret_cast<sabah_rt_last_error_fn>(GetProcAddress(g_sabah, "sabah_rt_last_error"));
        g_mul_mat_id = reinterpret_cast<sabah_rt_mul_mat_id_v2_fn>(GetProcAddress(g_sabah, "sabah_rt_mul_mat_id_v2"));

        if (!g_init || !g_shutdown || !g_last_error || !g_mul_mat_id) {
            g_failed = true;
            g_error = "Sabah runtime missing required exports";
            std::fprintf(stderr, "[SABAH] EXPORT ERROR: %s\n", g_error.c_str());
            std::fflush(stderr);
            return;
        }

        g_enabled = true;
        std::fprintf(stderr, "[SABAH] DLL bridge loaded successfully! Fast MoE kernels active.\n");
        std::fflush(stderr);
    });
}

static bool sabah_type_to_qtype(ggml_type type, int & qt) {
    switch (type) {
        case GGML_TYPE_Q1_0: qt = 41; return true;
        case GGML_TYPE_Q2_0: qt = 42; return true;
        default: return false;
    }
}

static void sabah_compare_capture(const ggml_tensor * dst) {
    if (!dst || dst->type != GGML_TYPE_F32) return;
    const size_t elements = ggml_nelements(dst);
    const size_t bytes = elements * sizeof(float);
    if (ggml_nbytes(dst) != bytes) return;

    std::vector<float> host(elements);
    if (cudaDeviceSynchronize() != cudaSuccess) return;
    if (cudaMemcpy(host.data(), dst->data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;

    {
        std::lock_guard<std::mutex> lock(g_sabah_compare_mutex);
        g_sabah_compare_state.sabah_result = std::move(host);
        g_sabah_compare_state.elements = elements;
        g_sabah_compare_state.pending = true;
    }
}

bool ggml_cuda_sabah_mul_mat_id(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst,
    void * stream) {

    sabah_load_once();

    if (!g_enabled || !g_mul_mat_id) {
        return false;
    }

    if (!src0 || !src1 || !ids || !dst) return false;

    int qt = 0;
    if (!sabah_type_to_qtype(src0->type, qt)) return false;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;

    const int k       = (int) src0->ne[0];
    const int rows    = (int) src0->ne[1];
    const int experts = (int) src0->ne[2];
    const int x_rows  = (int) src1->ne[1];
    const int n_ids   = (int) ids->ne[0];
    const int tokens  = (int) ids->ne[1];

    if (k <= 0 || rows <= 0 || experts <= 0 || x_rows <= 0 || n_ids <= 0 || tokens <= 0) return false;
    if ((k % 32) != 0) return false;
    if (x_rows != 1 && x_rows != n_ids) return false;

    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t)) return false;

    const size_t expected_row_bytes = qt == 41 ? ((size_t) k / 128) * 18 : ((size_t) k / 64) * 18;
    if (src0->nb[1] < expected_row_bytes || src0->nb[2] < src0->nb[1] * (size_t) rows) return false;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;

    {
        std::lock_guard<std::mutex> lock(g_init_mutex);
        if (g_initialized_device < 0) {
            if (g_init(device) != 0) {
                return false;
            }
            g_initialized_device = device;
        }
        if (g_initialized_device != device) return false;
    }

    if (stream != nullptr) {
        cudaStream_t llama_stream = reinterpret_cast<cudaStream_t>(stream);
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(llama_stream, &capture_status) != cudaSuccess || capture_status != cudaStreamCaptureStatusNone) {
            return false;
        }
        if (cudaStreamSynchronize(llama_stream) != cudaSuccess) return false;
    }

    const int rc = g_mul_mat_id(
        src0->data, src0->nb[2], src0->nb[1],
        src1->data, src1->nb[1], x_rows, src1->nb[2],
        ids->data, ids->nb[0], ids->nb[1],
        n_ids, tokens,
        static_cast<float *>(dst->data), dst->nb[1], dst->nb[2],
        rows, k, experts, qt,
        nullptr);

    if (rc == 0 && std::getenv("SABAH_COMPARE") != nullptr && std::strcmp(std::getenv("SABAH_COMPARE"), "0") != 0) {
        sabah_compare_capture(dst);
    }

    return (rc == 0);
}

void ggml_cuda_sabah_compare_after_fallback(const ggml_tensor * dst, void * stream) {
    if (stream != nullptr) {
        cudaStream_t compare_stream = reinterpret_cast<cudaStream_t>(stream);
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(compare_stream, &capture_status) != cudaSuccess || capture_status != cudaStreamCaptureStatusNone) {
            return;
        }
    }

    if (!dst || dst->type != GGML_TYPE_F32) return;

    std::vector<float> sabah;
    size_t elements = 0;

    {
        std::lock_guard<std::mutex> lock(g_sabah_compare_mutex);
        if (!g_sabah_compare_state.pending) return;
        elements = g_sabah_compare_state.elements;
        sabah = std::move(g_sabah_compare_state.sabah_result);
        g_sabah_compare_state.elements = 0;
        g_sabah_compare_state.pending = false;
    }

    const size_t bytes = elements * sizeof(float);
    if (ggml_nelements(dst) != elements || ggml_nbytes(dst) != bytes) return;

    std::vector<float> fallback(elements);
    if (cudaDeviceSynchronize() != cudaSuccess) return;
    if (cudaMemcpy(fallback.data(), dst->data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;

    float max_abs = 0.0f;
    for (size_t i = 0; i < elements; ++i) {
        float abs_err = std::fabs(sabah[i] - fallback[i]);
        if (abs_err > max_abs) max_abs = abs_err;
    }

    std::fprintf(stderr, "[SABAH_COMPARE] Elements: %zu | Max Error: %.6f\n", elements, max_abs);
    std::fflush(stderr);
}