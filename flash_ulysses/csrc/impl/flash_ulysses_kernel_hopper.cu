// Native ThunderKittens, pinned by the thirdparty/ThunderKittens submodule.
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace flash_ulysses {
using namespace kittens;

// Compile-time geometry, storage layout and TK launch requirements.
struct kernel_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int CONSUMER_WARPS = CONSUMER_WARPGROUPS * WARPGROUP_WARPS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 192;

    static constexpr int INPUT_STAGES = 3;
    static constexpr int OUTPUT_STAGES = 2;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int HEAD_DIM = 128;
    static constexpr int COL_BLOCK = HEAD_DIM;  // One complete Ulysses head per task.
    static constexpr int RED_BLOCK = 64;
    static constexpr int ROWS_PER_WG = ROW_BLOCK / CONSUMER_WARPGROUPS;

    using A_tile = st_bf<ROWS_PER_WG, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROWS_PER_WG, COL_BLOCK>;
    using accumulator_tile = rt_fl<ROWS_PER_WG / WARPGROUP_WARPS, COL_BLOCK>;

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, -1, 1, -1, -1, C_tile>;  // [source SP rank, 1, L, head columns]
    using positions_gl = gl<int64_t, 1, 1, 1, -1>;
    using norm_gl = gl<float, 1, 1, 1, HEAD_DIM>;
    using rope_gl = gl<float, 1, 1, -1, HEAD_DIM>;

    struct pipeline_inputs {
        A_tile A[CONSUMER_WARPGROUPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[CONSUMER_WARPGROUPS];
    };

    struct alignas(1024) shared_state {
        pipeline_inputs inputs[INPUT_STAGES];
        pipeline_outputs outputs[OUTPUT_STAGES];
        semaphore inputs_arrived[INPUT_STAGES];
        semaphore inputs_finished[INPUT_STAGES];
        semaphore outputs_arrived[OUTPUT_STAGES];
        semaphore outputs_finished[OUTPUT_STAGES];
    };

    static constexpr int DYNAMIC_SHARED_MEMORY = sizeof(shared_state);
};

// Per-launch tensor descriptors, topology and persistent-grid scheduling.
template<int NumDevices>
struct kernel_args {
    using C_pgl = pgl<kernel_config::C_gl, NumDevices, false>;

    kernel_config::A_gl A;
    kernel_config::B_gl B;
    C_pgl Q, K, V;
    kernel_config::positions_gl positions;
    kernel_config::norm_gl q_gamma, k_gamma;
    kernel_config::rope_gl rope;
    int group_start;
    int sp_rank;
    int num_tasks;
    int q_heads, kv_heads;  // Total heads in this base-TP projection, before SP exchange.
    float q_epsilon, k_epsilon;

    int num_blocks;

    __host__ dim3 grid() const { return dim3(num_blocks, 1, 1); }
};

// Apply NeoX RoPE directly to the WGMMA register layout.
template<int N>
__device__ void apply_neox_rope(const kernel_args<N>& G,
                              kernel_config::accumulator_tile& accumulator,
                              int row_idx) {
    using RT = kernel_config::accumulator_tile;
    constexpr int half_dim = kernel_config::HEAD_DIM / 2;
    // NeoX partners are four base tiles apart in the same lane. Each float2
    // holds adjacent columns, so rotation needs no inter-lane communication.
    const int lane = warp::laneid();
    const int first_row = row_idx * kernel_config::ROW_BLOCK +
        warpgroup::groupid() * kernel_config::ROWS_PER_WG +
        warpgroup::warpid() * RT::rows + lane / 4;

    #pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
        const int m = first_row + row_half * 8;
        // The four lanes sharing a row take the same branch. Padded rows
        // must never read positions, even though their TMA stores are clipped.
        if (m >= G.A.rows()) continue;
        const int pos = static_cast<int>(G.positions[{m}]);

        #pragma unroll
        for (int j = 0; j < RT::width / 2; ++j) {
            #pragma unroll
            for (int col_half = 0; col_half < 2; ++col_half) {
                const int k = row_half + col_half * 2;
                const int col = j * RT::tile_size_col + col_half * 8 + (lane % 4) * 2;
                const float2 lo = accumulator.tiles[0][j].data[k];
                const float2 hi = accumulator.tiles[0][j + RT::width / 2].data[k];
                const float2 cosine = {G.rope[{0, 0, pos, col}], G.rope[{0, 0, pos, col + 1}]};
                const float2 sine = {G.rope[{0, 0, pos, col + half_dim}], G.rope[{0, 0, pos, col + half_dim + 1}]};
                accumulator.tiles[0][j].data[k] = {
                    lo.x * cosine.x - hi.x * sine.x,
                    lo.y * cosine.y - hi.y * sine.y};
                accumulator.tiles[0][j + RT::width / 2].data[k] = {
                    hi.x * cosine.x + lo.x * sine.x,
                    hi.y * cosine.y + lo.y * sine.y};
            }
        }
    }
}

// Keep GEMM, RMSNorm and RoPE results in FP32 registers until the BF16 shared store.
template<int N>
__device__ void rmsnorm_and_rope(const kernel_args<N>& G,
                               kernel_config::accumulator_tile& accumulator,
                               int row_idx, int col_idx) {
    using RT = kernel_config::accumulator_tile;
    const bool is_q = col_idx < G.q_heads;
    const float epsilon = is_q ? G.q_epsilon : G.k_epsilon;

    RT::row_vec gamma;
    RT::col_vec scale;

    {
        RT squared;
        warp::mul(squared, accumulator, accumulator);
        warp::row_sum(scale, squared);
    }
    warp::apply(scale, scale, [epsilon](int, float sum) {
        return rsqrtf(sum / RT::cols + epsilon);
    });
    warp::load(gamma, is_q ? G.q_gamma : G.k_gamma, {0});
    warp::mul_row(accumulator, accumulator, scale);
    warp::mul_col(accumulator, accumulator, gamma);

    apply_neox_rope(G, accumulator, row_idx);
}

// All CTAs have the same roles, following TK parallel/gemm_rs/gemm_rs_h100.cu:
// WG0/WG1: WGMMA + RMSNorm/RoPE; WG2 warp0: TMA load; WG2 warp1: peer TMA store.
template<int N>
__device__ void kernel(const kernel_args<N>& G) {
    extern __shared__ __align__(1024) int storage[];
    tma_swizzle_allocator allocator(storage);

    auto& sm = allocator.allocate<kernel_config::shared_state>();

    if (threadIdx.x == 0) {
        for (int i = 0; i < kernel_config::INPUT_STAGES; ++i) {
            init_semaphore(sm.inputs_arrived[i], 0, 1);
            init_semaphore(sm.inputs_finished[i], kernel_config::CONSUMER_WARPS, 0);
        }
        for (int i = 0; i < kernel_config::OUTPUT_STAGES; ++i) {
            init_semaphore(sm.outputs_arrived[i], kernel_config::CONSUMER_WARPGROUPS, 0);
            init_semaphore(sm.outputs_finished[i], 1, 0);
        }
    }
    __syncthreads();

    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();
    int input_stage = 0;
    int output_stage = 0;
    // Per-thread phase state, carried across tasks. Lower half: inputs_arrived/outputs_arrived;
    // upper half: inputs_finished/outputs_finished. Output rings follow the input rings.
    // Initial finished phase 1 lets the first use of each empty slot proceed.
    uint32_t phasebits = 0xFFFF0000;
    const int num_iters = G.A.cols() / kernel_config::RED_BLOCK;
    const int col_blocks = G.B.cols() / kernel_config::COL_BLOCK;

    if (warpgroup_id == kernel_config::CONSUMER_WARPGROUPS) {
        warpgroup::decrease_registers<kernel_config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < G.num_tasks; task_id += gridDim.x) {
                const int row_idx = task_id / col_blocks;
                const int col_idx = task_id % col_blocks;  // Packed Q/K/V head ID.

                for (int red_idx = 0; red_idx < num_iters; ++red_idx) {
                    wait(sm.inputs_finished[input_stage], get_phasebit<1>(phasebits, input_stage));
                    update_phasebit<1>(phasebits, input_stage);
                    tma::expect(sm.inputs_arrived[input_stage], sm.inputs[input_stage]);
                    #pragma unroll
                    for (int i = 0; i < kernel_config::CONSUMER_WARPGROUPS; ++i)
                        tma::load_async(sm.inputs[input_stage].A[i], G.A,
                            {row_idx * kernel_config::CONSUMER_WARPGROUPS + i, red_idx}, sm.inputs_arrived[input_stage]);
                    tma::load_async(sm.inputs[input_stage].B, G.B, {red_idx, col_idx}, sm.inputs_arrived[input_stage]);
                    input_stage = (input_stage + 1) % kernel_config::INPUT_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < G.num_tasks; task_id += gridDim.x) {
                const int row_idx = task_id / col_blocks;
                const int col_idx = task_id % col_blocks;
                const int component = col_idx < G.q_heads ? 0 :
                    (col_idx < G.q_heads + G.kv_heads ? 1 : 2);
                const int head_id = col_idx - (component == 0 ? 0 :
                    G.q_heads + (component == 2 ? G.kv_heads : 0));
                const auto& output = component == 0 ? G.Q : (component == 1 ? G.K : G.V);
                const int heads_per_gpu = output[G.group_start].cols() / kernel_config::HEAD_DIM;
                const int dst_gpu = G.group_start + head_id / heads_per_gpu;
                const int dst_col_idx = head_id % heads_per_gpu;

                wait(sm.outputs_arrived[output_stage], get_phasebit<0>(phasebits, kernel_config::INPUT_STAGES + output_stage));
                update_phasebit<0>(phasebits, kernel_config::INPUT_STAGES + output_stage);

                // Each source is a separate batch with L rows. TMA drops OOB
                // rows within that batch instead of writing into the next source.
                #pragma unroll
                for (int i = 0; i < kernel_config::CONSUMER_WARPGROUPS; ++i) {
                    const int dst_row_idx = row_idx * kernel_config::CONSUMER_WARPGROUPS + i;
                    tma::store_async(output[dst_gpu], sm.outputs[output_stage].C[i],
                                     {G.sp_rank, 0, dst_row_idx, dst_col_idx});
                }
                // Complete peer writes before releasing this shared output slot.
                tma::store_async_wait();
                arrive(sm.outputs_finished[output_stage]);
                output_stage = (output_stage + 1) % kernel_config::OUTPUT_STAGES;
            }
        }
    } else {
        warpgroup::increase_registers<kernel_config::CONSUMER_REGISTERS>();

        for (int task_id = blockIdx.x; task_id < G.num_tasks; task_id += gridDim.x) {
            const int row_idx = task_id / col_blocks;
            const int col_idx = task_id % col_blocks;

            kernel_config::accumulator_tile accumulator;
            warp::zero(accumulator);

            for (int red_idx = 0; red_idx < num_iters; ++red_idx) {
                wait(sm.inputs_arrived[input_stage], get_phasebit<0>(phasebits, input_stage));
                update_phasebit<0>(phasebits, input_stage);
                warpgroup::mma_AB(accumulator, sm.inputs[input_stage].A[warpgroup_id], sm.inputs[input_stage].B);
                warpgroup::mma_async_wait();
                warp::arrive(sm.inputs_finished[input_stage]);
                input_stage = (input_stage + 1) % kernel_config::INPUT_STAGES;
            }

            if (col_idx < G.q_heads + G.kv_heads)
                rmsnorm_and_rope(G, accumulator, row_idx, col_idx);

            wait(sm.outputs_finished[output_stage], get_phasebit<1>(phasebits, kernel_config::INPUT_STAGES + output_stage));
            update_phasebit<1>(phasebits, kernel_config::INPUT_STAGES + output_stage);
            warpgroup::store(sm.outputs[output_stage].C[warpgroup_id], accumulator);
            warpgroup::sync(warpgroup_id + 1);
            warpgroup::arrive(sm.outputs_arrived[output_stage]);
            output_stage = (output_stage + 1) % kernel_config::OUTPUT_STAGES;
        }
    }
}

struct barrier_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

template<int N>
struct barrier_args {
    barrier_t<N> barrier;
    int dev_idx;
};

template<int N>
__device__ void barrier_kernel(const barrier_args<N>& G) {
    kittens::barrier_all(G.barrier, {0}, G.dev_idx);
    // TK polls with a relaxed system load. Order subsequent consumption after
    // observing every peer's release, before this stream continues.
    __threadfence_system();
}

// TK requires PGL::num_devices to match the native tensor's physical world.
template<typename F>
decltype(auto) dispatch_world(int world, F&& f) {
    switch (world) {
        case 2: return f.template operator()<2>();
        case 3: return f.template operator()<3>();
        case 4: return f.template operator()<4>();
        case 5: return f.template operator()<5>();
        case 6: return f.template operator()<6>();
        case 7: return f.template operator()<7>();
        case 8: return f.template operator()<8>();
        default: TORCH_CHECK(false, "Expected 2 to 8 GPUs");
    }
}

void launch(const at::Tensor& x, const at::Tensor& qkv_weight,
            const at::Tensor& positions, const at::Tensor& q_gamma,
            const at::Tensor& k_gamma, const at::Tensor& rope,
            kittens::py::TKParallelTensor& Q, kittens::py::TKParallelTensor& K,
            kittens::py::TKParallelTensor& V, kittens::py::TKParallelTensor& barrier,
            int q_heads, int kv_heads, int sp, double q_epsilon, double k_epsilon) {
    const int world_size = Q.local_world_size_;
    const int sp_rank = Q.local_rank_ % sp;
    const int64_t local_tokens = x.size(0);
    const int64_t num_tasks = kittens::cdiv<int64_t>(local_tokens, kernel_config::ROW_BLOCK) *
                              (q_heads + 2 * kv_heads);
    const int num_blocks = static_cast<int>(std::min<int64_t>(
        num_tasks, at::cuda::getCurrentDeviceProperties()->multiProcessorCount));

    dispatch_world(world_size, [&]<int N>() {
        kernel_args<N> kernel_G{
            .A = kittens::py::tensor_to_gl<kernel_config::A_gl>(x),
            .B = kittens::py::tensor_to_gl<kernel_config::B_gl>(qkv_weight),
            .Q = kittens::py::parallel_tensor_to_pgl<typename kernel_args<N>::C_pgl>(
                Q, sp, 1, local_tokens, Q.data_.size(1)),
            .K = kittens::py::parallel_tensor_to_pgl<typename kernel_args<N>::C_pgl>(
                K, sp, 1, local_tokens, K.data_.size(1)),
            .V = kittens::py::parallel_tensor_to_pgl<typename kernel_args<N>::C_pgl>(
                V, sp, 1, local_tokens, V.data_.size(1)),
            .positions = kittens::py::tensor_to_gl<kernel_config::positions_gl>(positions),
            .q_gamma = kittens::py::tensor_to_gl<kernel_config::norm_gl>(q_gamma),
            .k_gamma = kittens::py::tensor_to_gl<kernel_config::norm_gl>(k_gamma),
            .rope = kittens::py::tensor_to_gl<kernel_config::rope_gl>(rope),
            .group_start = Q.local_rank_ / sp * sp,
            .sp_rank = sp_rank,
            .num_tasks = static_cast<int>(num_tasks),
            .q_heads = q_heads,
            .kv_heads = kv_heads,
            .q_epsilon = static_cast<float>(q_epsilon),
            .k_epsilon = static_cast<float>(k_epsilon),
            .num_blocks = num_blocks,
        };

        barrier_args<N> barrier_G{
            .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<N>>(barrier),
            .dev_idx = barrier.local_rank_,
        };

        // Match TK GEMM-RS: stream order joins all local CTAs before the
        // single-thread multicast barrier joins the physical GPU world.
        kittens::py::launch_kernel<kernel_config, kernel_args<N>, kernel<N>>(kernel_G);
        kittens::py::launch_kernel<barrier_config, barrier_args<N>, barrier_kernel<N>>(barrier_G);
    });
}
}  // namespace flash_ulysses
