// Native ThunderKittens, pinned by the thirdparty/ThunderKittens submodule.
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace flash_ulysses {
using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    // Persistent compute grid; work is assigned using gridDim.x.
    static constexpr int NUM_BLOCKS = 132;
    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int CONSUMER_WARPS = CONSUMER_WARPGROUPS * WARPGROUP_WARPS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 192;
};

struct tiles {
    static constexpr int INPUT_STAGES = 3;
    static constexpr int OUTPUT_STAGES = 2;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int HEAD_DIM = 128;
    static constexpr int COL_BLOCK = HEAD_DIM;  // One complete Ulysses head per task.
    static constexpr int RED_BLOCK = 64;
    static constexpr int ROWS_PER_WG = ROW_BLOCK / config::CONSUMER_WARPGROUPS;

    using A_tile = st_bf<ROWS_PER_WG, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROWS_PER_WG, COL_BLOCK>;
    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, -1, 1, -1, -1, C_tile>;  // [source SP rank, 1, L, head columns]
    using positions_gl = gl<int64_t, 1, 1, 1, -1>;
    using norm_gl = gl<float, 1, 1, 1, HEAD_DIM>;
    using rope_gl = gl<float, 1, 1, -1, HEAD_DIM>;

    struct pipeline_inputs {
        A_tile A[config::CONSUMER_WARPGROUPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[config::CONSUMER_WARPGROUPS];
    };
};

template<int NumDevices>
struct globals : tiles {
    static constexpr int NUM_DEVICES = NumDevices;
    using C_pgl = pgl<C_gl, NumDevices, false>;

    A_gl A;
    B_gl B;
    C_pgl Q, K, V;
    positions_gl positions;
    norm_gl q_gamma, k_gamma;
    rope_gl rope;
    int group_start;
    int sp_rank;
    int num_tasks;
    int q_heads, kv_heads;  // Total heads in this base-TP projection, before SP exchange.
    float q_epsilon, k_epsilon;
};

struct alignas(1024) shared_state {
    tiles::pipeline_inputs inputs[tiles::INPUT_STAGES];
    tiles::pipeline_outputs outputs[tiles::OUTPUT_STAGES];
    semaphore inputs_arrived[tiles::INPUT_STAGES];
    semaphore inputs_finished[tiles::INPUT_STAGES];
    semaphore outputs_arrived[tiles::OUTPUT_STAGES];
    semaphore outputs_finished[tiles::OUTPUT_STAGES];
};

// Called by the 8 consumer warps after FP32 -> BF16 shared-store rounding.
template<int N>
__device__ void epilogue(const globals<N>& G, shared_state& sm,
                         int row_idx, int col_idx, int output_stage) {
    if (col_idx >= G.q_heads + G.kv_heads) return;

    const bool is_q = col_idx < G.q_heads;
    const int warpgroup_id = warpgroup::groupid();
    const float epsilon = is_q ? G.q_epsilon : G.k_epsilon;
    auto& output = sm.outputs[output_stage].C[warpgroup_id];

    rv_fl<tiles::HEAD_DIM> gamma;
    warp::load(gamma, is_q ? G.q_gamma : G.k_gamma, {0});

    for (int row = warpgroup::warpid(); row < tiles::ROWS_PER_WG; row += WARPGROUP_WARPS) {
        const int m = row_idx * tiles::ROW_BLOCK + warpgroup_id * tiles::ROWS_PER_WG + row;
        if (m >= G.A.rows()) continue;

        rv_bf<tiles::HEAD_DIM> rounded;
        rv_fl<tiles::HEAD_DIM> value, squared;

        warp::load(rounded, output, {row, 0});
        warp::copy(value, rounded);
        warp::mul(squared, value, value);
        const float scale = rsqrtf(warp::sum(squared) / tiles::HEAD_DIM + epsilon);
        warp::mul(value, value, scale);
        warp::mul(value, value, gamma);
        warp::copy(rounded, value);
        warp::copy(value, rounded);

        const int64_t pos = G.positions[{m}];
        rv_fl<tiles::HEAD_DIM / 2> lo, hi, cosine, sine, lo_cos, hi_sin;

        // NeoX pairs the two halves of the head. In naive RV layout each
        // partner is in the same lane, two registers away.
        #pragma unroll
        for (int i = 0; i < decltype(lo)::outer_dim; ++i) {
            lo[i][0] = value[i][0];
            hi[i][0] = value[i + decltype(lo)::outer_dim][0];
        }
        warp::load(cosine, G.rope, {0, 0, static_cast<int>(pos), 0});
        warp::load(sine, G.rope, {0, 0, static_cast<int>(pos), 1});
        warp::mul(lo_cos, lo, cosine);
        warp::mul(hi_sin, hi, sine);
        warp::sub(lo_cos, lo_cos, hi_sin);
        warp::mul(hi, hi, cosine);
        warp::mul(lo, lo, sine);
        warp::add(hi, hi, lo);

        #pragma unroll
        for (int i = 0; i < decltype(lo)::outer_dim; ++i) {
            value[i][0] = lo_cos[i][0];
            value[i + decltype(lo)::outer_dim][0] = hi[i][0];
        }
        warp::copy(rounded, value);

        warp::store(output, rounded, {row, 0});
    }
}

// All CTAs have the same roles, following TK parallel/gemm_rs/gemm_rs_h100.cu:
// WG0/WG1: WGMMA + epilogue; WG2 warp0: TMA load; WG2 warp1: peer TMA store.
template<int N>
__device__ void kernel(const globals<N>& G) {
    extern __shared__ __align__(1024) int storage[];
    tma_swizzle_allocator allocator(storage);

    auto& sm = allocator.allocate<shared_state>();

    if (threadIdx.x == 0) {
        for (int i = 0; i < tiles::INPUT_STAGES; ++i) {
            init_semaphore(sm.inputs_arrived[i], 0, 1);
            init_semaphore(sm.inputs_finished[i], config::CONSUMER_WARPS, 0);
        }
        for (int i = 0; i < tiles::OUTPUT_STAGES; ++i) {
            init_semaphore(sm.outputs_arrived[i], config::CONSUMER_WARPGROUPS, 0);
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
    const int num_iters = G.A.cols() / tiles::RED_BLOCK;
    const int col_blocks = G.B.cols() / tiles::COL_BLOCK;

    if (warpgroup_id == config::CONSUMER_WARPGROUPS) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < G.num_tasks; task_id += gridDim.x) {
                const int row_idx = task_id / col_blocks;
                const int col_idx = task_id % col_blocks;  // Packed Q/K/V head ID.

                for (int red_idx = 0; red_idx < num_iters; ++red_idx) {
                    wait(sm.inputs_finished[input_stage], get_phasebit<1>(phasebits, input_stage));
                    update_phasebit<1>(phasebits, input_stage);
                    tma::expect(sm.inputs_arrived[input_stage], sm.inputs[input_stage]);
                    #pragma unroll
                    for (int i = 0; i < config::CONSUMER_WARPGROUPS; ++i)
                        tma::load_async(sm.inputs[input_stage].A[i], G.A,
                            {row_idx * config::CONSUMER_WARPGROUPS + i, red_idx}, sm.inputs_arrived[input_stage]);
                    tma::load_async(sm.inputs[input_stage].B, G.B, {red_idx, col_idx}, sm.inputs_arrived[input_stage]);
                    input_stage = (input_stage + 1) % tiles::INPUT_STAGES;
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
                const int heads_per_gpu = output[G.group_start].cols() / tiles::HEAD_DIM;
                const int dst_gpu = G.group_start + head_id / heads_per_gpu;
                const int dst_col_idx = head_id % heads_per_gpu;

                wait(sm.outputs_arrived[output_stage], get_phasebit<0>(phasebits, tiles::INPUT_STAGES + output_stage));
                update_phasebit<0>(phasebits, tiles::INPUT_STAGES + output_stage);

                // Each source is a separate batch with L rows. TMA drops OOB
                // rows within that batch instead of writing into the next source.
                #pragma unroll
                for (int i = 0; i < config::CONSUMER_WARPGROUPS; ++i) {
                    const int dst_row_idx = row_idx * config::CONSUMER_WARPGROUPS + i;
                    tma::store_async(output[dst_gpu], sm.outputs[output_stage].C[i],
                                     {G.sp_rank, 0, dst_row_idx, dst_col_idx});
                }
                // Complete peer writes before releasing this shared output slot.
                tma::store_async_wait();
                arrive(sm.outputs_finished[output_stage]);
                output_stage = (output_stage + 1) % tiles::OUTPUT_STAGES;
            }
        }
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int task_id = blockIdx.x; task_id < G.num_tasks; task_id += gridDim.x) {
            const int row_idx = task_id / col_blocks;
            const int col_idx = task_id % col_blocks;

            rt_fl<tiles::ROWS_PER_WG / WARPGROUP_WARPS, tiles::COL_BLOCK> accumulator;
            warp::zero(accumulator);

            for (int red_idx = 0; red_idx < num_iters; ++red_idx) {
                wait(sm.inputs_arrived[input_stage], get_phasebit<0>(phasebits, input_stage));
                update_phasebit<0>(phasebits, input_stage);
                warpgroup::mma_AB(accumulator, sm.inputs[input_stage].A[warpgroup_id], sm.inputs[input_stage].B);
                warpgroup::mma_async_wait();
                warp::arrive(sm.inputs_finished[input_stage]);
                input_stage = (input_stage + 1) % tiles::INPUT_STAGES;
            }

            // Accumulation can overlap an earlier send. Do not overwrite its
            // shared slot until the store warp has finished reading it.
            wait(sm.outputs_finished[output_stage], get_phasebit<1>(phasebits, tiles::INPUT_STAGES + output_stage));
            update_phasebit<1>(phasebits, tiles::INPUT_STAGES + output_stage);
            warpgroup::store(sm.outputs[output_stage].C[warpgroup_id], accumulator);
            warpgroup::arrive_and_wait(barrier<WARPGROUP_WARPS>(warpgroup_id + 1));

            epilogue(G, sm, row_idx, col_idx, output_stage);

            // Every shared writer crosses the generic -> async proxy fence.
            // The consumer-only barrier joins both WG epilogues before handoff.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            group<config::CONSUMER_WARPS>::arrive_and_wait(barrier<config::CONSUMER_WARPS>(3));
            warpgroup::arrive(sm.outputs_arrived[output_stage]);
            output_stage = (output_stage + 1) % tiles::OUTPUT_STAGES;
        }
    }
}

struct compute_config : config {
    static constexpr int DYNAMIC_SHARED_MEMORY = sizeof(shared_state);
};

struct barrier_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

template<int N>
struct barrier_globals {
    barrier_t<N> barrier;
    int dev_idx;
};

template<int N>
__device__ void barrier_kernel(const barrier_globals<N>& G) {
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
    const int64_t num_tasks = kittens::cdiv<int64_t>(local_tokens, tiles::ROW_BLOCK) *
                              (q_heads + 2 * kv_heads);

    dispatch_world(world_size, [&]<int N>() {
        using G_type = globals<N>;

        G_type G{
            .A = kittens::py::tensor_to_gl<typename G_type::A_gl>(x),
            .B = kittens::py::tensor_to_gl<typename G_type::B_gl>(qkv_weight),
            .Q = kittens::py::parallel_tensor_to_pgl<typename G_type::C_pgl>(
                Q, sp, 1, local_tokens, Q.data_.size(1)),
            .K = kittens::py::parallel_tensor_to_pgl<typename G_type::C_pgl>(
                K, sp, 1, local_tokens, K.data_.size(1)),
            .V = kittens::py::parallel_tensor_to_pgl<typename G_type::C_pgl>(
                V, sp, 1, local_tokens, V.data_.size(1)),
            .positions = kittens::py::tensor_to_gl<typename G_type::positions_gl>(positions),
            .q_gamma = kittens::py::tensor_to_gl<typename G_type::norm_gl>(q_gamma),
            .k_gamma = kittens::py::tensor_to_gl<typename G_type::norm_gl>(k_gamma),
            .rope = kittens::py::tensor_to_gl<typename G_type::rope_gl>(rope),
            .group_start = Q.local_rank_ / sp * sp,
            .sp_rank = sp_rank,
            .num_tasks = static_cast<int>(num_tasks),
            .q_heads = q_heads,
            .kv_heads = kv_heads,
            .q_epsilon = static_cast<float>(q_epsilon),
            .k_epsilon = static_cast<float>(k_epsilon),
        };

        barrier_globals<N> barrier_G{
            .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<N>>(barrier),
            .dev_idx = barrier.local_rank_,
        };

        // Match TK GEMM-RS: stream order joins all local CTAs before the
        // single-thread multicast barrier joins the physical GPU world.
        kittens::py::launch_kernel<compute_config, G_type, kernel<N>>(G);
        kittens::py::launch_kernel<barrier_config, barrier_globals<N>, barrier_kernel<N>>(barrier_G);
    });
}
}  // namespace flash_ulysses
