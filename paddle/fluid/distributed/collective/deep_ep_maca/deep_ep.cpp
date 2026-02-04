// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <memory>

#include "paddle/fluid/distributed/collective/deep_ep_maca/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep_maca/deep_ep.hpp"
#include "paddle/fluid/distributed/collective/deep_ep_maca/kernels/api.cuh"
#include "paddle/fluid/distributed/collective/deep_ep_maca/kernels/configs.cuh"

#include "paddle/fluid/distributed/collective/deep_ep_maca/include/CUDADataType.h"
#include "paddle/fluid/distributed/collective/deep_ep_maca/include/ScalarType.h"
#include "paddle/fluid/distributed/collective/process_group_custom.h"

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

COMMON_DECLARE_int64(deep_ep_comm_prealloc_in_mb);

namespace deep_ep {
std::once_flag pre_alloc_once_flag;

namespace detail {
void SetAllocatorStreamForGPUContext(cudaStream_t stream,
                                     phi::CustomContext* ctx) {
  ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                        .GetAllocator(ctx->GetPlace(), stream)
                        .get());
}
}  // namespace detail

void PreAlloc(paddle::Tensor tensor, cudaStream_t stream) {
  int64_t numel = tensor.numel();
  auto alloc_size = FLAGS_deep_ep_comm_prealloc_in_mb * 1000000;
  std::cout << "alloc once here, size: " << alloc_size << " numel: " << numel
            << std::endl;
  std::cout << tensor.place() << "\t" << stream << std::endl;
  paddle::memory::allocation::AllocatorFacade::Instance()
      .GetAllocator(tensor.place(), stream)
      ->Allocate(alloc_size);
}

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               int context_ring_id)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode) {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
  paddle::distributed::ProcessGroup* pg = map->get(context_ring_id);
  const auto& place = phi::CustomPlace("metax_gpu", device_id); 
  comm_ctx =
      reinterpret_cast<paddle::distributed::ProcessGroupCustom*>(pg)
          ->GetOrCreateCommContext(place);
  comm_stream = deep_ep::detail::CUDAStream(reinterpret_cast<cudaStream_t>(comm_ctx->GetStream()->raw_stream()));
  calc_ctx = reinterpret_cast<phi::CustomContext*>(
      reinterpret_cast<paddle::distributed::ProcessGroupCustom*>(pg)
          ->GetDeviceContext(place, true));

  VLOG(3) << "DeepEP buffer device_id " << device_id << " context_ring_id "
          << context_ring_id << " comm_stream "
          << comm_stream << " compute_stream "
          << calc_ctx->stream();

  // Task fifo memory
  int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
  int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;
  int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;

  // Common checks
  EP_HOST_ASSERT(
      num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
      ((low_latency_mode || num_nvl_bytes <= std::numeric_limits<int>::max()) ||
       num_rdma_bytes == 0));
  EP_HOST_ASSERT(
      num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 &&
      (low_latency_mode || num_rdma_bytes <= std::numeric_limits<int>::max()));
  EP_HOST_ASSERT(0 <= rank && rank < num_ranks &&
                 (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS ||
                  low_latency_mode));
  EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS ||
                 num_ranks % NUM_MAX_NVL_PEERS == 0);
  if (num_rdma_bytes > 0)
    EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS || low_latency_mode);

  // Get ranks
  // CUDA_CHECK(cudaGetDevice(&device_id));
  rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS),
  num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

  // Get device info
  cudaDeviceProp device_prop = {};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

  if (num_nvl_bytes > 0) {
    // Local IPC: alloc local memory and set local IPC handle
    CUDA_CHECK(cudaMalloc(
        &buffer_ptrs[nvl_rank],
        num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes));
    CUDA_CHECK(
        cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
    buffer_ptrs_gpu = reinterpret_cast<void**>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
        fifo_bytes);

    // Set task fifo
    EP_HOST_ASSERT(NUM_MAX_FIFO_SLOTS % num_nvl_ranks == 0);
    task_fifo_ptrs[nvl_rank] = reinterpret_cast<int*>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    task_fifo_ptrs_gpu = reinterpret_cast<int**>(
        reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
        fifo_bytes + buffer_ptr_bytes);

    // No need to synchronize, will do a full device sync during `sync`
    CUDA_CHECK(cudaMemsetAsync(
        buffer_ptrs[nvl_rank],
        0,
        num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes,
        comm_stream.raw_stream()));
  }

  // Create 32 MiB workspace
  // Note(ZKK):  here we allocate more(2 * M2N_NUM_WORKSPACE) to support M2N!
  // Later we will optimize here!
  CUDA_CHECK(
      cudaMalloc(&workspace, 2 * M2N_NUM_WORKSPACE * NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(
      workspace, 0, 2 * M2N_NUM_WORKSPACE * NUM_WORKSPACE_BYTES, comm_stream));

  // MoE counter
  CUDA_CHECK(
      cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(
      &moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
  *moe_recv_counter = -1;

  // MoE expert-level counter
  CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter,
                            sizeof(int) * NUM_MAX_LOCAL_EXPERTS,
                            cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped,
                                      const_cast<int*>(moe_recv_expert_counter),
                                      0));
  for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
    moe_recv_expert_counter[i] = -1;

  // MoE RDMA-level counter
  if (num_rdma_ranks > 0) {
    CUDA_CHECK(cudaMallocHost(
        &moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped,
                                        const_cast<int*>(moe_recv_rdma_counter),
                                        0));
    *moe_recv_rdma_counter = -1;
  }
}

Buffer::~Buffer() noexcept(false) { CUDA_CHECK(cudaDeviceSynchronize()); }

bool Buffer::is_available() const { return available; }

bool Buffer::is_internode_available() const {
  return is_available() && num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const { return num_rdma_ranks; }

int Buffer::get_rdma_rank() const { return rdma_rank; }

int Buffer::get_root_rdma_rank(bool global) const {
  return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const { return device_id; }

cudaStream_t Buffer::get_comm_stream() const { return comm_stream; }

#ifndef PADDLE_NO_PYTHON
pybind11::bytearray Buffer::get_local_ipc_handle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

// pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
//   return {reinterpret_cast<const char*>(""), sizeof(BKCLUniqueId)};
// }

// void Buffer::sync(
//     const std::vector<int>& device_ids,
//     const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
//     const std::optional<pybind11::bytearray>& root_unique_id_opt) {
//   if (num_rdma_ranks > 1 || low_latency_mode) {
//     int ret = bkcl_xshmem_init(comm_ctx->GetBKCLComm());
//     EP_HOST_ASSERT(ret == 0 && "bkcl_xshmem_init failed");
//   }
// }
#endif

std::tuple<deep_ep::detail::Tensor,
           std::optional<deep_ep::detail::Tensor>,
           deep_ep::detail::Tensor,
           deep_ep::detail::Tensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout(const deep_ep::detail::Tensor& topk_idx,
                            int num_experts,
                            std::optional<EventHandle>& previous_event,
                            bool async,
                            bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(topk_idx.dim() == 2);
  EP_HOST_ASSERT(topk_idx.is_contiguous());
  EP_HOST_ASSERT(num_experts > 0);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() && async);
    deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  auto num_tokens = static_cast<int>(topk_idx.size(0)),
       num_topk = static_cast<int>(topk_idx.size(1));
  auto num_tokens_per_rank =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_ranks}, phi::DataType::INT32, phi::CustomPlace("metax_gpu", device_id)));
  auto num_tokens_per_rdma_rank = std::optional<deep_ep::detail::Tensor>();
  auto num_tokens_per_expert =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {num_experts}, phi::DataType::INT32, phi::CustomPlace("metax_gpu", device_id)));
  auto is_token_in_rank = ConvertPaddleTensorToDetailTensor(
      paddle::experimental::empty({num_tokens, num_ranks},
                                  phi::DataType::BOOL,
                                  phi::CustomPlace("metax_gpu", device_id)));
  if (is_internode_available()) {
    num_tokens_per_rdma_rank =
        ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
            {num_rdma_ranks}, phi::DataType::INT32, phi::CustomPlace("metax_gpu", device_id)));
  }

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) {
    deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
  }

  return {num_tokens_per_rank,
          num_tokens_per_rdma_rank,
          num_tokens_per_expert,
          is_token_in_rank,
          event};
}

// std::tuple<deep_ep::detail::Tensor,
//            std::optional<deep_ep::detail::Tensor>,
//            std::optional<deep_ep::detail::Tensor>,
//            std::optional<deep_ep::detail::Tensor>,
//            std::vector<int>,
//            deep_ep::detail::Tensor,
//            deep_ep::detail::Tensor,
//            deep_ep::detail::Tensor,
//            deep_ep::detail::Tensor,
//            deep_ep::detail::Tensor,
//            std::optional<EventHandle>>
// Buffer::intranode_dispatch(
//     const deep_ep::detail::Tensor& x,
//     const std::optional<deep_ep::detail::Tensor>& x_scales,
//     const std::optional<deep_ep::detail::Tensor>& topk_idx,
//     const std::optional<deep_ep::detail::Tensor>& topk_weights,
//     const std::optional<deep_ep::detail::Tensor>& num_tokens_per_rank,
//     const deep_ep::detail::Tensor& is_token_in_rank,
//     const std::optional<deep_ep::detail::Tensor>& num_tokens_per_expert,
//     int cached_num_recv_tokens,
//     const std::optional<deep_ep::detail::Tensor>& cached_rank_prefix_matrix,
//     const std::optional<deep_ep::detail::Tensor>& cached_channel_prefix_matrix,
//     int expert_alignment,
//     const Config& config,
//     std::optional<EventHandle>& previous_event,  // NOLINT
//     bool async,
//     bool allocate_on_comm_stream) {
//   if (topk_idx.has_value()) {
//     EP_HOST_ASSERT(topk_idx.has_value() && topk_weights.has_value() &&
//                    num_tokens_per_rank.has_value() &&
//                    num_tokens_per_expert.has_value());
//     last_topk_idx = ConvertPaddleTensorToDetailTensor(
//         cast_ad_func(topk_idx->raw_tensor(), phi::DataType::INT32));
//     last_topk_weights = ConvertPaddleTensorToDetailTensor(
//         assign_ad_func(topk_weights->raw_tensor()));
//     last_num_experts = static_cast<int>(num_tokens_per_expert->size(0));
//   } else {  // cache mode
//     EP_HOST_ASSERT(last_topk_idx.has_value() && last_topk_weights.has_value() &&
//                    last_num_experts != 0);
//   }

//   // Shape and contiguous checks
//   EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
//   auto num_tokens = static_cast<int>(x.size(0));
//   int hidden_size = static_cast<int>(x.size(1));
//   int num_topk = static_cast<int>(last_topk_idx->size(1));
//   auto num_local_experts = last_num_experts / num_ranks;
//   int ret = 0;

//   // For int8 dispatch, the corresponding combine would be bf16,
//   // so we must init buffer with bf16 here to avoid buffer overflow of combine.
//   if (!init_normal_buffer) {
//     ret = bkcl_init_normal_buffer(
//         comm_ctx->GetBKCLComm(), hidden_size, num_ranks, BKCL_BFLOAT16);
//     EP_HOST_ASSERT(ret == 0 && "bkcl_init_normal_buffer failed");
//     init_normal_buffer = true;
//   }

//   int num_scales = 0;
//   bool use_int8 = false;
//   if (x_scales.has_value()) {
//     num_scales = static_cast<int>(x_scales->size(1));
//     use_int8 = true;
//   }

//   // Allocate all tensors on comm stream if set
//   // NOTES: do not allocate tensors upfront!
//   auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

//   if (allocate_on_comm_stream) {
//     EP_HOST_ASSERT(previous_event.has_value() && async);
//     deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
//   }

//   // Wait previous tasks to be finished
//   if (previous_event.has_value()) {
//     stream_wait(comm_stream, previous_event.value());
//   } else {
//     stream_wait(comm_stream, compute_stream);
//   }

//   auto d_num_recv_tokens_per_expert_list =
//       ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//           {num_local_experts}, phi::DataType::INT32, x.place()));
//   auto h_num_recv_tokens_per_expert_list =
//       std::vector<int>(num_local_experts, 0);
//   auto rank_prefix_matrix =
//       ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//           {num_ranks, num_ranks}, phi::DataType::INT32, x.place()));
//   auto channel_prefix_matrix =
//       ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//           {num_ranks, 12}, phi::DataType::INT32, x.place()));
//   auto recv_channel_prefix_matrix =
//       ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//           {num_ranks, 12}, phi::DataType::INT32, x.place()));
//   auto recv_src_idx = ConvertPaddleTensorToDetailTensor(
//       paddle::experimental::empty({10}, phi::DataType::INT32, x.place()));
//   auto send_head =
//       ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//           {num_tokens, num_ranks}, phi::DataType::INT32, x.place()));

//   int num_recv_tokens =
//       bkcl_notify_dispatch_standard_with_num_recv_tokens_per_expert_list_cpu(
//           comm_ctx->GetBKCLComm(),
//           x.data_ptr(),
//           last_topk_idx->data_ptr<int>(),
//           last_topk_weights->data_ptr<float>(),
//           num_scales,
//           hidden_size,
//           num_tokens,
//           num_topk,
//           last_num_experts,
//           d_num_recv_tokens_per_expert_list
//               .data_ptr<int>(),  // should not be nullptr
//           h_num_recv_tokens_per_expert_list.data(),
//           ToBKCLDataType(x.dtype()),
//           use_int8,
//           async ? reinterpret_cast<cudaStream_t>(comm_stream)
//                 : reinterpret_cast<cudaStream_t>(compute_stream));
//   // num_tokens maybe 0, and num_recv_tokens also can be 0.
//   EP_HOST_ASSERT(num_recv_tokens >= 0 &&
//                  "bkcl_notify_dispatch_standard failed");

//   auto recv_x = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//       {num_recv_tokens, hidden_size}, x.dtype(), x.place()));
//   std::optional<deep_ep::detail::Tensor> recv_topk_idx =
//       ConvertPaddleTensorToDetailTensor(
//           paddle::experimental::empty({num_recv_tokens, num_topk},
//                                       last_topk_idx->dtype(),
//                                       last_topk_idx->place()));
//   std::optional<deep_ep::detail::Tensor> recv_topk_weights =
//       ConvertPaddleTensorToDetailTensor(
//           paddle::experimental::empty({num_recv_tokens, num_topk},
//                                       last_topk_weights->dtype(),
//                                       last_topk_weights->place()));

//   auto recv_x_scales = std::optional<deep_ep::detail::Tensor>();
//   float* x_scales_ptr = nullptr;
//   float* recv_x_scales_ptr = nullptr;

//   if (x_scales.has_value()) {
//     x_scales_ptr = const_cast<float*>(x_scales->data_ptr<float>());
//     recv_x_scales = ConvertPaddleTensorToDetailTensor(
//         paddle::experimental::empty({num_recv_tokens, num_scales},
//                                     x_scales->dtype(),
//                                     x_scales->place()));
//     recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
//   }

//   VLOG(3) << "DeepEP intranode_dispatch num_local_experts " << num_local_experts
//           << " num_scales " << num_scales << " hidden_size " << hidden_size
//           << " num_tokens " << num_tokens << " last_num_experts "
//           << last_num_experts << " num_recv_tokens " << num_recv_tokens;
//   VLOG(3) << "DeepEP intranode_dispatch x dim " << x.dim()
//           << " last_topk_idx dim " << last_topk_idx->dim()
//           << " last_topk_weights dim " << last_topk_weights->dim();

//   ret = bkcl_normal_dispatch_standard(
//       comm_ctx->GetBKCLComm(),
//       x.data_ptr(),  // sendbuf
//       x_scales_ptr,
//       last_topk_idx->data_ptr<int>(),
//       last_topk_weights->data_ptr<float>(),
//       recv_x.data_ptr(),
//       recv_x_scales_ptr,
//       recv_topk_idx->data_ptr<int>(),
//       recv_topk_weights->data_ptr<float>(),
//       num_scales,
//       -1,  // UNUSED
//       hidden_size,
//       num_tokens,
//       num_topk,
//       last_num_experts,
//       ToBKCLDataType(x.dtype()),
//       use_int8,
//       async ? reinterpret_cast<cudaStream_t>(comm_stream)
//             : reinterpret_cast<cudaStream_t>(compute_stream));
//   EP_HOST_ASSERT(ret == 0 && "bkcl_normal_dispatch_standard failed");

//   // Wait streams
//   std::optional<EventHandle> event;
//   if (async) {
//     event = EventHandle(comm_stream);
//     for (auto& t : {x, recv_x}) {
//       t.record_stream(comm_stream);
//       if (allocate_on_comm_stream) t.record_stream(compute_stream);
//     }
//     for (auto& to : {x_scales,
//                      topk_idx,
//                      topk_weights,
//                      recv_topk_idx,
//                      recv_topk_weights,
//                      recv_x_scales}) {
//       to.has_value() ? to->record_stream(comm_stream) : void();
//       if (allocate_on_comm_stream)
//         to.has_value() ? to->record_stream(compute_stream) : void();
//     }
//   } else {
//     stream_wait(compute_stream, comm_stream);
//   }

//   // Switch back compute stream
//   if (allocate_on_comm_stream) {
//     deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
//   }

//   // Return values
//   return {recv_x,
//           recv_x_scales,
//           recv_topk_idx,
//           recv_topk_weights,
//           h_num_recv_tokens_per_expert_list,
//           rank_prefix_matrix,
//           channel_prefix_matrix,
//           recv_channel_prefix_matrix,
//           recv_src_idx,
//           send_head,
//           event};
// }

// std::tuple<deep_ep::detail::Tensor,
//            std::optional<deep_ep::detail::Tensor>,
//            std::optional<EventHandle>>
// Buffer::intranode_combine(
//     const deep_ep::detail::Tensor& x,
//     const std::optional<deep_ep::detail::Tensor>& topk_weights,
//     const deep_ep::detail::Tensor& src_idx,
//     const deep_ep::detail::Tensor& rank_prefix_matrix,
//     const deep_ep::detail::Tensor& channel_prefix_matrix,
//     const deep_ep::detail::Tensor& send_head,
//     const Config& config,
//     std::optional<EventHandle>& previous_event,
//     bool async,
//     bool allocate_on_comm_stream) {
//   EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
//   auto num_tokens = static_cast<int>(x.size(0)),
//        hidden_size = static_cast<int>(x.size(1));
//   auto num_combined_tokens = static_cast<int>(send_head.size(0));

//   int ret = BKCL_SUCCESS;
//   if (!init_normal_buffer) {
//     ret = bkcl_init_normal_buffer(
//         comm_ctx->GetBKCLComm(), hidden_size, num_ranks, BKCL_BFLOAT16);
//     EP_HOST_ASSERT(ret == 0 && "bkcl_init_normal_buffer failed");
//     init_normal_buffer = true;
//   }

//   // Allocate all tensors on comm stream if set
//   // NOTES: do not allocate tensors upfront!
//   auto compute_stream = reinterpret_cast<cudaStream_t>(calc_ctx->stream());

//   if (allocate_on_comm_stream) {
//     EP_HOST_ASSERT(previous_event.has_value() && async);
//     deep_ep::detail::SetAllocatorStreamForGPUContext(comm_stream, calc_ctx);
//   }

//   // Wait previous tasks to be finished
//   if (previous_event.has_value()) {
//     stream_wait(comm_stream, previous_event.value());
//   } else {
//     stream_wait(comm_stream, compute_stream);
//   }

//   // Top-k checks
//   int num_topk = 0;
//   auto combined_topk_weights = std::optional<deep_ep::detail::Tensor>();
//   float* topk_weights_ptr = nullptr;
//   float* combined_topk_weights_ptr = nullptr;
//   if (topk_weights.has_value()) {
//     EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
//     EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
//     EP_HOST_ASSERT(topk_weights->scalar_type() == deep_ep::detail::kFloat32);
//     num_topk = static_cast<int>(topk_weights->size(1));
//     topk_weights_ptr = topk_weights->data_ptr<float>();
//     combined_topk_weights = ConvertPaddleTensorToDetailTensor(
//         paddle::experimental::empty({num_combined_tokens, num_topk},
//                                     topk_weights->dtype(),
//                                     topk_weights->place()));
//     combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
//   }

//   // Combine data
//   auto recv_x = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
//       {num_combined_tokens, hidden_size}, x.dtype(), x.place()));

//   VLOG(3) << "DeepEP intranode_combine x.dim " << x.dim() << " num_tokens "
//           << num_tokens << " num_combined_tokens " << num_combined_tokens
//           << " num_topk " << num_topk << " topk_weights_ptr "
//           << topk_weights_ptr << " combined_topk_weights_ptr "
//           << combined_topk_weights_ptr;

//   ret = bkcl_normal_combine_standard(
//       comm_ctx->GetBKCLComm(),
//       x.data_ptr(),
//       topk_weights_ptr,
//       recv_x.data_ptr(),
//       combined_topk_weights_ptr,
//       hidden_size,
//       num_tokens,
//       num_combined_tokens,
//       num_topk,
//       0 /*num_experts*/,
//       ToBKCLDataType(x.scalar_type()),
//       async ? reinterpret_cast<cudaStream_t>(comm_stream)
//             : reinterpret_cast<cudaStream_t>(compute_stream));

//   // Wait streams
//   std::optional<EventHandle> event;
//   if (async) {
//     event = EventHandle(comm_stream);
//     for (auto& to : {x, recv_x}) {
//       to.record_stream(comm_stream);
//       if (allocate_on_comm_stream) to.record_stream(compute_stream);
//     }

//     for (auto& to : {topk_weights, combined_topk_weights}) {
//       to.has_value() ? to->record_stream(comm_stream) : void();
//       if (allocate_on_comm_stream)
//         to.has_value() ? to->record_stream(compute_stream) : void();
//     }
//   } else {
//     stream_wait(compute_stream, comm_stream);
//   }

//   // Switch back compute stream
//   if (allocate_on_comm_stream) {
//     deep_ep::detail::SetAllocatorStreamForGPUContext(compute_stream, calc_ctx);
//   }

//   return {recv_x, combined_topk_weights, event};
// }


void Buffer::barrier_all() {}

std::tuple<paddle::Tensor,
           std::optional<paddle::Tensor>,
           paddle::Tensor,
           paddle::Tensor,
           std::optional<EventHandle>>
Buffer::get_dispatch_layout_api(const paddle::Tensor& topk_idx,
                                int num_experts,
                                std::optional<EventHandle>& previous_event,
                                bool async,
                                bool allocate_on_comm_stream) {
  const auto& topk_idx_ = ConvertPaddleTensorToDetailTensor(topk_idx);
  auto res = get_dispatch_layout(
      topk_idx_, num_experts, previous_event, async, allocate_on_comm_stream);
  const auto& num_tokens_per_rank = std::get<0>(res);
  const auto& num_tokens_per_rdma_rank = std::get<1>(res);
  const auto& num_tokens_per_expert = std::get<2>(res);
  const auto& is_token_in_rank = std::get<3>(res);
  const auto& event = std::get<4>(res);
  auto num_tokens_per_rank_ =
      ConvertDetailTensorToPaddleTensor(num_tokens_per_rank);
  std::optional<paddle::Tensor> num_tokens_per_rdma_rank_ = std::nullopt;
  if (num_tokens_per_rdma_rank.has_value()) {
    num_tokens_per_rdma_rank_ =
        ConvertDetailTensorToPaddleTensor(num_tokens_per_rdma_rank.value());
  }
  auto num_tokens_per_expert_ =
      ConvertDetailTensorToPaddleTensor(num_tokens_per_expert);
  auto is_token_in_rank_ = ConvertDetailTensorToPaddleTensor(is_token_in_rank);
  return {num_tokens_per_rank_,
          num_tokens_per_rdma_rank_,
          num_tokens_per_expert_,
          is_token_in_rank_,
          event};
}

// std::tuple<paddle::Tensor,
//            std::optional<paddle::Tensor>,
//            std::optional<paddle::Tensor>,
//            std::optional<paddle::Tensor>,
//            std::vector<int>,
//            paddle::Tensor,
//            paddle::Tensor,
//            paddle::Tensor,
//            paddle::Tensor,
//            paddle::Tensor,
//            std::optional<EventHandle>>
// Buffer::intranode_dispatch_api(
//     const paddle::Tensor& x,
//     const std::optional<paddle::Tensor>& x_scales,
//     const std::optional<paddle::Tensor>& topk_idx,
//     const std::optional<paddle::Tensor>& topk_weights,
//     const std::optional<paddle::Tensor>& num_tokens_per_rank,
//     const paddle::Tensor& is_token_in_rank,
//     const std::optional<paddle::Tensor>& num_tokens_per_expert,
//     int cached_num_recv_tokens,
//     const std::optional<paddle::Tensor>& cached_rank_prefix_matrix,
//     const std::optional<paddle::Tensor>& cached_channel_prefix_matrix,
//     int expert_alignment,
//     const Config& config,
//     std::optional<EventHandle>& previous_event,  // NOLINT
//     bool async,
//     bool allocate_on_comm_stream) {
//   const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
//   std::optional<deep_ep::detail::Tensor> x_scales_;
//   if (x_scales.has_value()) {
//     x_scales_ = ConvertPaddleTensorToDetailTensor(x_scales.value());
//   }
//   std::optional<deep_ep::detail::Tensor> topk_idx_;
//   if (topk_idx.has_value()) {
//     topk_idx_ = ConvertPaddleTensorToDetailTensor(topk_idx.value());
//   }
//   std::optional<deep_ep::detail::Tensor> topk_weights_;
//   if (topk_weights.has_value()) {
//     topk_weights_ = ConvertPaddleTensorToDetailTensor(topk_weights.value());
//   }
//   std::optional<deep_ep::detail::Tensor> num_tokens_per_rank_;
//   if (num_tokens_per_rank.has_value()) {
//     num_tokens_per_rank_ =
//         ConvertPaddleTensorToDetailTensor(num_tokens_per_rank.value());
//   }
//   const auto& is_token_in_rank_ =
//       ConvertPaddleTensorToDetailTensor(is_token_in_rank);
//   std::optional<deep_ep::detail::Tensor> num_tokens_per_expert_;
//   if (num_tokens_per_expert.has_value()) {
//     num_tokens_per_expert_ =
//         ConvertPaddleTensorToDetailTensor(num_tokens_per_expert.value());
//   }
//   std::optional<deep_ep::detail::Tensor> cached_rank_prefix_matrix_;
//   if (cached_rank_prefix_matrix.has_value()) {
//     cached_rank_prefix_matrix_ =
//         ConvertPaddleTensorToDetailTensor(cached_rank_prefix_matrix.value());
//   }
//   std::optional<deep_ep::detail::Tensor> cached_channel_prefix_matrix_;
//   if (cached_channel_prefix_matrix.has_value()) {
//     cached_channel_prefix_matrix_ =
//         ConvertPaddleTensorToDetailTensor(cached_channel_prefix_matrix.value());
//   }

//   auto res = intranode_dispatch(x_,
//                                 x_scales_,
//                                 topk_idx_,
//                                 topk_weights_,
//                                 num_tokens_per_rank_,
//                                 is_token_in_rank_,
//                                 num_tokens_per_expert_,
//                                 cached_num_recv_tokens,
//                                 cached_rank_prefix_matrix_,
//                                 cached_channel_prefix_matrix_,
//                                 expert_alignment,
//                                 config,
//                                 previous_event,
//                                 async,
//                                 allocate_on_comm_stream);

//   const auto& recv_x = std::get<0>(res);
//   const auto& recv_x_scales = std::get<1>(res);
//   const auto& recv_topk_idx = std::get<2>(res);
//   const auto& recv_topk_weights = std::get<3>(res);
//   const auto& num_recv_tokens_per_expert_list = std::get<4>(res);
//   const auto& rank_prefix_matrix = std::get<5>(res);
//   const auto& channel_prefix_matrix = std::get<6>(res);
//   const auto& recv_channel_prefix_matrix = std::get<7>(res);
//   const auto& recv_src_idx = std::get<8>(res);
//   const auto& send_head = std::get<9>(res);
//   const auto& event = std::get<10>(res);

//   auto recv_x_ = ConvertDetailTensorToPaddleTensor(recv_x);
//   std::optional<paddle::Tensor> recv_x_scales_;
//   if (recv_x_scales.has_value()) {
//     recv_x_scales_ = ConvertDetailTensorToPaddleTensor(recv_x_scales.value());
//   }
//   std::optional<paddle::Tensor> recv_topk_idx_;
//   if (recv_topk_idx.has_value()) {
//     recv_topk_idx_ = ConvertDetailTensorToPaddleTensor(recv_topk_idx.value());
//   }
//   std::optional<paddle::Tensor> recv_topk_weights_;
//   if (recv_topk_weights.has_value()) {
//     recv_topk_weights_ =
//         ConvertDetailTensorToPaddleTensor(recv_topk_weights.value());
//   }
//   auto rank_prefix_matrix_ =
//       ConvertDetailTensorToPaddleTensor(rank_prefix_matrix);
//   auto channel_prefix_matrix_ =
//       ConvertDetailTensorToPaddleTensor(channel_prefix_matrix);
//   auto recv_channel_prefix_matrix_ =
//       ConvertDetailTensorToPaddleTensor(recv_channel_prefix_matrix);
//   auto recv_src_idx_ = ConvertDetailTensorToPaddleTensor(recv_src_idx);
//   auto send_head_ = ConvertDetailTensorToPaddleTensor(send_head);

//   return {recv_x_,
//           recv_x_scales_,
//           recv_topk_idx_,
//           recv_topk_weights_,
//           num_recv_tokens_per_expert_list,
//           rank_prefix_matrix_,
//           channel_prefix_matrix_,
//           recv_channel_prefix_matrix_,
//           recv_src_idx_,
//           send_head_,
//           event};
// }

// std::tuple<paddle::Tensor,
//            std::optional<paddle::Tensor>,
//            std::optional<EventHandle>>
// Buffer::intranode_combine_api(const paddle::Tensor& x,
//                               const std::optional<paddle::Tensor>& topk_weights,
//                               const paddle::Tensor& src_idx,
//                               const paddle::Tensor& rank_prefix_matrix,
//                               const paddle::Tensor& channel_prefix_matrix,
//                               const paddle::Tensor& send_head,
//                               const Config& config,
//                               std::optional<EventHandle>& previous_event,
//                               bool async,
//                               bool allocate_on_comm_stream) {
//   const auto& x_ = ConvertPaddleTensorToDetailTensor(x);
//   std::optional<deep_ep::detail::Tensor> topk_weights_;
//   if (topk_weights.has_value()) {
//     topk_weights_ = ConvertPaddleTensorToDetailTensor(topk_weights.value());
//   }
//   const auto& src_idx_ = ConvertPaddleTensorToDetailTensor(src_idx);
//   const auto& rank_prefix_matrix_ =
//       ConvertPaddleTensorToDetailTensor(rank_prefix_matrix);
//   const auto& channel_prefix_matrix_ =
//       ConvertPaddleTensorToDetailTensor(channel_prefix_matrix);
//   const auto& send_head_ = ConvertPaddleTensorToDetailTensor(send_head);

//   auto res = intranode_combine(x_,
//                                topk_weights_,
//                                src_idx_,
//                                rank_prefix_matrix_,
//                                channel_prefix_matrix_,
//                                send_head_,
//                                config,
//                                previous_event,
//                                async,
//                                allocate_on_comm_stream);

//   const auto& recv_x = std::get<0>(res);
//   const auto& recv_topk_weights = std::get<1>(res);
//   const auto& event = std::get<2>(res);

//   auto recv_x_ = ConvertDetailTensorToPaddleTensor(recv_x);
//   std::optional<paddle::Tensor> recv_topk_weights_;
//   if (recv_topk_weights.has_value()) {
//     recv_topk_weights_ =
//         ConvertDetailTensorToPaddleTensor(recv_topk_weights.value());
//   }
//   auto event_ = event;
//   return {recv_x_, recv_topk_weights_, event_};
// }

deep_ep::detail::Tensor ConvertPaddleTensorToDetailTensor(
    const paddle::Tensor& tensor) {
  deep_ep::detail::Tensor res(tensor);
  return res;
}

paddle::Tensor ConvertDetailTensorToPaddleTensor(
    const deep_ep::detail::Tensor& tensor) {
  return tensor.raw_tensor();
}

std::optional<deep_ep::detail::Tensor>
ConvertOptionalPaddleTensorToDetailTensor(
    const std::optional<paddle::Tensor>& tensor) {
  std::optional<deep_ep::detail::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertPaddleTensorToDetailTensor(tensor.value());
  }
  return res;
}

std::optional<paddle::Tensor> ConvertOptionalDetailTensorToPaddleTensor(
    const std::optional<deep_ep::detail::Tensor>& tensor) {
  std::optional<paddle::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertDetailTensorToPaddleTensor(tensor.value());
  }
  return res;
}

}  // namespace deep_ep
