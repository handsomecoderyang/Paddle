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

#include <Python.h>
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_DEEP_EP_MACA
#include "paddle/fluid/distributed/collective/deep_ep_maca/deep_ep.hpp"
#endif

#include "paddle/fluid/pybind/deep_ep_api_maca.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle::pybind {

void BindDeepEPApiMACA(pybind11::module *m) {
#ifdef PADDLE_WITH_DEEP_EP_MACA
  // ============ Config (simplified, only NVL related) ============
  pybind11::class_<deep_ep::Config>(*m, "ConfigMACA", py::module_local())
      .def(pybind11::init<int, int, int>(),
           py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6,
           py::arg("num_max_nvl_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint",
           &deep_ep::Config::get_nvl_buffer_size_hint);

  // ============ EventHandle ============
  pybind11::class_<deep_ep::EventHandle>(*m, "EventHandleMACA", py::module_local())
      .def(pybind11::init<>())
      .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait)
      .def("calc_stream_wait", &deep_ep::EventHandle::CalcStreamWait)
      .def("comm_stream_wait", &deep_ep::EventHandle::CommStreamWait);

  m->def("maca_get_event_handle_from_calc_stream",
         &deep_ep::GetEventHandleFromCalcStream);
  m->def("maca_get_event_handle_from_comm_stream",
         &deep_ep::GetEventHandleFromCommStream);

  // ============ Buffer (intranode only) ============
  pybind11::class_<deep_ep::Buffer>(*m, "BufferMACA", py::module_local())
      // Constructor: only NVL buffer, no RDMA
      // Buffer(num_nvl_bytes, num_rdma_bytes=0, low_latency_mode=false, num_qps=0)
      .def(pybind11::init([](int group_id,
                             int num_nvl_ranks,
                             int64_t num_nvl_bytes,
                             int context_ring_id) {
             // Create buffer with only NVL support, no RDMA
             return std::make_unique<deep_ep::Buffer>(
                 group_id,
                 num_nvl_ranks,
                 num_nvl_bytes,
                 0,      // num_rdma_bytes = 0 (no internode)
                 false,  // low_latency_mode = false
                 context_ring_id);
           }),
           py::arg("group_id"),
           py::arg("num_nvl_ranks"),
           py::arg("num_nvl_bytes"),
           py::arg("context_ring_id"))

      // Basic info
      .def("is_available", &deep_ep::Buffer::is_available)
      .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
      .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
      .def("sync", &deep_ep::Buffer::sync)

      // Get comm stream
      .def("get_comm_stream",
           [](deep_ep::Buffer &self) {
             int device_id = self.get_local_device_id();
             auto comm_stream = self.get_comm_stream();
             auto s = phi::Stream(reinterpret_cast<phi::StreamId>(comm_stream));
             // For MACA CustomDevice, return a generic stream representation
             // You may need to adjust this based on your CustomDevice stream type
             return s;
           })

      // ============ Intranode APIs only ============

      // get_dispatch_layout: compute communication layout from topk_idx
      .def("get_dispatch_layout",
           [](deep_ep::Buffer &self,
              py::handle topk_idx,
              int num_experts,
              std::optional<deep_ep::EventHandle> &previous_event,
              bool async,
              bool allocate_on_comm_stream) {
             auto topk_idx_tensor = CastPyArg2Tensor(topk_idx.ptr(), 0);
             return self.get_dispatch_layout_api(topk_idx_tensor,
                                                 num_experts,
                                                 previous_event,
                                                 async,
                                                 allocate_on_comm_stream);
           },
           py::arg("topk_idx"),
           py::arg("num_experts"),
           py::arg("previous_event") = std::nullopt,
           py::arg("async_op") = false,
           py::arg("allocate_on_comm_stream") = true)

      // intranode_dispatch: All-to-All dispatch within node
      .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch_api,
           py::arg("x"),
           py::arg("x_scales") = std::nullopt,
           py::arg("topk_idx") = std::nullopt,
           py::arg("topk_weights") = std::nullopt,
           py::arg("num_tokens_per_rank") = std::nullopt,
           py::arg("is_token_in_rank"),
           py::arg("num_tokens_per_expert") = std::nullopt,
           py::arg("cached_num_recv_tokens") = -1,
           py::arg("cached_rank_prefix_matrix") = std::nullopt,
           py::arg("cached_channel_prefix_matrix") = std::nullopt,
           py::arg("expert_alignment") = 1,
           py::arg("config"),
           py::arg("previous_event") = std::nullopt,
           py::arg("async_op") = false,
           py::arg("allocate_on_comm_stream") = true)

      // intranode_combine: All-to-All combine within node
      .def("intranode_combine", &deep_ep::Buffer::intranode_combine_api,
           py::arg("x"),
           py::arg("topk_weights") = std::nullopt,
           py::arg("src_idx"),
           py::arg("rank_prefix_matrix"),
           py::arg("channel_prefix_matrix"),
           py::arg("send_head"),
           py::arg("config"),
           py::arg("previous_event") = std::nullopt,
           py::arg("async_op") = false,
           py::arg("allocate_on_comm_stream") = true)

      // barrier for synchronization
      .def("barrier_all", &deep_ep::Buffer::barrier_all);

#endif  // PADDLE_WITH_DEEP_EP_MACA
}

}  // namespace paddle::pybind
