# deep_ep_custom CMakeLists.txt
# set(DEEPEP_KERNEL_SRCS 
#     kernels/intranode.cpp   # MACA 实现
#     )

# cc_library(deepep_kernels_maca SRCS)

set(MACA_PATH "/opt/maca/" CACHE PATH "MACA SDK path")

cc_library(
  deep_ep_maca
  SRCS deep_ep.cpp
       src/event_pool.cc
       src/event.cc
       src/CUDAStream.cc
  DEPS phi common)

target_include_directories(deep_ep_maca PUBLIC
    ${MACA_PATH}/tools/cu-bridge/include
    ${MACA_PATH}/include
    ${MACA_PATH}/include/mcr
    ${MACA_PATH}/include/common
    # /usr/local/cuda-11.7/targets/x86_64-linux/include/
)

target_link_libraries(deep_ep_maca
    ${MACA_PATH}/lib/libruntime_cu.so
)

target_compile_options(deep_ep_maca PRIVATE -Wno-reorder -Wno-unused-variable)
