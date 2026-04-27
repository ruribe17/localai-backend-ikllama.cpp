# 🔧 NUEVO: Versión y repo de ik_llama.cpp
IK_LLAMA_VERSION?=542988773c00f4862c7fdd75e36fffc365d8af86
#IK_LLAMA_VERSION?=afa6439
IK_LLAMA_REPO?=https://github.com/ikawrakow/ik_llama.cpp.git

CMAKE_ARGS?=
BUILD_TYPE?=
NATIVE?=false
ONEAPI_VARS?=/opt/intel/oneapi/setvars.sh
TARGET?=--target grpc-server
JOBS?=$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
ARCH?=$(shell uname -m)

CMAKE_ARGS+=-DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF

CURRENT_MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ifeq ($(NATIVE),false)
    CMAKE_ARGS+=-DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF
endif

ifeq ($(BUILD_TYPE),cublas)
    CMAKE_ARGS+=-DGGML_CUDA=ON
else ifeq ($(BUILD_TYPE),openblas)
    CMAKE_ARGS+=-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
else ifeq ($(BUILD_TYPE),clblas)
    CMAKE_ARGS+=-DGGML_CLBLAST=ON -DCLBlast_DIR=/some/path
else ifeq ($(BUILD_TYPE),hipblas)
    ROCM_HOME ?= /opt/rocm
    ROCM_PATH ?= /opt/rocm
    export CXX=$(ROCM_HOME)/llvm/bin/clang++
    export CC=$(ROCM_HOME)/llvm/bin/clang
    AMDGPU_TARGETS?=gfx803,gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1030,gfx1032,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201
    CMAKE_ARGS+=-DGGML_HIP=ON -DAMDGPU_TARGETS=$(AMDGPU_TARGETS)
else ifeq ($(BUILD_TYPE),vulkan)
    CMAKE_ARGS+=-DGGML_VULKAN=1
else ifeq ($(OS),Darwin)
    ifeq ($(BUILD_TYPE),)
        BUILD_TYPE=metal
    endif
    ifneq ($(BUILD_TYPE),metal)
        CMAKE_ARGS+=-DGGML_METAL=OFF
    else
        CMAKE_ARGS+=-DGGML_METAL=ON
        CMAKE_ARGS+=-DGGML_METAL_EMBED_LIBRARY=ON
        CMAKE_ARGS+=-DGGML_METAL_USE_BF16=ON
        CMAKE_ARGS+=-DGGML_OPENMP=OFF
    endif
    TARGET+=--target ggml-metal
endif

ifeq ($(BUILD_TYPE),sycl_f16)
    CMAKE_ARGS+=-DGGML_SYCL=ON \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-fsycl" \
        -DGGML_SYCL_F16=ON
endif

ifeq ($(BUILD_TYPE),sycl_f32)
    CMAKE_ARGS+=-DGGML_SYCL=ON \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-fsycl"
endif

INSTALLED_PACKAGES=$(CURDIR)/../grpc/installed_packages
INSTALLED_LIB_CMAKE=$(INSTALLED_PACKAGES)/lib/cmake
ADDED_CMAKE_ARGS=-Dabsl_DIR=${INSTALLED_LIB_CMAKE}/absl \
                 -DProtobuf_DIR=${INSTALLED_LIB_CMAKE}/protobuf \
                 -Dutf8_range_DIR=${INSTALLED_LIB_CMAKE}/utf8_range \
                 -DgRPC_DIR=${INSTALLED_LIB_CMAKE}/grpc \
                 -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=${INSTALLED_PACKAGES}/include

build-llama-cpp-grpc-server:
ifdef BUILD_GRPC_FOR_BACKEND_LLAMA
	$(MAKE) -C ../../grpc build
	_PROTOBUF_PROTOC=${INSTALLED_PACKAGES}/bin/proto \
	_GRPC_CPP_PLUGIN_EXECUTABLE=${INSTALLED_PACKAGES}/bin/grpc_cpp_plugin \
	PATH="${INSTALLED_PACKAGES}/bin:${PATH}" \
	CMAKE_ARGS="${CMAKE_ARGS} ${ADDED_CMAKE_ARGS}" \
	IK_LLAMA_VERSION=$(IK_LLAMA_VERSION) \
	$(MAKE) -C $(CURRENT_MAKEFILE_DIR) grpc-server
else
	echo "BUILD_GRPC_FOR_BACKEND_LLAMA is not defined."
	IK_LLAMA_VERSION=$(IK_LLAMA_VERSION) $(MAKE) -C $(CURRENT_MAKEFILE_DIR) grpc-server
endif

# ── AVX2 optimizado para Broadwell-EP (E5-2690v4 / E5-2697Av4 dual socket) ──
#
# DISEÑO: en lugar de crear un build-directory separado (que requeriría
# copiar el Makefile completo), este target:
#   1. Asegura que prepare.sh fue ejecutado (grpc-server ya tiene sus archivos)
#   2. Limpia solo el build/ de cmake (no los fuentes)
#   3. Recompila con CMAKE_ARGS específicos para Broadwell + BLAS + C++20
#   4. Copia el binario resultante como llama-cpp-avx2
#
# Flags Broadwell-EP:
#   -march=broadwell -mtune=broadwell  instrucciones específicas del Xeon
#   -DGGML_BMI2=ON                     Broadwell soporta BMI2
#   -DGGML_BLAS=ON + OpenBLAS          aceleración BLAS multi-thread
#   -DBLAS_LIBRARIES=libopenblaso.so.0 variante OpenMP de OpenBLAS
#   -DCMAKE_CXX_STANDARD=20            C++20 requerido por grpc-server.cpp
#   -DGGML_LTO=ON                      Link Time Optimization
llama-cpp-avx2: ik_llama.cpp ik_llama.cpp/examples/grpc-server
	$(info ${GREEN}I ik_llama-cpp build info:avx2 broadwell-optimized${RESET})
	rm -rf ik_llama.cpp/build
	CMAKE_ARGS="$(CMAKE_ARGS) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_STANDARD=17 \
	    -DCMAKE_C_FLAGS='-O3 -march=broadwell -mtune=broadwell -funroll-loops' \
	    -DCMAKE_CXX_FLAGS='-O3 -march=broadwell -mtune=broadwell -funroll-loops' \
	    -DGGML_AVX=ON \
	    -DGGML_AVX2=ON \
	    -DGGML_AVX512=OFF \
	    -DGGML_FMA=ON \
	    -DGGML_F16C=ON \
	    -DGGML_BMI2=ON \
	    -DGGML_OPENMP=ON \
	    -DGGML_LTO=ON \
	    -DGGML_BLAS=ON \
	    -DGGML_BLAS_VENDOR=OpenBLAS \
	    -DBLAS_LIBRARIES=/usr/lib64/libopenblaso.so.0 \
	    -DBLAS_INCLUDE_DIRS=/usr/include/openblas" \
	$(MAKE) build-llama-cpp-grpc-server
	cp -fv grpc-server llama-cpp-avx2

# Dependencia: asegura que prepare.sh se ejecutó y grpc-server tiene sus archivos
ik_llama.cpp/examples/grpc-server: ik_llama.cpp
	mkdir -p ik_llama.cpp/examples/grpc-server
	bash prepare.sh

llama-cpp-avx512: ik_llama.cpp ik_llama.cpp/examples/grpc-server
	$(info ${GREEN}I ik_llama-cpp build info:avx512${RESET})
	rm -rf ik_llama.cpp/build
	CMAKE_ARGS="$(CMAKE_ARGS) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_STANDARD=20 \
	    -DGGML_AVX=ON \
	    -DGGML_AVX2=OFF \
	    -DGGML_AVX512=ON \
	    -DGGML_FMA=ON \
	    -DGGML_F16C=ON \
	    -DGGML_OPENMP=ON" \
	$(MAKE) build-llama-cpp-grpc-server
	cp -fv grpc-server llama-cpp-avx512

llama-cpp-avx: ik_llama.cpp ik_llama.cpp/examples/grpc-server
	$(info ${GREEN}I ik_llama-cpp build info:avx${RESET})
	rm -rf ik_llama.cpp/build
	CMAKE_ARGS="$(CMAKE_ARGS) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_STANDARD=20 \
	    -DGGML_AVX=ON \
	    -DGGML_AVX2=OFF \
	    -DGGML_AVX512=OFF \
	    -DGGML_FMA=OFF \
	    -DGGML_F16C=OFF \
	    -DGGML_BMI2=OFF \
	    -DGGML_OPENMP=ON" \
	$(MAKE) build-llama-cpp-grpc-server
	cp -fv grpc-server llama-cpp-avx

llama-cpp-fallback: ik_llama.cpp ik_llama.cpp/examples/grpc-server
	$(info ${GREEN}I ik_llama-cpp build info:fallback${RESET})
	rm -rf ik_llama.cpp/build
	CMAKE_ARGS="$(CMAKE_ARGS) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_STANDARD=20 \
	    -DGGML_AVX=OFF \
	    -DGGML_AVX2=OFF \
	    -DGGML_AVX512=OFF \
	    -DGGML_FMA=OFF \
	    -DGGML_F16C=OFF \
	    -DGGML_BMI2=OFF \
	    -DGGML_OPENMP=ON" \
	$(MAKE) build-llama-cpp-grpc-server
	cp -fv grpc-server llama-cpp-fallback

llama-cpp-grpc: ik_llama.cpp ik_llama.cpp/examples/grpc-server
	$(info ${GREEN}I ik_llama-cpp build info:grpc${RESET})
	rm -rf ik_llama.cpp/build
	CMAKE_ARGS="$(CMAKE_ARGS) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_STANDARD=20 \
	    -DGGML_RPC=ON \
	    -DGGML_AVX=OFF \
	    -DGGML_AVX2=OFF \
	    -DGGML_AVX512=OFF \
	    -DGGML_FMA=OFF \
	    -DGGML_F16C=OFF \
	    -DGGML_BMI2=OFF" \
	TARGET="--target grpc-server --target rpc-server" \
	$(MAKE) build-llama-cpp-grpc-server
	cp -fv grpc-server llama-cpp-grpc

llama-cpp-rpc-server: llama-cpp-grpc
	cp -fv ik_llama.cpp/build/bin/rpc-server llama-cpp-rpc-server

# 🔧 CLONACIÓN CON IK_LLAMA_REPO Y IK_LLAMA_VERSION
ik_llama.cpp:
	mkdir -p ik_llama.cpp
	cd ik_llama.cpp && \
	git init && \
	git remote add origin $(IK_LLAMA_REPO) && \
	git fetch origin && \
	git checkout -b build $(IK_LLAMA_VERSION) && \
	git submodule update --init --recursive --depth 1 --single-branch

rebuild:
	bash prepare.sh
	rm -rf grpc-server ik_llama.cpp/build
	$(MAKE) grpc-server

package:
	bash package.sh

purge:
	rm -rf ik_llama.cpp/build
	rm -rf ik_llama.cpp/examples/grpc-server
	rm -rf grpc-server
	rm -rf llama-cpp-avx2-build

clean: purge
	rm -rf ik_llama.cpp
	rm -f llama-cpp-avx2 llama-cpp-avx llama-cpp-avx512 llama-cpp-fallback llama-cpp-grpc

grpc-server: ik_llama.cpp ik_llama.cpp/examples/grpc-server
	@echo "Building grpc-server with $(BUILD_TYPE) build type and $(CMAKE_ARGS)"
ifneq (,$(findstring sycl,$(BUILD_TYPE)))
	+bash -c "source $(ONEAPI_VARS); \
	cd ik_llama.cpp && mkdir -p build && cd build && cmake .. $(CMAKE_ARGS) && cmake --build . --config Release -j $(JOBS) $(TARGET)"
else
	+cd ik_llama.cpp && mkdir -p build && cd build && cmake .. $(CMAKE_ARGS) && cmake --build . --config Release -j $(JOBS) $(TARGET)
endif
	cp ik_llama.cpp/build/bin/grpc-server .
