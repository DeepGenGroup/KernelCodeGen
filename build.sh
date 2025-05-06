#!/bin/bash
cd /home/xushilong/KernelCodeGen/build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
 -DCMAKE_C_COMPILER=/usr/bin/gcc \
 -DMLIR_DIR=/home/xushilong/rocm-llvm-install/lib/cmake/mlir 
 
make -j8
