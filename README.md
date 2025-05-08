# 项目说明

## 1.简介

本项目为Deepgen的工具项目，主要功能为：将并行IR转换为对应的cuda代码，以便于进行IR变换过程、算法的debug。

## 2.项目构建 & 第三方依赖
项目依赖 rocm-LLVM (https://github.com/DeepGenGroup/rocm-llvm-project.git)  deepgen-dev 分支

在 `build.sh` 中，修改对应路径和编译器位置即可。

```shell
#!/bin/bash
cd $HOME/KernelCodeGen/build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
 -DCMAKE_C_COMPILER=/usr/bin/gcc \
 -DMLIR_DIR=$HOME/rocm-llvm-install/lib/cmake/mlir 
 
make -j8

```
