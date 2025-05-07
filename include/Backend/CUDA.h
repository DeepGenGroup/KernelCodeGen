#pragma once
#include "IR/IR.h"

#ifndef LOG_DEBUG
#define LOG_DEBUG(msgstr) \
llvm::outs() << msgstr << "\n";llvm::outs().flush();
#endif

namespace KernelCodeGen {

std::string CUDAGen(mlir::ModuleOp &module);

}