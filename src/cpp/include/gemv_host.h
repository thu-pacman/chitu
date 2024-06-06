#pragma once
#include "common.h"

void gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out);