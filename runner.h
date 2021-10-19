#pragma once

#include <torch/torch.h>

#include "model.h"


class Runner{

    private: NNModel network;

    public:
        Runner(int64_t input_channels, int64_t num_actions);
        void print_modules(const torch::jit::script::Module& imodule);
        void loadstatedict(const std::string& param_path);
        void run(void);

};
