//
// Created by Navneet Madhu Kumar on 2019-07-12.
//
#pragma once

#include <torch/torch.h>

#include "model.h"
#include "/Users/teggsung/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"


class Trainer{

    private: NNModel network, target_network;
    private: ALEInterface ale;
    private: int64_t batch_size = 32;

    public:
        Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity);
        void load_environment(int64_t random_seed, std::string rom_path);
        torch::Tensor get_tensor_observation(std::vector<unsigned char> state);
        void run(int64_t random_seed, std::string rom_path, int64_t num_epochs);
};
