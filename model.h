//
// Created by Navneet Madhu Kumar on 2019-07-10.
//

#pragma once

#include "model.h"
#include <torch/torch.h>


struct NNModel : torch::nn::Module {
    NNModel(int64_t input_channels, int64_t num_actions)
            : fc_1(torch::nn::Linear(input_channels, 256)),
              fc_2(torch::nn::Linear(256, 256)),
              value(torch::nn::Linear(256, 2)),
              output(torch::nn::Linear(256, num_actions)){}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        torch::Tensor v, act;

        input = torch::relu(fc_1(input));
        input = torch::relu(fc_2(input));
        v = value(input);
        act = output(input);
        return {v, act};
    }

    torch::Tensor act(torch::Tensor state) {
        auto [q_value, actions] = forward(state);
        torch::Tensor action = std::get<1>(q_value.max(1));
        return action;
    }

    torch::nn::Linear fc_1, fc_2, value, output;
};
