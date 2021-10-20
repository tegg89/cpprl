//
// Created by Navneet Madhu Kumar on 2019-07-10.
//

#pragma once

#include "model.h"
#include <torch/torch.h>


struct NNModel : torch::nn::Module {
    NNModel(int64_t input_channels, int64_t num_actions)
            : fc_1(register_module("fc_1", torch::nn::Linear(8, 8))),
              fc_value_1(register_module("fc_value_1", torch::nn::Linear(8, 8))),
              fc_2(register_module("fc_2", torch::nn::Linear(8, 8))),
              fc_value_2(register_module("fc_value_2", torch::nn::Linear(8, 8))),
              value_out(register_module("value_out", torch::nn::Linear(8, 2))),
              fc_out(register_module("fc_out", torch::nn::Linear(8, num_actions))){}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        torch::Tensor v, act, act_input, v_input;

        act_input = torch::relu(fc_1(input));
        act_input = torch::relu(fc_2(input));
        act = torch::relu(fc_out(act_input));

        v_input = torch::relu(fc_value_1(input));
        v_input = torch::relu(fc_value_1(input));        
        v = torch::relu(value_out(input));

        return {v, act};
    }

    torch::Tensor act(torch::Tensor state) {
        auto [q_value, actions] = forward(state);
        torch::Tensor action = std::get<1>(q_value.max(1));
        return action;
    }

    torch::nn::Linear fc_1, fc_2, fc_out, fc_value_1, fc_value_2, value_out;
};
