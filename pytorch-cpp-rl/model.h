//
// Created by Navneet Madhu Kumar on 2019-07-10.
//

#pragma once

#include "model.h"
#include <torch/torch.h>


struct NNModel : torch::nn::Module{
    NNModel(int64_t input_channels, int64_t num_actions)
            :
            linear1(torch::nn::Linear(input_channels, 8)),  // hidden feat = 8
            // linear2(torch::nn::Linear(32, 32)),
            output(torch::nn::Linear(8, num_actions)){}

    torch::Tensor forward(torch::Tensor input) {
        input = torch::relu(feat1(input));
        // Flatten the output
        input = input.view({input.size(0), -1});
        // input = torch::relu(linear2(input));
        input = output(input);
        return input;
    }

    torch::Tensor act(torch::Tensor state){
        torch::Tensor q_value = forward(state);
        torch::Tensor action = std::get<1>(q_value.max(1));
        return action;
    }

    torch::nn::Linear linear1, output;
};
