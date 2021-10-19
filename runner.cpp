#include "runner.h"
#include "model.h"
#include <iostream>
#include <torch/script.h>


Runner::Runner(int64_t input_channels, int64_t num_actions):
        network(input_channels, num_actions){}

void Runner::loadstatedict(const std::string& param_path) {
    // Load model parameters
    torch::autograd::GradMode::set_enabled(false);  
    torch::jit::script::Module new_params = torch::jit::load(param_path);
    std::cout << "model loaded!" << std::endl;
    
    // Load values by name
    torch::Tensor fc_1_kernel = new_params.attr("default_policy/fc_1/kernel").toTensor();
    torch::Tensor fc_1_bias = new_params.attr("default_policy/fc_1/bias").toTensor();
    torch::Tensor fc_value_1_kernal = new_params.attr("default_policy/fc_value_1/kernel").toTensor();
    torch::Tensor fc_value_1_bias = new_params.attr("default_policy/fc_value_1/bias").toTensor();
    torch::Tensor fc_2_kernel = new_params.attr("default_policy/fc_2/kernel").toTensor();
    torch::Tensor fc_2_bias = new_params.attr("default_policy/fc_2/bias").toTensor();
    torch::Tensor fc_value_2_kernal = new_params.attr("default_policy/fc_value_2/kernel").toTensor();
    torch::Tensor fc_value_2_bias = new_params.attr("default_policy/fc_value_2/bias").toTensor();
    torch::Tensor fc_out_kernel = new_params.attr("default_policy/fc_out/kernel").toTensor();
    torch::Tensor fc_out_bias = new_params.attr("default_policy/fc_out/bias").toTensor();
    torch::Tensor value_out_kernel = new_params.attr("default_policy/value_out/kernel").toTensor();
    torch::Tensor value_out_bias = new_params.attr("default_policy/value_out/bias").toTensor();

    // bias returns same dimension, whereas weight returns transposed dimensions
    // LibTorch (apparently) uses row-major storage, while Eigen by default uses column-major storage
    // Therefore, we have to transpose weight tensors before copying to the network
    network.fc_1->named_parameters()["weight"].permute({1, 0}).copy_(fc_1_kernel);
    network.fc_1->named_parameters()["bias"].copy_(fc_1_bias);
    network.fc_2->named_parameters()["weight"].permute({1, 0}).copy_(fc_2_kernel);
    network.fc_2->named_parameters()["bias"].copy_(fc_2_bias);
    network.value->named_parameters()["weight"].permute({1, 0}).copy_(fc_out_kernel);
    network.value->named_parameters()["bias"].copy_(fc_out_bias);
    network.output->named_parameters()["weight"].permute({1, 0}).copy_(value_out_kernel);
    network.output->named_parameters()["bias"].copy_(value_out_bias);
    
    std::cout << "Model parameters loaded!!" << std::endl; 

    // for (auto& val : params) {
    //     auto name = val.key();
        // auto* t = params.find(name);
        // if (t != nullptr) {
        //     t->copy_(val.value());
        // } else {
        //     if (t != nullptr) {
        //         t->copy_(val.value());
        //     }
        // }
}

void Runner::run(void) {
    // init state and action vectors
    torch::Tensor state_tensor = torch::rand({1, 4});  // random vector (1, 8)
    std::cout << "observation vector (random): " << state_tensor << std::endl;

    // always explotation in inference stage
    torch::Tensor action_tensor = network.act(state_tensor);
    int64_t index = action_tensor[0].item<int64_t>();
    std::cout << "action index: " << index << std::endl;
}
