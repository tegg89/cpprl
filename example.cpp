#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#include "pytorch-cpp-rl/model.h"

int main() {
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // TODO: Load model parameters from checkpoint file
    // torch::jit::script::Module module;
    // module = torch::jit::load("./checkpoint_000100/checkpoint-100");
    // std::cout << "model loaded!" << std::endl;

    // init NN model
    network(input_channels, num_actions);
    network->to(device)
    network.eval()

    // init state and action vectors
    torch::Tensor state_tensor = torch::rand({1, 8});  // random vector (1, 8)
    Action a;
    std::cout << "observation vector (random): " << state_tensor << std::endl;

    // always explotation in inference stage
    torch::Tensor action_tensor = network.act(state_tensor);
    int64_t index = action_tensor[0].item<int64_t>();
    std::cout << "action index: " << index << std::endl;
}