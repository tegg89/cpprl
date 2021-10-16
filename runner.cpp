#include "runner.h"
#include "model.h"
#include <iostream>


Runner::Runner(int64_t input_channels, int64_t num_actions):
        network(input_channels, num_actions){}

    // Device
    // auto cuda_available = torch::cuda::is_available();
    // torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    // std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // network(input_channels, num_actions);
    // network->to(device)
    // network.eval()
    
void Runner::run(void) {
    // TODO: Load model parameters from checkpoint file
    // torch::jit::script::Module module;
    // module = torch::jit::load("./checkpoint_000100/checkpoint-100");
    // std::cout << "model loaded!" << std::endl;
    
    // init state and action vectors
    torch::Tensor state_tensor = torch::rand({1, 8});  // random vector (1, 8)
    std::cout << "observation vector (random): " << state_tensor << std::endl;

    // always explotation in inference stage
    torch::Tensor action_tensor = network.act(state_tensor);
    int64_t index = action_tensor[0].item<int64_t>();
    std::cout << "action index: " << index << std::endl;
}
