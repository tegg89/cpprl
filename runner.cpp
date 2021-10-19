#include "runner.h"
#include "model.h"
#include <iostream>
#include <torch/script.h>


Runner::Runner(int64_t input_channels, int64_t num_actions):
        network(input_channels, num_actions){}

    // Device
    // auto cuda_available = torch::cuda::is_available();
    // torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    // std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // network(input_channels, num_actions);
    // network->to(device)
    // network.eval()

void print_modules(const torch::jit::script::Module& imodule) {
    for (const auto& module : imodule.named_children()) {
        if(module.value.children().size() > 0){
            print_modules(module.value);
        }
        else{
            std::cout << module.name << std::endl;
        }
    }
}

void Runner::loadstatedict(const std::string& param_path) {
    // torch::load(network, "/Users/teggsung/code/example/all_weights.pt");
    // std::cout << "model2 loaded!" << std::endl;

    // Load values by name
    // torch::Tensor a = containter.attr("default_policy/value_out/kernel").toTensor();
    // std::cout << a << "\n";

    // torch::Tensor b = containter.attr("default_policy/value_out/bias").toTensor();
    // std::cout << b << "\n";

    // torch::jit::script::Module containter = torch::jit::load(data_path);
    // std::cout << "model loaded!" << std::endl
    // std::cout << network.named_parameters() << std::endl;

    // WORK FINE!
    // for (const auto& pair : network.named_parameters()) {
    //     std::cout << pair.key() << ": " <<  pair.value() << std::endl;
    // }
    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    auto new_params = torch::jit::load(param_path); // implement this
    print_modules(new_params);
    // for (const auto& pair : new_params.parameters()) {
    //     std::cout << pair.values() << std::endl;
    // }
    auto params = network.named_parameters(true /*recurse*/);
    for (auto& val : new_params) {
        auto name = val.key();
        auto* t = params.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        } else {
            if (t != nullptr) {
                t->copy_(val.value());
            }
        }
    }
}
    
void Runner::run(void) {
    // init state and action vectors
    torch::Tensor state_tensor = torch::rand({1, 8});  // random vector (1, 8)
    std::cout << "observation vector (random): " << state_tensor << std::endl;

    // always explotation in inference stage
    torch::Tensor action_tensor = network.act(state_tensor);
    int64_t index = action_tensor[0].item<int64_t>();
    
    std::cout << "Action index: " << index << std::endl;
}
