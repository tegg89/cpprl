//
// Created by Navneet Madhu Kumar on 2019-07-10.
//
#include "Trainer.h"
#include "model.h"
#include "/Users/teggsung/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"
#include <math.h>
#include <chrono>


Trainer::Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity, char* saved_model):
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    network(input_channels, num_actions);
    network->to(device)
    network.eval()
    
    void Trainer::load_environment(int64_t random_seed, std::string rom_path) {
        ale.setInt("random_seed", random_seed);
        ale.setBool("display_screen", true);
        ale.loadROM(rom_path);
    }

    torch::Tensor Trainer::get_tensor_observation(std::vector<unsigned char> state) {
        std::vector<int64_t > state_int;
        state_int.reserve(state.size());

        for (int i=0; i<state.size(); i++){
            state_int.push_back(int64_t(state[i]));
        }
        
        // TODO: Change obs dimension (4D -> 3D -- [batch, feat1, feat2])
        torch::Tensor state_tensor = torch::from_blob(state_int.data(), {1, 3, 210, 160});
        return state_tensor;
    }

    void Trainer::run(int64_t random_seed, std::string rom_path, int64_t num_epochs) {
        // init environment
        load_environment(random_seed, rom_path);
        ActionVect legal_actions = ale.getLegalActionSet();
        ale.reset_game();
        std::vector<unsigned char> state;
        ale.getScreenRGB(state);
        float episode_reward = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        // load ML-trained model
        torch::load(network, "model.pt");

        for(int i=1; i<=num_epochs; i++) {
            // init obs & act
            torch::Tensor state_tensor = get_tensor_observation(state);
            Action a;

            // always explotation in inference stage
            torch::Tensor action_tensor = network.act(state_tensor);
            int64_t index = action_tensor[0].item<int64_t>();
            a = legal_actions[index];

            // reward
            float reward = ale.act(a);  // env.step(a)
            episode_reward += reward;

            // next state & done
            std::vector<unsigned char> new_state;
            ale.getScreenRGB(new_state);
            torch::Tensor new_state_tensor = get_tensor_observation(new_state);
            bool done = ale.game_over();

            state = new_state;

            if (done) {
                ale.reset_game();
                std::vector<unsigned char> state;
                ale.getScreenRGB(state);
                std::cout << episode_reward << std::endl;
                episode_reward = 0.0;
            }

        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    }
