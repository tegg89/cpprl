#include "Trainer.h"
#include <iostream>
// Move torch imports before ale because ale uses namespace std which interferes with the torch imports.
#include "/Users/teggsung/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"


int main() {

    Trainer trainer(3, 2, 100000);  // (input channel, num_actions, capacity, device)
    trainer.run(123, "/Users/teggsung/Codes/Pytorch-RL/CPP/atari_roms/pong.bin", 1000000); // (random seed, rom_path, num_epochs)

}