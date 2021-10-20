#include <iostream>

#include "runner.h"

int main() {
    Runner runner(8, 1);
    runner.loadstatedict("./all_shared_weights.pt");
    runner.run();
}