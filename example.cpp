#include <iostream>

#include "runner.h"

int main() {
    Runner runner(8, 1);
    runner.loadstatedict("./all_weights.pt");
    runner.run();
}