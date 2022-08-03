#include "catch.hpp"
#include "../src/neuralnetwork.h"
//#include <cmath> // std::isnan in C++11

using namespace std;
using namespace neuralnetworkfirstprinciples;

TEST_CASE("Neural Network implementation", "[neuralnetworktest]") {

    SECTION("Check we are working") {
        NeuralNetwork nn = NeuralNetwork();
        REQUIRE(nn.isOK());
    }

    // SECTION("Check put / call parity") {
    //     F = 100, X = 110, sd = 0.2, df = 0.97;
    //     call = make_unique<Black76Call>(F,X,sd,df);
    //     put = make_unique<Black76Put>(F,X,sd,df);
    //     REQUIRE(abs(put->value() - call->value() - (X-F) * df) < 1e-14);

    // }
}