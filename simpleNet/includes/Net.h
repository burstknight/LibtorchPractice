#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/serialize/input-archive.h"
#include <bits/stdint-intn.h>
#include <torch/torch.h>

struct ParameterNet: torch::nn::Module{
	ParameterNet(int64_t N, int64_t M){
		W = register_parameter("W", torch::randn({N, M}));
		b = register_parameter("b", torch::randn(M));
	} // constructor
	
	torch::Tensor forward(torch::Tensor input){
		return torch::addmm(b, input, W);
	}


	torch::Tensor W, b;
}; // End of struct Net

struct SubmoduleNet: torch::nn::Module{
	torch::nn::Linear linear = nullptr;
	torch::Tensor another_bias;

	SubmoduleNet(int N, int M){
		linear = register_module("linear", torch::nn::Linear(N, M));
		another_bias = register_parameter("b", torch::randn(M));
	} // End of constructor

	torch::Tensor forward(torch::Tensor input){
		return linear(input) + another_bias;
	} // End of forward

}; // End of struct SubmoduleNet

