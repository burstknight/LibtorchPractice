#include "../includes/header.h"
#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "torch/nn/module.h"
#include "torch/nn/modules/batchnorm.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/options/conv.h"
#include "torch/nn/pimpl.h"

struct DCGANGeneratorImpl: torch::nn::Module{
	torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
	torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;

	DCGANGeneratorImpl(int iNoiseSize):
		conv1(torch::nn::ConvTranspose2dOptions(iNoiseSize, 256, 4).bias(false)),
		batch_norm1(256),
		conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
		batch_norm2(128),
		conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
		batch_norm3(64),
		conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false)){
			register_module("conv1", conv1);
			register_module("batch_norm1", batch_norm1);
			register_module("conv2", conv2);
			register_module("batch_norm2", batch_norm2);
			register_module("conv3", conv3);
			register_module("batch_norm3", batch_norm3);
			register_module("conv4", conv4);
	} // constructor

	torch::Tensor forward(torch::Tensor x){
		x = torch::relu(batch_norm1(conv1(x)));
		x = torch::relu(batch_norm2(conv2(x)));
		x = torch::relu(batch_norm3(conv3(x)));
		x = torch::tanh(conv4(x));

		return x;
	}

}; // End of struct DCGANGeneratorImpl


TORCH_MODULE(DCGANGenerator);
