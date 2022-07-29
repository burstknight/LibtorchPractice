#include "../includes/header.h"
#include "../includes/DCGANGenerator.h"
#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "torch/data/dataloader.h"
#include "torch/data/datasets/mnist.h"
#include "torch/data/example.h"
#include "torch/data/transforms/stack.h"
#include "torch/data/transforms/tensor.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/batchnorm.h"
#include "torch/nn/modules/container/sequential.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/options/activation.h"
#include "torch/nn/options/conv.h"
#include "torch/optim/adam.h"
#include "torch/serialize/input-archive.h"
#include <cstdio>
#include <tuple>
#include <utility>

// Description: Store the parameters for training
struct TrainParams{
	// The size of the noise vector fed to the generator
	unsigned int iNoiseSize;

	// The batch size for training
	unsigned int iBatchSize;

	// The number of epochs
	unsigned int iNumOfEpochs;

	// Where to find the MNIST dataset
	char *pcDatasetFolder;

	// After how many batches to create a new checkpoint periodically
	unsigned int iCheckPoint;

	// How many images to sample at every checkpoint
	unsigned int iNumOfSamplesPerCheckPoint;

	// Set to `true` to restore models and optimizers from previously saved
	// checkpoints
	bool isResume;

	// After how many batches to log a new update with the loss value
	unsigned int iLogInterval;
}; // End of struct TrainParams


/*
 * Description: Train DCGAN models
 * @param pParams: The trainign parameters
 */
void train(const TrainParams *pParams);

int main(){
	return 0;	
} // End of main

void train(const TrainParams *pParams){
	DCGANGenerator poGeneratorNet(pParams->iNoiseSize);

	torch::nn::Sequential poDiscriminator(
		// Layer 1
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

		// Layer 2
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(128),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

		// Layer 3
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

		// Layer 4
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(1).bias(false)),
		torch::nn::Sigmoid() );	


	/* Loading MNIST dataset for trainig
	 */
	auto poDataset = torch::data::datasets::MNIST(pParams->pcDatasetFolder)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	auto poDataLoader = torch::data::make_data_loader(std::move(poDataset), 
			torch::data::DataLoaderOptions().batch_size(pParams->iBatchSize).workers(2));

	torch::optim::Adam oGeneratorOptimizer(poGeneratorNet->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
	torch::optim::Adam oDiscriminatorOptimizer(poDiscriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));

	/* Run for-loop to train model
	 */
	for(unsigned int i = 1; i <= pParams->iNumOfEpochs; i++){
		unsigned long iBatchIndex = 0;	
		for(torch::data::Example<> &oBatchExample : *poDataLoader){
			poDiscriminator->zero_grad();
			torch::Tensor tRealImages = oBatchExample.data;
			torch::Tensor tRealLabel = torch::empty(oBatchExample.data.sizes(0));
			torch::Tensor tRealOutput = poDiscriminator->forward(tRealImages);
		} // End of for-loop

	} // End of for-loop


} // End of train

