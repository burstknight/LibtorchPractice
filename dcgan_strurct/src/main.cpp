#include "../includes/header.h"
#include "../includes/DCGANGenerator.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/imgcodecs.hpp"
#include "torch/cuda.h"
#include "torch/types.h"
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <utility>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

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

	// Set the folder to save model
	char *pcModelFolder;

	// After how many batches to create a new checkpoint periodically
	unsigned int iCheckPoint;

	// How many images to sample at every checkpoint
	unsigned int iNumOfSamplesPerCheckPoint;

	// Set to `true` to restore models and optimizers from previously saved
	// checkpoints
	bool isResume;

	// Set to `-1` to use CPU for training.
	int iDevice;

	// After how many batches to log a new update with the loss value
	unsigned int iLogInterval;
}; // End of struct TrainParams


/*
 * Description: Train DCGAN models
 * @param pParams: The trainign parameters
 */
void train(const TrainParams *pParams);

/* Description: Initialzie training parameters
 * @param pParams[out]: Return the initialized training parameters
 */
void initTrainingParams(TrainParams *pParams);

/* Description: Parse the arguments from user.
 * @param argc[in]: The number of arguments
 * @param argv[in]: The arguments from user.
 * @param pParams[out]: The training parameters.
 * @return Return 0 if succeed to parse arguments, otherwise return -1.
 */
int parseArgs(int argc, char **argv, TrainParams *pParams);

int main(int argc, char **argv){
	TrainParams *pParams = (TrainParams*)malloc(sizeof(TrainParams));
	initTrainingParams(pParams);

	if(0 == parseArgs(argc, argv, pParams)){
		train(pParams);	
	} // End of if-conditon

	free(pParams->pcDatasetFolder);
	pParams->pcDatasetFolder = NULL;
	free(pParams->pcModelFolder);
	pParams->pcModelFolder = NULL;
	free(pParams);
	pParams = NULL;
	return 0;	
} // End of main

void train(const TrainParams *pParams){
	char buffer[4096];
	torch::manual_seed(1);
	
	// Set the device to train.
	// If user want to train model on GPU and GPU is avalailable, 
	// creating a struct to move all datum and models into GPU.
	torch::Device device(torch::kCPU);
	if(pParams->iDevice >= 0 && torch::cuda::is_available()){
		printf("CUDA is available! Training on GPU.\n");
		device = torch::Device(torch::kCUDA);
		device.set_index(pParams->iDevice);
	} // End of if-condition


	DCGANGenerator poGeneratorNet(pParams->iNoiseSize);
	poGeneratorNet->to(device);

	// Create a network to evaluate the factuality of the fake images that are generate by GAN.
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
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
		torch::nn::Sigmoid() );	
	poDiscriminator->to(device);


	/* Loading MNIST dataset for trainig
	 */
	auto poDataset = torch::data::datasets::MNIST(pParams->pcDatasetFolder)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	unsigned int uiBatchSizePerEpochs = std::ceil(poDataset.size().value()/(1.0*pParams->iBatchSize));

	auto poDataLoader = torch::data::make_data_loader(std::move(poDataset), 
			torch::data::DataLoaderOptions().batch_size(pParams->iBatchSize).workers(2));

	torch::optim::Adam oGeneratorOptimizer(poGeneratorNet->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
	torch::optim::Adam oDiscriminatorOptimizer(poDiscriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));


	if(pParams->isResume){
		printf("Loading previous checkpoint...");
		sprintf(buffer, "%s/generator-checkpoint.pt", pParams->pcModelFolder);
		torch::load(poGeneratorNet, buffer);
		sprintf(buffer, "%s/generator-optimizer-checkpoint.pt", pParams->pcModelFolder);
		torch::load(oGeneratorOptimizer, buffer);
		sprintf(buffer, "%s/discrimator-checkpoint.pt", pParams->pcModelFolder);
		torch::load(poDiscriminator, buffer);
		sprintf(buffer, "%s/discrimator-optimizer-checkpoint.pt", pParams->pcModelFolder);
		torch::load(oDiscriminatorOptimizer, buffer);
		printf(" Done!\n");
	} // End of if-conditon

	unsigned long uiCheckpointCounter = 1;
	/* Run for-loop to train model
	 */
	for(unsigned int i = 1; i <= pParams->iNumOfEpochs; i++){
		unsigned long iBatchIndex = 0;	
		for(torch::data::Example<> &oBatchExample : *poDataLoader){
			// Train MNIST detector with real images
			poDiscriminator->zero_grad();
			torch::Tensor tRealImages = oBatchExample.data.to(device);
			torch::Tensor tRealLabel = torch::empty(oBatchExample.data.size(0), device).uniform_(0.8, 1.0);
			torch::Tensor tRealOutput = poDiscriminator->forward(tRealImages).reshape({oBatchExample.data.size(0)});
			torch::Tensor tLossReal = torch::binary_cross_entropy(tRealOutput, tRealLabel);
			tLossReal.backward();

			// Train MNIST detector with fake images
			torch::Tensor tNoise = torch::randn({oBatchExample.data.size(0), pParams->iNoiseSize, 1, 1}, device);
			torch::Tensor tFakeImages = poGeneratorNet->forward(tNoise);
			torch::Tensor tFakeLabels = torch::zeros(oBatchExample.data.size(0), device);
			torch::Tensor tFakeOutput = poDiscriminator->forward(tFakeImages.detach()).reshape({oBatchExample.data.size(0)});
			torch::Tensor tLossFake = torch::binary_cross_entropy(tFakeOutput, tFakeLabels);
			tLossFake.backward();

			torch::Tensor tTotalLoss = tLossReal + tLossFake;
			oDiscriminatorOptimizer.step();

			// Train generator
			poGeneratorNet->zero_grad();
			tFakeLabels.fill_(1);
			tFakeOutput = poDiscriminator->forward(tFakeImages).reshape({oBatchExample.data.size(0)});
			torch::Tensor tGenLoss = torch::binary_cross_entropy(tFakeOutput, tFakeLabels);
			tGenLoss.backward();
			oGeneratorOptimizer.step();

			if(iBatchIndex % pParams->iLogInterval == 0){
				printf("[%2d/%2d][%3lu/%3lu] D_loss: %.6f | G_loss: %.6f\n", i, pParams->iNumOfEpochs,
						iBatchIndex, uiBatchSizePerEpochs,
						tTotalLoss.item<float>(), tGenLoss.item<float>());
			} // End of if-conditon

			if(iBatchIndex % pParams->iCheckPoint == 0){
				// Checkpoint the model and optimizer state
				sprintf(buffer, "%s/generator-checkpoint.pt", pParams->pcModelFolder);
				torch::save(poGeneratorNet, buffer);
				sprintf(buffer, "%s/generator-optimizer-checkpoint.pt", pParams->pcModelFolder);
				torch::save(oGeneratorOptimizer, buffer);
				sprintf(buffer, "%s/discrimator-checkpoint.pt", pParams->pcModelFolder);
				torch::save(poDiscriminator, buffer);
				sprintf(buffer, "%s/discrimator-optimizer-checkpoint.pt", pParams->pcModelFolder);
				torch::save(oDiscriminatorOptimizer, buffer);

				// Sample the generator and save the images
				torch::Tensor tSamples = poGeneratorNet->forward(torch::randn({pParams->iNumOfSamplesPerCheckPoint, pParams->iNoiseSize, 1, 1}, device));
				tSamples = (tSamples + 1.0)/2.0;
				tSamples = tSamples.mul(255).clamp(0, 255).to(torch::kU8);
				tSamples = tSamples.to(torch::kCPU);
				for(int j = 0; j < pParams->iNumOfSamplesPerCheckPoint; j++){
					/* tSamples.size() can get its dimension
					 * tSamples.size(0) is the number of the batches for the tensor.
					 * tSamples.size(1) is the number of th channels for the tensor
					 * tSamples.size(2) is the columns for the tensor.
					 * tSamples.size(3) is the rows for the tensor.
					 */

					int iCols = tSamples.size(2);
					int iRows = tSamples.size(3);
					
					// Convert tensor to cv::Mat.
					Mat mSample(iCols, iRows, CV_8UC1);
					std::memcpy((void*) mSample.data, 
							(void*)((uchar*)tSamples.data_ptr() + iRows*iCols),  // This tensor has batches images, so must move pointer to split each images.
							sizeof(torch::kU8)*iRows*iCols);

					sprintf(buffer, "%s/dcfan-sample-%06lu_%04d.bmp", pParams->pcModelFolder, uiCheckpointCounter, j);
					cv::imwrite(buffer, mSample);
				} // End of for-loop

				printf("\n-> checkpoint %06ld\n", uiCheckpointCounter);
				uiCheckpointCounter++;
			} // End of if-condition

			iBatchIndex++;
		} // End of for-loop
	} // End of for-loop

	printf("Training complete!\n");
} // End of train

void initTrainingParams(TrainParams *pParams){
	pParams->iNoiseSize = 100;
	pParams->iBatchSize = 64;
	pParams->iNumOfEpochs = 30;
	pParams->isResume = false;
	pParams->iCheckPoint = 200;
	pParams->iLogInterval = 10;
	pParams->iNumOfSamplesPerCheckPoint = 10;
	pParams->iDevice = -1;
	pParams->pcModelFolder = (char*)malloc(sizeof(char)*4096);
	pParams->pcDatasetFolder = (char*)malloc(sizeof(char)*4096);

	sprintf(pParams->pcDatasetFolder, "./dataset");
	sprintf(pParams->pcModelFolder, "./models");
} // End of initTrainingParams

int parseArgs(int argc, char **argv, TrainParams *pParams){
	while(1){
		int iArgs = getopt(argc, argv, "hn:b:e:d:m:c:r:s:l:v:");	
		if(-1 == iArgs)
			break;

		switch (iArgs) {
			case 'h':
				printf("dcgan_struct:\nThis program is implemented by libtorch to generate fake MNIST images with GAN.\n\nUsage:\n");
				printf("\t-n:\tThe size of the noise vector fed to the generator for training. \x1b[32m[Default value: 100]\x1b[0m\n");
				printf("\t-b:\tThe batch size for training model. \x1b[32m[Default value: 64]\x1b[0m\n");
				printf("\t-e:\tThe number of the epoches for training. \x1b[32mDefault value: 30\x1b[0m\n");
				printf("\t-d:\tSet the directory to find the MNIST dataset. \x1b[32m[Default value: './dataset']\x1b[0m\n");
				printf("\t-m:\tSet the directory to save model. Default value: \x1b[32m['./models']\x1b[0m\n");
				printf("\t-c:\tSet it to how many epoches to create a new checkpoint periodically. \x1b[32m[Default value: 200]\x1b[0m\n");
				printf("\t-r:\tSet to positive integer to restore training progress from previously checkpoint. \x1b[32m[Default value: -1]\x1b[0m\n");
				printf("\t-s:\tHow many images to sample at every checkpoint. \x1b[32m[Default value: 100]\x1b[0m\n");
				printf("\t-l:\tSet to log a new update with the loss value. \x1b[32m[Default value: 10]\x1b[0m\n");
				printf("\t-v:\tSet the device to train. If you want to use CPU, you can set '-1', \n\t\totherwise set the ID of GPU. \x1b[32m[Default value: -1]\x1b[0m\n");
				printf("\t-h:\tShow the usage of this program.\n");
				return -1;
			case 'b':
				pParams->iBatchSize = atoi(optarg);
				break;
			case 'n':
				pParams->iNoiseSize = atoi(optarg);
				break;
			case 'e':
				pParams->iNumOfEpochs = atoi(optarg);
				break;
			case 'd':
				sprintf(pParams->pcDatasetFolder, "%s", optarg);
				break;
			case 'm':
				sprintf(pParams->pcModelFolder, "%s", optarg);
				break;
			case 'c':
				pParams->iCheckPoint = atoi(optarg);
				break;
			case 'r':
				pParams->isResume = atoi(optarg) > 0 ? true : false;
				break;
			case 's':
				pParams->iNumOfSamplesPerCheckPoint = atoi(optarg);
				break;
			case 'l':
				pParams->iLogInterval = atoi(optarg);
				break;
			case 'v':
				pParams->iDevice = atoi(optarg);
				break;
			default:
				printf("\x1b[31mError: The argument -%c is invalid! Please use '-h' to check the usage of this program.\x1b[0m\n", optopt);
				return  -1;
			} // End of switch
	} // End of while-loop

	return 0;
} // End of parseArgs

