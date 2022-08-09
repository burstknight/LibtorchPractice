#include <cstring>
#include <deque>
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "../../../includes/myMnistDataset.h"
#include "ATen/core/TensorBody.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "torch/data.h"
#include "torch/data/dataloader.h"
#include "torch/data/dataloader_options.h"
#include "torch/data/transforms/stack.h"
#include "torch/types.h"

struct Params{
	std::string sImagePath;
	std::string sLabelPath;
	std::string sOutputDir;
};

/* Description: Parse the arguments from user
 * @param argc[in]: The number of the arguments
 * @param argv[in]: The list of the arguments
 * @param poParam[out]: The struct is to store the parsed arguments
 * @return Return 0 if succeed to parse arguments, otherwise return -1
 */
int parseArgs(int argc, char **argv, Params *poParam);

/* Description: Convert the binary files to images and txt file for MNIST dataset
 * @param poParam[in]: This struct contains all config for converting
 */
void convertDataset(const Params *poParam);

int main(int argc, char **argv){
	Params *poParam = (Params*)malloc(sizeof(Params));

	if(0 == parseArgs(argc, argv, poParam)){
		try {
			convertDataset(poParam);
		} catch (std::exception &oEx) {
			printf("%s", oEx.what());
		} // End of try-catch
	} // End of if-condition


	free(poParam);
	poParam = NULL;
	return 0;
} // End od main

int parseArgs(int argc, char **argv, Params *poParam){
	while (1) {
		int iArg = getopt(argc, argv, "hi:a:o:");
		switch (iArg) {
			case 'h':
				printf("Name:\n\ttestDataset\n\n");
				printf("Description:\n\tThis program will convert MNIST Dataset to images for testing the class 'myMnistDataset'\n\n");
				printf("Usage:\n");
				printf("\t-h\tShow the usage\n");
				printf("\t-i\tSet to load the image path.\n");
				printf("\t-a\tSet to load the labels.\n");
				printf("\t-o\tSet the output directry.\n");
				return 1;
			case 'i':
				poParam->sImagePath = optarg;
				break;
			case 'a':
				poParam->sLabelPath = optarg;
				break;
			case 'o':
				poParam->sOutputDir = optarg;
				break;
			default:
				printf("Error: The argument '-%c' is invalid. Please use '-h' to get the usage.\n", iArg);
				return -1;
		} // End of switch
	} // End of while-loop

	return 0;
} // End of parseArgs

void convertDataset(const Params *poParam){
	char acBuffer[4096];
	auto poDataset = myMnistDataset(poParam->sImagePath, poParam->sLabelPath)
		.map(torch::data::transforms::Stack<>());

	auto poDataLoader = torch::data::make_data_loader(std::move(poDataset), 
			torch::data::DataLoaderOptions().batch_size(1));

	printf("Start to convert dataset!\n");

	int index = 0;
	for(auto &oExample: *poDataLoader){
		torch::Tensor tImage = oExample.data;
		torch::Tensor tLabel = oExample.target;
		tImage = tImage.to(torch::kU8);

		int iCols = tImage.size(2);
		int iRows = tImage.size(3);

		cv::Mat mImage(iRows, iCols, CV_8UC1);
		std::memcpy((void*)mImage.data, (void*)tImage.data_ptr(), sizeof(torch::kU8)*iRows*iCols);

		iCols = tLabel.size(2);
		iRows = tLabel.size(3);
		cv::Mat mLabel(iRows, iCols, CV_8UC1);
		tLabel = tLabel.to(torch::kU8);
		memcpy((void*)mLabel.data, (void*)tLabel.data_ptr(), sizeof(torch::kU8)*iRows*iCols);

		sprintf(acBuffer, "%s/%08d_%d.bmp", poParam->sOutputDir.c_str(), index, mLabel.at<uchar>(0, 0));
		printf("%08d: %d\r", index, mLabel.at<uchar>(0, 0));
		cv::imwrite(acBuffer, mImage);
		index++;
	} // End of for-loop

	printf("\nFinish!\n");
} // End of convertDataset

