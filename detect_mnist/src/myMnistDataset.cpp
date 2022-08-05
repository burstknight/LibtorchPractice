#include "../includes/myMnistDataset.h"
#include "opencv2/core/hal/interface.h"
#include "torch/types.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

myMnistDataset::myMnistDataset(std::string sImagePath, std::string sLabelsPath){
	if(0 != loadImages(sImagePath)){
		throw std::runtime_error("Not found file to load images!");
	} // End of if-condition

	if(0 != loadLabels(sLabelsPath)){
		throw std::runtime_error("Not found to load labels!");
	} // End of if-condition

} // End of constructor

int myMnistDataset::loadImages(std::string sImagePath){
	std::FILE *pFileReader = fopen(sImagePath.c_str(), "rb");	
	if(NULL == pFileReader){
		return -1;
	} // End of if-condition

	unsigned int iNumOfImages = 0;
	unsigned int iRows, iCols;

	// Skip the magic number
	fseek(pFileReader, 4, SEEK_SET);

	// Read the number of the images
	fread(&iNumOfImages, sizeof(unsigned int), 1, pFileReader);

	// Read the rows and columns for each images
	fread(&iRows, sizeof(unsigned int), 1, pFileReader);
	fread(&iCols, sizeof(unsigned int), 1, pFileReader);

	unsigned char *pcBuffer = (unsigned char*)malloc(sizeof(unsigned char) * iRows * iCols);
	m_vmImages.resize(iNumOfImages);
	
	for(unsigned int i = 0; i < iNumOfImages; i++){
		fread(pcBuffer, sizeof(unsigned char), iRows * iCols, pFileReader);	
		
		cv::Mat mImage(iCols, iRows, CV_8UC1);
		std::memcpy((void*)mImage.data, (void*)pcBuffer, sizeof(unsigned char) * iRows * iCols);
		m_vmImages[i] = mImage;
	} // End of for-loop

	free(pcBuffer);
	pcBuffer = NULL;
	fclose(pFileReader);
	pFileReader = NULL;

	return 0;
} // End of myMnistDataset::loadImages

int myMnistDataset::loadLabels(std::string sLabelsPath){
	std::FILE *pFileReader = fopen(sLabelsPath.c_str(), "rb");	
	if(NULL == pFileReader){
		return -1;
	} // End of if-condition

	unsigned int iNumOfItems = 0;
	unsigned char cLabel;

	// Skip the magic number
	fseek(pFileReader, 4, SEEK_SET);

	// Read the number of the labels for the dataset
	fread(&iNumOfItems, sizeof(unsigned int), 1, pFileReader);
	m_viLabels.resize(iNumOfItems);

	for(unsigned int i = 0; i < iNumOfItems; i++){
		fread(&cLabel, sizeof(unsigned char), 1, pFileReader);	
		m_viLabels[i] = cLabel;
	} // End of for-loop

	fclose(pFileReader);
	pFileReader = NULL;
	return 0;
} // End of myMnistDataset:::loadLabels

torch::optional<size_t> myMnistDataset::size() const{
	return m_vmImages.size();	
} // End of myMnistDataset::size

