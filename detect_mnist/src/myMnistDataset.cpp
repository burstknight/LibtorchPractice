#include "../includes/myMnistDataset.h"
#include "opencv2/core/hal/interface.h"
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
		m_vmImages.push_back(mImage);
	} // End of for-loop

	free(pcBuffer);
	pcBuffer = NULL;
	fclose(pFileReader);
	pFileReader = NULL;

	return 0;
} // End of myMnistDataset::loadImages

