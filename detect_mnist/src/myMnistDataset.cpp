#include "../includes/myMnistDataset.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/from_blob.h"
#include "ATen/ops/full.h"
#include "c10/core/ScalarType.h"
#include "opencv2/core/hal/interface.h"
#include "torch/data/example.h"
#include "torch/types.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

myMnistDataset::myMnistDataset(const std::string &sImagePath, const std::string &sLabelsPath){
	if(0 != loadImages(sImagePath)){
		throw std::runtime_error("Not found file to load images!");
	} // End of if-condition

	if(0 != loadLabels(sLabelsPath)){
		throw std::runtime_error("Not found to load labels!");
	} // End of if-condition

	if((m_vmImages.size() != m_viLabels.size()) || (m_vmImages.size() * m_viLabels.size() <= 0)){
		throw std::runtime_error("The number of the labels 'm_viLabels' != the number of the images 'm_vmImages' for the given dataset!");
	} // End of if-condition

} // End of constructor

int myMnistDataset::loadImages(std::string sImagePath){
	std::FILE *pFileReader = fopen(sImagePath.c_str(), "rb");	
	if(NULL == pFileReader){
		return -1;
	} // End of if-condition

	int iNumOfImages = 0;
	int iRows, iCols;
	IntBuffer buffer;

	int iMagic;
	// Skip the magic number
	fseek(pFileReader, 4, SEEK_SET);


	// Read the number of the images
	fread(&(buffer.iData), sizeof(int), 1, pFileReader);
	convertFormat(buffer);
	iNumOfImages = buffer.iData;

	// Read the rows and columns for each images
	fread(&(buffer.iData), sizeof(int), 1, pFileReader);
	convertFormat(buffer);
	iRows = buffer.iData;

	fread(&(buffer.iData), sizeof(int), 1, pFileReader);
	convertFormat(buffer);
	iCols = buffer.iData;


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
	IntBuffer buffer;

	// Skip the magic number
	fseek(pFileReader, 4, SEEK_SET);

	// Read the number of the labels for the dataset
	fread(&(buffer.iData), sizeof(unsigned int), 1, pFileReader);
	convertFormat(buffer);
	iNumOfItems = buffer.iData;
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

torch::data::Example<> myMnistDataset::get(size_t index){
	torch::Tensor tImage = torch::from_blob(m_vmImages[index].data, 
			{m_vmImages[index].rows, m_vmImages[index].cols, 1}, torch::kByte)
			.permute({2, 0, 1});	// convert cols*rows*channels to cols*rows*channels

	torch::Tensor tLabel = torch::full({1}, m_viLabels[index]);

	return {tImage.clone(), tLabel.clone()};
} // End of myMnistDataset::get

void convertFormat(IntBuffer &buffer){
	for(int i = 0; i < 2; i++){
		unsigned char tmp = buffer.acData[3 - i];
		buffer.acData[3 - i] = buffer.acData[i];
		buffer.acData[i] = tmp;
	} // End of for-loop
} // End of convertFormat

