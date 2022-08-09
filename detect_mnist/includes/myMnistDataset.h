#ifndef MYMNISTDATASET_H
#define MYMNISTDATASET_H

#include "opencv2/core/mat.hpp"
#include "torch/data.h"
#include "torch/data/datasets/base.h"
#include "torch/data/example.h"
#include "torch/types.h"
#include <cstddef>
#include <string>
#include <torch/torch.h>
#include <vector>
#include <opencv2/opencv.hpp>

union IntBuffer{
	int iData;
	unsigned char acData[4];
}; // End of union Buffer

/* Description: Convert the data from MSB to LSB
 * @param buffer[in/out]: The input data
 */
void convertFormat(IntBuffer &buffer);

class myMnistDataset: public torch::data::datasets::Dataset<myMnistDataset>{
	// private fields
	private:
		std::vector<cv::Mat> m_vmImages;
		std::vector<int> m_viLabels;

	// public methods
	public:
		/* Description: Constructor
		 * @param sImagePath[in]: The images path
		 * @param sLabelsPath[in]: The label file path
		 */
		myMnistDataset(const std::string &sImagePath, const std::string &sLabelsPath);

		/* Description: Get the number of the datum
		 * @return Return the number oif the datum
		 */
		torch::optional<size_t> size() const override;

		/* Description: Obtain the tensor at the current index
		 * @param index[in]: The current index that expects to get the example in the dataset
		 * @return Return the tensor of the example at the current index
		 */
		torch::data::Example<> get(size_t index) override;
	
	// private methods
	private:
		/* Description: Load the images
		 * @param sImagePath[in]: The images path
		 * @return Return 0 if succeed to load images, otherwise return -1
		 */
		int loadImages(std::string sImagePath);

		/* Description: Load the labels for the dataset
		 * @param sLabelsPath[in]: The label file path
		 * @return Return 0 if succeed to load the labels for the dataset, otherwise return -1
		 */
		int loadLabels(std::string sLabelsPath);
}; // End of class myMnistDataset

#endif
