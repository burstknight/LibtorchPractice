#include "../includes/Net.h"
#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/Logging.h"
#include "torch/nn/module.h"
#include "torch/ordered_dict.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Description: Show all parameters in the net
// @param oNet: the net
void showParameter(torch::nn::Module &oNet);

// Description: Show all parameters with each name in the net
// @param oNet: the net
void showNamedParameter(torch::nn::Module &oNet);

int main(){
	ParameterNet oParameterNet(4, 5);	
	SubmoduleNet oSubmoduleNet(4, 3);

	cout << "============== ParameterNet ==============\n";
	showParameter(oParameterNet);
	cout << "\n";
	showNamedParameter(oParameterNet);

	cout << "\nForward:\n";
	cout << oParameterNet.forward(torch::ones({2, 4})) << "\n";

	cout << "\n============== SubmoduleNet ==============\n";
	showParameter(oSubmoduleNet);
	cout << "\n";
	showNamedParameter(oSubmoduleNet);

	cout << "\nForward:\n";
	cout << oSubmoduleNet.forward(torch::ones({2, 4})) << "\n";

	return 0;
} // End of main

void showParameter(torch::nn::Module &oNet){
	// get the vector of the parameters of the net
	vector<torch::Tensor> vtParamerters = oNet.parameters();	

	for(int i = 0; i < vtParamerters.size(); i++){
		cout << vtParamerters[i] << "\n";
	} // End of for-loop

} // End of showParameter

void showNamedParameter(torch::nn::Module &oNet){
	// get the ordered_dict of the parameters of the net
	torch::OrderedDict<std::string, torch::Tensor> oDictParams = oNet.named_parameters();

	for(const auto &p : oDictParams){
		cout << p.key() << ": \n" << p.value() << "\n";
	} // End of for-loop

} // End of showNamedParameter

