#include <torch/torch.h>
#include <iostream>

using namespace std;

int main(){
	torch::Tensor oTensor = torch::eye(3);
	cout << oTensor << "\n";
	return 0;
} // End of main

