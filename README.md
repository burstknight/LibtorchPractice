# LibtorchPractice
This is my libtorch practice. All examples follow the [tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html#), so you can see the web to get more details of all examples.

Libtorch is C++ interface of Pytorch that is machine learning framework for Python. We can use libtorch to create neural network, like using PyTorch.

## Dependencies
In my examples, I use makefile and pkg-config to build code. You can use my another [repository](https://github.com/burstknight/InstallLibtorch) to install develop enviroment.

## Build
If you want to debug code, you can follow:
```bash
make clean
make debug=1
```

Maybe you want to build all example for release. You can use these commands:
```bash
make clean
make
```

## Examples
I implemented three examples following this [web](https://pytorch.org/tutorials/advanced/cpp_frontend.html#). The examples are:
* [tensorEye](./tensorEye/Readme.md): It is a simple example that generates an identity matrix and show on terminal.
* [simpleNet](): It demostracted how to create a neural network by libtorch.
* [dcgan_struct](): It used GAN to demostract how to create a program to train a network.