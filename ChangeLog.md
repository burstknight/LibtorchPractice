# Change Log

------------
## [Unrelease]
### [Added]
- Add a simple example `tensorEye` to create eye tensor.
- Add an example `simpleNet` to show how to create a network.
- Add an example `dcgan_struct` to show how to train a network with struct.

### [Changed]
- Update the example `simpleNet` to run network forward.
- Update the example `dcgan_struct` to support GPU for training.
- Update the example `dcgan_struct` to show more detail for usage.
- Update the example `dcgan_struct` to convert the samples that are generated from GAN to images.

### [Fixed]
- Fixed the dimension error for the example `dcgan_struct`.
- Fixed the bug that cannot get the iteration number per batches for the example `dcgan_struct`.
- Fixed the bug that the example `dcgan_struct` could not parse arguments from user.
- Fixed the bug that the training flow is wrong for the example `dcgan_struct`.
- Fixed the bug that the example `dcgan_struct` would save the samples into wrong path at every checkpoint.
- Fixed the bug that the example `dcgan_struct` could not save the samples at every checkpoint.
- Fixed the bug that the example `dcgan_struct` could not resume previous checkpoint.
- Fixed the bug that the type of the samples is same as the type of the modules at every checkpoint for the example `dcgan_struct`.
- Fixed the bug thatt warning about `output not match` happened during training on GPU for the example `dcgan_struct`.

### [Removed]

