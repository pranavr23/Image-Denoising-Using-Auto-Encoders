Data Set used: MNIST Dataset that contains a Handwritten Digit and Labels



Image denoising is a popular application in which autoencoders attempt to reconstruct noise -free images from noisy source images.
Autoencoder is a type of feed-forward neural network used for unsupervised learning. They compress the input and then reconstruct the output. 
An autoencoder consists of three components: encoder, code, and decoder. The encoder compresses and generates the code, which the decoder only uses to reconstruct the input.
The Denoising Autoencoder is a modified autoencoder. It is made up of an encoder, which compresses the data into the latent code while extracting the most relevant features, and a decoder, which decompresses it and reconstructs the original input, just like a standard autoencoder
Denoising Autoencoder takes a noisy image as input and uses the original input without noise as the target for the output layer.
