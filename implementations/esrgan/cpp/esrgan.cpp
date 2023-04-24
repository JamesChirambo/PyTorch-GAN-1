#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <chrono>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
int load(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

    // Load the Torch model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Load the input image using OpenCV
    cv::Mat image1 = cv::imread("/home/ubuntu/PyTorch-GAN/data/img_align_celeba/000001.jpg", cv::IMREAD_COLOR);

    cv::Mat image;
    cv::Size size(64, 64);  // define the new size of the image
    cv::resize(image1, image, size);  // resize the image
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Convert the input image to a Torch tensor
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);
    input_tensor = input_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA);

    // Run inference on the input tensor
    at::Tensor output_tensor = module.forward({input_tensor}).toTensor();

    // // Convert the output tensor to a cv::Mat object
    cv::Mat output_image(output_tensor.size(2), output_tensor.size(3), CV_32FC3, output_tensor.data_ptr<float>());
    // output_image *= 255.0;

    // Display the output image using OpenCV

  std::cout << "ok\n";
}