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


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({8, 3, 64, 64}).to(torch::kCUDA));

    Mat image;
    image = imread("/home/ubuntu/PyTorch-GAN/data/img_align_celeba/000001.jpg", IMREAD_COLOR);


    // Convert the image to a 3D tensor with dimensions (C, H, W)
    cv::Mat image_float;
    image.convertTo(image_float, CV_32F, 1.0/255.0);
    cv::Mat image_transposed = image_float.t();
    std::vector<int64_t> dims = {1, 3, image_transposed.rows, image_transposed.cols};
    torch::Tensor input_tensor = torch::from_blob(image_transposed.data, torch::IntList(dims));

    // Normalize the tensor to match the mean and standard deviation used during training
    input_tensor[0][0] = (input_tensor[0][0] - 0.485) / 0.229;
    input_tensor[0][1] = (input_tensor[0][1] - 0.456) / 0.224;
    input_tensor[0][2] = (input_tensor[0][2] - 0.406) / 0.225;

    // Convert the tensor to a variable and move it to the device (CPU or GPU)
    torch::Tensor input_variable = torch::autograd::make_variable(input_tensor, false);
    input_variable = input_variable.to(at::kCUDA);

    // Forward pass through the network
    at::Tensor output = module.forward({input_variable}).toTensor();
    // Get the current time after executing the code

    // Compute the elapsed time in milliseconds

    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    // Convert the output tensor to a CPU tensor and detach it from the computation graph
    at::Tensor output_cpu = output.to(at::kCPU).detach();

    // Convert the tensor to a 3D CxHxW array
    std::vector<int64_t> dim = output_cpu.sizes();
    int C = dim[1];
    int H = dim[2];
    int W = dim[3];
    float* data = output_cpu.data<float>();
    cv::Mat output_image(H, W, CV_32FC(C), data);

    // Convert the output image from float to unsigned char (0-255)
    output_image.convertTo(output_image, CV_8UC(C), 255.0);

    // Convert the output image from CxHxW to HxWxC format
    cv::Mat output_image_transposed = output_image.t();
    cv::Mat output_image_swapped;
    cv::cvtColor(output_image_transposed, output_image_swapped, cv::COLOR_BGR2RGB);

    // Save the output image to a file
    cv::imwrite("output.jpg", output_image_swapped);

  std::cout << "ok\n";
}