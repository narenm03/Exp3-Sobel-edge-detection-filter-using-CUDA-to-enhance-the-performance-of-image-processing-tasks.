# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>ENTER YOUR NAME : NARENDHARAN M</h3>
<h3>ENTER YOUR REGISTER NO:212223230134</h3>
<h3>EX. NO:3</h3>
<h3>DATE:17.08.2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(
    unsigned char *srcImage,
    unsigned char *dstImage,
    unsigned int width,
    unsigned int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    float Ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    // Ignore border pixels
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {

        float Gx = 0.0f;
        float Gy = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {

                unsigned char pixel =
                    srcImage[(y + ky) * width + (x + kx)];

                Gx += pixel * Kx[ky + 1][kx + 1];
                Gy += pixel * Ky[ky + 1][kx + 1];
            }
        }

        float magnitude = fabsf(Gx) + fabsf(Gy);

        // Keep output within 0-255
        if (magnitude > 255)
            magnitude = 255;

        dstImage[y * width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    // Read WEBP image
    Mat image = imread("/content/shoe.webp", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    // Host output memory
    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Device memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(
        cudaMalloc(&d_inputImage, imageSize)
    );

    checkCudaErrors(
        cudaMalloc(&d_outputImage, imageSize)
    );

    // Copy input image to GPU
    checkCudaErrors(
        cudaMemcpy(
            d_inputImage,
            image.data,
            imageSize,
            cudaMemcpyHostToDevice
        )
    );

    // CUDA block and grid
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // CUDA timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch Sobel kernel
    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height
    );

    checkCudaErrors(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    // Copy result from GPU
    checkCudaErrors(
        cudaMemcpy(
            h_outputImage,
            d_outputImage,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    // Create output image
    Mat outputImage(
        height,
        width,
        CV_8UC1,
        h_outputImage
    );

    // Save output as WEBP
    if (!imwrite("/content/output_sobel.webp", outputImage)) {
        printf("Error: Could not save output image.\n");
        return -1;
    }

    // Free memory
    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Image processed successfully.\n");
    printf("Output: /content/output_sobel.webp\n");
    printf("CUDA kernel time: %f milliseconds\n", milliseconds);

    return 0;
}

```


## OUTPUT:
<img width="437" height="411" alt="image" src="https://github.com/user-attachments/assets/c838f673-6332-466c-99fb-a43742d467ea" />


## RESULT:
Thus the program has been executed by using CUDA to accelerate Sobel edge detection and improve image processing performance using parallel computation on GPU.
