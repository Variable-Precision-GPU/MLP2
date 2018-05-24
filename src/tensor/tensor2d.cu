#include "tensor2d.h"

__global__
void kAdd1D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] += b[x];
    }
}

__global__
void kAdd2D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] += b[y*sizeX + x];
    }
}

__global__
void kSubtract(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] -= b[y*sizeX + x];
    }
}

__global__
void kScale(float *a, float factor, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] *= factor;
    }
}

__global__
void kMultiply(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
{
    // TODO: This implementation is very basic. Please improve me!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < bX && row < aY) {
        float sum = 0.0f;
        for (int i = 0; i < aX; i++) {
            sum += a[row*aX+i] * b[i*bX+col];
        }
        c[row*bX+col] = sum;
    }
}

__global__
void kMultiplyByTransposition(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
{
    // TODO: This implementation is very basic. Please improve me!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < bY && row < aY) {
        float sum = 0.0f;
        for (int i = 0; i < aX; i++) {
            sum += a[row*aX+i] * b[col*bX+i];
        }
        c[row*bY+col] = sum;
    }
}

__global__
void kTransposeAndMultiply(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
{
    // TODO: This implementation is very basic. Please improve me!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < bX && row < aX) {
        float sum = 0.0f;
        for (int i = 0; i < bY; i++) {
            sum += a[i*aX+row] * b[i*bX+col];
        }
        c[row*bX+col] = sum;
    }
}

__global__
void kMeanX(float* a, int aX, int aY, float* b)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < aX) {
        float sum = 0.0;
        for (int i = 0; i < aY; i++) {
            sum += a[i*aX + col];
        }
        b[col] = sum / aY;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    if (this->sizeX && this->sizeY) {
        cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
    } else {
        this->devData = NULL;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float** hostData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    if (this->sizeX && this->sizeY) {
        //printf("Created by allocation %dx%d\n", this->sizeX, this->sizeY);
        cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
        cudaMemcpy(this->devData, *hostData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        this->devData = NULL;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float* devData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->devData = devData;
    //printf("Created by assigning %dx%d\n", this->sizeX, this->sizeY);
}

Tensor2D::~Tensor2D() {
    //printf("Released %dx%d\n", this->sizeX, this->sizeY);
    cudaFree(this->devData);
}

float* Tensor2D::getDeviceData() {
    return this->devData;
}

float** Tensor2D::fetchDataFromDevice() {
    float** hostData = new float*[this->sizeY];
    *hostData = new float[this->sizeY * this->sizeX];
    for (int i = 1; i < this->sizeY; i++) hostData[i] = hostData[i-1] + this->sizeX;

    cudaDeviceSynchronize(); 
    cudaMemcpy(*hostData, this->devData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor2D::add(Tensor1D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->size) {
        printf("ERROR! Cannot add vector with size %d to matrix %dx%d.\n", tensor->size, this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd1D<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::add(Tensor2D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->sizeX || this->sizeY != tensor->sizeY) {
        printf("ERROR! Cannot add matrix with size %dx%d to matrix %dx%d.\n", tensor->sizeX, tensor->sizeY, this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd2D<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::subtract(Tensor2D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->sizeX || this->sizeY != tensor->sizeY) {
        printf("ERROR! Cannot subtract matrix with size %dx%d to matrix %dx%d.\n", tensor->sizeX, tensor->sizeY, this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kSubtract<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::scale(float factor) {
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kScale<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), factor, this->sizeX, this->sizeY);
}

Tensor2D* Tensor2D::multiply(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->sizeY) {
        printf("ERROR! Cannot multiply matrices with shape %dx%d and %dx%d.\n", this->sizeX, this->sizeY, tensor->sizeX, tensor->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((tensor->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kMultiply<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, tensor->getDeviceData(), tensor->sizeX, tensor->sizeY, output->getDeviceData());
    return output;
}

Tensor2D* Tensor2D::multiplyByTransposition(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->sizeX) {
        printf("ERROR! Cannot multiply matrix with shape %dx%d by transposition of matrix %dx%d.\n", this->sizeX, this->sizeY, tensor->sizeX, tensor->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((tensor->sizeY + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kMultiplyByTransposition<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, tensor->getDeviceData(), tensor->sizeX, tensor->sizeY, output->getDeviceData());
    return output;
}

Tensor2D* Tensor2D::transposeAndMultiply(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeY != tensor->sizeY) {
        printf("ERROR! Cannot multiply transposition of matrix with shape %dx%d by matrix %dx%d.\n", this->sizeX, this->sizeY, tensor->sizeX, tensor->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(8, 8);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((tensor->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeX + threadsPerBlock.y)/threadsPerBlock.y);
    kTransposeAndMultiply<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, tensor->getDeviceData(), tensor->sizeX, tensor->sizeY, output->getDeviceData());
    return output;
}

Tensor1D* Tensor2D::meanX(Tensor1D* output) {
    int threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    int numBlocks = (this->sizeX + threadsPerBlock)/threadsPerBlock;
    kMeanX<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, output->getDeviceData());
    return output;
}

void Tensor2D::debugPrint() {
    float** values = this->fetchDataFromDevice();
    for (int y = 0; y < this->sizeY; y++) {
        for (int x = 0; x < this->sizeX; x++) {
            printf("%8.5f; ", values[y][x]);
        }
        printf("\n");
    }
    delete[] values;
}