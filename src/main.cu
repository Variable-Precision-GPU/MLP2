#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <assert.h>

#include "layers/dense.hpp"
#include "layers/relu.cuh"
#include "optimizers/sgd.hpp"
#include "loss/crossentropy.cuh"
#include "models/sequential.cuh"
#include "datasets/mnist.hpp"
#include "loggers/csv_logger.hpp"
#include "utils.hpp"
#include "configuration.cuh"


void train(SequentialModel* model, CrossEntropyLoss* loss, int startEpoch, int endEpoch) {
  MNISTDataSet* trainDataset = new MNISTDataSet(TRAIN);

  // Run some epochs
  // int epochs = Configuration::numberOfEpochs;
  int batchSize = Configuration::batchSize;
  int numberOfTrainBatches = trainDataset->getSize() / batchSize;
  for (int epoch = startEpoch; epoch <= endEpoch; epoch++) {
      float trainingLoss = 0.0, trainingAccuracy = 0.0;
      double trainingForwardTime = 0.0, trainingBackwardTime = 0.0;
      printf("Epoch %d:\n", epoch);
      for (int batch = 0; batch < numberOfTrainBatches; batch++) {
          // Fetch batch from dataset
          Tensor2D* images = trainDataset->getBatchOfImages(batch, batchSize);
          Tensor2D* labels = trainDataset->getBatchOfLabels(batch, batchSize);

          // Forward pass
          Tensor2D* output = model->forward(images);
          trainingLoss += loss->getLoss(output, labels);
          trainingAccuracy += loss->getAccuracy(output, labels);

          // Backward pass
          model->backward(output, labels);

          // Clean data for this batch
          delete images;
          delete labels;
      }

      // Calculate mean training metrics
      trainingLoss /= numberOfTrainBatches;
      trainingAccuracy /= numberOfTrainBatches;
      printf("  - [Train] Loss=%.5f\n", trainingLoss);
      printf("  - [Train] Accuracy=%.5f%%\n", trainingAccuracy);
      printf("  - [Train] Total Forward Time=%.5fms\n", trainingForwardTime);
      printf("  - [Train] Total Backward Time=%.5fms\n", trainingBackwardTime);
      printf("  - [Train] Batch Forward Time=%.5fms\n", trainingForwardTime / numberOfTrainBatches);
      printf("  - [Train] Batch Backward Time=%.5fms\n", trainingBackwardTime / numberOfTrainBatches);

      // Shuffle both datasets before next epoch!
      trainDataset->shuffle();
    }

}

void test(SequentialModel* model, CrossEntropyLoss* loss) {
  MNISTDataSet* testDataset = new MNISTDataSet(TEST);
  int batchSize = Configuration::batchSize;
  int numberOfTestBatches = testDataset->getSize() / batchSize;
  // Check model performance on test set
  float testLoss = 0.0, testAccuracy = 0.0;
  for (int batch = 0; batch < numberOfTestBatches; batch++) {
      // Fetch batch from dataset
      Tensor2D* images = testDataset->getBatchOfImages(batch, batchSize);
      Tensor2D* labels = testDataset->getBatchOfLabels(batch, batchSize);

      // Forward pass
      Tensor2D* output = model->forward(images);

      // Print error
      testLoss += loss->getLoss(output, labels);
      testAccuracy += loss->getAccuracy(output, labels);

      // Clean data for this batch
      delete images;
      delete labels;
  }

  // Calculate mean testing metrics
  testLoss /= numberOfTestBatches;
  testAccuracy /= numberOfTestBatches;
  printf("  - [Test] Loss=%.5f\n", testLoss);
  printf("  - [Test] Accuracy=%.5f%%\n", testAccuracy);
  printf("\n");
}

int main(int argc, char *argv[]) {
  // Always initialize seed to some random value
  // srand(static_cast<unsigned>(time(0)));
  srand(0);
  assert(argc > 1 && "Run with mode -train, -train-increment or -test");

  // Print our current configuration for this training
  Configuration::printCurrentConfiguration();
  Configuration::printCUDAConfiguration();

  // Prepare optimizer and loss function
  float learningRate = Configuration::learningRate;
  SGDOptimizer* optimizer = new SGDOptimizer(learningRate);
  CrossEntropyLoss* loss = new CrossEntropyLoss();

  // Prepare model
  SequentialModel* model = new SequentialModel(optimizer, loss);
  model->addLayer(new DenseLayer(28*28, 256));
  model->addLayer(new ReLuLayer(256));
  model->addLayer(new DenseLayer(256, 128));
  model->addLayer(new ReLuLayer(128));
  model->addLayer(new DenseLayer(128, 64));
  model->addLayer(new ReLuLayer(64));
  model->addLayer(new DenseLayer(64, 10));

  // model->addLayer(new DenseLayer(28*28, 512));
  // model->addLayer(new ReLuLayer(512));
  // model->addLayer(new DenseLayer(512, 10));

  if (strcmp(argv[1], "-train") == 0) {
      assert(argc == 4 && "Please provide the number of epochs and weights file");
      int epochs = atoi(argv[2]);

      train(model, loss, 1, epochs);
      model->saveWeights(argv[3]);
  } else if (strcmp(argv[1], "-train-increment") == 0) {
      assert(argc == 6 && "Please provide the start epoch, end epoch, input weights file and output weights file");
      int startEpoch = atoi(argv[2]);
      int endEpoch = atoi(argv[3]);

      model->loadWeights(argv[4]);
      train(model, loss, startEpoch, endEpoch);
      model->saveWeights(argv[5]);
  } else if (strcmp(argv[1], "-test") == 0) {
      assert(argc == 3 && "Please provide the weights file");
      model->loadWeights(argv[2]);
      test(model, loss);
  } else {
      assert(0 && "Run with mode -train, -train-increment or -test");
  }
  return 0;
}