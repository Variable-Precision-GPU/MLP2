#include "sequential.cuh"

SequentialModel::SequentialModel(Optimizer* optimizer, LossFunction* lossFunction) {
    this->optimizer = optimizer;
    this->lossFunction = lossFunction;
    this->gradients = NULL;
}

void SequentialModel::addLayer(Layer* layer) {
    DEBUG_PRINT("Adding Layer to the model: %d\n", layer);
    this->layers.push_back(layer);
}

Tensor2D* SequentialModel::forward(Tensor2D* input) {
    Tensor2D* values = input;
    int i = 0;
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        if (i == 0) {
            setenv("VF_SIGNIFICAND","8",1);
        } else if (i == 2) {
            setenv("VF_SIGNIFICAND","8",1);
        } else if (i == 4) {
            setenv("VF_SIGNIFICAND","4",1);
        } else if (i == 6) {
            setenv("VF_SIGNIFICAND","4",1);
        }
        i++;
        values = (*layer)->forward(values);
        #if defined(DEBUG) && DEBUG >= 2
        DEBUG_PRINT("Forward pass for Layer %d:\n", (*layer));
        values->debugPrint();
        #endif
    }
    return values;
}

void SequentialModel::backward(Tensor2D* output, Tensor2D* labels) {
    // Compute gradients with loss function
    if (!this->gradients) {
        this->gradients = new Tensor2D(output->getSize(X), output->getSize(Y));
    }
    this->lossFunction->calculate(output, labels, this->gradients);
    #if defined(DEBUG) && DEBUG >= 2
    DEBUG_PRINT("Backward pass gradients:\n");
    gradients->debugPrint();
    #endif

    // Pass these gradients with backpropagation
    Tensor2D* values = gradients;
    for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); layer++) {
        values = (*layer)->backward(values);
        #if defined(DEBUG) && DEBUG >= 2
        DEBUG_PRINT("\nBackward pass for Layer %d:\n", (*layer));
        values->debugPrint();
        #endif
    }

    // Updates all layers with optimizer
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        optimizer->optimize(*layer);
    }
}

void SequentialModel::saveWeights(const char *weights_file) {
  std::ofstream file(weights_file);
  printf("Starting save\n");
  for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
      (*layer)->write(file);
  }
}

void SequentialModel::loadWeights(const char *weights_file) {
    FILE *file = fopen(weights_file, "r");
    printf("Starting load\n");
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        (*layer)->read(file);
    }
}
