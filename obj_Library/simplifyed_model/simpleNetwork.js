class SmpNetwork {
    constructor(size) {
        this.size = size; // hid + output (== index of Output Layer)
        this.network = new Array(size + 1); // 0 is input layer
        
    }

    initializeLayer(index, size, connections, activFunction) {
        this.network[index] = new SmpLayer(size, connections, activFunction);
    }
    
    feedForward(input) {  // Works!
        let activation = input;
        for (let layerIndex = 1; layerIndex <= this.size; ++layerIndex) {
            let w = this.network[layerIndex].getWeights();
            let b = this.network[layerIndex].getBiases();
            activation = math.map(math.add(math.multiply(w, activation), b), this.network[layerIndex].getActivationFunction());
        }
        return activation;
    }   

    SGD(miniBatchSize, epochs, learningRate, trainingImages, trainingLabels, testImages, testLabels) {
        let trainingIndex = [];
        for (let i = 0; i < trainingImages.length; ++i) {
            trainingIndex.push(i);
        }
        
        for (let i = 1; i <= epochs; ++i) {
            trainingIndex.sort(() => Math.random() - 0.5);

            let miniBatches = [];
            for (let j = 0; j < trainingImages.length; j += miniBatchSize) {
                let miniBatch = trainingIndex.slice(j, j + miniBatchSize);
                miniBatches.push(miniBatch);
            }
            // update
            for (let j = 0; j < miniBatches.length; ++j) {
                this.update(miniBatches[j], learningRate, trainingImages, trainingLabels);
            }
            // evaluate
            console.log("epoch: ", i, "total: ", numInputs, "; guessed: ", this.evaluate());
        }
    }

    update(miniBatch, learningRate, trainingImages, trainingLabels) {
        // calculate avgGradient(undivided by batchsize)/layer
        for (let i = 0; i < miniBatch.length; ++i) {
            let index = miniBatch[i],
                input = trainingImages[index],
                label = trainingLabels[index];
            this.backpropagate(input, label);
        }
        // update-itself
        for (let layerIndex = 1; layerIndex <= this.size; ++layerIndex) {
            let currentLayer = this.network[layerIndex]
                avgWGradients = math.multiply(currentLayer.getWeightsGradient(), learningRate / miniBatch.length),
                avgBGradients = math.multiply(currentLayer.getBiasesGradient(), learningRate / miniBatch.length);
            
            currentLayer.subtractFromWeights(avgWGradients);
            currentLayer.subtractFromBiases(avgBGradients);  
        }
    }   

    backpropagate(input, label) {
        // forward pass
        let activation = input;
        let activations = [];
        let weightedSums = [];
        activation.push(null);
        weightedSums.push(null);
        for (let layerIndex = 1; layerIndex <= this.size; ++layerIndex) {
            let w = this.network[layerIndex].getWeights();
            let b = this.network[layerIndex].getBiases();
            let weightedSum = math.add(math.multiply(w, activation), b);
            activation = math.map(weightedSum, this.network[layerIndex].getActivationFunction());
            weightedSums.push(weightedSum);
            activations.push(activation);
        }
        // backward pass
        let delta = math.dotMultiply(
                    this.costDerivative(activations[this.size], label)
                    , math.map(weightedSums[this.size], derivate(this.network[this.size].getActivationFunction())));
        this.network[this.size].accumulateGradient(delta);
        for (let layerIndex = this.size - 1; layerIndex > 0; --layerIndex) {
            let w = this.network[layerIndex + 1].getWeights(); // w of L + 1
            let primeWs = math.map(weightedSums[layerIndex], derivate(this.network[layerIndex].getActivationFunction()));
            delta = math.dotMultiply(math.multiply(math.transpose(w), delta), primeWs);
            this.network[layerIndex].accumulateGradient(delta);
        }
    }

    costDerivative(outputActivations, label) {
        let y = math.matrix(math.zeros(10, 1)); // hot encoded
        y.set([label, 0], 1);
        return math.multiply(math.subtract(outputActivations, y), 2);
    }

    evaluate(testImages, testLabels) {
        let total = testImages.length, correct = 0 
        for (let i = 0; i < total; ++i) {
            let output = this.feedForward(testImages[i]);
            let guess = math.argmax(output);
            if (guess === testLabels[i]) {
                ++correct
            }
        }
        return correct;
    }
}