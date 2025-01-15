class SmpNetwork {
    constructor(size) {
        this.size = size; // hid + output (== index of Output Layer)
        this.network = new Array(size + 1); // 0 is input layer
    }

    initializeLayer(index, size, connections, activFunction) {
        this.network[index] = new SmpLayer(size, connections, activFunction);
        //console.log("Layer ", index, " initialized with ", size, " neurons");
        //console.log("Weights: ", this.network[index].getWeights(), "Biases: ", this.network[index].getBiases());
    }
    
    feedForward(input) {  // Works!
        let activation = math.reshape(input, [784, 1]);
        for (let layerIndex = 1; layerIndex <= this.size; ++layerIndex) {
            //console.log("Layer ", layerIndex, " input: ", activation);
            let w = this.network[layerIndex].getWeights();
            let b = this.network[layerIndex].getBiases();
            //console.log(math.size(w), math.size(activation), math.size(b)); 
            activation = math.map(math.add(math.multiply(w, activation), b), this.network[layerIndex].getActivationFunction());
        }
        //console.log("Output: ", activation);
        return activation;
    }   

    SGD(miniBatchSize, epochs, learningRate, trainingImages, trainingLabels, testImages, testLabels) {
        let trainingIndex = [];
        for (let i = 0; i < trainingImages.length; ++i) {
            trainingIndex.push(i);
        }
        
        for (let i = 1; i <= 1/*epochs*/; ++i) {
            trainingIndex.sort(() => Math.random() - 0.5);

            let miniBatches = [];
            for (let j = 0; j < trainingImages.length; j += miniBatchSize) {
                let miniBatch = trainingIndex.slice(j, j + miniBatchSize);
                miniBatches.push(miniBatch);
            }
            //console.log("Epoch: ", i, "MiniBatches: ", miniBatches);
            // update
            for (let j = 0; j < miniBatches.length; ++j) {
                this.update(miniBatches[j], learningRate, trainingImages, trainingLabels);
            }
            // evaluate
            console.log("epoch: ", i, "total: ", testImages.length, "; guessed: ", this.evaluate(testImages, testLabels));
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
            let currentLayer = this.network[layerIndex],
                avgWGradients = math.multiply(currentLayer.getWeightsGradient(), learningRate / miniBatch.length),
                avgBGradients = math.multiply(currentLayer.getBiasesGradient(), learningRate / miniBatch.length);
            //console.log("Layer ", layerIndex, " avgWGradients: ", avgWGradients, " avgBGradients: ", avgBGradients);
            currentLayer.subtractFromWeights(avgWGradients);
            currentLayer.subtractFromBiases(avgBGradients);  
        }
    }   

    backpropagate(input, label) {
        // forward pass
        let activation =  math.reshape(input, [784, 1]);
        let activations = [];
        activations.push(activation);
        for (let layerIndex = 1; layerIndex <= this.size; ++layerIndex) {
            let w = this.network[layerIndex].getWeights();
            let b = this.network[layerIndex].getBiases();
            let weightedSum = math.matrix(math.add(math.multiply(w, activation), b));
            activation = math.matrix(math.map(weightedSum, this.network[layerIndex].getActivationFunction()));
            activations.push(activation);
        }
        // backward pass
        let delta = math.dotMultiply(
                    this.costDerivative(activations[this.size], label)
                    , math.map(activations[this.size], derivate(this.network[this.size].getActivationFunction())));
        this.network[this.size].accumulateGradient(delta, activations[this.size - 1]);
        for (let layerIndex = this.size - 1; layerIndex > 0; --layerIndex) {
            let w = this.network[layerIndex + 1].getWeights(); // w of L + 1
            let primeWs = math.map(activations[layerIndex], derivate(this.network[layerIndex].getActivationFunction()));
            delta = math.dotMultiply(math.multiply(math.transpose(w), delta), primeWs);
            this.network[layerIndex].accumulateGradient(delta, activations[layerIndex - 1]);
        }
    }

    costDerivative(outputActivations, label) {
        let y = math.matrix(math.zeros(10, 1)); // hot encoded
        y.set([label, 0], 1);
        //console.log (math.subtract(outputActivations, y));
        return math.subtract(outputActivations, y);
        //return math.multiply(math.subtract(outputActivations, y), 2);
    }

    evaluate(testImages, testLabels) {
        let total = testImages.length, correct = 0 
        for (let i = 0; i < total; ++i) {
            let output = this.feedForward(testImages[i]);
            //console.log( this.feedForward(testImages[i]), "Output: ", output);
            let guess = outputDigit(output);
            //console.log("Guess: ", guess, "Label: ", testLabels[i]);
            if (guess === testLabels[i]) {
                ++correct
            }
        }
        return correct;
    }
}