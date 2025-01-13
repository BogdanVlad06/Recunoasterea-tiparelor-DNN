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

            //this.update()
            //this.evaluate()
        }
    }

    update(miniBatch, learningRate, trainingImages, trainingLabels) {
        let getAvgGradientMatrix
        for (let i = 0; i < miniBatch.length; ++i) {
            // backpropagate
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
        return math.multiply(math.substract(outputActivations, y), 2);
    }
    ////////////////////////////////////////////////////////////////////////
    
    computeCaseProb() { // Works!
        let outputLayerIndex = this.numHiddenLayers + 1;
        let activations = this.network[outputLayerIndex].getActivationArr();
        let expActivations = activations.map(Math.exp);
        let expSum = expActivations.reduce((acc, curr) => acc + curr, 0);
        
        for (let digit = 0; digit < this.outputSize; ++digit) {
            this.caseProbability[digit] = softmax(activations[digit], expSum);
        }
    }
    
    calculateLoss(label) { // cu CCEL, (am omis sa folosesc MSE)
        let loss = -Math.log(this.caseProbability[label]);
        return loss;
    }
    
    backpropagate(label) { // works !
        let gradArr = this.caseProbability;
        gradArr[label] -= 1;
    
        this.network[this.numHiddenLayers + 1].setGradientArr(gradArr);
        this.network[this.numHiddenLayers + 1].updateAvgGradientArr();
        
        for (let layer = this.numHiddenLayers; layer > 0; --layer) {
            let actFunctionDerivate = derivate(this.network[layer].getActivationFunction());
            let delta_matrix = math.multiply(math.transpose(this.network[layer + 1].getWeightsMatrix()), this.network[layer + 1].getGradientMatrix());
            let derivateAct_Marix = this.network[layer].getActivationMatrix().map(actFunctionDerivate);
            //Hadamard product (element-wise multiplication) (both column vectors)
            let computedGradient_matrix = math.dotMultiply(delta_matrix, derivateAct_Marix);
            this.network[layer].setGradientArrFromMatrix(computedGradient_matrix);
            this.network[layer].updateAvgGradientArr();
        }
    }
    
    update(numBackporpagations) {
        let denominatorForAvgGradientCalculus = this.learningRate / numBackporpagations;
        
        for (let layer = 1; layer < this.numHiddenLayers + 2; ++ layer) {
            this.network[layer].computeAvgGradientArr(denominatorForAvgGradientCalculus);

            let error_matrix = this.network[layer].getAvgGradientMatrix();
            let biases_matrix = this.network[layer].getBiasesMatrix();
            let weights_matrix = this.network[layer].getWeightsMatrix();
            let prev_activation_matrix = this.network[layer - 1].getActivationMatrix();
            
            let newBiasesMatrix = math.subtract(biases_matrix, error_matrix);
            this.network[layer].setBiasesArrFromMatrix(newBiasesMatrix);
            
            let weightGradient_matrix = math.multiply(error_matrix, math.transpose(prev_activation_matrix));
            let newWeightsMatrix = math.subtract(weights_matrix, weightGradient_matrix);
            this.network[layer].setWeightsArrFromMatrix(newWeightsMatrix);

            this.network[layer].resetAvgGradientArr();
        }
    }
    // ---------------------- SETTER -----------------------
    setLearningRate(value) {
        this.learningRate = value;
    }

    // ---------------------- Utils -----------------------
    test(inputsArr, labelsArr) {
        let numInputs = inputsArr.length;
        let testLoss = 0, correctPredictions = 0;
        for (let index = 0; index < numInputs; ++index) {
            let input = inputsArr[index];
            let label = labelsArr[index];

            this.feedForward(input);
            this.computeCaseProb();
            this.predict();

            let prediction = this.getPrediction();
            if (prediction == label) correctPredictions++;  

            testLoss += this.calculateLoss(label);
        }
        let accuracy = correctPredictions / numInputs;
        testLoss /= numInputs;
        updateMetrics(testLoss, accuracy);
        console.log("total: ", numInputs, "; guessed: ", correctPredictions);
    }

    stopTraining(value) {
        this.trainingStop = value;
    }

    /*async*/ train(startingLearingValue, decay, inputsArr, labelsArr, numCycles, batchSize = 10) {
        const numInputs = inputsArr.length;
        
        for (let epoch = 0; epoch < numCycles; ++epoch) {
            let epochLoss = 0;
            this.learningRate = startingLearingValue / (1 + (decay * epoch));
            let correctPredictions = 0;
            // Shuffle data each epoch for better convergence
            const shuffledIndices = [...Array(numInputs).keys()].sort(() => Math.random() - 0.5);
            
            for (let startIdx = 0; startIdx < numInputs; startIdx += batchSize) {
                if (this.trainingStop) {
                    return;
                }
                // Accumulate gradients over the mini-batch
                let batchLoss = 0;
                
                for (let i = startIdx; i < Math.min(startIdx + batchSize, numInputs); ++i) {
                    const index = shuffledIndices[i];
                    const input = inputsArr[index];
                    const label = labelsArr[index];
                    
                    this.feedForward(input);
                    this.computeCaseProb();
                    this.predict();
                    
                    let prediction = this.getPrediction();
                    if (prediction == label) correctPredictions++;

                    let exampleLoss = this.calculateLoss(label);
                    batchLoss += exampleLoss;
                    
                    this.backpropagate(label);
                }

                // Update weights after each mini-batch
                this.update(batchSize);

                epochLoss += batchLoss / batchSize;
                //await new Promise(resolve => setTimeout(resolve, 0));

            }
            let accuracy = correctPredictions / numInputs;
            console.log(`Epoch ${epoch + 1} Loss: ${epochLoss / (numInputs / batchSize)}, Accuracy: ${(accuracy * 100).toFixed(2)}%`);
            console.log(this.learningRate);
        }
        console.log(this.learningRate, decay, batchSize);
    }
    
    predict() {
        let digitPrediction = -1, maxAct = 0;
    
        for (let digit = 0; digit < 10; ++digit) {
            if (maxAct < this.caseProbability[digit]) {
                maxAct = this.caseProbability[digit];
                digitPrediction = digit;
            }
        }
        
        this.prediction = digitPrediction;
    }

    // ---------------------- GETTERS -----------------------
    getCaseProbabilities() {
        return this.caseProbability;
    }

    getPrediction() {
        return this.prediction;
    }
    // ------------------------------ SAVE / LOAD -----------------------------------------
    saveNetworkConfigToFile(fileName = "network_config.json") {
        let data = {
            layers: []
        };
    
        for (let i = 1; i <= this.numHiddenLayers + 1; ++i) {
            let layer = this.network[i]; // Assuming `this.layers` holds the Layer objects
            data.layers.push({
                weights: layer.getWeightsArr(), // Save weights as a 2D array
                biases: layer.getBiasesArr()   // Save biases as a 1D array
            });
        }
    
        let json = JSON.stringify(data, null, 2);
        saveJSON(data, fileName);
    }
    // trb sa o fac sa poata incarca indif de numaru de layere si nr de neuroni ? 
    loadNetworkConfigFromFile(fileName = "network_config.json") {
        loadJSON(fileName, (data) => {
            for (let i = 1; i <= this.numHiddenLayers + 1; ++i) {
                let layerData = data.layers[i - 1]; // JSON layers are 0-indexed
                
                if (!this.network[i]) {
                    console.error(`Layer ${i} not found in network.`);
                    continue;
                }
                // Set weights and biases from the saved data
                this.network[i].setBiasesArr(layerData.biases);
                this.network[i].setWeightsArr(layerData.weights);
            }
        });
    }
}