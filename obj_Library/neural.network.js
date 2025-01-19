class NeuralNetwork {
    constructor(outputDim, numOfLayers, learningRate) {
        this.numHiddenLayers = numOfLayers - 1;
        this.outputSize = outputDim; 
        this.learningRate = learningRate; 
        this.trainingStop = false;

        this.loss;
        this.accuracy;

        this.network = new Array(this.numHiddenLayers + 2);
        this.network[0] = new Layer(0, 0);

        this.caseProbability = new Array(this.outputSize); 
        this.prediction = -1;
        
    }

    initializeLayer(index, size, connections, activFunction) {
        this.network[index] = new Layer(size, connections, activFunction);
    }
    
    feedForward(input) {  // Works!
        this.network[0].setActivationsArr(input);

        let W_matrix, prevA_matrix, B_matrix, Z_matrix;
        for (let layerIndex = 1; layerIndex <= this.numHiddenLayers + 1; ++layerIndex) {
            W_matrix = this.network[layerIndex].getWeightsMatrix();
            prevA_matrix = this.network[layerIndex - 1].getActivationMatrix();
            B_matrix = this.network[layerIndex].getBiasesMatrix();

            Z_matrix = math.add(math.multiply(W_matrix, prevA_matrix), B_matrix);
            this.network[layerIndex].computeActivationFromMatrix(Z_matrix);
        }
    }
    
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
        let struggles = new Array(10).fill(0);
        let numInputs = inputsArr.length;
        let testLoss = 0, correctPredictions = 0;
        for (let index = 0; index < numInputs; ++index) {
            let input = inputsArr[index];
            let label = labelsArr[index];

            this.feedForward(input);
            this.computeCaseProb();
            this.predict();

            let prediction = this.getPrediction();
            if (prediction == label) {
                correctPredictions++;  
            } else {
                ++struggles[label];
            }
            testLoss += this.calculateLoss(label);
        }
        let accuracy = correctPredictions / numInputs;
        testLoss /= numInputs;
        updateMetrics(testLoss, accuracy);
        this.loss = testLoss;
        this.accuracy = accuracy;
        console.log("total: ", numInputs, "; guessed: ", correctPredictions, "struggled with: ", struggles);
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

    getLayerWeights(layerIndex) {
        return this.network[layerIndex].getWeightsArr();
    }

    getLayerBiases(layerIndex) {
        return this.network[layerIndex].getBiasesArr();
    }

    getLayerActivation(layerIndex) {
        return this.network[layerIndex].getActivationArr();
    }
    // ------------------------------ SAVE / LOAD -----------------------------------------
    saveNetworkConfigToFile(fileName = "network_config.json") {
        let data = {
            layers: [],
            accuracy: this.accuracy,
            loss: this.loss
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
            this.accuracy = data.accuracy;
            this.loss = data.loss;    
            updateMetrics(this.loss, this.accuracy);
        });
    }
}