// ramane sa realizez antrenarea retelei folosindu-ma de datele MINST
class NeuralNetwork {
    constructor(outputDim, numOfLayers, learningRate) {
        this.numHiddenLayers = numOfLayers - 1;
        this.outputSize = outputDim; 
        this.learningRate = learningRate; 
        this.label;

        this.network = new Array(this.numHiddenLayers + 2);

        this.caseProbability = new Array(this.outputSize); 
        this.prediction = -1;
        
    }

    initializeLayer(index, size, connections, activFunction) {
        this.network[index] = new Layer(size, connections, activFunction);
    }
    
    feedForward(input) {  // Works!
        this.network[0] = new Layer(0, 0);
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
    train(decay, inputsArr, labelsArr, numCycles, batchSize = 10) {
        const numInputs = inputsArr.length;
        
        for (let epoch = 0; epoch < numCycles; ++epoch) {
            let epochLoss = 0;
            this.learningRate = this.learningRate / (1 + (decay * epoch));
            let correctPredictions = 0;
            // Shuffle data each epoch for better convergence
            const shuffledIndices = [...Array(numInputs).keys()].sort(() => Math.random() - 0.5);
            
            for (let startIdx = 0; startIdx < numInputs; startIdx += batchSize) {
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
            }
            let accuracy = correctPredictions / numInputs;
            updateMetrics(epochLoss / (numInputs / batchSize), accuracy);
            console.log(`Epoch ${epoch + 1} Loss: ${epochLoss / (numInputs / batchSize)}, Accuracy: ${(accuracy * 100).toFixed(2)}%`);
        }
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
    
    getPrediction() {
        return this.prediction;
    }

    saveNetworkConfigToFile(fileName = "network_config.json") {
        let data = {
            architecture: []
        };

        for(let layer = 1; layer < this.numHiddenLayers + 2; ++layer) {
            let layerData = [];
            let size = this.network[layer].length;
            for (let i = 0; i < size; ++i) {
                layerData.push({
                    weights: this.network[layer][i].w,
                    bias: this.network[layer][i].b
                });
            }
            data.architecture.push(layerData);
        }
        let json = JSON.stringify(data, null, 2);
        //let blob = new Blob([json], { type: 'application/json' });
        //saveAs(blob, 'obj_Library/SavedNetworkConfig.json');
        saveJSON(data, "networkConfig.json");
    }

    loadNetworkConfigFromFile(file){ // trb sa o fac sa poata incarca indif de numaru de layere si nr de neuroni
        let data = JSON.parse(file.data);
        const {architecture} = data;
        let auxNetwork = [];
        auxNetwork.push(new Array(784));
        for (let layer = 1; layer <= this.numHiddenLayers + 2; ++layer) {
            let layerData = architecture[layer - 1];
            auxNetwork.push(new Array(layerData.length));
            for (let i = 0; i < layerData.length; ++i) {
                auxNetwork[layer][i] = new Neuron(auxNetwork[layer - 1].length);
                auxNetwork[layer][i].w = layerData[i].weights;
                auxNetwork[layer][i].b = layerData[i].bias;
            }
        }
        this.network = auxNetwork;
    } // ma gandesc sa creez un network aux care sa inlocuiasca networku curent
}