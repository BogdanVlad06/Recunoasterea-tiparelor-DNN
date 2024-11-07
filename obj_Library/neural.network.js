// ramane sa realizez antrenarea retelei folosindu-ma de datele MINST
class NeuralNetwork {
    constructor(inputDim, outputDim, noHiddenLayers, learningRate) {
        this.numHiddenLayers = noHiddenLayers;
        this.outputSize = outputDim; 
        this.learningRate = learningRate; 
        this.label;
        this.sigmoidActivations = new Array(this.numHiddenLayers + 1);
        
        this.network = new Array(this.numHiddenLayers + 2);
        this.initializeNetwork(inputDim);
        
        this.gradient = new Array(this.numHiddenLayers + 2)
        
        this.caseProbability = new Array(this.outputSize); 
        this.prediction = -1;
        
    }
    
    initializeNetwork(inputLayerDim) {
        for (let layer = 1; layer <= this.numHiddenLayers + 1; ++layer) {
            let numNeurons; // the number of neurons for the current Layer
            let prevNumNeurons = Math.floor(inputLayerDim / (2 ** (layer - 1)));
            if (layer > this.numHiddenLayers) { // Output layer case
                numNeurons = this.outputSize;
            } else {
                numNeurons = Math.floor(inputLayerDim / (2 ** layer));
            }
            this.network[layer] = new Array(numNeurons);
            if (layer <= this.numHiddenLayers) {
                this.sigmoidActivations[layer] = new Array(numNeurons);
            }
            for (let i = 0; i < numNeurons; ++i) {
                this.network[layer][i] = new Neuron(prevNumNeurons);
            }
        }
    }
    
    feedForward(input) {
        this.network[0] = input;
        for (let layer = 1; layer <= this.numHiddenLayers + 1; ++layer) {
            let numNeurons = this.network[layer].length;
            let prevLayerActivationArr = this.getLayerActivationArr(layer - 1, "sigmoid");
            for (let i = 0; i < numNeurons; ++i) {
                this.network[layer][i].feed(prevLayerActivationArr);
                if (layer <= this.numHiddenLayers) {
                    this.sigmoidActivations[layer][i] = sigmoid(this.network[layer][i].getWeightedSum());
                }
            }
        }
    } 
    
    computeCaseProb() { // calculateOutput
        let outputActivationArr = this.getLayerActivationArr(this.numHiddenLayers + 1, "none");
        let expSum = 0;
        for (let i = 0; i < this.outputSize; ++i) {
            expSum += Math.exp(outputActivationArr[i]);
        }
        for (let digit = 0; digit < this.outputSize; ++digit) {
            this.caseProbability[digit] = softmax(outputActivationArr[digit], expSum);
        }
    }
    
    calculateLoss(label) { // cu CCEL, (am omis sa folosesc MSE)
        let loss = -Math.log(this.caseProbability[label]);
        return loss;
    }
    
    backpropagate(label) {
        this.gradient[this.numHiddenLayers + 1] = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; ++i) {
            this.gradient[this.numHiddenLayers + 1][i] = this.caseProbability[i];
        }
        this.gradient[this.numHiddenLayers + 1][label] -= 1;
        
        for (let layer = this.numHiddenLayers; layer > 0; --layer) {
            let layerSize = this.network[layer].length;
            this.gradient[layer] = new Array(layerSize);
            let prevLayerSize = this.network[layer + 1].length;
            
            for (let i = 0; i < layerSize; ++i) {
                let deltaOfPrevLayerGradients = 0; 
                for (let j = 0; j < prevLayerSize; ++j) { // neuron ([l][i]) conectat de layer-ul din dreapta prin [l+1][j].weight(i) 
                    deltaOfPrevLayerGradients += this.gradient[layer + 1][j] * this.network[layer + 1][j].getWeight(i);
                }
                this.gradient[layer][i] = sigmoidDerivate(sigmoid(this.network[layer][i].getWeightedSum())) * deltaOfPrevLayerGradients;
            }
        }
    }
    
    update() {
        for (let layer = 1; layer < this.numHiddenLayers + 2; ++ layer) {
            let layerSize = this.network[layer].length;
            let prevLayerActivationArr = this.getLayerActivationArr(layer - 1, sigmoid);
            // update neuron(l)(i) weights
            for (let i = 0; i < layerSize; ++i) {
                let costGradient = this.learningRate * this.gradient[layer][i];
                
                let newBias = this.network[layer][i].getBias() - costGradient;
                this.network[layer][i].setBias(newBias);
    
                for (let wi = 0, numWeights = this.network[layer][i].getNoWeights(); wi < numWeights; ++wi) {
                    let newWeightValue = this.network[layer][i].getWeight(wi) - (costGradient * prevLayerActivationArr[wi]);
                    this.network[layer][i].setWeight(wi, newWeightValue);
                }
            }
        }
    }
    // ---------------------- GETTERS -----------------------
    getLayerActivationArr(layerIndex, activFunction) {
        if (layerIndex == 0) {
            return this.network[0];
        }

        if (activFunction == "sigmoid") {
            return this.sigmoidActivations[layerIndex];
        }

        let numNeurons = this.network[layerIndex].length;
        let activationsArr = new Array(numNeurons);
        for (let i = 0; i < numNeurons; ++i) {
            activationsArr[i] = this.network[layerIndex][i].getWeightedSum();
        }
        return activationsArr;
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
}