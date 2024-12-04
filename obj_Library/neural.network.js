// ramane sa realizez antrenarea retelei folosindu-ma de datele MINST
class NeuralNetwork {
    constructor(inputDim, outputDim, noHiddenLayers, learningRate) {
        this.numHiddenLayers = noHiddenLayers;
        this.outputSize = outputDim; 
        this.learningRate = learningRate; 
        this.label;
        // -- useless
        this.sigmoidActivations = new Array(this.numHiddenLayers + 1);
        // -- useless
        this.activation = new Array(this.numHiddenLayers + 2);

        this.network = new Array(this.numHiddenLayers + 2);
        this.averageGradient = new Array(this.numHiddenLayers + 2);
        
        this.initializeNetwork(inputDim);
        
        this.gradient = new Array(this.numHiddenLayers + 2);

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
            this.averageGradient[layer] = new Array(numNeurons);
            this.activation[layer] = new Array(numNeurons);
            
            for (let i = 0; i < numNeurons; ++i) {
                this.network[layer][i] = new Neuron(prevNumNeurons);
                this.averageGradient[layer][i] = 0;
            }
        }
    }
    
    feedForward(input, activFunction) {
        this.activation[0] = input;
        this.network[0] = input;
        for (let layer = 1; layer <= this.numHiddenLayers + 1; ++layer) {
            let numNeurons = this.network[layer].length;
            //let prevLayerActivationArr = this.getLayerActivationArr(layer - 1, "sigmoid");
            for (let i = 0; i < numNeurons; ++i) {
                this.network[layer][i].feed(this.activation[layer - 1]);
                this.activation[layer][i] = this.network[layer][i].getWeightedSum();
                if (layer <= this.numHiddenLayers) {
                    this.activation[layer][i] = activFunction(this.activation[layer][i]);
                }
            }
        }
    } 
    
    computeCaseProb() { // calculateOutput
        //let outputActivationArr = this.getLayerActivationArr(this.numHiddenLayers + 1, "none");
        let expSum = 0, outputLayer = this.numHiddenLayers + 1;
        for (let i = 0; i < this.outputSize; ++i) {
            expSum += Math.exp(this.activation[outputLayer][i]);
        }
        for (let digit = 0; digit < this.outputSize; ++digit) {
            this.caseProbability[digit] = softmax(this.activation[outputLayer][digit], expSum);
        }
    }
    
    calculateLoss(label) { // cu CCEL, (am omis sa folosesc MSE)
        let loss = -Math.log(this.caseProbability[label]);
        return loss;
    }
    
    backpropagate(label, actFunctionDerivate) {
        this.gradient[this.numHiddenLayers + 1] = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; ++i) {
            this.gradient[this.numHiddenLayers + 1][i] = this.caseProbability[i];
            if (i == label) {
                this.gradient[this.numHiddenLayers + 1][i] -= 1;
            }
            this.averageGradient[this.numHiddenLayers + 1][i] += this.gradient[this.numHiddenLayers + 1][i];
        }
        
        for (let layer = this.numHiddenLayers; layer > 0; --layer) {
            let layerSize = this.network[layer].length;
            this.gradient[layer] = new Array(layerSize);
            let prevLayerSize = this.network[layer + 1].length;
            
            for (let i = 0; i < layerSize; ++i) {
                let deltaOfPrevLayerGradients = 0; 
                for (let j = 0; j < prevLayerSize; ++j) { // neuron ([l][i]) conectat de layer-ul din dreapta prin [l+1][j].weight(i) 
                    deltaOfPrevLayerGradients += this.gradient[layer + 1][j] * this.network[layer + 1][j].getWeight(i);
                }
                this.gradient[layer][i] = actFunctionDerivate(this.activation[layer][i]) * deltaOfPrevLayerGradients;
                
                this.averageGradient[layer][i] += this.gradient[layer][i];
            }
        }
    }
    
    update(numBackporpagations) {
        for (let layer = 1; layer < this.numHiddenLayers + 2; ++ layer) {
            let layerSize = this.network[layer].length;
            // update neuron(l)(i) weights
            for (let i = 0; i < layerSize; ++i) {
                this.averageGradient[layer][i] /= numBackporpagations; // calculez media aritmetica a gradientelor acumulate in backprop-urile unei Epoch
                
                let costGradient = this.learningRate * this.averageGradient[layer][i];
                
                this.averageGradient[layer][i] = 0; // resetez gradientul mediei aritmetice pt urm Epoch

                let newBias = this.network[layer][i].getBias() - costGradient;
                this.network[layer][i].setBias(newBias);
    
                for (let wi = 0, numWeights = this.network[layer][i].getNoWeights(); wi < numWeights; ++wi) {
                    let newWeightValue = this.network[layer][i].getWeight(wi) - (costGradient * this.activation[layer - 1][wi]);
                    this.network[layer][i].setWeight(wi, newWeightValue);
                }
            }
        }
    }
    // ---------------------- SETTER -----------------------
    setLearningRate(value) {
        this.learningRate = value;
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