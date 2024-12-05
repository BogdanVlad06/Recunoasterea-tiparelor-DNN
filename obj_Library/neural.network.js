// ramane sa realizez antrenarea retelei folosindu-ma de datele MINST
class NeuralNetwork {
    constructor(outputDim, numOfLayers, learningRate) {
        this.numHiddenLayers = numOfLayers - 1;
        this.outputSize = outputDim; 
        this.learningRate = learningRate; 
        this.label;

        this.activation = new Array(this.numHiddenLayers + 2);

        this.network = new Array(this.numHiddenLayers + 2);
        this.averageGradient = new Array(this.numHiddenLayers + 2);
                
        this.gradient = new Array(this.numHiddenLayers + 2);

        this.caseProbability = new Array(this.outputSize); 
        this.prediction = -1;
        
    }

    initializeLayer(index, size, connections, activFunction) {
        this.network[index] = new Layer(size, connections, activFunction);
        this.averageGradient[index] = new Array(numNeurons).fill(0);
    }
    
    feedForwardUsingMatrix(input) {
        this.network[0] = new Layer(0, 0);
        this.network[0].setActivations(input);

        let W_matrix, prevA_matrix, B_matrix, Z_matrix;
        for (let layerIndex = 1; layerIndex <= this.numHiddenLayers + 1; ++layerIndex) {
            W_matrix = this.network[layerIndex].getWeightsMatrix();
            prevA_matrix = this.network[layerIndex - 1].getActivationMatrix();
            B_matrix = this.network[layerIndex].getBiasesMatrix();

            Z_matrix = math.add(math.multiply(W_matrix, prevA_matrix), B_matrix);

            this.network[layerIndex].computeActivation(Z_matrix);
        }
    }
    
    computeCaseProb() { // calculateOutput
        //let outputActivationArr = this.getLayerActivationArr(this.numHiddenLayers + 1, "none");
        let outputLayerIndex = this.numHiddenLayers + 1;
        let activations = this.network[outputLayerIndex].getActivationMatrix(),
            expActivations = activations.map(Math.exp),
            expSum = expActivations.reduce((acc, curr) => acc + curr, 0);
        
        for (let digit = 0; digit < this.outputSize; ++digit) {
            this.caseProbability[digit] = softmax(activations[digit], expSum);
        }
    }
    
    calculateLoss(label) { // cu CCEL, (am omis sa folosesc MSE)
        let loss = -Math.log(this.caseProbability[label]);
        return loss;
    }
    
    backpropagate(label, actFunctionDerivate) {
        this.gradient[this.numHiddenLayers + 1] = this.caseProbability;
        this.gradient[this.numHiddenLayers + 1][label] -= 1;
        this.averageGradient[this.numHiddenLayers + 1] = Math.add(this.averageGradient[this.numHiddenLayers + 1], this.gradient[this.numHiddenLayers + 1]);
        
        for (let layer = this.numHiddenLayers; layer > 0; --layer) {
            let prevLayerDeltaGradients_matrix = Math.multiply(this.gradient[layer + 1], this.network[layer + 1].getWeightsMatrix()),
                derivateActivation_matrix = this.network[layer].getActivationMatrix().map(actFunctionDerivate);
            this.gradient[layer] = Math.multiply(derivateActivation_matrix, prevLayerDeltaGradients_matrix)
            
            this.averageGradient[layer] = Math.add(this.averageGradient[layer], this.gradient[layer]);
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
                    let newWeightValue = this.network[layer][i].getWeightAt(wi) - (costGradient * this.activation[layer - 1][wi]);
                    this.network[layer][i].setWeight(wi, newWeightValue);
                }
            }
        }
    }
    // ---------------------- SETTER -----------------------
    setLearningRate(value) {
        this.learningRate = value;
    }
    // ---------------------- Utils -----------------------
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