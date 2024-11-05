class NeuralNetwork {
    constructor(valueArr, noHiddenLayers, learningRate) {
        this.neuronArr_perLayer = new Array(noHiddenLayers + 1); 
        this.activationArr_perLayer = new Array(noHiddenLayers + 1); // imi pare ca e inutila si ar putea fi ceva local in FHL
        //oare ar fi mai logic sa integrez output-ul in datele de mai sus, sa fie stocate pe indexul NoHiddenLayers?
        this.outputNeuronArr = new Array(10);
        this.outputActivationArr = new Array(10);
    
        this.caseProbability = new Array(10); // probabilitatile
        this.prediction = -1;

        this.forwardHiddenLayers(valueArr, 1, noHiddenLayers);
        this.forwardOutputLayer(noHiddenLayers);
        this.predict();

        let networkGradient = this.gradientCalculus(3,noHiddenLayers);
    }

    // (array of activations), (current currentLayer number), (No of HidLayers); 
    forwardHiddenLayers(input, currentLayer, noHiddenLayers) {
        let newNoNeurons = Math.floor(input.length / 2);        // the new number of neurons for the current currentLayer
        this.neuronArr_perLayer[currentLayer] = new Array(newNoNeurons);   // create the array of neurons for the current layer
        this.activationArr_perLayer[currentLayer] = new Array(newNoNeurons);

        for (let i = 0; i < newNoNeurons; ++i) {
            this.neuronArr_perLayer[currentLayer][i] = new Neuron(input);
            this.activationArr_perLayer[currentLayer][i] = this.neuronArr_perLayer[currentLayer][i].getActivation();
        }
        
        if (currentLayer >= noHiddenLayers) { // conditia de final;
            return;
        } else {
            this.forwardHiddenLayers(this.activationArr_perLayer[currentLayer], currentLayer + 1, noHiddenLayers);
        }

    }

    forwardOutputLayer(noHiddenLayers) {
        let lastHidLayerIndex = noHiddenLayers;
        let expSum = 0;
        for (let i = 0; i < 10; ++i) {
            this.outputNeuronArr[i] = new Neuron(this.getLayerActivationArr(lastHidLayerIndex));
            this.outputActivationArr[i] = this.outputNeuronArr[i].getActivation();
            expSum += Math.exp(this.outputActivationArr[i]);
        }
        for (let digit = 0; digit < 10; ++digit) {
            this.caseProbability[digit] = softmax(this.outputActivationArr[digit], expSum);
        }
    }

    predict() {
        let prediction = -1, maxAct = 0;

        for (let digit = 0; digit < 10; ++digit) {
            if (maxAct < this.caseProbability[digit]) {
                maxAct = this.caseProbability[digit];
                prediction = digit;
            }
        }
        
        this.prediction = prediction;
    }

    calculateLoss(label) { // cu CCEL, (am omis sa folosesc MSE)
        let loss = -Math.log(this.caseProbability[label]);
        return loss;
    }

    // calculeaza gradientul (delta) : “δj(l)​\= σ′(zj(l)​) k ∑ ​wkj(l+1) ​δk(l+1)​”
    gradientCalculus(label, noHiddenLayers) { // cifra coresp inputului = label 
        let outputGradient = new Array(10); 
        for (let i = 0; i < 10; ++i) {
            outputGradient[i] = this.caseProbability[i];
            if (i == label) {
                outputGradient[i] -= 1;
            }
        }
        
        let gradientArr_perLayer = new Array(noHiddenLayers + 2);
        gradientArr_perLayer[noHiddenLayers + 1] = outputGradient;

        let size = this.neuronArr_perLayer[noHiddenLayers].length;
        gradientArr_perLayer[noHiddenLayers] = new Array(size);
        // loop lastHiddenLayer 
        for (let ind = 0; ind < size; ++ind) {
            let gradientValue = 0;
            for(let j = 0; j < 10; ++j) { // output neurons loop, The sum of the gradients
                gradientValue += outputGradient[j] * this.outputNeuronArr[j].getWeight(ind);
            }
            gradientArr_perLayer[noHiddenLayers][ind] = gradientValue * sigmoidDerivate(this.neuronArr_perLayer[noHiddenLayers][ind].getActivation());
        }
        // loop layers
        for (let layer = noHiddenLayers - 1; layer > 0; --layer) {
           let noNeurons = this.neuronArr_perLayer[layer].length;
           gradientArr_perLayer[layer] = new Array(noNeurons) //loop neurons of layer
            for (let neuronInd = 0; neuronInd < noNeurons; ++neuronInd) {
                let gradientValue = 0; 
                size = this.neuronArr_perLayer[layer + 1].length;
                // loop corespWeights to calculate gradient
                for (let ind = 0; ind < size; ++ind) { // k ∑ ​wkj(l+1) ​δk(l+1)
                    gradientValue += gradientArr_perLayer[layer + 1][ind] * this.neuronArr_perLayer[layer + 1][ind].getWeight(ind);
                }
                gradientArr_perLayer[layer][neuronInd] = gradientValue * sigmoidDerivate(this.neuronArr_perLayer[layer][neuronInd].getActivation())
            }
        }
        return gradientArr_perLayer;
    }

    update(input, gradientArr_perLayer, noHiddenLayers, learningRate) { // Testata, valorile se modifica dupa ce trec prin fctie!
        //Layer 1 caz particular: in loc de this.neuronArr_perLayer[layer - 1][ind].getActivation(), folosim direct inputu
        this.updateLayerWeightsAndBiases(input, gradientArr_perLayer[1], 1, learningRate, this.neuronArr_perLayer[1]);
        
        // loop Hidden
        for (let layer = 2; layer <= noHiddenLayers; ++layer) {
            this.updateLayerWeightsAndBiases(this.activationArr_perLayer[layer - 1], gradientArr_perLayer[layer], layer, learningRate, this.neuronArr_perLayer[layer]);
        }
        // loop output
        this.updateLayerWeightsAndBiases(this.activationArr_perLayer[noHiddenLayers], gradientArr_perLayer[noHiddenLayers + 1], noHiddenLayers + 1, learningRate, this.outputNeuronArr);
    }

    updateLayerWeightsAndBiases(activations, gradientArr, layer, learningRate, layerNeurons) {
        for (let neuronInd = 0; neuronInd < layerNeurons.length; ++neuronInd) {
            let delta = gradientArr[neuronInd];
            let noWeights = layerNeurons[neuronInd].getNoWeights();
            for (let i = 0; i < noWeights; ++i) {
                let weightGradient = activations[i] * delta;
                let newWeight = layerNeurons[neuronInd].getWeight(i) - (weightGradient * learningRate);
                layerNeurons[neuronInd].setWeight(i, newWeight);
            }
            let newBias = layerNeurons[neuronInd].getBias() - (delta * learningRate);
            layerNeurons[neuronInd].setBias(newBias);
        }
    }

// ---------------------- GETTERS -----------------------
    getLayerActivationArr(layerIndex) {
        return this.activationArr_perLayer[layerIndex];
    }

    getPrediction() {
        return this.prediction;
    }
}

// propunere : sa mut FOL in FHL si sa il fac ForwardPass simplu