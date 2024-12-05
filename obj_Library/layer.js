class Layer {
    constructor(numNeurons, connectionsToPrevL, activFunction = null) {
        this.neuronArr = new Array(numNeurons);
        this.weightsMatrix = new Array(numNeurons);
        this.biasesMatrix = new Array(numNeurons);
        this.activationMatrix;
        this.activationFunction = activFunction;
        

        this.genLayer(numNeurons, connectionsToPrevL);
    }
    
    genLayer(numNeurons, connectionsToPrevL) {
        for (let i = 0; i < numNeurons; ++i) {
            this.neuronArr[i] = new Neuron(connectionsToPrevL);

            this.weightsMatrix[i] = this.neuronArr[i].getWeights();
            this.biasesMatrix = this.neuronArr[i].getBias();
        }
    }

    computeActivation(weightedSum) {
        this.activationMatrix = Math.map(weightedSum, this.activationFunction);
    }

    getWeightsMatrix() {
        return this.weightsMatrix;
    }

    getBiasesMatrix() {
        return this.biasesMatrix;
    }

    getActivationMatrix() {
        return this.activationMatrix;
    }

    setActivations(activArr) {
        this.activationMatrix = activArr;
    }
    // update
    // get activations
}