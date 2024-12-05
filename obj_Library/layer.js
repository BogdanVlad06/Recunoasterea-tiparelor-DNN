// Ce scop mai are sa folosesc obiectul neuron, daca pot salva la un layer matricea Weights s.a.m.d ?
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
        this.activationMatrix = math.map(weightedSum, this.activationFunction);
    }
    // ----------------- GETTERS ---------------------------
    getWeightsMatrix() {
        return this.weightsMatrix;
    }

    getBiasesMatrix() {
        return this.biasesMatrix;
    }

    getActivationMatrix() {
        return this.activationMatrix;
    }
    // ----------------------- SETTERS ----------------------
    
    setActivations(activArr) {
        this.activationMatrix = activArr;
    }

    setBiasesMatrix(newMatrix) {
        this.biasesMatrix = newMatrix;
    }
}