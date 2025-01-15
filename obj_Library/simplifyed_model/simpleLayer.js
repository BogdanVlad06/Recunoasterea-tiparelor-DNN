class SmpLayer {
    constructor(numNeurons, connectionsToPrevL, activFunction = null) {
        this.activFunction = activFunction
        this.weights; // Li x Li - 1
        this.biases; // Li x 1
        this.avgWGradients = math.matrix(math.zeros(numNeurons, connectionsToPrevL));
        this.avgBGradients = math.matrix(math.zeros(numNeurons, 1));

        this.basicInit(numNeurons, connectionsToPrevL);
    }

    basicInit(numNeurons, connectionsToPrevL) { // Initialize weights and biases with random values
        this.weights = math.random([numNeurons, connectionsToPrevL], -1, 1);
        this.biases = math.random([numNeurons, 1], 0, 1); 
        //console.log("weights: ", this.weights, "biases: ", this.biases);
    }

    accumulateGradient(delta, activToTheLeft) {
        //console.log("delta: ", delta, "activToTheLeft: ", activToTheLeft, "avgWGradients: ", this.avgWGradients, "avgBGradients: ", this.avgBGradients);
        this.avgBGradients = math.add(this.avgBGradients, delta);
        this.avgWGradients = math.add(this.avgWGradients,
             math.multiply(delta, math.transpose(activToTheLeft))); // pt a obtine o matrice de NL x NL-1
    }

    subtractFromWeights(valueMatrix) {
        valueMatrix = valueMatrix.toArray();
        console.log("weights: ", this.weights);
        this.weights = math.subtract(this.weights, valueMatrix);
        console.log("after substraction: ", this.weights);
    }

    subtractFromBiases(valueMatrix) {
        valueMatrix = valueMatrix.toArray();
        console.log("biases: ", this.biases);
        console.log("valueMatrix: ", valueMatrix);
        this.biases = math.subtract(this.biases, valueMatrix);
        console.log("after substraction: ", this.biases);
    }

// ---------------------------------- GETTERS -----------------------------------------
    getWeights() {
        return this.weights;
    }
    
    getBiases() {
        return this.biases;
    }

    getActivationFunction() {
        return this.activFunction;
    }

    getWeightsGradient() {
        return this.avgWGradients;
    }

    getBiasesGradient() {
        return this.avgBGradients;
    }
}