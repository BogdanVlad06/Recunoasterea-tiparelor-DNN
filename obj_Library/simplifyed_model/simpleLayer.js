class SmpLayer {
    constructor(numNeurons, connectionsToPrevL, activFunction = null) {
        this.activFunction = activFunction
        this.weights; // Li x Li - 1
        this.biases; // Li x 1
        this.avgWGradients = math.matrix(math.zeros(numNeurons, 1));
        this.avgBGradients = math.matrix(math.zeros(numNeurons, 1));

        this.basicInit(numNeurons, connectionsToPrevL);
    }

    basicInit(numNeurons, connectionsToPrevL) { // Initialize weights and biases with random values
        this.weights = math.random([numNeurons, connectionsToPrevL], -1, 1);
        this.biases = math.random([numNeurons, 1], 0, 1); 
    }

    accumulateGradient(delta, activToTheLeft) {
        this.avgBGradients = math.add(this.avgBGradients, delta);
        this.avgWGradients = math.add(this.avgWGradients,
             math.multiply(delta, math.transpose(activToTheLeft))); // pt a obtine o matrice de NL x NL-1
    }

    getActivationFunction() {
        return this.activFunction;
    }
}