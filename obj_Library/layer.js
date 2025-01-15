class Layer {
    constructor(numNeurons, connectionsToPrevL, activFunction = null) {
        this.weightsArr = new Array(numNeurons);
        this.biasesArr = new Array(numNeurons);
        this.activationArr;
        this.activationFunction = activFunction;
        this.gradientArr = new Array(numNeurons);
        this.avgGradientArr = new Array(numNeurons).fill(0);
        

        this.genLayer(numNeurons, connectionsToPrevL);
    }    
    genLayer(numNeurons, connectionsToPrevL) { 
        const stdDev = Math.sqrt(2 / connectionsToPrevL); // He Initialization
        this.weightsArr = Array.from({ length: numNeurons }, () =>
            Array.from({ length: connectionsToPrevL }, () => (Math.random() * 2 - 1) * stdDev)
        );
        this.biasesArr = Array.from({ length: numNeurons }, () => 0); // Initialize biases to zero
    }
    // --------------------- UPDATES -----------------------
    computeActivationFromMatrix(weightedSumMatrix) { // works!
        let weightedSumArr = weightedSumMatrix.toArray();
        weightedSumArr = weightedSumArr.flat();
        this.activationArr = weightedSumArr.map(this.activationFunction);
    }

    computeAvgGradientArr(scalar) {
        this.avgGradientArr = this.avgGradientArr.map(value => value * scalar);
    }

    updateAvgGradientArr() {
        this.avgGradientArr = math.add(this.avgGradientArr, this.gradientArr);
    }
    // ----------------- GETTERS ---------------------------
    getWeightsArr() {
        return this.weightsArr;
    }

    getWeightsMatrix() {
        return math.matrix(this.weightsArr);
    }

    getBiasesArr() {
        return this.biasesArr;
    }

    getBiasesMatrix() { // needs formating for column matrix
        return math.reshape(math.matrix(this.biasesArr), [this.biasesArr.length, 1]) ;
    }

    getActivationArr() {
        return this.activationArr;
    }

    getActivationMatrix() { // needs formating for column matrix
        return math.reshape(math.matrix(this.activationArr), [this.activationArr.length, 1]);
    }

    getActivationFunction() {
        return this.activationFunction;
    }

    getGradientMatrix() { // column matrix
        return math.reshape(math.matrix(this.gradientArr), [this.gradientArr.length, 1]);
    }

    getGradientArr() {
        return this.gradientArr;
    }

    getAvgGradientMatrix() {
        return math.reshape(math.matrix(this.avgGradientArr), [this.avgGradientArr.length, 1]);
    }
    // ----------------------- SETTERS ----------------------
    
    setActivationsArr(activArr) {
        this.activationArr = activArr; //math.reshape(math.matrix(activArr), [activArr.length, 1]);
    }

    setGradientArr(grdArr) {
        this.gradientArr = grdArr;
    }

    setGradientArrFromMatrix(grdMatrix) {
        let newGradietArr = grdMatrix.toArray();
        newGradietArr = newGradietArr.flat();
        this.gradientArr = newGradietArr;
    }

    setBiasesArrFromMatrix(biasesMatrix) {
        let newBiasesArr = biasesMatrix.toArray();
        newBiasesArr = newBiasesArr.flat();
        this.biasesArr = newBiasesArr;
    }

    setBiasesArr(newBiasesArr) {
        this.biasesArr = newBiasesArr;
    }

    setWeightsArrFromMatrix(newWeightsMatrix) {
        let newWeightsArr = newWeightsMatrix.toArray();
        this.weightsArr = newWeightsArr;
    }

    setWeightsArr(newWeightsArr) {
        this.weightsArr = newWeightsArr;

    }

    resetAvgGradientArr() {
        this.avgGradientArr.fill(0);
    }
}