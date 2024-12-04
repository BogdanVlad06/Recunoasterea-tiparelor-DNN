class Neuron {
    constructor(inputDim) {
        this.weightedSum = 0;
        this.numWeights = inputDim;
        this.w;// pentru Random(-1, 1) this.w = new Array(inputDim); // am ales sa asociez w[indx] cu neuronul din [L-1][indx]
        this.b = 0;
        this.HeWeightInit(inputDim);
    }

    HeWeightInit(inpSize) {
        const stdDev = Math.sqrt(2 / inpSize);
        this.w = Array.from({ length: inpSize }, () => (Math.random() * 2 - 1) * stdDev);
    }

    genW() {
        for (let i = 0; i < this.numWeights; ++i) {
            this.w[i] = random(-1, 1);
        }
    }

    feed(input) { // compute weighted sum
        this.weightedSum = 0;
        for (let i = 0; i < this.numWeights; ++i) {
            this.weightedSum += (this.w[i] * input[i]);
        }
        this.weightedSum += this.b;
    }
// ----------- Getters --------------
    getWeightedSum() {
        return this.weightedSum;
    }

    getWeight(index) {
        return this.w[index];
    }

    getBias() {
        return this.b;
    }

    getNoWeights() {
        return this.numWeights;
    }
// --------- Setters -----------
    setWeight(index, value) {
        //console.log("w de la[" + index + "] devine din : " + this.w[index] + " -> " + value);
        this.w[index] = value;
    }

    setBias(value) {
        this.b = value;
    }
}