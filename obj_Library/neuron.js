class Neuron {
    constructor(input) {
        this.weightedSum = 0;
        this.size = input.length;
        this.b = random(-1, 1);
        this.w = new Array(this.size); // am ales sa asociez w[indx] cu neuronul din [L-1][indx]
        this.genW();
        this.computeSum(input);
    }

    genW() {
        for (let i = 0; i < this.size; ++i) {
            this.w[i] = random(-1, 1);
        }
    }

    computeSum(input) {
        this.weightedSum = 0;
        for (let i = 0; i < this.size; ++i) {
            this.weightedSum += (this.w[i] * input[i]);
        }
        this.weightedSum += this.b;
    }
// ----------- Getters --------------
    getActivation() {
        return sigmoid(this.weightedSum)
    }

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
        return this.size;
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