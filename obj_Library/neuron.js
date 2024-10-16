class neuron {
    constructor(input) {
        this.z = 0;
        this.size = input.length;
        this.b = random(-1, 1);
        this.w = new Array(this.size);
        this.genW();
        this.feedforward(input);
    }

    genW() {
        for (let i = 0; i < this.size; ++i) {
            this.w[i] = random(-1, 1);
        }
    }

    activFunction(z) {
        return Math.exp(z) / (1 + Math.exp(z));
    }

    feedforward(input) {
        this.z = 0;
        for (let i = 0; i < this.size; ++i) {
            this.z += (this.w[i] * input[i]);
        }
        this.z += this.b;
        this.z = this.activFunction(this.z);
    }

    getOutput() {
        return this.z;
    }
}