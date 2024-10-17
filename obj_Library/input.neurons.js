class inpNeuron {
    constructor(input) { // 2D array
        this.neuron = new Array(input.length ** 2);
        this.flatten(input);
    }

    flatten(input) {
        for (let i = 0; i < input.length; ++i) {
            for (let j = 0; j < input.length; ++j) {
                this.neuron[i * input.length + j] = input[i][j].getVal();    
            }
        }
    }
// ---------------- Getter ---------------
    getInputNeurons() {
        return this.neuron;
    }
    
}