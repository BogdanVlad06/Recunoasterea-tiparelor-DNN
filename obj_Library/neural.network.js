class NeuralNetwork {
    constructor(valueArr, noHiddenLayers) {
        this.neuronArr_perLayer = new Array(noHiddenLayers); // Must be initialized
        this.outputNeuronArr = new Array(10);
        this.prediction = this.forwardOutputLayer(this.forwardHiddenLayers(valueArr, 1, noHiddenLayers));
    }

    // (array of activations), (current currentLayer number), (No of HidLayers); 
    forwardHiddenLayers(input, currentLayer, noHiddenLayers) {
        let newNoNeurons = Math.floor(input.length / 2);        // the new number of neurons for the current currentLayer
        let activationArr = new Array(newNoNeurons);    // create the array of activations, which is the "output" of a currentLayer
        this.neuronArr_perLayer[currentLayer - 1] = new Array(newNoNeurons);   // create the array of neurons for the current currentLayer
        for (let i = 0; i < newNoNeurons; ++i) {
            this.neuronArr_perLayer[currentLayer - 1][i] = new Neuron(input);
            activationArr[i] = this.neuronArr_perLayer[currentLayer - 1][i].getActivation();
        }
        
        if (currentLayer >= noHiddenLayers) { // conditia de final;
            return activationArr;
        } else {
            return this.forwardHiddenLayers(activationArr, currentLayer + 1, noHiddenLayers);
        }

    }

    forwardOutputLayer(input) {
        for (let i = 0; i < 10; ++i) {
            this.outputNeuronArr[i] = new Neuron(input);
        }
        
        let prediction = -1, maxAct = 0;

        for (let i = 0; i < 10; ++i) {
            let digitActivation =  this.outputNeuronArr[i].getActivation();
            if (maxAct < digitActivation) {
                maxAct = digitActivation;
                prediction = i;
            }
        }
        
        return prediction;
    }
    
    getPrediction() {
        return this.prediction;
    }
}