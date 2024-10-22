class NeuralNetwork {
    constructor(valueArr, noHiddenLayers) {
        this.neuronArr_perLayer = new Array(noHiddenLayers + 1); 
        this.activationArr_perLayer = new Array(noHiddenLayers + 1);
        //oare ar fi mai logic sa integrez output-ul in datele de mai sus, sa fie stocate pe indexul NoHiddenLayers?
        this.outputNeuronArr = new Array(10);
        this.outputActivationArr = new Array(10);
    
        this.prediction = -1;

        this.forwardHiddenLayers(valueArr, 1, noHiddenLayers);
        this.forwardOutputLayer(noHiddenLayers);
    }

    // (array of activations), (current currentLayer number), (No of HidLayers); 
    forwardHiddenLayers(input, currentLayer, noHiddenLayers) {
        let newNoNeurons = Math.floor(input.length / 2);        // the new number of neurons for the current currentLayer
        this.neuronArr_perLayer[currentLayer] = new Array(newNoNeurons);   // create the array of neurons for the current layer
        this.activationArr_perLayer[currentLayer] = new Array(newNoNeurons);

        for (let i = 0; i < newNoNeurons; ++i) {
            this.neuronArr_perLayer[currentLayer][i] = new Neuron(input);
            this.activationArr_perLayer[currentLayer][i] = this.neuronArr_perLayer[currentLayer][i].getActivation;
        }
        
        if (currentLayer >= noHiddenLayers) { // conditia de final;
            return;
        } else {
            this.forwardHiddenLayers(this.activationArr_perLayer[currentLayer], currentLayer + 1, noHiddenLayers);
        }

    }

    forwardOutputLayer(noHiddenLayers) {
        let lastHidLayerIndex = noHiddenLayers;
        for (let i = 0; i < 10; ++i) {
            this.outputNeuronArr[i] = new Neuron(this.getLayerActivationArr(lastHidLayerIndex));
            this.outputActivationArr[i] = this.outputNeuronArr[i].getActivation();
        }
    }
    
    predict() {
        let prediction = -1, maxAct = 0;

        for (let i = 0; i < 10; ++i) {
            let digitActivation =  this.outputActivationArr[i];
            if (maxAct < digitActivation) {
                maxAct = digitActivation;
                prediction = i;
            }
        }
        
        this.prediction = prediction;
    }
// ---------------------- GETTERS -----------------------
    getLayerActivationArr(layerIndex) {
        return this.activationArr_perLayer[layerIndex];
    }

    getPrediction() {
        return this.prediction;
    }
}