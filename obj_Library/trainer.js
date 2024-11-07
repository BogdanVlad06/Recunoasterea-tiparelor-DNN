class Trainer {
    constructor(network, learningRate) {
        this.network = new NeuralNetwork(3, 3, 3, 3); // network;
        this.learningRate = learningRate;
    }

    train(inputsArr, labelsArr, numCycles) {
        for (let epoch = 0; epoch < numCycles; ++epoch) {
            let epochLoss = 0, numInputs = inputsArr.length;
            for (let inputInd = 0; inputInd < numInputs ; ++inputInd) {
                let input = inputsArr[inputInd], label = labelsArr[inputInd]; 
            //Forward
                this.network.feedForward(input);
                
                this.network.computeCaseProb();
                
                let exampleLoss = this.network.calculateLoss(label);
                epochLoss += exampleLoss;
            //Backward
                this.network.backpropagate();
                
                this.network.update();
            }
            console.log(epochLoss / numInputs);
        }
    }
}