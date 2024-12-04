class Trainer {
    constructor(network, learningRate, decay) {
        this.network = network;//new NeuralNetwork(3, 3, 3, 3); // network;
        this.learningRate = learningRate;
        this.decay = decay;
    }
    train(inputsArr, labelsArr, numCycles, batchSize = 10) {
        const numInputs = inputsArr.length;
        
        for (let epoch = 0; epoch < numCycles; ++epoch) {
            let epochLoss = 0, currentLearningRate = this.learningRate / (1 + (this.decay * epoch));
            let correctPredictions = 0;
            this.network.setLearningRate(currentLearningRate);
            // Shuffle data each epoch for better convergence
            const shuffledIndices = [...Array(numInputs).keys()].sort(() => Math.random() - 0.5);
            
            for (let startIdx = 0; startIdx < numInputs; startIdx += batchSize) {
                // Accumulate gradients over the mini-batch
                let batchLoss = 0;
                
                for (let i = startIdx; i < Math.min(startIdx + batchSize, numInputs); ++i) {
                    const index = shuffledIndices[i];
                    const input = inputsArr[index];
                    const label = labelsArr[index];
        
                    this.network.feedForward(input, ReLU);
                    this.network.computeCaseProb();
                    this.network.predict();
                    
                    let prediction = this.network.getPrediction();
                    if (prediction == label) correctPredictions++;

                    let exampleLoss = this.network.calculateLoss(label);
                    batchLoss += exampleLoss;
                    
                    this.network.backpropagate(label, ReLUderivate);
                }

                // Update weights after each mini-batch
                this.network.update(batchSize);

                epochLoss += batchLoss / batchSize;
            }
            let accuracy = correctPredictions / numInputs;
            updateMetrics(epochLoss / (numInputs / batchSize), accuracy);
            console.log(`Epoch ${epoch + 1} Loss: ${epochLoss / (numInputs / batchSize)}, Accuracy: ${(accuracy * 100).toFixed(2)}%`);
        }
    }
    // train(inputsArr, labelsArr, numCycles) {
    //     for (let epoch = 0; epoch < numCycles; ++epoch) {
    //         let epochLoss = 0, numInputs = inputsArr.length;
    //         for (let inputInd = 0; inputInd < numInputs ; ++inputInd) {
    //             if (inputInd % 1000 == 0) {
    //                 console.log('banane');
    //             }
    //             let input = inputsArr[inputInd], label = labelsArr[inputInd]; 
    //         //Forward
    //             this.network.feedForward(input);
                
    //             this.network.computeCaseProb();
                
    //             let exampleLoss = this.network.calculateLoss(label);
    //             //console.log(exampleLoss);
    //             epochLoss += exampleLoss;
    //         //Backward
    //             this.network.backpropagate();
                
    //         }
    //         this.network.update(numInputs);

    //         console.log(epochLoss / numInputs);
    //     }
    // }
}