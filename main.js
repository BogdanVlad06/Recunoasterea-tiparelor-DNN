//---------------------------------- MINST Dataset table conversion ----------------------------------
//IMPORTANT!!! coordonatele carteziene(x, y) in p5js canvas sunt INVERSATE x -> y si y -> x.
// din acest motiv voi schimba logicile astfel incat coord x, y la care ma voi referi in cod vor fi corect logic
// in alte cuvinte inversez alocuri;

//---------------------------------- MINST Dataset table conversion ---------------------------------
let trainData;
//---------------------------------- Variables ----------------------------------
let noPixels = 28, imgWidth = 15 * noPixels;
let grid = new Grid(noPixels, imgWidth);
let canvas;
// ----------------------- Buttons ---------------------------
let predictionButton, resetGridButton, trainButton, saveButton, loadButton;
let PredictionText;

//---------------------------------- NN parameters ----------------------------------
let learningRate = 0.02;
let numEpoch = 1;
let inputDim = noPixels ** 2;
let outputDim = 10;
let numOfLayers = 3;

//-------------------------------------- LOADING DATASET ---------------------------------------
let trainImages = new Array(), trainLabels = new Array();

function preload() {
    trainData = loadTable("MINST_dataset/mnist_train.csv", "csv", "header");
}

//---------------------------------- Program ----------------------------------
function setup() {
    processMNISTdata();
    canvas = createCanvas(imgWidth + 1, imgWidth + 1);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    let NN = new NeuralNetwork(outputDim, numOfLayers, learningRate);
    
    NN.initializeLayer(1, 16, inputDim, ReLU); // HidL1
    NN.initializeLayer(2, 16, 16, ReLU); // HidL2 
    NN.initializeLayer(3, outputDim, 16, ReLU); // OutL
    
    resetGridButton = createButton("resetCanvas");
    resetGridButton.mouseClicked(
        function() {
            grid = new Grid(noPixels, imgWidth);
        }
    ) 

    trainButton = createButton("train");
    trainButton.mouseClicked(
        function() {
            let networkTrainer = new Trainer(NN, learningRate, 0.05);
            networkTrainer.train(trainImages, trainLabels, numEpoch);
        }
    ) 

    predictionButton = createButton('Predict');
    PredictionText = createDiv();
    predictionButton.position(canvas.x, canvas.y - 21);
    predictionButton.mouseClicked(
        function() {
            let input = flatten(grid.getGrid());
            NN.feedForward(input);
            NN.computeCaseProb();
            NN.predict();
            console.log(NN);
            PredictionText.html('Predicted: ' + NN.getPrediction());
            PredictionText.position(canvas.x + (imgWidth / 2) - 60, canvas.y + imgWidth);
            PredictionText.style('font-size', '32px');
        }
    );

    saveButton = createButton("Save");
    saveButton.mouseClicked(
        function() {
            NN.saveNetworkConfigToFile("obj_Library/SavedNetworkConfig.json");
        }
    )

    loadButton = createButton("Load");
    loadButton.mouseClicked(
        function() {
            NN.loadNetworkConfigFromFile("obj_Library/SavedNetworkConfig.json");
        }
    )

    lossText = createDiv("Loss: 0").style('font-size', '16px').position(10, height - 40);
    accuracyText = createDiv("Accuracy: 0%").style('font-size', '16px').position(10, height - 20);
}

function draw() {
    background(220);
    grid.show();
}

function mouseDragged() { 
    grid.colour(mouseY, mouseX); // mouseX se refera la Y (mergand pe axa oY) analog pt mouseY;
}

function windowResized() {
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    predictionButton.position(canvas.x, canvas.y - 21);
    PredictionText.position(canvas.x + (imgWidth / 2) - 60, canvas.y + imgWidth);
}

function processMNISTdata() {
    for (let i = 0; i < trainData.getRowCount(); i++) {
        let row = trainData.getRow(i).arr;
    
        // First value is the label
        let label = Number(row[0]);
        trainLabels.push(label);
    
        // Rest are pixel values
        let image = row.slice(1).map(x => Number(x) / 255);
        trainImages.push(image);
      }
}

function updateMetrics(epochLoss, accuracy) {
    if (epochLoss != 0) {
        lossText.html("Loss: " + epochLoss.toFixed(4));
        accuracyText.html("Accuracy: " + (accuracy * 100).toFixed(2) + "%");
    }
}

// PROBLEM -se schimba valorile si devin negative mari