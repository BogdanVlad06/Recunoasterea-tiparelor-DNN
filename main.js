//---------------------------------- MNIST Dataset table conversion ----------------------------------
//IMPORTANT!!! coordonatele carteziene(x, y) in p5js canvas sunt INVERSATE x -> y si y -> x.
// din acest motiv voi schimba logicile astfel incat coord x, y la care ma voi referi in cod vor fi corect logic
// in alte cuvinte inversez alocuri;

//---------------------------------- MNIST Dataset table conversion ---------------------------------
let trainData, testData;
//---------------------------------- Variables ----------------------------------
let noPixels = 28, imgWidth = 15 * noPixels;
let grid = new Grid(noPixels, imgWidth);
let canvas;
let brushHardness = 0.15;   
// ----------------------- Buttons ---------------------------
let predictionButton, resetCanvasButton, testButton, trainButton, stopButton, saveButton, loadButton, loadRandomTestImgButton;
let PredictionText;

let brushSlider;
//---------------------------------- NN parameters ----------------------------------
let learningRate = 0.03;
let learningRateDecay = 0.03;
let numEpoch = 50;
let miniBatchSize = 15;
let inputDim = noPixels ** 2;
let outputDim = 10;
let numOfLayers = 3;

//-------------------------------------- LOADING DATASET ---------------------------------------
let trainImages = new Array(), trainLabels = new Array();
let testImages = new Array(), testLabels = new Array();

function preload() {
    trainData = loadTable("MNIST_dataset/mnist_train.csv", "csv", "header");
    testData = loadTable("MNIST_dataset/mnist_test.csv", "csv", "header");
}

// ---------------------- NETWORK -----------------------
let NN;
//---------------------------------- Program ----------------------------------
function setup() {
    processMNISTdata();
    canvas = createCanvas(imgWidth + 50, imgWidth + 50);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    NN = new NeuralNetwork(outputDim, numOfLayers, learningRate);
    
    NN.initializeLayer(1, 16, inputDim, ReLU); // HidL1
    NN.initializeLayer(2, 16, 16, ReLU); // HidL2 
    NN.initializeLayer(3, outputDim, 16, ReLU); // OutL
    
    resetCanvasButton = createButton("reset Canvas");
    resetCanvasButton.mouseClicked(
        function() {
            grid = new Grid(noPixels, imgWidth);
        }
    );

    trainButton = createButton("train");
    trainButton.mouseClicked(
        function() {
            noLoop();
            NN.stopTraining(false);
            NN.train(learningRate, learningRateDecay, trainImages, trainLabels, numEpoch, miniBatchSize);
            loop();
        }
    );

    stopButton = createButton("Stop Training");
    stopButton.mouseClicked(function() {
        NN.stopTraining(true);
    });

    testButton = createButton("test");
    testButton.mouseClicked(
        function() {
            NN.test(testImages, testLabels);
        }
    );

    predictionButton = createButton('Predict');
    PredictionText = createDiv();
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
            NN.saveNetworkConfigToFile("SavedNetworkConfig.json");
        }
    );

    loadButton = createButton("Load");
    loadButton.mouseClicked(
        function() {
            NN.loadNetworkConfigFromFile("obj_Library/SavedNetworkConfig.json");
        }
    );

    loadRandomTestImgButton = createButton("Random test image");
    loadRandomTestImgButton.mouseClicked(
        function() {
            let randomIndex = Math.floor(Math.random() * testImages.length);
            console.log(randomIndex, testLabels[randomIndex]);
            let bidimArr = MNISTto2DArr(testImages[randomIndex]);
            grid.setGrid(bidimArr);
        }
    );

    brushSlider = createSlider(0.01, 0.5, brushHardness, 0.01);
    brushSlider.size(80);
    
    lossText = createDiv("Loss: 0").style('font-size', '16px').position(10, height - 40);
    accuracyText = createDiv("Accuracy: 0%").style('font-size', '16px').position(10, height - 20);
    windowResized();
}

function drawOutputLayer(outputValues) { // Visualize the output layer as a series of squares
    const squareSize = 20; // Size of each square
    const spacing = 5; // Spacing between squares
    const startX = imgWidth + 10; // Starting X position (to the right of the grid)
    const startY = 10; // Starting Y position

    for (let i = 0; i < outputValues.length; i++) {
        const value = outputValues[i];
        const fillHeight = value * squareSize; // Height of the filled part of the square

        // Draw the square outline
        noFill();
        rect(startX, startY + i * (squareSize + spacing), squareSize, squareSize);

        // Draw the filled part from bottom to top
        fill(0); // Use a single color for the "liquid" effect
        rect(startX, startY + i * (squareSize + spacing) + (squareSize - fillHeight), squareSize, fillHeight);
        // Draw the corresponding digit next to the square
        fill(0);
        textSize(16);
        text(i, startX + squareSize + 10, startY + i * (squareSize + spacing) + squareSize / 2 + 5);
    }
}

function draw() {
    background(255);
    brushHardness = brushSlider.value();
    grid.show();
    
    let input = flatten(grid.getGrid());
    NN.feedForward(input);
    NN.computeCaseProb();
    NN.predict();
    PredictionText.html('Predicted: ' + NN.getPrediction());
    PredictionText.position(canvas.x + (imgWidth / 2) - 60, canvas.y + imgWidth);
    PredictionText.style('font-size', '32px');

    drawOutputLayer(NN.getCaseProbabilities());
}

function mouseDragged() { 
    grid.colour(mouseY, mouseX, brushHardness); // mouseX se refera la Y (mergand pe axa oY) analog pt mouseY;
}

function windowResized() {
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    // Reposition all buttons relative to the canvas
    let xOffset = 0, yOffset = 21; 
    resetCanvasButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += resetCanvasButton.width;

    trainButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += trainButton.width;   

    testButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += testButton.width;

    saveButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += saveButton.width;

    loadButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += loadButton.width;

    loadRandomTestImgButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += loadRandomTestImgButton.width;

    predictionButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += predictionButton.width;

    PredictionText.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset = 0;
    yOffset += 21;

    brushSlider.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += resetCanvasButton.width;

    stopButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += stopButton.width;
    
    lossText.position(canvas.x , canvas.y + imgWidth + 10);
    accuracyText.position(canvas.x, canvas.y + imgWidth + 30);
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
    for (let i = 0; i < testData.getRowCount(); i++) {
        let row = testData.getRow(i).arr;
    
        // First value is the label
        let label = Number(row[0]);
        testLabels.push(label);
    
        // Rest are pixel values
        let image = row.slice(1).map(x => Number(x) / 255);
        testImages.push(image);
    }
}

function updateMetrics(testLoss, accuracy) {
    if (testLoss != 0) {
        lossText.html("Loss: " + testLoss.toFixed(4));
        accuracyText.html("Accuracy: " + (accuracy * 100).toFixed(2) + "%");
    }
}

// PROBLEM -se schimba valorile si devin negative mari