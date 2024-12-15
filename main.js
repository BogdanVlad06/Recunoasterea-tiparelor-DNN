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
let predictionButton, resetCanvasButton, testButton, trainButton, saveButton, loadButton, loadRandomTestImgButton;
let PredictionText;

let brushSlider;
//---------------------------------- NN parameters ----------------------------------
let learningRate = 0.03;
let numEpoch = 35;
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

//---------------------------------- Program ----------------------------------
function setup() {
    processMNISTdata();
    canvas = createCanvas(imgWidth + 1, imgWidth + 1);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    let NN = new NeuralNetwork(outputDim, numOfLayers, learningRate);
    
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
            NN.train(0.05, trainImages, trainLabels, numEpoch);
        }
    );

    testButton = createButton("test");
    testButton.mouseClicked(
        function() {
            NN.test(testImages, testLabels);
        }
    );

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
}

function draw() {
    background(220);
    brushHardness = brushSlider.value();
    grid.show();
}

function mouseDragged() { 
    grid.colour(mouseY, mouseX, brushHardness); // mouseX se refera la Y (mergand pe axa oY) analog pt mouseY;
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