//---------------------------------- MNIST Dataset table conversion ----------------------------------
//IMPORTANT!!! coordonatele carteziene(x, y) in p5js canvas sunt INVERSATE x -> y si y -> x.
// din acest motiv voi schimba logicile astfel incat coord x, y la care ma voi referi in cod vor fi corect logic
// in alte cuvinte inversez alocuri;

//---------------------------------- MNIST Dataset table conversion ---------------------------------
let trainData, testData;
//---------------------------------- Variables ----------------------------------
let noPixels = 28, 
    cellSize = 15, 
    imgWidth = cellSize * noPixels;
let grid = new Grid(noPixels, imgWidth);
let canvas;
let brushHardness = 0.15;   
let painting = false;

// ----------------------- Buttons ---------------------------
let predictionButton, resetCanvasButton, testButton, trainButton, stopButton, saveButton, loadButton, loadRandomTestImgButton,
    reinitNetworkButton;
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

//-------------------------------------- LOADING DATASET - DEV ONLY ---------------------------------------
let trainImages = new Array(), trainLabels = new Array();
let testImages = new Array(), testLabels = new Array();

function preload() {
    // trainData = loadTable("MNIST_dataset/mnist_train.csv", "csv", "header");
    testData = loadTable("MNIST_dataset/mnist_test.csv", "csv", "header");
}

function processMNISTdata() {
    // for (let i = 0; i < trainData.getRowCount(); i++) {
    //     let row = trainData.getRow(i).arr;
    
    //     // First value is the label
    //     let label = Number(row[0]);
    //     trainLabels.push(label);
    
    //     // Rest are pixel values
    //     let image = row.slice(1).map(x => Number(x) / 255);
    //     trainImages.push(image);
    // }
    // ---------------------- Test data -----------------------
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

// ---------------------- NETWORK -----------------------
let NN;

function initNetwork() {
    NN.initializeLayer(1, 16, inputDim, ReLU); // HidL1
    NN.initializeLayer(2, 16, 16, ReLU); // HidL2 
    NN.initializeLayer(3, outputDim, 16, ReLU); // OutL
}

//---------------------------------- Program ----------------------------------
function setup() {
    processMNISTdata();
    canvas = createCanvas(imgWidth + 400, imgWidth + 100);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    NN = new NeuralNetwork(outputDim, numOfLayers, learningRate);
    
    initNetwork();

    reinitNetworkButton = createButton("Reinit Network");
    reinitNetworkButton.mouseClicked(
        function() {
            initNetwork();
            updateMetrics(NaN, NaN);
        }
    );
    
    resetCanvasButton = createButton("reset Canvas");
    resetCanvasButton.mouseClicked(
        function() {
            grid = new Grid(noPixels, imgWidth);
        }
    );
/* ---------------------- DEV buttons for training and testing -----------------------
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

    */
    testButton = createButton("test");
    testButton.mouseClicked(
        function() {
            NN.test(testImages, testLabels);
        }
    );
    PredictionText = createDiv();
/* ---------------------- Prediction button - when live prediction is in use, the button is pointless-----------------------
    predictionButton = createButton('Predict');
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
*/
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
    brushSlider.size(100);
    
    lossText = createDiv("Loss: NaN").style('font-size', '16px').position(10, height - 40);
    accuracyText = createDiv("Accuracy: NaN%").style('font-size', '16px').position(10, height - 20);
    windowResized();
}

function drawNetwork() { // Visualize the output layer as a series of squares
    const spacingVal = 5;
    const widthVal = 5
    let squareSize = 20; // Size of each square
    let spacing = spacingVal; // Spacing between squares
    let startX = 10 + imgWidth; // Starting X position (to the right of the grid)
    let startY = 10; // Starting Y position

    for (let i = 1; i <= numOfLayers; i++) {
        let layerA;
        if (i == 3) {
            layerA = NN.getCaseProbabilities();
        } else {
            layerA = NN.getLayerActivation(i);
        }
        let squareX = startX + widthVal * (i - 1) * (squareSize + spacing);
        spacing = spacingVal;
        if (i == numOfLayers) {
            spacing = 30;
        } 

        for (let j = 0; j < layerA.length; j++) {
            let val = Math.min(1, layerA[j]);
            let volume = val * squareSize;
            let squareY = startY + j * (squareSize + spacing);
           
            noFill();
           
            // Draw the square outline
            stroke('purple');
            strokeWeight(1.5);
            rect(squareX, squareY, squareSize, squareSize);
            strokeWeight(1);
            stroke(0);
            // Draw the filled part from bottom to top
            fill(0); // Use a single color for the "liquid" effect
            rect(squareX, squareY + (squareSize - volume), squareSize, volume);
            if (i == numOfLayers) {
                fill(0);
                textSize(16);
                text(j, squareX + (squareSize + spacing), squareY + squareSize / 2 + 5);
            }
        }
    }
    // Visualize weights
    for (let i = 1; i < numOfLayers; i++) {
        spacing = spacingVal;
        let currentLayer = NN.network[i + 1];
        let nextLayer = NN.network[i + 1];
        let currentLayerA = currentLayer.getActivationArr();
        let nextLayerA = nextLayer.getActivationArr();
        let weights = currentLayer.getWeightsArr();
        
        let currentSquareX = startX + widthVal * (i - 1) * (squareSize + spacing);
        let nextSquareX = startX + widthVal * i * (squareSize + spacing);
        
        if (i + 1 == numOfLayers) {
            spacing = 30;
        }
        for (let j = 0; j < weights[1].length; j++) {
            let currentSquareY = startY + j * (squareSize + spacingVal);
            for (let k = 0; k < nextLayerA.length; k++) {
                let nextSquareY = startY + k * (squareSize + spacing);
                let weight = weights[k][j];
                strokeWeight(Math.abs(weight) * 2); // Line thickness represents weight magnitude
                stroke(weight > 0 ? 'green' : 'red'); // Color represents positive or negative weight
                line(currentSquareX + squareSize, currentSquareY + squareSize / 2, nextSquareX, nextSquareY + squareSize / 2);
            }
        }
    }

    // Reset stroke settings to default
    strokeWeight(1);
    stroke(0);
}

function draw() {
    background(255);
    brushHardness = brushSlider.value();
    grid.show();
    if (painting) { brush() }
    // live prediction
    let input = flatten(grid.getGrid());
    NN.feedForward(input);
    NN.computeCaseProb();
    NN.predict();
    PredictionText.html('Predicted: ' + NN.getPrediction());
    PredictionText.position(canvas.x + (imgWidth / 2) - 60, canvas.y + imgWidth);
    PredictionText.style('font-size', '32px');
    
    drawNetwork();
}

function mousePressed() {
    painting = true;
}

function mouseReleased() {
    painting = false;
}

function brush() {
    if (mouseX >= 0 && mouseX <= imgWidth && mouseY >= 0 && mouseY <= imgWidth) {
        let gridX = Math.floor(mouseX / cellSize), gridY = Math.floor(mouseY / cellSize);
        let pGridX = Math.floor(pmouseX / cellSize), pGridY = Math.floor(pmouseY / cellSize);
        let deltaX = gridX - pGridX, deltaY = gridY - pGridY;
        let dist = sqrt((deltaX) ** 2 + (deltaY) ** 2);
        if (dist > 0) {
            grid.drawLine(pGridX, pGridY, deltaX, deltaY, min(0.4, brushHardness * 2));
        } else {
            grid.colour(gridY, gridX, brushHardness * 0.99); // mouseX se refera la Y (mergand pe axa oY) analog pt mouseY;
        }    
    }
}

function windowResized() {
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    reinitNetworkButton.position(canvas.x + imgWidth - reinitNetworkButton.width, canvas.y + imgWidth);

    // Reposition all buttons relative to the canvas
    let xOffset = 0, yOffset = 21, spacing = 20; 
    resetCanvasButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += spacing + resetCanvasButton.width;

    // trainButton.position(canvas.x + xOffset, canvas.y - yOffset);
    // xOffset += spacing + trainButton.width;   

    testButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += spacing + testButton.width;

    saveButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += spacing + saveButton.width;

    loadButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += spacing + loadButton.width;

    loadRandomTestImgButton.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += spacing + loadRandomTestImgButton.width;

    // predictionButton.position(canvas.x + xOffset, canvas.y - yOffset);
    // xOffset += spacing + predictionButton.width;

    PredictionText.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset = 0;
    yOffset += 21;

    brushSlider.position(canvas.x + xOffset, canvas.y - yOffset);
    xOffset += spacing + resetCanvasButton.width;

    // stopButton.position(canvas.x + xOffset, canvas.y - yOffset);
    // xOffset += spacing + stopButton.width;
    
    lossText.position(canvas.x , canvas.y + imgWidth + 10);
    accuracyText.position(canvas.x, canvas.y + imgWidth + 30);
}

function updateMetrics(testLoss, accuracy) {
    if (testLoss != 0) {
        lossText.html("Loss: " + testLoss.toFixed(4));
        accuracyText.html("Accuracy: " + (accuracy * 100).toFixed(2) + "%");
    }
}

// PROBLEM -se schimba valorile si devin negative mari