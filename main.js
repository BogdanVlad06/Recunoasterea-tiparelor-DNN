//IMPORTANT!!! coordonatele carteziene(x, y) in p5js canvas sunt INVERSATE x -> y si y -> x.
// din acest motiv voi schimba logicile astfel incat coord x, y la care ma voi referi in cod vor fi corect logic
// in alte cuvinte inversez alocuri;
//-------------------------------------- LOADING DATASET ---------------------------------------
// let mnistData; // In data MINST, pixel alb(255) este echivalent cu activare maxima, adica negrul repr fundal;
// function preload() {
//   // Load the MNIST dataset CSV file
//   mnistData = loadTable('MINST_dataset/mnist_train.csv', 'csv', 'header');
// }
//---------------------------------- PROGRAM ----------------------------------
let noPixels = 28, imgWidth = 15 * noPixels;
let grid = new Grid(noPixels, imgWidth);
let canvas;
let predictionButton;
let PredictionText;
let learningRate = 0.3;

function setup() {
    canvas = createCanvas(imgWidth + 1, imgWidth + 1);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    console.log(grid);
    predictionButton = createButton('Predict');
    PredictionText = createDiv();
    predictionButton.position(canvas.x, canvas.y - 21);
    predictionButton.mouseClicked(
        function() {
            console.log(grid);
            let input = flatten(grid.getGrid());
            let NN = new NeuralNetwork(input, 3, learningRate);
            PredictionText.html('Predicted: ' + NN.getPrediction());
            PredictionText.position(canvas.x + (imgWidth / 2) - 60, canvas.y + imgWidth);
            PredictionText.style('font-size', '32px');
        }
    );
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