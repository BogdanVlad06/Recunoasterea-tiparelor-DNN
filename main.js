let noPixels = 28, imgWidth = 15 * 28;
let grid = new Grid(noPixels, imgWidth);
let canvas;
let predictionButton;
let PredictionText;

function setup() {
    canvas = createCanvas(imgWidth + 1, imgWidth + 1);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
        
    predictionButton = createButton('Predict');
    PredictionText = createDiv();
    predictionButton.position(canvas.x, canvas.y - 21);
    predictionButton.mouseClicked(
        function() {
            let input = flatten(grid.getGrid());
            let NN = new NeuralNetwork(input, 3);
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
    grid.colour(mouseX, mouseY);
}

function windowResized() {
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    predictionButton.position(canvas.x, canvas.y - 21);
    PredictionText.position(canvas.x + (imgWidth / 2) - 60, canvas.y + imgWidth);
}