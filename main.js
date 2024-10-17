let noPixels = 28, imgWidth = 15 * 28;
let grid = new Grid(noPixels, imgWidth);
let canvas;

function setup() {
    canvas = createCanvas(imgWidth + 1, imgWidth + 1);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
    
    let PredictionText = createDiv();
    let predictionButton = createButton('Predict');
    predictionButton.position(canvas.x, canvas.y - 21);
    predictionButton.mouseClicked(
        function() {
            let input = flatten(grid.getGrid());
            let NN = new NeuralNetwork(input, 3);
            let offSet = imgWidth;
            PredictionText.html('Predicted: ' + NN.getPrediction());
            PredictionText.style('font-size', '32px');
            PredictionText.position(canvas.x + (offSet / 2) - 60, canvas.y + offSet);
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
}