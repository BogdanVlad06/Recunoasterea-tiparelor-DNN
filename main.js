let noPixels = 28, imgWidth = 15 * 28;
let grid = new Grid(noPixels, imgWidth);
let canvas;

function setup() {
    canvas = createCanvas(imgWidth + 1, imgWidth + 1);
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
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