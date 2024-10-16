let dim = 28, dp = 15;
let grid = new Grid(dim, dp);
let canvas;

function setup() {
    canvas = createCanvas(dim * dp + 1, dim * dp + 1);
    grid.genCells();
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
}

function draw() {
    background(220);
    grid.visualise();
}

function mouseDragged() {
    grid.colour(mouseX, mouseY);
}

function windowResized() {
    centerCanvas(canvas, windowWidth, width, windowHeight, height);
}