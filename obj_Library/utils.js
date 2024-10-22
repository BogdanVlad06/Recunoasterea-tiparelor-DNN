function mk2DArr(rows, cols) {
    let arr = new Array(rows);
    for (let i = 0; i < rows; ++i) {
        arr[i] = new Array(cols);
    }
    return arr;
}

function flatten(grid) {
    let flattenGrid = new Array(grid.length ** 2);
    for (let i = 0; i < grid.length; ++i) {
        for (let j = 0; j < grid.length; ++j) {
            flattenGrid[i * grid.length + j] = grid[i][j].getVal();    
        }
    }
    return flattenGrid;
}

function logArr(arr) {
    for (let i = 0; i < arr.length; ++i) {
        console.log(arr[i] + ' ');
    }
}

function sigmoid(number) {
    return Math.exp(number) / (1 + Math.exp(number));
}

function softmax(activ, outputActivExpSum) {
    return Math.exp(activ) / outputActivExpSum;
}

function centerCanvas(canvas, wW, w, wH, h) {
    let x = (wW - w) / 2;
    let y = (wH - h) / 2;
    canvas.position(x, y);  
}

function dataPixelToValue(pixelColorArr) {
    let convertedValuesArr = new Array(pixelColorArr.length);
    let white = 255;
    for (let i = 0; i < pixelColorArr.length; ++i) {
        convertedValuesArr[i] = pixelColorArr[i] / white;
        // 255 e alb(activ maximain MINST) si eu vr sa ii dau val de 1, atunci trb sa impart la 255.
    }
    return convertedValuesArr;
}
/* ---------------------------------- verificare a datelor MINST ----------------------------------
// setup{
    let row = mnistData.getRow(0);  // For example, let's load the first image
    let label = row.getString(0);   // The label (first value in row)
    let pixels = row.arr.slice(1);  // The pixel values (remaining 784 values)
    
    // Convert the pixel array to a 28x28 grid and visualize it
    visualizeImage(pixels, label);
}

function visualizeImage(pixels, label) {
    background(255);  // White background
    let imgSize = 28;  // 28x28 grid size
    let pixelIndex = 0;
    
    for (let i = 0; i < imgSize; i++) {
        for (let j = 0; j < imgSize; j++) {
            let pixelValue = int(pixels[pixelIndex]);  // Get the pixel value (0-255)
            fill(pixelValue);  // Set the color based on pixel intensity (grayscale)
            rect(j * 10, i * 10, 10, 10);  // Draw the pixel (scaled up for visibility)
            pixelIndex++;
        }
    }
    
    // Display the label
    textSize(32);
    fill(0);
    text(`Label: ${label}`, 10, 300);
}
*/