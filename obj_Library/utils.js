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