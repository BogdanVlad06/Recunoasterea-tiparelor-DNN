function mk2DArr(rows, cols) {
    let arr = new Array(rows);
    for (let i = 0; i < rows; ++i) {
        arr[i] = new Array(cols);
    }
    return arr;
}