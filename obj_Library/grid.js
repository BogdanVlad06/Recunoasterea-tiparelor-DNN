// grid ul merge coloana cate coloana si la fiecare coloana ia toate randurile

class Grid {
    constructor(size, width) {
        this.size = size; // No of pixels
        this.width = width; // Total width of grid
        this.grid = mk2DArr(this.size, this.size);
        this.genCells();
    }

    genCells() {
        for (let y = 0; y < this.size; ++y) {
            for (let x = 0; x < this.size; ++x) {
                this.grid[y][x] = new Cell(x,  y, this.width / this.size, 0);
          }
        }
    }

    show() { 
        for (let y = 0; y < this.size; ++y) {
            for (let x = 0; x < this.size; ++x) {
                this.grid[y][x].show();
            }
        }
    }

    inBounds(x, y) {
        return (x >= 0 && x < this.size && y >= 0 && y < this.size);
    }

    colour(x, y) {
        x = Math.floor(x / (this.width / this.size)), y = Math.floor(y / (this.width / this.size));
        if (this.inBounds(x, y)) {
            this.grid[x][y].setVal(1); 
            // lee care porneste din  X,Y si are intensitatea 1, si sa isi piarda din intensitate (importanta redusa)
            let v = [
                {dx: 0, dy: -1, val: 0.75},  // sus
                {dx: 0, dy: 1, val: 0.75},   // jos
                {dx: -1, dy: 0, val: 0.75},  // stânga
                {dx: 1, dy: 0, val: 0.75},   // dreapta
                {dx: -1, dy: -1, val: 0.5},  // stânga sus
                {dx: -1, dy: 1, val: 0.5},   // stânga jos
                {dx: 1, dy: -1, val: 0.5},   // dreapta sus
                {dx: 1, dy: 1, val: 0.5}     // dreapta jos
            ];
            // random(w/ gpt help)
            for (let i = 0; i < v.length; ++i) {
               if(random() > 0.15) {
                    let xV = x + v[i].dx, yV = y + v[i].dy;
                    if (this.inBounds(xV, yV)) {
                        let randomVal = v[i].val + random(-0.15, 0.15); // Randomize the value
                        randomVal = constrain(randomVal, 0, 1);
                        this.grid[xV][yV].setVal(max(this.grid[xV][yV].getVal(), randomVal));
                    }
                }
            }
            // w/o gpt
            /*for (let i = 0; i < v.length; ++i) {
                let xV = x + v[i].dx, yV = y + v[i].dy;
                if (this.inBounds(xV, yV)) {
                    this.grid[xV][yV].setVal(max(this.grid[xV][yV].getVal(), v[i].val));
                }
            } */
        }
    }
// ------------- Getter -------------
    getGrid() {
        return this.grid;
    }
}