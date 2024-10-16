class Grid {
    constructor(dim, dimPatrat) {
        this.d = dim;
        this.dP = dimPatrat;
        this.grid = mk2DArr(this.d, this.d);
    }

    genCells() {
        for (let i = 0; i < this.d; ++i) {
            for (let j = 0; j < this.d; ++j) {
                this.grid[i][j] = new Cell(i * this.dP, j * this.dP, this.dP, 0);
          }
        }
    }

    visualise() {
        for (let i = 0; i < this.d; ++i) {
            for (let j = 0; j < this.d; ++j) {
                this.grid[i][j].show();
            }
        }
    }

    inBounds(x, y) {
        return (x >= 0 && x < this.d && y >= 0 && y < this.d);
    }

    colour(x, y) {
        x = Math.floor(x / this.dP), y = Math.floor(y / this.dP);
        if (this.inBounds(x, y)) {
            this.grid[x][y].changeVal(1); 
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
                        this.grid[xV][yV].changeVal(max(this.grid[xV][yV].getVal(), randomVal));
                    }
                }
            }
            // w/o gpt
            /*for (let i = 0; i < v.length; ++i) {
                let xV = x + v[i].dx, yV = y + v[i].dy;
                if (this.inBounds(xV, yV)) {
                    this.grid[xV][yV].changeVal(max(this.grid[xV][yV].getVal(), v[i].val));
                }
            } */
        }
    }

    getGrid() {
        return this.grid;
    }
}