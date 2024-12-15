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

    colour(x, y, decay) {
        x = Math.floor(x / (this.width / this.size)), y = Math.floor(y / (this.width / this.size));
        if (this.inBounds(x, y)) {
            let marked = mk2DArr(this.size, this.size);
            
            let neighbour = [
                {f: 0, s: -1},
                {f: 0, s: 1},
                {f: -1, s: 0},
                {f: 1, s: 0}   
            ];
            let queue = [];
            queue.push({cx : x, cy : y, intensity : 1});
            marked[x][y] = 1;

            while (queue.length != 0) {
                let cx = queue[0].cx, cy = queue[0].cy, intensity = queue[0].intensity;
                queue.shift();
                let newVal = this.grid[cx][cy].getVal() + intensity;
                this.grid[cx][cy].setVal(min(newVal, 1));

                for (let i = 0; i < neighbour.length; ++i) {
                    if (Math.random() > 0.25) {
                        let nx = cx + neighbour[i].f, 
                            ny = cy + neighbour[i].s,
                            nIntensity = intensity * decay;
                        if (this.inBounds(nx, ny) && marked[nx][ny] != 1) {
                            queue.push({cx : nx, cy : ny, intensity : nIntensity}); // next
                            marked[nx][ny] = 1;
                        }
                    }
                }
            }
        }
    }
// ------------- Getter -------------
    getGrid() {
        return this.grid;
    }
// ------------- Setter -------------
    setGrid(bidimArr) { 
        for (let i = 0; i < this.size; ++i) {
            for (let j = 0; j < this.size; ++j) {
                this.grid[i][j].setVal(bidimArr[i][j]);
            }
        }
    }
}