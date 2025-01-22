// grid ul merge coloana cate coloana si la fiecare coloana ia toate randurile
// x si y vor fii coordonate direct in sistemul grid, nu din canvas
class Grid {
    constructor(size, width) {
        this.cellSize = width / size;
        this.size = size; // No of pixels
        this.width = width; // Total width of grid
        this.grid = mk2DArr(this.size, this.size);
        this.genCells();
    }

    genCells() {
        for (let y = 0; y < this.size; ++y) {
            for (let x = 0; x < this.size; ++x) {
                this.grid[y][x] = new Cell(x,  y, this.cellSize, 0);
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

    drawLine(x0, y0, dx, dy, decay) {
        let step = max(abs(dx), abs(dy));
        if (step != 0) {
            let stepX = dx / step, stepY = dy / step; 
            for (let i = 0; i <= step; ++i) {
                this.colour(Math.floor(y0 + stepY * i), Math.floor(x0 + stepX * i), decay);
            } 
        }
    }

    colour(x, y, decay) {
        if (!this.inBounds(x, y)) {
            return;
        }
        let marked = mk2DArr(this.size, this.size);
            
        let neighbour = [
            {f: 0, s: -1},
            {f: 0, s: 1},
            {f: -1, s: 0},
            {f: 1, s: 0}   
        ];
        let queue = [];
        queue.push({cx : x, cy : y, intensity : 0.88});
        marked[x][y] = true;

        while (queue.length != 0) {
            let {cx, cy, intensity} = queue.shift();
            let newVal = min(1, this.grid[cx][cy].getVal() + intensity);
            if (newVal  > 0.01) {
            this.grid[cx][cy].setVal(newVal);//min(newVal, 1));
                for (let i = 0; i < neighbour.length; ++i) {
                    if (Math.random() > 0.25) {
                        let nx = cx + neighbour[i].f, 
                            ny = cy + neighbour[i].s,
                            nIntensity = intensity * decay;
                        if (this.inBounds(nx, ny) && !marked[nx][ny]) {
                            queue.push({cx : nx, cy : ny, intensity : nIntensity}); // next
                            marked[nx][ny] = true;
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