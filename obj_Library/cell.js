class Cell {
    constructor(x, y, dp, val){        
        this.x = x;
        this.y = y;
        this.dp = dp;
        this.val = val; // e cuprinsa intre 0 si 1, si indica cat de activat e pixelul (0 deloc; 1 activat maxim)
    }

    show() {
        fill((1 - this.val) * 255); // pentru rep vizuala, voi umple un patrat cu negru daca e activat, sau alb in caz contrat
        square(this.x, this.y, this.dp);
    }

    changeVal(newVal) {
        this.val = newVal;
    }

    getVal() {
        return this.val;
    }
}
