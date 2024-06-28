#constant(valid, false).
#constant(valid, true).
#modeh(valid_output(const(valid))).
#modeb(1, sameRow(var(cell), var(cell))).
#modeb(1, sameCol(var(cell), var(cell))).
#modeb(1, sameBlock(var(cell), var(cell))).
#modeb(2, cellVal(var(cell), var(element))).