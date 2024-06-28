element(0).
element(1).
element(2).
element(3).
blockSize(2).
cellIdx(1..M) :- blockSize(N), M = N * N.
blockIdx(1..N) :- blockSize(N).
cell((X, Y)) :- cellIdx(X), cellIdx(Y).block((BX, BY)) :- blockIdx(BX), blockIdx(BY).
sameRow((X, Y), (X, Z)) :- cell((X, Y)), cell((X, Z)), Y != Z.
sameCol((X, Y), (Z, Y)) :- cell((X, Y)), cell((Z, Y)), X != Z.
inBlock((X, Y), (BX, BY)) :- blockSize(N), cell((X, Y)), block((BX, BY)), BX = (X - 1) / N + 1, BY = (Y - 1) / N + 1.
sameBlock(C1, C2) :- inBlock(C1, B), inBlock(C2, B), C1 != C2.
valid_output(true) :- not valid_output(false).
cellVal((X, Y), V) :- holds(cell(X, Y), V).