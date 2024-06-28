num_elements(5).
super_set_size(4).
hitting_set_target_size(2).
element(0).
element(1).
element(2).
element(3).
element(4).
index(0..N) :- super_set_size(N+1).
set(S) :- index(S).
pow2(0, 1).
pow2(X, R) :- index(X), pow2(X-1, T), R = 2 * T.
value(0..M) :- super_set_size(N), pow2(N-1, M+1).
num_ones(0, 0).
num_ones(1, 1).
num_ones(X, N) :- value(X), num_ones(X / 2, M), num_ones(X \ 2, L), N = M+L.
num_ones_until(T, I, N) :- template(T), index(I), pow2(I, R), M = T \ R, num_ones(M, N).
set_assignment(I, S) :- index(I), template(T), num_ones_until(T, I, S).
template(T) :- holds(input_template, T).
super_set_entry(I, V) :- holds(input_element(I), V).