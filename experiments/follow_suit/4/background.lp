rank(two).
rank(three).
rank(four).
rank(five).
rank(six).
rank(seven).
rank(eight).
rank(nine).
rank(ten).
rank(jack).
rank(queen).
rank(king).
rank(ace).
rank_value(two,2).
rank_value(three,3).
rank_value(four,4).
rank_value(five,5).
rank_value(six,6).
rank_value(seven,7).
rank_value(eight,8).
rank_value(nine,9).
rank_value(ten,10).
rank_value(jack,11).
rank_value(queen,12).
rank_value(king,13).
rank_value(ace,14).
suit(diamonds).
suit(spades).
suit(hearts).
suit(clubs).
rank_higher(R1,R2) :- rank_value(R1,V1), rank_value(R2,V2), V1 > V2.
player(1..4).
priority_player(P1, P2) :- player(P1), player(P2), P1 < P2.
player_card(P, R, S) :- holds(player_card(P), R, S).
player_suit(P, S) :- player_card(P, _, S).
player_rank_higher(P1, P2) :- player_card(P1, R1, _), player_card(P2, R2, _), rank_higher(R1, R2).
maybe_winner(P) :- player(P), not loses(P).
actually_loses(P1) :- maybe_winner(P1), maybe_winner(P2), priority_player(P2, P1).
winner(P) :- maybe_winner(P), not actually_loses(P).