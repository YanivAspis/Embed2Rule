#constant(player, 1).
#constant(player, 2).
#constant(player, 3).
#constant(player, 4).
#modeh(loses(var(player))).
#modeb(1, var(suit) != var(suit)).
#modeb(2, player_suit(var(player), var(suit)), (positive)).
#modeb(1, player_suit(const(player), var(suit)), (positive)).
#modeb(1, player_rank_higher(var(player), var(player)), (positive)).