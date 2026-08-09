# V028 history

V028 tested quantity-preserving permutations of v22 premium SELL slots with a
per-unit lockstep simulator and a public v22 market shadow. The first smoke
implementation exposed and then fixed a simulator error that merged all shed
products into one availability counter. After the fix, structural tests and
complete-game smoke passed.

The final candidates made no action changes in the tested v22, v022c, and
v13-r3 games. The 3-seed, 24-game v22 comparison also had identical cash,
minimum cash, and action traces for all three candidates. V22 remains the
control; these files are retained as a negative market-search result rather
than a replacement submission.
