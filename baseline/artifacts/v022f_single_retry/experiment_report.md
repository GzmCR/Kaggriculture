# V022f single-retry ablation report

V022f removes the second `DIG -> retry` attempt from V022e.  It was compared
with the unchanged V022c control on 192 games: 2 candidates × 8 opponents × 6
seeds × 2 seats.

| Candidate | Games | Mean cash | Min cash | W/T/L | DONE | Errors | Invalid | Max p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V022c control | 96 | 125,821.1 | 1,020 | 80/0/16 | 96/96 | 0 | 0 | 0.281 |
| V022f single retry | 96 | 125,390.5 | 1,852 | 78/0/18 | 96/96 | 0 | 0 | 0.274 |

Compared with V022e, deleting the second retry recovered most of the cash
regression: V022e averaged 123,048.1 while V022f averaged 125,390.5.  It also
reduced catch-up actions from 1,001 to 936.  However, V022f still lost two more
games than V022c, so it fails the win-rate gate and is not promoted.

V022f recovery totals:

- visible WEED repairs: 285;
- first retry successes: 217;
- retry failures: 36;
- early releases: 20;
- catch-up actions: 936;
- abandoned transactions: 36.

The result suggests the second retry was a real source of cash damage, but the
remaining adaptive release/suppression path is still not reliably better than
V022c's fixed recovery on the current opponent matrix.
