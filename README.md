SynCord Linux
=============

This repo has the linux source code modified for SynCord.

We implemented SynCord on top of linux v5.4 with three different underlying
locks. Each branch has respective lock implementation modified for SynCord.

- stock    : Linux v5.4
- cna      : CNA lock ([paper](https://dl.acm.org/doi/10.1145/3302424.3303984),
  [LWN](https://lwn.net/Articles/798629/))
- shfllock : ShflLock ([paper](https://dl.acm.org/doi/10.1145/3341301.3359629))
