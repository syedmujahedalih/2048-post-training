# Third-party components

- `vendor/2048-ai/`: unmodified `2048.cpp`, `2048.h`, and `platdefs.h` from
  https://github.com/nneonneo/2048-ai, at the revision in `UPSTREAM_COMMIT`.
  MIT license retained in `LICENSE`. Our `bridge.cpp` exposes an explicit search
  depth without changing upstream source. The upstream bitboard cannot represent
  tiles above 32768; our Python bridge rejects such input.
- `vendor/original-2048/`: unmodified `game_manager.js`, `grid.js`, and `tile.js`
  from https://github.com/gabrielecirulli/2048, at `UPSTREAM_COMMIT`. MIT license
  retained in `LICENSE.txt`. Used only for an independent deterministic rule check.
- Strands Decider: https://github.com/strands-labs/strands-decider, Apache-2.0.
  Installed in its own local environment; upstream code and weights are not
  committed here. Its exact source revision is in `DECIDER_UPSTREAM_COMMIT`.
- Teacher demonstrations are generated locally from the simulation and the
  pinned bounded-search teacher. They contain no private user data.
