// Local bridge. The included source is unmodified upstream MIT-licensed code.
#include "2048.cpp"

extern "C" DLL_PUBLIC float score_move_limited(board_t board, int move, int depth) {
    eval_state state;
    state.depth_limit = depth;
    return _score_toplevel_move(state, board, move);
}
