import math
import os
import time
import copy

# =====================================================================
# 1. ENVIRONMENT FORMALISATION (Week 1 & Week 2 Slides)
# States {S_1, ..., S_i}, Actions {A_1, ..., A_n}, Update P(S'|S, A)
# Fully Observable, Multi-agent, Deterministic, Sequential, Static, Discrete
# =====================================================================

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_1 = 1  # The Player we are advising ('X')
PLAYER_2 = 2  # The Opponent ('O')

class Connect4Environment:
    def __init__(self):
        # State S: 6x7 grid initialised to 0 (empty)
        self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]

    def print_board(self):
        print("\n  0   1   2   3   4   5   6  <- Columns (Actions)")
        print("+---+---+---+---+---+---+---+")
        for r in range(ROWS):
            row_str = "| "
            for c in range(COLS):
                symbol = " "
                if self.board[r][c] == PLAYER_1:
                    symbol = "X"
                elif self.board[r][c] == PLAYER_2:
                    symbol = "O"
                row_str += symbol + " | "
            print(row_str)
            print("+---+---+---+---+---+---+---+")

    def get_valid_actions(self, board):
        """Returns available Actions {A_1, ..., A_n} (columns that are not full)."""
        # Order columns from centre outwards [3, 2, 4, 1, 5, 0, 6]
        # This improves Alpha-Beta pruning efficiency!
        col_order = [3, 2, 4, 1, 5, 0, 6]
        return [c for c in col_order if board[0][c] == EMPTY]

    def transition_model(self, board, col, player):
        """Update function P(S'|S, A): returns new board state S' after action A."""
        new_board = [row[:] for row in board]
        for r in range(ROWS - 1, -1, -1):
            if new_board[r][col] == EMPTY:
                new_board[r][col] = player
                break
        return new_board

    def check_winner(self, board, player):
        """Goal / Terminal state check: 4 in a row horizontally, vertically, or diagonally."""
        # Horizontal check
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(board[r][c + i] == player for i in range(4)):
                    return True
        # Vertical check
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(board[r + i][c] == player for i in range(4)):
                    return True
        # Positive diagonal check
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(board[r + i][c + i] == player for i in range(4)):
                    return True
        # Negative diagonal check
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(board[r - i][c + i] == player for i in range(4)):
                    return True
        return False

    def is_terminal_state(self, board):
        """Checks if the game is won or drawn."""
        return (self.check_winner(board, PLAYER_1) or
                self.check_winner(board, PLAYER_2) or
                len(self.get_valid_actions(board)) == 0)


# =====================================================================
# 2. INFORMED SEARCH & HEURISTIC FUNCTION h(n) (Week 2 & Week 3 Slides)
# Handcrafted evaluation function + Sigmoid Win Probability
# =====================================================================

class Connect4PredictorAI:
    def __init__(self, env):
        self.env = env
        self.nodes_expanded = 0
        self.search_tree = None  # list of node dicts when tree recording is on

    def evaluate_window(self, window, player):
        """Scores a 4-cell window based on piece alignments."""
        score = 0
        opp_player = PLAYER_2 if player == PLAYER_1 else PLAYER_1

        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(EMPTY) == 1:
            score += 10
        elif window.count(player) == 2 and window.count(EMPTY) == 2:
            score += 3

        # Penalise opponent threats
        if window.count(opp_player) == 3 and window.count(EMPTY) == 1:
            score -= 12

        return score

    def heuristic_evaluation(self, board, player):
        """
        Heuristic function h(n) (Week 2): Estimates how close state n is to a goal state.
        Positive score favours Player 1 ('X'); Negative score favours Player 2 ('O').
        """
        score = 0

        # 1. Centre column preference (historically proven optimal by Allis, 1988)
        centre_array = [board[r][COLS // 2] for r in range(ROWS)]
        centre_count = centre_array.count(player)
        score += centre_count * 4

        # 2. Horizontal windows
        for r in range(ROWS):
            row_array = board[r]
            for c in range(COLS - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, player)

        # 3. Vertical windows
        for c in range(COLS):
            col_array = [board[r][c] for r in range(ROWS)]
            for r in range(ROWS - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, player)

        # 4. Diagonal windows
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, player)
                window_neg = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window_neg, player)

        return score

    def score_to_win_probability(self, heuristic_score, scale=25.0):
        """
        Uses the Logistic (Sigmoid) Function from Week 3 Slides (Page 26):
        p(x) = 1 / (1 + e^(-(x - mu) / s)) where mu = 0.
        Converts heuristic score into a Win Probability for Player 1 (0% to 100%).
        """
        # Clamp extreme terminal scores to avoid math overflow
        clamped_score = max(-500, min(500, heuristic_score))
        prob = 1.0 / (1.0 + math.exp(-clamped_score / scale))
        return round(prob * 100, 1)

    # =================================================================
    # 3. SEARCH ALGORITHMS: MINIMAX & ALPHA-BETA PRUNING
    # Returns: (heuristic_score, predicted_move_sequence)
    # =================================================================

    def _record_node(self, parent_id, move, maximizing_player, depth_from_root):
        """Adds a node to the search tree (only when tree recording is switched on)."""
        if self.search_tree is None:
            return None
        node_id = len(self.search_tree)
        self.search_tree.append({
            "id": node_id,
            "parent": parent_id,
            "move": move,                  # (player, col) that led to this state, None for root
            "is_max": maximizing_player,   # whose turn it is AT this node
            "level": depth_from_root,
            "score": None,
            "children": [],
            "best_child": None,
            "pruned": False,
            "leaf_type": None,             # 'heuristic', 'win', 'loss', 'draw' or None
        })
        if parent_id is not None:
            self.search_tree[parent_id]["children"].append(node_id)
        return node_id

    def _record_pruned(self, parent_id, player, cols, maximizing_player, depth_from_root):
        """Marks the columns skipped by an alpha-beta cut-off as pruned branches."""
        if self.search_tree is None:
            return
        for col in cols:
            node_id = self._record_node(parent_id, (player, col), maximizing_player, depth_from_root)
            self.search_tree[node_id]["pruned"] = True

    def minimax_search(self, board, depth, alpha, beta, maximizing_player, use_pruning=True,
                       _parent_id=None, _move=None, _level=0):
        """
        Explores the future game tree up to 'depth' turns ahead.
        Returns both the heuristic score AND the predicted sequence of future moves!
        If self.search_tree is a list, every node visited is recorded for visualisation.
        """
        self.nodes_expanded += 1
        node_id = self._record_node(_parent_id, _move, maximizing_player, _level)
        node = self.search_tree[node_id] if node_id is not None else {}
        valid_actions = self.env.get_valid_actions(board)
        is_terminal = self.env.is_terminal_state(board)

        # Base case: reached depth limit or terminal game-over state
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.env.check_winner(board, PLAYER_1):
                    result, leaf_type = (10000 + depth, []), "win"  # Prefer faster wins
                elif self.env.check_winner(board, PLAYER_2):
                    result, leaf_type = (-10000 - depth, []), "loss"
                else:
                    result, leaf_type = (0, []), "draw"  # Draw
            else:
                result, leaf_type = (self.heuristic_evaluation(board, PLAYER_1), []), "heuristic"
            node["score"], node["leaf_type"] = result[0], leaf_type
            return result

        if maximizing_player:
            max_eval = -math.inf
            best_sequence = []
            for i, col in enumerate(valid_actions):
                next_board = self.env.transition_model(board, col, PLAYER_1)
                eval_score, future_seq = self.minimax_search(
                    next_board, depth - 1, alpha, beta, False, use_pruning,
                    node_id, (PLAYER_1, col), _level + 1
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_sequence = [(PLAYER_1, col)] + future_seq
                    if node_id is not None:
                        node["best_child"] = node["children"][-1]
                if use_pruning:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        self._record_pruned(node_id, PLAYER_1, valid_actions[i + 1:], False, _level + 1)
                        break  # Beta cut-off (pruning suboptimal branches)
            node["score"] = max_eval
            return max_eval, best_sequence

        else:
            min_eval = math.inf
            best_sequence = []
            for i, col in enumerate(valid_actions):
                next_board = self.env.transition_model(board, col, PLAYER_2)
                eval_score, future_seq = self.minimax_search(
                    next_board, depth - 1, alpha, beta, True, use_pruning,
                    node_id, (PLAYER_2, col), _level + 1
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_sequence = [(PLAYER_2, col)] + future_seq
                    if node_id is not None:
                        node["best_child"] = node["children"][-1]
                if use_pruning:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        self._record_pruned(node_id, PLAYER_2, valid_actions[i + 1:], True, _level + 1)
                        break  # Alpha cut-off
            node["score"] = min_eval
            return min_eval, best_sequence

    def predict_future(self, board, current_player, depth=4, use_pruning=True, record_tree=False):
        """
        Runs the predictor and returns a full analysis dictionary.
        If record_tree=True, the explored search tree is stored in self.search_tree
        so it can be drawn with SearchTreeVisualiser.
        """
        self.nodes_expanded = 0
        self.search_tree = [] if record_tree else None
        start_time = time.perf_counter()

        maximizing = (current_player == PLAYER_1)
        score, predicted_line = self.minimax_search(
            board, depth, -math.inf, math.inf, maximizing, use_pruning
        )

        elapsed_time = time.perf_counter() - start_time
        p1_win_prob = self.score_to_win_probability(score)

        return {
            "recommended_move": predicted_line[0][1] if predicted_line else None,
            "predicted_sequence": predicted_line,
            "heuristic_score": score,
            "p1_win_probability": p1_win_prob,
            "p2_win_probability": round(100.0 - p1_win_prob, 1),
            "nodes_expanded": self.nodes_expanded,
            "time_seconds": elapsed_time
        }


# =====================================================================
# 4. SEARCH TREE VISUALISATION (Graphviz)
# Draws the Minimax / Alpha-Beta game tree with the chosen move highlighted
# Requires: pip install graphviz  AND  the Graphviz program (https://graphviz.org/download/)
# =====================================================================

class SearchTreeVisualiser:
    # Colours
    MAX_FILL = "#dbe9ff"     # Player 1 ('X') to move  -> MAX node
    MIN_FILL = "#ffe0e0"     # Player 2 ('O') to move  -> MIN node
    PV_COLOUR = "#2e7d32"    # Principal variation (predicted best line of play)
    CHOSEN_FILL = "#ffd54f"  # The move the AI actually recommends
    PRUNED_COLOUR = "#9e9e9e"

    def __init__(self, search_tree):
        self.tree = search_tree
        self.pv_nodes = self._principal_variation()

    def _principal_variation(self):
        """Follows best_child links from the root to get the predicted line of play."""
        pv = set()
        node_id = 0
        while node_id is not None:
            pv.add(node_id)
            node_id = self.tree[node_id]["best_child"]
        return pv

    def _subtree_size(self, node_id):
        return 1 + sum(self._subtree_size(c) for c in self.tree[node_id]["children"])

    @staticmethod
    def _format_score(node):
        if node["leaf_type"] == "win":
            return "X WINS"
        if node["leaf_type"] == "loss":
            return "O WINS"
        return str(node["score"])

    def _label(self, node):
        if node["move"] is None:
            to_move = "X" if node["is_max"] else "O"
            return f"ROOT ({to_move} to move)\nscore: {self._format_score(node)}"
        player, col = node["move"]
        symbol = "X" if player == PLAYER_1 else "O"
        if node["pruned"]:
            return f"{symbol} -> col {col}\n(pruned)"
        role = "MAX" if node["is_max"] else "MIN"
        if node["leaf_type"] == "heuristic":
            role = "h(n)"
        return f"{symbol} -> col {col}\n{role}: {self._format_score(node)}"

    def build_graph(self, max_levels=3, title=None):
        """
        Builds a graphviz.Digraph of the search tree.
        Levels deeper than max_levels are collapsed into '+N nodes' summaries, except
        along the principal variation, which is always drawn down to the leaf.
        """
        import graphviz

        dot = graphviz.Digraph("SearchTree")
        dot.attr(rankdir="TB", nodesep="0.15", ranksep="0.6")
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")
        dot.attr("edge", arrowsize="0.6")
        if title:
            dot.attr(label=title, labelloc="t", fontsize="16", fontname="Helvetica-Bold")

        chosen_id = self.tree[0]["best_child"]
        stack = [0]
        while stack:
            node_id = stack.pop()
            node = self.tree[node_id]
            on_pv = node_id in self.pv_nodes

            # --- Node styling ---
            attrs = {"fillcolor": self.MAX_FILL if node["is_max"] else self.MIN_FILL}
            if node["pruned"]:
                attrs = {"fillcolor": "#f5f5f5", "color": self.PRUNED_COLOUR,
                         "fontcolor": self.PRUNED_COLOUR, "style": "rounded,dashed,filled"}
            if on_pv:
                attrs.update(color=self.PV_COLOUR, penwidth="2.5")
            if node_id == chosen_id:
                attrs.update(fillcolor=self.CHOSEN_FILL, penwidth="3.5",
                             fontname="Helvetica-Bold")
            dot.node(str(node_id), self._label(node), **attrs)

            # --- Children ---
            hidden = 0
            for child_id in node["children"]:
                child = self.tree[child_id]
                if child["level"] > max_levels and child_id not in self.pv_nodes:
                    hidden += self._subtree_size(child_id)
                    continue
                edge_attrs = {}
                if child_id in self.pv_nodes:
                    edge_attrs = {"color": self.PV_COLOUR, "penwidth": "3"}
                elif child["pruned"]:
                    edge_attrs = {"color": self.PRUNED_COLOUR, "style": "dashed"}
                dot.edge(str(node_id), str(child_id), **edge_attrs)
                stack.append(child_id)

            if hidden:
                summary_id = f"{node_id}_hidden"
                dot.node(summary_id, f"+{hidden} nodes", shape="plaintext",
                         style="", fontcolor="#757575", fontsize="9")
                dot.edge(str(node_id), summary_id, style="dotted", color="#bdbdbd", arrowhead="none")

        # --- Legend ---
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Key", fontsize="11", style="rounded", color="#bdbdbd")
            legend.node("legend_chosen", "Chosen move", fillcolor=self.CHOSEN_FILL,
                        color=self.PV_COLOUR, penwidth="3.5")
            legend.node("legend_pv", "Predicted best line", fillcolor="white",
                        color=self.PV_COLOUR, penwidth="2.5")
            legend.node("legend_max", "MAX node (X to move)", fillcolor=self.MAX_FILL)
            legend.node("legend_min", "MIN node (O to move)", fillcolor=self.MIN_FILL)
            legend.node("legend_pruned", "Pruned branch", fillcolor="#f5f5f5",
                        color=self.PRUNED_COLOUR, fontcolor=self.PRUNED_COLOUR,
                        style="rounded,dashed,filled")
            # Invisible edges just stack the key entries vertically
            for a, b in [("legend_chosen", "legend_pv"), ("legend_pv", "legend_max"),
                         ("legend_max", "legend_min"), ("legend_min", "legend_pruned")]:
                legend.edge(a, b, style="invis")

        return dot

    def render(self, filename="search_tree", max_levels=3, title=None, fmt="png", view=True):
        """
        Renders the tree to an image. If the Graphviz program isn't installed, the
        .dot source is still saved so it can be pasted into https://dreampuf.github.io/GraphvizOnline
        """
        try:
            import graphviz
        except ImportError:
            print("[Search Tree] The 'graphviz' Python package is missing - run: pip install graphviz")
            return None

        # Save next to this script, regardless of which folder it was run from
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

        dot = self.build_graph(max_levels=max_levels, title=title)
        try:
            path = dot.render(filename, format=fmt, view=view, cleanup=True)
            print(f"[Search Tree] Saved to {path}")
            return path
        except graphviz.ExecutableNotFound:
            if os.path.exists(filename):
                os.remove(filename)  # extension-less source graphviz leaves behind on failure
            dot_path = f"{filename}.dot"
            dot.save(dot_path)
            print("[Search Tree] Graphviz program not found on PATH, so no image was drawn.")
            print(f"[Search Tree] Saved the source to {dot_path} instead - install Graphviz from "
                  "https://graphviz.org/download/ to render it.")
            return dot_path


def show_search_tree(ai, board, current_player, depth=4, max_levels=3,
                     filename="search_tree", view=True):
    """Runs an Alpha-Beta search with tree recording on and draws the result."""
    prediction = ai.predict_future(board, current_player, depth=depth,
                                   use_pruning=True, record_tree=True)
    symbol = "X" if current_player == PLAYER_1 else "O"
    title = (f"Alpha-Beta search tree (depth {depth}, {symbol} to move) - "
             f"chosen column {prediction['recommended_move']}, "
             f"{prediction['nodes_expanded']} nodes expanded")
    SearchTreeVisualiser(ai.search_tree).render(filename, max_levels=max_levels,
                                                title=title, view=view)
    return prediction


# =====================================================================
# 5. CASE STUDY BENCHMARK & INTERACTIVE ADVISOR
# =====================================================================

def run_report_experiments():
    """
    Generates performance data comparing Standard Minimax vs Alpha-Beta Pruning
    across depths 1 to 5 so the team can copy-paste the table into the report!
    """
    print("=" * 70)
    print("CASE STUDY EXPERIMENT: STANDARD MINIMAX VS. ALPHA-BETA PRUNING")
    print("=" * 70)

    env = Connect4Environment()
    ai = Connect4PredictorAI(env)

    # Set up a realistic mid-game test state (6 moves played)
    moves = [(3, PLAYER_1), (3, PLAYER_2), (4, PLAYER_1),
             (2, PLAYER_2), (4, PLAYER_1), (4, PLAYER_2)]
    for col, player in moves:
        env.board = env.transition_model(env.board, col, player)

    print("Mid-Game Test Board State (Player 1 'X' to move):")
    env.print_board()

    print(f"\n{'Depth':<7} | {'Method':<18} | {'Best Col':<8} | {'Win Prob (P1)':<14} | {'Nodes Expanded':<15} | {'Time (s)':<10}")
    print("-" * 82)

    for depth in [2, 3, 4, 5]:
        # 1. Standard Minimax (No Pruning)
        res_mm = ai.predict_future(env.board, PLAYER_1, depth=depth, use_pruning=False)
        print(f"{depth:<7} | {'Standard Minimax':<18} | Col {res_mm['recommended_move']:<4} | "
              f"{res_mm['p1_win_probability']:>11.1f}%  | {res_mm['nodes_expanded']:<15} | {res_mm['time_seconds']:.5f}s")

        # 2. Minimax + Alpha-Beta Pruning
        res_ab = ai.predict_future(env.board, PLAYER_1, depth=depth, use_pruning=True)
        print(f"{depth:<7} | {'Alpha-Beta Pruned':<18} | Col {res_ab['recommended_move']:<4} | "
              f"{res_ab['p1_win_probability']:>11.1f}%  | {res_ab['nodes_expanded']:<15} | {res_ab['time_seconds']:.5f}s")
        print("-" * 82)

    # Show the predicted future move sequence at Depth 5
    seq_str = " -> ".join([f"P{p}:Col {c}" for p, c in res_ab["predicted_sequence"]])
    print(f"\n[AI Move Forecast (Next 5 Turns)]: {seq_str}")
    print(f"[AI Evaluation]: Player 1 ('X') has a {res_ab['p1_win_probability']}% chance of winning from this state.\n")

    # Draw the Alpha-Beta search tree for this test state (for the report)
    show_search_tree(ai, env.board, PLAYER_1, depth=4, max_levels=2,
                     filename="search_tree_midgame", view=False)
    print()


def interactive_advisor_mode():
    """Allows players to play Connect 4 while the AI predicts future moves at every turn."""
    env = Connect4Environment()
    ai = Connect4PredictorAI(env)
    current_player = PLAYER_1
    search_depth = 5

    print("=" * 70)
    print("INTERACTIVE CONNECT 4 AI MOVE PREDICTOR & ADVISOR")
    print("Enter a column (0-6) to drop a piece, 't' to view the AI's search tree, or 'q' to quit.")
    print("=" * 70)

    while not env.is_terminal_state(env.board):
        env.print_board()
        symbol = "X" if current_player == PLAYER_1 else "O"

        # AI generates prediction for the current state
        prediction = ai.predict_future(env.board, current_player, depth=search_depth, use_pruning=True)
        seq_formatted = " -> ".join([f"{'X' if p==1 else 'O'}:Col {c}" for p, c in prediction["predicted_sequence"]])

        print(f"\n--- AI PREDICTOR ANALYSIS (Lookahead Depth = {search_depth}) ---")
        print(f" * Win Probability : Player 1 (X): {prediction['p1_win_probability']}% | Player 2 (O): {prediction['p2_win_probability']}%")
        print(f" * Recommended Move: Drop in Column {prediction['recommended_move']} to maximise outcome")
        print(f" * Predicted Future: {seq_formatted}")
        print(f" * Search Metrics  : {prediction['nodes_expanded']} states explored in {prediction['time_seconds']:.4f}s")

        user_input = input(f"\nPlayer {current_player} ({symbol}), choose column (0-6) [AI suggests {prediction['recommended_move']}]: ").strip()
        if user_input.lower() == 'q':
            print("Exiting interactive mode.")
            break

        if user_input.lower() == 't':
            # Full depth-5 search, showing the top 2 levels plus the whole predicted line
            show_search_tree(ai, env.board, current_player, depth=search_depth, max_levels=2)
            continue

        if not user_input.isdigit() or int(user_input) not in env.get_valid_actions(env.board):
            print("Invalid column! Please pick a non-full column between 0 and 6.")
            continue

        col = int(user_input)
        env.board = env.transition_model(env.board, col, current_player)

        if env.check_winner(env.board, current_player):
            env.print_board()
            print(f"\nGame Over! Player {current_player} ({symbol}) wins!")
            return

        current_player = PLAYER_2 if current_player == PLAYER_1 else PLAYER_1


if __name__ == "__main__":
    # 1. Run the automated benchmark for your report
    run_report_experiments()

    # 2. Launch interactive predictor mode
    choice = input("Would you like to test the Interactive Advisor Mode? (y/n): ").strip().lower()
    if choice == 'y':
        interactive_advisor_mode()