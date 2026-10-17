import math
import random

ROWS = 6
COLS = 7
EMPTY = "."
HUMAN = "X"
BOT = "O"
PLAYERS = (HUMAN, BOT)
SEARCH_DEPTH = 5


def create_board():
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def print_board(board):
    print()
    for row in board:
        print(" ".join(row))
    print(" ".join(str(c + 1) for c in range(COLS)))
    print()


def is_valid_move(board, col):
    return 0 <= col < COLS and board[0][col] == EMPTY


def drop_piece(board, col, piece):
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == EMPTY:
            board[row][col] = piece
            return row
    return -1


def is_winner(board, piece):
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in directions:
                if all(
                    0 <= r + i * dr < ROWS
                    and 0 <= c + i * dc < COLS
                    and board[r + i * dr][c + i * dc] == piece
                    for i in range(4)
                ):
                    return True
    return False


def is_board_full(board):
    return all(cell != EMPTY for cell in board[0])


def get_valid_moves(board):
    order = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))
    return [c for c in order if is_valid_move(board, c)]


def score_window(window, piece):
    opponent = HUMAN if piece == BOT else BOT
    mine = window.count(piece)
    theirs = window.count(opponent)
    empty = window.count(EMPTY)

    if mine == 4:
        return 100
    if mine == 3 and empty == 1:
        return 5
    if mine == 2 and empty == 2:
        return 2
    if theirs == 3 and empty == 1:
        return -4
    return 0


def score_position(board, piece):
    score = 0

    centre = [board[r][COLS // 2] for r in range(ROWS)]
    score += centre.count(piece) * 3

    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in directions:
                end_r, end_c = r + 3 * dr, c + 3 * dc
                if 0 <= end_r < ROWS and 0 <= end_c < COLS:
                    window = [board[r + i * dr][c + i * dc] for i in range(4)]
                    score += score_window(window, piece)
    return score


def minimax(board, depth, alpha, beta, maximising):
    valid_moves = get_valid_moves(board)

    if is_winner(board, BOT):
        return 1_000_000 + depth, None # win sooner = better
    if is_winner(board, HUMAN):
        return -1_000_000 - depth, None  # lose later = better
    if not valid_moves:
        return 0, None
    if depth == 0:
        return score_position(board, BOT), None

    best_col = valid_moves[0]
    if maximising:
        best_score = -math.inf
        for col in valid_moves:
            child = [row[:] for row in board]
            drop_piece(child, col, BOT)
            score, _ = minimax(child, depth - 1, alpha, beta, False)
            if score > best_score:
                best_score, best_col = score, col
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
    else:
        best_score = math.inf
        for col in valid_moves:
            child = [row[:] for row in board]
            drop_piece(child, col, HUMAN)
            score, _ = minimax(child, depth - 1, alpha, beta, True)
            if score < best_score:
                best_score, best_col = score, col
            beta = min(beta, best_score)
            if alpha >= beta:
                break
    return best_score, best_col


def get_bot_move(board):
    _, col = minimax(board, SEARCH_DEPTH, -math.inf, math.inf, True)
    if col is None:
        col = random.choice(get_valid_moves(board))
    return col


def get_move(board, piece):
    while True:
        choice = input(f"Player {piece}, choose a column (1-{COLS}) or 'q' to quit: ").strip()
        if choice.lower() == "q":
            return None
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        col = int(choice) - 1
        if not is_valid_move(board, col):
            print("That column is full or out of range. Try again.")
            continue
        return col


def play():
    board = create_board()
    turn = 0
    print("Welcome to Connect 4!")
    print(f"You are {HUMAN}, the bot is {BOT}.")
    print_board(board)

    while True:
        piece = PLAYERS[turn % 2]
        if piece == BOT:
            print("Bot is thinking...")
            col = get_bot_move(board)
            print(f"Bot plays column {col + 1}.")
        else:
            col = get_move(board, piece)
            if col is None:
                print("Game ended.")
                return

        drop_piece(board, col, piece)
        print_board(board)

        if is_winner(board, piece):
            print("You win!" if piece == HUMAN else "The bot wins!")
            return
        if is_board_full(board):
            print("It's a draw!")
            return

        turn += 1


if __name__ == "__main__":
    play()
