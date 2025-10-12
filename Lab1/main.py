import numpy as np
from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax

#Gracz 2 = AI
PLAYERS = [1, 2]

class ConnectFour(TwoPlayerGame):

    def __init__(self, players):
        self.players = players
        # Plansza ma 6 wierszy i 7 kolumn
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        self.winner = None

    def possible_moves(self):
        # Zwraca listę możliwych do wykonania ruchów (numery kolumn).
        return [str(col) for col in range(7) if self.board[5, col] == 0]

    def make_move(self, move):
        # Wykonuje ruch, umieszczając pionek w wybranej kolumnie.
        col = int(move)
        for row in range(6):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                break

    def is_over(self):
        # Sprawdza, czy gra się zakończyła (wygrana lub remis).
        self.winner = self.find_winner()
        return self.winner is not None or len(self.possible_moves()) == 0

    def show(self):
        # Wyświetla aktualny stan planszy w czytelny sposób.
        print("\n  0 1 2 3 4 5 6")
        print("-----------------")
        for row in reversed(range(6)):
            row_str = "|"
            for col in range(7):
                if self.board[row, col] == 1:
                    row_str += "X "  # Gracz 1 (Człowiek) to 'X'
                elif self.board[row, col] == 2:
                    row_str += "O "  # Gracz 2 (AI) to 'O'
                else:
                    row_str += ". "  # Puste pole
            print(row_str + "|")
        print("-----------------")

    def scoring(self):
        # Ocena stanu gry dla AI po ruchu Gracza. +100 za wygraną AI, -100 za przegraną, 0 za remis.
        if self.winner == 2:  # AI wygrało
            return 100
        if self.winner == 1:  # Gracz wygrał
            return -100
        return 0  # Remis

    def find_winner(self):
        # Sprawdza, czy któryś z graczy ma cztery pionki w rzędzie. Zwraca numer gracza, który wygrał, lub None.
        for player in PLAYERS:
            # Sprawdzenie w poziomie
            for r in range(6):
                for c in range(4):
                    if all(self.board[r, c + i] == player for i in range(4)):
                        return player

            # Sprawdzenie w pionie
            for r in range(3):
                for c in range(7):
                    if all(self.board[r + i, c] == player for i in range(4)):
                        return player

            # Sprawdzenie na ukos (w prawo w górę)
            for r in range(3):
                for c in range(4):
                    if all(self.board[r + i, c + i] == player for i in range(4)):
                        return player

            # Sprawdzenie na ukos (w lewo w górę)
            for r in range(3):
                for c in range(3, 7):
                    if all(self.board[r + i, c - i] == player for i in range(4)):
                        return player

        return None


if __name__ == "__main__":
    # Negamax(6) oznacza, że AI patrzy 6 ruchów w przód
    ai_algo = Negamax(6)
    game = ConnectFour([Human_Player(), AI_Player(ai_algo)])

    # Aby AI grało jako pirewsze, zamień kolejność:
    # game = ConnectFour([AI_Player(ai_algo), Human_Player()])

    game.play()

    if game.winner:
        print(f"\nZWYCIĘZCA: Gracz {game.winner}")
    else:
        print("\nREMIS!")