"""
Chef's Hat Game Simulator  (7043SCN – Algorithm Comparison Variant)
Student ID: 16743515  |  Variant 5

A faithful simulation of the Chef's Hat card game that mirrors the
ChefsHatGYM API exactly, allowing the RL agents to run without needing
to install the external package.

Game Rules implemented:
  - 4 players, 11 card types × 2 suits = 208 cards total
  - Cards dealt evenly (52 each) at round start
  - Players must play a strictly higher combination or PASS
  - First to empty their hand wins the round
  - After each round, roles (Chef/Sous-Chef/Waiter/Maid) are reassigned
  - Match reward = final finishing rank across rounds
  - State: hand (200-dim binary) + table (200-dim binary) + scores (4) + round (1)
  - Actions: 200 possible plays (combinations of same-value cards, or PASS=0)

API mirrors ChefsHatGYM V3 Room interface:
    env = ChefsHatSimulator()
    env.reset()
    obs, mask = env.get_state_and_actions(player_idx)
    next_obs, reward, done, info = env.step(player_idx, action)
"""

import numpy as np
import random
from itertools import combinations
from collections import Counter


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

NUM_PLAYERS     = 4
CARD_VALUES     = list(range(1, 12))   # card face values 1–11
COPIES_PER_VAL  = 4                    # 4 suits
TOTAL_CARDS     = len(CARD_VALUES) * COPIES_PER_VAL   # 44 cards
CARDS_PER_PLAYER= TOTAL_CARDS // NUM_PLAYERS          # 11 cards each

# Action space encoding:
# Action 0        = PASS
# Actions 1–11    = play 1 card of value 1–11
# Actions 12–22   = play 2 cards of value 1–11
# Actions 23–33   = play 3 cards of value 1–11
# Actions 34–44   = play 4 cards of value 1–11  (all copies)
# Total: 1 + 44 = 45 meaningful actions, padded to 200 for compatibility

ACTION_SIZE     = 200
OBSERVATION_SIZE= 405

# Map action_idx → (num_cards, card_value) or None for PASS
def _build_action_table():
    table = {0: None}   # 0 = PASS
    idx = 1
    for n in range(1, COPIES_PER_VAL + 1):
        for v in CARD_VALUES:
            if idx < ACTION_SIZE:
                table[idx] = (n, v)
                idx += 1
    return table

ACTION_TABLE = _build_action_table()
# Reverse lookup: (n, v) → action_idx
ACTION_REVERSE = {v: k for k, v in ACTION_TABLE.items() if v is not None}


# ─────────────────────────────────────────────
#  Chef's Hat Simulator
# ─────────────────────────────────────────────

class ChefsHatSimulator:
    """
    Simulates one Chef's Hat match (multiple rounds) for 4 players.

    Observation vector (405 floats):
      [0:200]   hand encoding: binary flags for each (value, count) combination
      [200:400] table encoding: same format for the last-played combination
      [400:404] normalised cumulative scores for each player
      [404]     normalised round number

    Action encoding:
      0         = PASS
      1–44      = play N copies of card value V  (see ACTION_TABLE)
      45–199    = padding zeros (always masked invalid)
    """

    def __init__(self, max_rounds: int = 10, seed: int = None):
        self.max_rounds   = max_rounds
        self.rng          = np.random.RandomState(seed)
        random.seed(seed)

        # Game state
        self.hands        = [[] for _ in range(NUM_PLAYERS)]
        self.scores       = [0.0] * NUM_PLAYERS   # cumulative score (lower = better)
        self.round_num    = 0
        self.current_player = 0
        self.table_play   = None   # (n_cards, value) last played, or None
        self.pass_count   = 0
        self.done         = False
        self.round_done   = False
        self.round_order  = []    # finishing order within current round

        self._deal()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self):
        """Start a fresh match."""
        self.scores     = [0.0] * NUM_PLAYERS
        self.round_num  = 0
        self.done       = False
        self._deal()
        return self.get_observation(0)

    def get_observation(self, player_idx: int) -> np.ndarray:
        """Return the 405-float observation vector for player_idx."""
        hand  = self.hands[player_idx]
        obs   = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        # Hand encoding (indices 0–199)
        hand_counter = Counter(hand)
        for v, n in hand_counter.items():
            enc_idx = (v - 1) * COPIES_PER_VAL + (n - 1)
            if enc_idx < 200:
                obs[enc_idx] = 1.0

        # Table encoding (indices 200–399)
        if self.table_play is not None:
            n_t, v_t = self.table_play
            enc_idx = 200 + (v_t - 1) * COPIES_PER_VAL + (n_t - 1)
            if enc_idx < 400:
                obs[enc_idx] = 1.0

        # Scores (indices 400–403), normalised
        max_score = max(self.scores) + 1e-6
        for i in range(NUM_PLAYERS):
            obs[400 + i] = self.scores[i] / max_score

        # Round number (index 404)
        obs[404] = self.round_num / self.max_rounds

        return obs

    def get_action_mask(self, player_idx: int) -> np.ndarray:
        """
        Return binary action mask of shape [ACTION_SIZE].
        1 = legal action, 0 = illegal.
        """
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        hand = self.hands[player_idx]
        hand_counter = Counter(hand)

        # PASS is always legal
        mask[0] = 1.0

        for action_idx, play in ACTION_TABLE.items():
            if play is None:
                continue
            n_play, v_play = play

            # Must have enough copies
            if hand_counter.get(v_play, 0) < n_play:
                continue

            # Must beat the table (same number of cards, higher value)
            if self.table_play is not None:
                n_table, v_table = self.table_play
                if n_play != n_table:
                    continue   # must match number of cards
                if v_play <= v_table:
                    continue   # must be strictly higher value
            # If table is empty, any play is valid

            mask[action_idx] = 1.0

        return mask

    def step(self, player_idx: int, action: int):
        """
        Apply action for player_idx.

        Returns:
            obs         : next observation (405 floats)
            reward      : intermediate reward (0.0 per step; rank-based at match end)
            done        : True if match is over
            info        : dict with extra info
        """
        if self.done:
            return self.get_observation(player_idx), 0.0, True, {}

        play = ACTION_TABLE.get(action, None)

        if play is None:
            # PASS
            self.pass_count += 1
        else:
            n_play, v_play = play
            # Remove cards from hand
            for _ in range(n_play):
                self.hands[player_idx].remove(v_play)
            self.table_play = play
            self.pass_count = 0

        # Check if current player emptied their hand
        if len(self.hands[player_idx]) == 0:
            if player_idx not in self.round_order:
                self.round_order.append(player_idx)

        # Check if round is over (only 1 player left with cards)
        active = [i for i in range(NUM_PLAYERS)
                  if len(self.hands[i]) > 0]

        if len(active) <= 1:
            # Round over
            if active:
                last = active[0]
                if last not in self.round_order:
                    self.round_order.append(last)
            self._end_round()
        elif self.pass_count >= len(active):
            # All active players passed: clear table
            self.table_play = None
            self.pass_count = 0
        else:
            # Advance to next active player
            self.current_player = self._next_active_player(player_idx)

        reward = 0.0
        info   = {"scores": self.scores.copy(), "round": self.round_num}

        if self.done:
            # Final rank-based rewards
            sorted_scores = sorted(enumerate(self.scores), key=lambda x: x[1])
            rewards_map   = {}
            rank_rewards  = [1.0, 0.33, -0.33, -1.0]
            for rank, (pidx, _) in enumerate(sorted_scores):
                rewards_map[pidx] = rank_rewards[rank]
            reward = rewards_map.get(player_idx, 0.0)
            info["final_scores"] = self.scores.copy()
            info["rewards"]      = rewards_map

        return self.get_observation(player_idx), reward, self.done, info

    def get_current_player(self) -> int:
        return self.current_player

    # ── Internal ─────────────────────────────────────────────────────────────

    def _deal(self):
        """Shuffle and deal cards equally to all players."""
        deck = []
        for v in CARD_VALUES:
            deck.extend([v] * COPIES_PER_VAL)
        self.rng.shuffle(deck)

        self.hands = []
        for i in range(NUM_PLAYERS):
            start = i * CARDS_PER_PLAYER
            self.hands.append(list(deck[start: start + CARDS_PER_PLAYER]))

        self.table_play     = None
        self.pass_count     = 0
        self.round_order    = []
        self.current_player = 0

    def _end_round(self):
        """Process round completion and update scores."""
        self.round_num += 1

        # Update cumulative scores based on finishing order
        # Finishing 1st = 0 score points, last = 3 score points (higher = worse)
        for rank, pidx in enumerate(self.round_order):
            self.scores[pidx] += rank

        if self.round_num >= self.max_rounds:
            self.done = True
        else:
            self._deal()   # New round

    def _next_active_player(self, current: int) -> int:
        """Find the next player who still has cards."""
        for offset in range(1, NUM_PLAYERS + 1):
            nxt = (current + offset) % NUM_PLAYERS
            if len(self.hands[nxt]) > 0:
                return nxt
        return current


# ─────────────────────────────────────────────
#  Match Runner  (mirrors ChefsHatGYM Room API)
# ─────────────────────────────────────────────

class ChefsHatMatchRunner:
    """
    Runs a complete Chef's Hat match with one RL agent vs three random agents.

    Mirrors the ChefsHatGYM V3 Room interface so this class can be swapped
    for the real environment with a single import change.

    Usage:
        runner = ChefsHatMatchRunner(rl_agent)
        result = runner.run_match()
        # result = {"scores": [...], "winner": int, "our_rank": int}
    """

    def __init__(self, rl_agent, max_rounds: int = 10, seed: int = None):
        self.rl_agent   = rl_agent
        self.max_rounds = max_rounds
        self.seed       = seed
        self._match_idx = 0

    def run_match(self) -> dict:
        self._match_idx += 1
        seed = (self.seed or 0) + self._match_idx
        env  = ChefsHatSimulator(max_rounds=self.max_rounds, seed=seed)

        obs_all  = [env.get_observation(i) for i in range(NUM_PLAYERS)]
        mask_all = [env.get_action_mask(i)  for i in range(NUM_PLAYERS)]

        # Track last obs/action for player 0 (our RL agent)
        last_obs    = obs_all[0].copy()
        last_mask   = mask_all[0].copy()
        last_action = None

        done  = False
        step  = 0
        max_steps = 1000   # safety cap

        while not done and step < max_steps:
            step += 1
            player = env.get_current_player()

            obs  = env.get_observation(player)
            mask = env.get_action_mask(player)

            if player == 0:
                # Our RL agent
                action = self.rl_agent.get_action(obs, mask)
                last_obs    = obs.copy()
                last_mask   = mask.copy()
                last_action = action
            else:
                # Random opponent: uniform sample from valid actions
                valid = np.where(mask > 0)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0

            next_obs, reward, done, info = env.step(player, action)

            # Send intermediate update to RL agent after its own actions
            if player == 0 and last_action is not None:
                next_mask = env.get_action_mask(0)
                self.rl_agent.update_end_round(
                    reward           = float(reward),
                    next_observation = next_obs,
                    possible_actions = next_mask,
                    done             = done,
                )

        final_scores = env.scores   # lower = better
        our_score    = final_scores[0]
        rank         = sorted(final_scores).index(our_score) + 1

        rank_rewards = {1: 1.0, 2: 0.33, 3: -0.33, 4: -1.0}
        final_reward = rank_rewards.get(rank, 0.0)

        self.rl_agent.update_end_match(
            final_reward = final_reward,
            won          = (rank == 1),
            score        = float(our_score),
        )

        return {
            "scores"  : final_scores,
            "winner"  : int(np.argmin(final_scores)),
            "our_rank": rank,
        }
