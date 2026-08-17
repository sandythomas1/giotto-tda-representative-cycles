# Learnings

Concepts, patterns, and AI-native practices picked up while building this project. This is a running curriculum — review it periodically.

<!-- New entries go at the top, most recent first -->

## 2026-08-16 — Probe before you spec
Every requirement in spec 001 was derived from a measurement, not from reading the code and
guessing. Three throwaway probe scripts produced the numbers that became acceptance criteria:
essential bars vanishing under `max_edge_length` (0 features found where 1 exists), 2-D
projection variance loss on torus tube-loops (0.50–0.69 kept vs 0.92–0.98 for the best-fit
plane), and reconstruction cost (2.37 s at n=1500). A spec whose acceptance criteria are
measured numbers is testable; one whose criteria are adjectives is not.
