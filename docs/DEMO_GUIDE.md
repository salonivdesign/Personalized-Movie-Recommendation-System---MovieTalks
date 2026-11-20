# 1–2 Minute Demo Guide

1. **0:00–0:15 — Introduction:** State the project goal: recommend movies from rating patterns using collaborative filtering.
2. **0:15–0:35 — Dataset and EDA:** Show the college dataset and one or two generated charts. Mention the dataset contains 500 original rows and a small catalogue of 8 distinct movie titles.
3. **0:35–1:05 — Algorithms:** Briefly explain user-based CF (similar users), item-based CF (similar rating patterns across movies), and SVD (low-rank factorisation of a user–movie rating matrix).
4. **1:05–1:35 — Prototype:** Run the existing application and demonstrate a recommendation if the application and model are configured. Use genuine output only.
5. **1:35–1:55 — Evaluation and limitations:** Show `docs/evaluation_results.json` or CSV. Mention RMSE and Precision@3 and explain that the dataset is too small and metadata too inconsistent to claim production-quality performance.
6. **1:55–2:00 — Close:** State that a larger, more reliable ratings dataset would be needed for stronger validation.

Do not claim a live feature works unless you have personally run it successfully. If the UI is unavailable during the demo, present the tested evaluation script and actual outputs instead.
