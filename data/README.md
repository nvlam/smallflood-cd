# Dataset contract

Raw data are immutable and excluded from version control. Each manifest row must identify
the pre-event tensor, post/during-event tensor, binary label, valid mask, event ID, and
optional coherence tensor. Event IDs define train/validation/test isolation.

Expected manifest columns:

- `patch_id`
- `event_id`
- `pre_path`
- `post_path`
- `label_path`
- `valid_mask_path`
- `coherence_path` (optional)
- `split`

