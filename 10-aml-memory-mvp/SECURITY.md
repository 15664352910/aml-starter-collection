# Security and evaluation-data handling

- Never commit `.env`, API keys, Leaderboard Keys, or Memory System Keys.
- Add and Search may be protected with `AML_API_KEY`; Health is intentionally public.
- Retrieval is filtered by the exact `user_id` supplied in each request.
- The service does not log request bodies or model credentials.
- Delete the configured SQLite database or Docker volume no later than 30 days after the evaluation finishes, unless written organizer permission says otherwise.
- Report accidental credential exposure by revoking the credential first, then opening a sanitized private report with the maintainer.

