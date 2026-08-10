# Security policy

## What must not be committed

Do not commit `.env` / `.dev.env` files, API keys, JWT secrets, database URLs, CRM or channel tokens, private keys, customer conversations, order records, or screenshots containing personal data.

Use environment variables or the deployment platform's secret manager for credentials. The public demo uses only local retrieval and synthetic or anonymized examples.

## Reporting a vulnerability

Please do not open a public issue for a suspected credential leak or access-control issue. Contact the repository owner privately with the affected route, reproduction steps, and the smallest safe proof of concept.
