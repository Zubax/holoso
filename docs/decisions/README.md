# Decision records

Durable records of the frontend-campaign decisions (see `docs/campaign.md`). One file per decision, named
`<topic>-ruling.md` or `<topic>-ledger.md`, containing: the question, the evidence considered, the positions
(including Codex's, with session ids), the ruling, and its consequences. Records are append-only after their
ruling lands; a reversed decision gets a new record referencing the old one.

Expected during the campaign: `scope-ruling.md` (S1), `h1-ledger.md` (S2.2), `arch-ruling.md` (S3.4).
