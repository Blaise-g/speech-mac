# Local tracker

The coordinator maintains the product requirements in `spec.md` and work items in `issues/<NN>-<slug>.md`. These files are the authoritative local tracker and stay committed; repository ignore rules must not hide them. Feature requirements and tickets are added by the coordinator after the setup handoff and approved Run Plan.

Tickets use one of these statuses:

- `ready-for-agent`: accepted and ready to implement.
- `in-progress`: an implementation worker owns the ticket.
- `done`: implementation, review, and required checks are complete.
- `blocked`: progress needs a specific unresolved dependency or decision.

Commit trailers use the ticket id, for example `Ticket: 00-project-setup`. Implementation workers follow the GPT-6 Luna xhigh `implement-lite` profile and complete independent review before commit.
