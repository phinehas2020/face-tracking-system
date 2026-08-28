# Prototype Instructions

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.

## Durable product decisions

- The selected visual target is **Event Time Machine**: a dense, desktop-first technical rehearsal console on a warm neutral canvas with dark camera wells, IBM Plex typography, orange time/navigation accents, teal verified-state accents, and violet comparison data.
- The primary screen must keep synchronized Lane A, Lane B, and Wide evidence visible together with model-run comparison, an actionable disagreement queue, a dense event timeline, camera health, and persistent replay controls.
- This is an event operations product, not a generic security dashboard. Language centers passages, unique attendees, identity links, rehearsal, camera topology, and reversible operator decisions.
- The core behavior works with the API and degrades to deterministic fixtures for visual review. Case decisions, replay controls, camera selection, overlays, model selection, navigation, and guarded lifecycle controls stay interactive.
- Use Phosphor for icons, Recharts for data visualization, and generated camera imagery from `public/assets/cameras`. Do not introduce emoji, inline SVG, decorative gradients, or generic placeholders.
