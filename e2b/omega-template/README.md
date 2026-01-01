## Omega E2B Template

This template installs the dependencies needed by the algorithm generator,
prefetches OpenML datasets, and caches benchmark splits on disk so evaluation
doesn’t reload data for every request.

### Prerequisites
- Node + npm (for the E2B CLI)
- E2B account

Install the CLI:

```bash
npm i -g @e2b/cli
```

Authenticate:

```bash
e2b auth login
```

### Build the template

From the repo root:

```bash
e2b template create omega-template -p e2b/omega-template -d e2b.Dockerfile
```

This builds a template named `omega-template`.

### Use the template in the app

Set the template name in your environment:

```bash
export E2B_TEMPLATE=omega-template
```

Then restart the API so it picks up the new template.

### Notes
- The Dockerfile prefetches OpenML datasets into `/data/openml_cache`.
- Benchmark splits are cached in `/data/benchmarks` as `.npz` files.
- If you add datasets in `data_loader.py`, update the ID list in
  `e2b/omega-template/e2b.Dockerfile` and rebuild the template.
- `e2b/omega-template/data_loader.py` is a copy of
  `src/algorithm_generator/data_loader.py` used at build time.
