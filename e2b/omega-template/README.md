## Omega E2B Template

This template installs the dependencies needed by the algorithm generator
and prefetches OpenML datasets used by `src/algorithm_generator/data_loader.py`.

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
- If you add datasets in `data_loader.py`, update the ID list in
  `e2b/omega-template/e2b.Dockerfile` and rebuild the template.
