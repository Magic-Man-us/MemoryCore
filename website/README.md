# MemoryCore website

Static product site for MemoryCore, built with Vite and deployed through GitHub Pages.

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm ci
npm run build
```

The production artifact is written to `website/dist/`. The repository Pages workflow
uploads that directory and deploys it at the project base path `/MemoryCore/`.
