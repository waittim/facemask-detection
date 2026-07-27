---
name: WearMask
description: Privacy-first in-browser face mask detection UI
colors:
  bg: "#0b0f19"
  bg-light: "#f8fafc"
  fg: "#f3f4f6"
  fg-light: "#0f172a"
  muted: "#94a3b8"
  muted-light: "#475569"
  accent: "#22d3ee"
  accent-soft: "#38bdf8"
  accent-strong: "#0891b2"
  accent-light: "#0369a1"
  accent-light-strong: "#0c4a6e"
  ok: "#34d399"
  ok-border: "#10b981"
  warn: "#fca5a5"
  danger: "#ef4444"
  danger-strong: "#dc2626"
  primary-from: "#2563eb"
  primary-to: "#0891b2"
  surface: "#111827"
  white: "#ffffff"
typography:
  body:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  title:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "-0.025em"
  label:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1
  micro:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.2
  mono:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
    fontSize: "0.875rem"
    fontWeight: 700
    lineHeight: 1.2
rounded:
  sm: "0.5rem"
  md: "0.75rem"
  lg: "1rem"
  xl: "1.25rem"
  pill: "9999px"
spacing:
  xs: "0.25rem"
  sm: "0.5rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "2rem"
  touch: "2.75rem"
components:
  button-primary:
    backgroundColor: "{colors.primary-from}"
    textColor: "{colors.white}"
    rounded: "{rounded.md}"
    padding: "0.65rem 1.25rem"
  button-primary-hover:
    backgroundColor: "{colors.accent-strong}"
    textColor: "{colors.white}"
  ctrl-btn:
    textColor: "{colors.muted}"
    rounded: "{rounded.pill}"
    padding: "0.55rem 0.95rem"
    height: "{spacing.touch}"
  glass-panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "1.75rem 2rem"
  status-pill:
    rounded: "{rounded.pill}"
    padding: "0.5rem 1.25rem"
    textColor: "{colors.accent-soft}"
---

# Design System: WearMask

## Overview

**Creative North Star: "Local Edge Instrument"**

WearMask’s interface reads like a compact on-device lab instrument: dark by default, cyan telemetry accents, glass panels over a quiet mesh glow, and the live camera stage as the single dominant artifact. Typography stays on the system stack so privacy, performance, and legibility outrank decorative branding.

The system is Operate-first on the demo surface (detect, pause, snapshot, inspect telemetry) and Read-first on documentation pages (about, technical overview, privacy). Light mode is a first-class theme with AA contrast for links and body text—not an afterthought invert.

**Key Characteristics:**
- Camera stage is the hero; chrome stays subordinate
- Cyan = live/local compute; green = mask OK; red = no-mask / error
- System UI type + monospace metrics
- Glass elevation with graceful blur/transparency fallthrough
- Touch targets ≥ 44px; visible `focus-visible` rings

## Colors

Cyan accent signals “running locally.” Semantic green/red encode detection outcomes. Neutrals stay cool slate so panels feel technical without purple-glass SaaS defaults.

### Primary
- **Signal Cyan** (`#22d3ee` dark / `#0369a1` light): links, live badges, control icons, focus rings.
- **Cyan Soft** (`#38bdf8`): status-neutral text, spinner accent.

### Secondary
- **Primary Gradient Blue→Cyan** (`#2563eb` → `#0891b2`): primary CTAs (retry camera, feedback).

### Neutral
- **Void Navy** (`#0b0f19`): dark page background.
- **Paper Slate** (`#f8fafc`): light page background.
- **Panel Ink** (`#111827` / translucent): glass surfaces.
- **Muted** (`#94a3b8` dark / `#475569` light): secondary copy.

### Semantic
- **Mask OK** (`#34d399` / border `#10b981`)
- **No Mask / Danger** (`#ef4444` → `#dc2626`, warn text `#fca5a5`)

### Named Rules
**The Signal Budget Rule.** Cyan is for live state, links, and focus—not large fills. Detection outcomes own green/red exclusively.

## Typography

**Display/Body Font:** system-ui stack (no webfont dependency)
**Mono Font:** ui-monospace stack for FPS, latency, BibTeX, code

**Character:** Utilitarian and calm; weight contrast carries hierarchy instead of a display face.

### Hierarchy
- **Title** (700, ~1.5–1.875rem, tight tracking): page headings on docs; brand wordmark on demo
- **Body** (400, 1rem, 1.5): docs prose; feature card copy ~0.75–0.875rem
- **Label** (600–700, 0.65–0.75rem): feature labels, accel badges
- **Micro** (600, 11px): dense telemetry captions (`text-[11px]`)
- **Mono metric** (700, ~1.375rem): telemetry numbers

### Named Rules
**The No-Webfont Rule.** Do not reintroduce Inter or other CDN fonts without an explicit product decision.

## Layout

- Content max widths: demo `max-w-7xl` (80rem); docs `max-w-4xl` / `max-w-5xl`
- Horizontal padding: `1rem` → `1.5rem` from `sm`
- Workspace: column stack below 1024px; ~2:1 video/telemetry row at `lg`
- Breakpoints in use: `640px` (sm), `1024px` (lg)
- Rhythm: tight control groups; generous separation before feature cards (`mt-14`+)

## Elevation & Depth

Hybrid: translucent panels + light blur, not heavy drop-shadow theater. Status can intensify border glow; prefer offset soft shadows over zero-offset neon halos when adding new elevation.

### Shadow Vocabulary
- **Panel** (`0 10px 30px -5px rgba(0,0,0,0.5)`): glass panels
- **Control bar** (`0 10px 25px rgba(0,0,0,0.5)`): floating toolbar
- **Toast** (`0 10px 25px rgba(0,0,0,0.6)`): ephemeral status

### Named Rules
**The Blur Budget Rule.** `backdrop-filter` is allowed on nav/panel/control chrome; disable under `prefers-reduced-transparency` / `prefers-reduced-motion`, and avoid stacking multiple large blurs.

## Shapes

- Controls: pill (`9999px`) for toolbar buttons; `0.75rem` for theme/lang chips
- Panels / stage: `1rem`–`1.25rem` radii
- Borders: 1px low-contrast hairlines; stage uses 2px semantic border

## Components

### Buttons
- **Primary:** blue→cyan gradient, white text, `0.75rem` radius, cyan-tinted border/shadow
- **Ctrl (toolbar):** ghost pill, cyan icons, `min-height: 2.75rem`
- **Theme/Lang:** elevated chip, same touch minimum
- **Focus:** 2px `var(--wm-focus)` ring, 2px offset

### Cards / Containers
- **Glass panel / feature card:** translucent slate, 1px border, optional hover accent border
- **Metric card:** denser padding, monospace value

### Status
- **Status pill:** neutral / ok / warn variants tied to detection state; announced via `aria-live`
- **Toast:** fixed bottom center, polite live region

### Navigation
- Sticky glass nav; brand mark is not the document `h1` on the demo (sr-only page title instead)
- Docs: breadcrumb + section switcher + footer links

### Signature: Video Stage
- Dominant rounded stage with semantic border (cyan / green / red)
- Loading and camera-error overlays in-place
- Control bar immediately beneath

## Do's and Don'ts

### Do:
- **Do** keep detection semantics on green/red and live chrome on cyan.
- **Do** route new colors through `--wm-*` tokens in `docs/static/css/style.css`.
- **Do** preserve research/demo disclaimer near privacy claims.
- **Do** keep touch targets ≥ `--wm-touch-min` (2.75rem).

### Don't:
- **Don't** load Tailwind Play CDN or Google Fonts for production pages.
- **Don't** put marketing card grids in the first viewport of the live demo.
- **Don't** invent medical efficacy claims in UI copy.
- **Don't** use zero-offset colored glow as the only elevation language for new components.
