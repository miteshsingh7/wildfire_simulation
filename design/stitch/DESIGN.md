---
name: Tactical Incident Command Interface
colors:
  surface: '#051424'
  surface-dim: '#051424'
  surface-bright: '#2c3a4c'
  surface-container-lowest: '#010f1f'
  surface-container-low: '#0d1c2d'
  surface-container: '#122131'
  surface-container-high: '#1c2b3c'
  surface-container-highest: '#273647'
  on-surface: '#d4e4fa'
  on-surface-variant: '#bbc9cc'
  inverse-surface: '#d4e4fa'
  inverse-on-surface: '#233143'
  outline: '#859396'
  outline-variant: '#3c494b'
  surface-tint: '#3dd9ed'
  primary: '#45def3'
  on-primary: '#00363d'
  primary-container: '#00c2d6'
  on-primary-container: '#004b53'
  inverse-primary: '#006874'
  secondary: '#45dfa4'
  on-secondary: '#003825'
  secondary-container: '#00bd85'
  on-secondary-container: '#00452e'
  tertiary: '#d3c3ff'
  on-tertiary: '#381385'
  tertiary-container: '#b9a2ff'
  on-tertiary-container: '#4b2d98'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#97f0ff'
  primary-fixed-dim: '#3dd9ed'
  on-primary-fixed: '#001f24'
  on-primary-fixed-variant: '#004f58'
  secondary-fixed: '#68fcbf'
  secondary-fixed-dim: '#45dfa4'
  on-secondary-fixed: '#002114'
  on-secondary-fixed-variant: '#005137'
  tertiary-fixed: '#e8ddff'
  tertiary-fixed-dim: '#cebdff'
  on-tertiary-fixed: '#21005e'
  on-tertiary-fixed-variant: '#4f319c'
  background: '#051424'
  on-background: '#d4e4fa'
  surface-variant: '#273647'
typography:
  headline-xl:
    fontFamily: IBM Plex Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: IBM Plex Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: IBM Plex Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: 0em
  headline-sm:
    fontFamily: IBM Plex Sans
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 12px
    letterSpacing: 0.06em
spacing:
  gutter: 0.75rem
  margin: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system delivers an austere, mission-critical operational interface engineered for high-consequence wildland fire evacuation modeling, route computation, and GIS situational telemetry. Targeted at Incident Commanders, Operations Section Chiefs, GIS Technical Specialists, and public safety dispatchers, the interface must prioritize uncompromised legibility under extreme cognitive load, variable field lighting conditions, and severe time compression.

The design movement balances **Tactical Command Brutalism** with **Cartographic Instrument Precision**:
- **Zero Decorative Noise:** Every pixel, boundary, and hue communicates an explicit geographic, temporal, or operational state. No soft airbrushed shadows, ambient glows, or gratuitous gradients.
- **Instrument Density:** High spatial efficiency that maximizes tactical screen real estate for multi-layered vector maps, clearance-time histograms, bottleneck matrices, and stream telemetry.
- **Factual, Non-Sensationalist Tone:** Operational alerts and system telemetry present unvarnished status codes, probabilistic arrival thresholds, and physical blockades without dramatization.

## Colors

The palette is anchored in an ultra-deep slate baseline designed to eliminate screen glare in dim emergency operations centers (EOC) and mobile command vehicles, while maintaining absolute visual separation for vector GIS line symbology.

### Canvas & Surfaces
- **Canvas Base (`#0F1419`):** Near-black deep slate for primary viewports and GIS background raster tiles.
- **Panel Surface (`#171D24`):** Secondary slate elevation for telemetry sidebars, route detail cards, and floating tool clusters.
- **Panel Elevated (`#1F2731`):** Active or hovered container surface.
- **Subtle Surface (`#242D38`):** Inactive controls, table header bands, and trackbeds.
- **Structural Stroke (`#2A323C`):** Strict 1px division boundary separating map viewports, panes, and inspector columns.
- **Focus / Active Stroke (`#475569`):** Enhanced delineation for active inspector panels and focused telemetry rows.

### Evacuation Route Symbology (Colorblind-Safe Hierarchy)
Routes are plotted with deliberate luminance separation and orthogonal chromatic channels to maintain distinction under protanopia, deuteranopia, and tritanopia:
- **Route Alpha (Primary Evacuation Vector):** Pure White (`#FFFFFF`) with solid 3px line width. Max luminance contrast against `#0F1419`.
- **Route Bravo (Secondary Tactical Vector):** Tactical Cyan (`#00C2D6`). High saturation, distinct from landcover layers.
- **Route Charlie (Egress Corridor Green):** Mint / Emerald Green (`#34D399`). High perceptual clarity for cleared or optimal vectors.
- **Route Delta (Auxiliary / Bypass Vector):** Tactical Violet (`#A78BFA`). High wavelength separation from cyan and green channels.
- **Route Inactive / Staged:** Muted Slate Grey (`#64748B`). Low-profile background trace for unassigned or deprecated corridors.

### Hazard & Incident Status Indicators
- **Critical / Burnover Hazard (`#EF4444` / `#991B1B`):** Deep, non-fluorescent vermilion red denoting active flame fronts, crowning behavior, or compromised escape routes.
- **Caution / Roadblock Obstruction (`#F59E0B`):** Amber warning tone indicating down trees, road closures, or dense particulate smoke.
- **Impediment / Abandoned Vehicle (`#94A3B8`):** Muted slate alert for stationary physical choke points and stalled vehicle clusters.
- **Nominal / Cleared State (`#10B981`):** Calibrated field green for verified corridors and operational gates.

## Typography

The typography architecture uses a dual-engine model: **Inter** and **IBM Plex Sans** govern administrative controls, tactical action buttons, and descriptive field notes, while **JetBrains Mono** strictly isolates coordinate data, clearance-time calculations, route hashes, and sensor telemetry.

### Font Roles
- **IBM Plex Sans:** Applied to system pane headers, modal identification, sector titles, and primary action verbs. Provides an authoritative, technical structure.
- **Inter:** Drives general UI controls, descriptive risk briefings, sector descriptions, and route instructions. Optimized for low-fatigue dense scanning.
- **JetBrains Mono:** Dedicated exclusively to numeric and spatial parameters: MGRS/Lat-Long coordinates, timestamps (UTC / Incident Local), remaining capacity figures, vehicle-per-hour metrics, and dynamic distance-to-front calculations. All tabular figures must retain zero-proportional drift during real-time data stream ticks.

### Sizing and Spacing Rules
- All monospace telemetry tokens (`label-sm`, `label-md`) use uppercase formatting for geographic designators (e.g., `SEC-NORTH-4A`, `T-CLEAR: 00:42:15`).
- Font weights are restricted to Regular (400), Medium (500), Semi-Bold (600), and Bold (700) to ensure rendering consistency on low-spec field workstations and ruggedized mobile tactical displays.

## Layout & Spacing

The layout is built upon an operational multi-pane console model designed for large, high-resolution operations monitors (1440p / 4K) with systematic docking configurations for 1080p field tablets.

### Grid & Layout Architecture
- **Console Grid:** A fluid, viewport-locked structure without document-style vertical page scrolling.
- **Left Telemetry Rail:** Fixed 320px width for live incident summary, perimeter velocity, wind vector conditions, and master corridor selector.
- **Center Canvas:** Fluid viewport dedicating maximum spatial continuity to the GIS route analysis map, heat vector overlays, and road network topological graphs.
- **Right Inspector Pane:** Collapsible 360px–420px structural dock for route segment breakdowns, clearance timelines, bottleneck alerts, and resource allocations.
- **Bottom Timeline / Stream Tray:** Fixed 240px drawer (collapsible to 36px) reserved for temporal evacuation progression curves, fire propagation time-sliders, and live dispatch logs.

### Responsive Behavior & Reflow
- **Desktop (1440px+):** Tri-pane concurrent view with map canvas as the central anchor. All telemetry rails open by default.
- **Tablet / Ruggedized Field Display (768px – 1439px):** Map canvas remains top-level. Left and right panels collapse into off-canvas sliding overlays accessed via tactile icon toggles in a persistent 48px top control bar.
- **Mobile Handheld (<768px):** Swaps to a tabbed situational view: Map (Full screen with bottom floating hazard sheet), Telemetry (Single column list), and Queue (Step-by-step route directions and choke points).

## Elevation & Depth

Visual hierarchy does not rely on soft ambient blur or dropshadows, which reduce cartographic precision and wash out vector lines. Depth is achieved via **Tonal Surface Layering** and **Crisp Structural Strokes**.

### Elevation Stack
- **Layer 0 (Base Raster Canvas):** `#0F1419`. Cartography basemap, satellite imagery, slope aspect shading, and unpaved terrain.
- **Layer 1 (Embedded Surfaces & Trays):** `#171D24` with a 1px border of `#2A323C`. Used for fixed sidebar containers, collapsed map trays, and fixed tool shelves.
- **Layer 2 (Floating Instrument Overlays):** `#1F2731` with a 1px border of `#384352`. Map scale instruments, dynamic route profile hovers, and coordinate inspectors.
- **Layer 3 (Critical Alert Modals & Interventions):** `#171D24` surrounded by a rigid 2px border matching the alert status color (e.g., `#EF4444` for active burnover triggers, `#F59E0B` for unverified route breaks). A dark backdrop overlay (`#000000` at 75% opacity) isolates the command action.

## Shapes

The interface employs a **strictly sharp shape language (`roundedness: 0`)**. Corner radii throughout the design system are locked to `0px`.

### Rationale & Rules
- **Tactical Utility:** Zero-radius corners maximize internal panel volume, preserve pixel density along container seams, and maintain alignment with Cartesian GIS coordinates and technical instrumentation.
- **Structural Integrity:** Panels, inputs, badges, notifications, and map overlays fit flush together with 1px dividers, eliminating decorative gaps and visual distortion in dense data tables.
- **Crosshair / Reticle Framing:** Selected waypoints, node vertices, and route endpoints use sharp geometric indicators (diamonds, squares, crosshairs) rather than rounded or circular pins.

## Components

### Buttons & Tactical Triggers
- **Primary Operational Action:** Background `#00C2D6`, text `#0F1419` (`headline-sm`, font-weight 600), border none, padding `0.5rem 1rem`. Active state shifts to `#00A3B5`.
- **Secondary / Action Frame:** Background transparent, text `#E2E8F0`, 1px border `#2A323C`. Hover creates background `#1F2731` with border `#475569`.
- **Critical Action (Evacuate Order / Override):** Background `#991B1B`, text `#FFFFFF`, 1px border `#EF4444`. Focus state adds 2px offset border in `#EF4444`.
- **Dimensions:** Strict 32px height for compact toolbar actions; 40px height for primary execution triggers.

### Chips & Route Badges
- **Route Key Badges:** Monospace `label-md` badge indicating vector IDs (e.g., `RT-01-ALPHA`). Dark slate background `#171D24` bordered by a 2px left-accent line matching the route vector color (`#FFFFFF`, `#00C2D6`, `#34D399`, or `#A78BFA`).
- **Hazard Indicator Chips:**
  - *Burnover Hazard:* Background `#2D1517`, text `#FCA5A5`, border 1px solid `#EF4444`.
  - *Roadblock / Debris:* Background `#2B1D0E`, text `#FCD34D`, border 1px solid `#F59E0B`.
  - *Impediment / Vehicle:* Background `#1E252E`, text `#CBD5E1`, border 1px solid `#64748B`.

### Lists & Telemetry Feeds
- **Segment Lists:** Row items render with alternating subtle zebra-striping (`#171D24` and `#131920`). Border-bottom 1px solid `#2A323C`.
- **Row Hover:** Highlights to `#1F2731` with an immediate highlight of the corresponding vector segment on the GIS canvas.
- **Telemetry Cells:** Numeric entries right-aligned using `JetBrains Mono` (`label-md` or `body-md`) with muted unit indicators (`#64748B`).

### Checkboxes & Segment Toggles
- **Checkboxes:** 14px × 14px sharp square boxes. Border 1px solid `#475569`, background `#0F1419`. Checked state fills with `#00C2D6` displaying an angular solid square core or sharp checkmark.
- **Layer Visibility Toggles:** Toggle switches use a flat rectangular slide design (28px × 14px) with a square thumb indicator; green/cyan active track (`#00C2D6`), muted inactive track (`#242D38`).

### Input Fields & Filter Bars
- **Telemetry Input:** Background `#0F1419`, border 1px solid `#2A323C`, text `#F1F5F9`, typography `label-lg`. Padding `0.375rem 0.5rem`.
- **Active Focus:** Border transitions to `#00C2D6` with zero glow.
- **Prefix / Suffix Metrics:** Integrated monospace unit blocks (e.g., `MPH`, `KM/H`, `VEH/HR`, `DEG`) styled in `#64748B` within the field frame.

### Panels & Inspector Cards
- **Inspector Header:** Background `#1F2731`, border-bottom 1px solid `#2A323C`, padding `0.5rem 0.75rem`. Uses `headline-sm` with right-aligned status lights.
- **Card Body:** Background `#171D24`, padding `0.75rem`. Contains modular data clusters arranged in strict key-value pairs (Key: `label-sm` in `#64748B`; Value: `label-lg` in `#F8FAFC`).

### Domain-Specific Components
- **Clearance Time Histogram:** Step-line histogram with capacity threshold lines. Normal flow represented in `#34D399`, surge flow in `#00C2D6`, and critical bottleneck (>100% capacity) capped in `#EF4444`.
- **Flame Front Proximity Meter:** Segmented 10-bar linear gauge transitioning from slate (`#2A323C`) to caution amber (`#F59E0B`) to critical vermilion red (`#EF4444`) with distance readouts in `JetBrains Mono`.
- **Coordinate Crosshair HUD:** Miniature pinned coordinate overlay in the canvas corner detailing viewport bounding box, camera zoom level, and crosshair lat/long with sub-second recalculation.