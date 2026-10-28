/**
 * Tactical Incident Command Interface - Core Application Logic
 * Pure ES6, zero external CDN dependencies, 100% offline.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    currentTimeMin: 120,
    fieldSource: "surrogate", // 'surrogate' or 'teacher'
    selectedOriginId: null,
    isPlaying: false,
    playIntervalId: null,
    layers: {
      predictedPerimeter: true,
      isochrones: true,
      observedPerimeter: true,
      hazards: true,
      routes: true,
      updates: true,
    },
    methods: {
      agent_a3: true,
      agent_b_on_a3: true,
      agent_a2_assim: true,
      agent_a2_pure: true,
      static: true,
    },
    activeTab: "console",
    // Map Viewport Transformations
    viewBox: { x: 0, y: 0, w: 1000, h: 700 },
    initialViewBox: { x: 0, y: 0, w: 1000, h: 700 },
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    // Table Sorting
    zoneSortColumn: "zone_or_corridor",
    zoneSortAsc: true,
  };

  // Geographic Projection to Web Mercator (EPSG:3857) normalized to SVG viewport
  const PROJECTION = {
    // Camp Fire Paradise bounding box (WGS84 EPSG:4326)
    minLon: -121.75,
    maxLon: -121.43,
    minLat: 39.67,
    maxLat: 39.88,
    // Cached projected Mercator meters
    minX: 0,
    maxX: 0,
    minY: 0,
    maxY: 0,

    init() {
      const p1 = this.projectLonLat(this.minLon, this.minLat);
      const p2 = this.projectLonLat(this.maxLon, this.maxLat);
      this.minX = Math.min(p1.x, p2.x);
      this.maxX = Math.max(p1.x, p2.x);
      this.minY = Math.min(p1.y, p2.y);
      this.maxY = Math.max(p1.y, p2.y);
    },

    projectLonLat(lon, lat) {
      const rad = Math.PI / 180;
      const x = 6378137.0 * lon * rad;
      const phi = lat * rad;
      const y = 6378137.0 * Math.log(Math.tan(Math.PI / 4 + phi / 2));
      return { x, y };
    },

    toSvg(lon, lat) {
      const p = this.projectLonLat(lon, lat);
      const normX = (p.x - this.minX) / (this.maxX - this.minX);
      const normY = (this.maxY - p.y) / (this.maxY - this.minY); // Invert Y for screen
      return {
        x: normX * 1000,
        y: normY * 700,
      };
    },
  };

  // Method Styling Configurations
  const METHOD_STYLES = {
    agent_a3: {
      name: "A3 surrogate stand-in",
      note: "WindConditionedUNet 30m stand-in forward",
      color: "#45dfa4",
      strokeWidth: 3.5,
      dashArray: "none",
    },
    agent_b_on_a3: {
      name: "B on A3",
      note: "Observation-driven provider with downwind cone heuristic",
      color: "#d3c3ff",
      strokeWidth: 2.5,
      dashArray: "8,3,2,3",
    },
    agent_a2_assim: {
      name: "A2-assim",
      note: "Teacher with causal NIST detection assimilation",
      color: "#45def3",
      strokeWidth: 3.5,
      dashArray: "none",
    },
    agent_a2_pure: {
      name: "A2-pure",
      note: "Teacher unassimilated physics",
      color: "#00c2d6",
      strokeWidth: 2.5,
      dashArray: "6,4",
    },
    static: {
      name: "A0 static",
      note: "Static baseline shortest path",
      color: "#FFFFFF",
      strokeWidth: 3,
      dashArray: "none",
    },
  };

  const ISOCHRONE_COLORS = {
    "0h-1h": "#fde047",
    "1h-2h": "#f59e0b",
    "2h-3h": "#f97316",
    "3h-4h": "#ea580c",
    "4h-5h": "#dc2626",
    "5h-6h": "#991b1b",
  };

  // DOM Elements Cache
  const dom = {};

  function init() {
    PROJECTION.init();
    cacheDom();
    setupEventListeners();
    populateOriginPicker();
    renderAll();
  }

  function cacheDom() {
    dom.headerClock = document.getElementById("headerClockReadout");
    dom.timelineClock = document.getElementById("timelineClockText");
    dom.replanCycleText = document.getElementById("replanCycleText");
    dom.timelineRange = document.getElementById("timelineRange");
    dom.btnPlay = document.getElementById("btnTimelinePlay");
    dom.btnReplay10 = document.getElementById("btnTimelineReplay10");
    dom.btnFwd10 = document.getElementById("btnTimelineFwd10");

    dom.srcBtnSurrogate = document.getElementById("srcBtnSurrogate");
    dom.srcBtnTeacher = document.getElementById("srcBtnTeacher");
    dom.surrogateWarningChip = document.getElementById("surrogateWarningChip");

    dom.originPicker = document.getElementById("originPickerSelect");
    dom.originCounterBadge = document.getElementById("originCounterBadge");
    dom.btnPrevOrigin = document.getElementById("btnPrevOrigin");
    dom.btnNextOrigin = document.getElementById("btnNextOrigin");
    dom.bannerOriginId = document.getElementById("bannerOriginId");
    dom.bannerOriginZone = document.getElementById("bannerOriginZone");
    dom.bannerDepartureTime = document.getElementById("bannerDepartureTime");
    dom.routeCardsContainer = document.getElementById("routeCardsContainer");
    dom.btnReplan = document.getElementById("btnReplanTrigger");
    dom.replanFeedback = document.getElementById("replanFeedbackMsg");

    dom.canvasWrapper = document.getElementById("canvasWrapper");
    dom.gisMapSvg = document.getElementById("gisMapSvg");
    dom.mapTooltip = document.getElementById("mapTooltip");

    dom.layerIsochrones = document.getElementById("layerIsochrones");
    dom.layerPerimeters = document.getElementById("layerPerimeters");
    dom.layerZones = document.getElementById("layerZones");
    dom.layerRoutes = document.getElementById("layerRoutes");
    dom.layerUpdates = document.getElementById("layerUpdates");
    dom.layerHazards = document.getElementById("layerHazards");
    dom.layerOriginMarker = document.getElementById("layerOriginMarker");

    dom.btnZoomIn = document.getElementById("btnZoomIn");
    dom.btnZoomOut = document.getElementById("btnZoomOut");
    dom.btnResetView = document.getElementById("btnResetView");
    dom.btnToggleSidebar = document.getElementById("btnToggleSidebar");
    dom.leftRail = document.getElementById("leftRail");

    dom.tabBtns = document.querySelectorAll(".nav-tab-btn");
    dom.tabOverlays = document.querySelectorAll(".tab-pane-overlay");
    dom.closePaneBtns = document.querySelectorAll(".btn-close-pane");

    dom.zonesTableBody = document.getElementById("zonesTableBody");
    dom.zonesTableHeaders = document.querySelectorAll("#zonesSummaryTable th");
    dom.btnShowLatencyHistory = document.getElementById("btnShowLatencyHistory");
    dom.recordedLatencyLog = document.getElementById("recordedLatencyLog");
    dom.latencyEventsList = document.getElementById("latencyEventsList");
  }

  function setupEventListeners() {
    // Timeline slider
    dom.timelineRange.addEventListener("input", (e) => {
      setTime(parseInt(e.target.value, 10));
    });

    // Timeline transport buttons
    dom.btnPlay.addEventListener("click", togglePlay);
    dom.btnReplay10.addEventListener("click", () => setTime(Math.max(0, state.currentTimeMin - 10)));
    dom.btnFwd10.addEventListener("click", () => setTime(Math.min(360, state.currentTimeMin + 10)));

    // Keyboard navigation for timeline
    window.addEventListener("keydown", (e) => {
      if (document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
      if (e.key === "ArrowLeft") {
        setTime(Math.max(0, state.currentTimeMin - 10));
      } else if (e.key === "ArrowRight") {
        setTime(Math.min(360, state.currentTimeMin + 10));
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    });

    // Field source engine toggle
    dom.srcBtnSurrogate.addEventListener("click", () => setFieldSource("surrogate"));
    dom.srcBtnTeacher.addEventListener("click", () => setFieldSource("teacher"));

    // Layer checkboxes
    const layerCheckboxes = {
      chkPredictedPerimeter: "predictedPerimeter",
      chkIsochrones: "isochrones",
      chkObservedPerimeter: "observedPerimeter",
      chkHazards: "hazards",
      chkRoutes: "routes",
      chkUpdates: "updates",
    };
    for (const [id, key] of Object.entries(layerCheckboxes)) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", (e) => {
          state.layers[key] = e.target.checked;
          renderMap();
        });
      }
    }

    // Method checkboxes
    const methodCheckboxes = {
      chkMethodA3: "agent_a3",
      chkMethodB: "agent_b_on_a3",
      chkMethodA2Assim: "agent_a2_assim",
      chkMethodA2Pure: "agent_a2_pure",
      chkMethodStatic: "static",
    };
    for (const [id, key] of Object.entries(methodCheckboxes)) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", (e) => {
          state.methods[key] = e.target.checked;
          renderMap();
          renderRouteCards();
        });
      }
    }

    // Origin selector
    dom.originPicker.addEventListener("change", (e) => {
      setOrigin(e.target.value);
    });

    dom.btnPrevOrigin.addEventListener("click", () => {
      const origins = getSortedOriginIds();
      const idx = origins.indexOf(state.selectedOriginId);
      if (idx > 0) setOrigin(origins[idx - 1]);
    });

    dom.btnNextOrigin.addEventListener("click", () => {
      const origins = getSortedOriginIds();
      const idx = origins.indexOf(state.selectedOriginId);
      if (idx < origins.length - 1) setOrigin(origins[idx + 1]);
    });

    // Re-plan trigger button
    dom.btnReplan.addEventListener("click", () => {
      dom.replanFeedback.style.display = "block";
      renderRouteCards();
      setTimeout(() => {
        dom.replanFeedback.style.display = "none";
      }, 3000);
    });

    // Map pan and zoom
    dom.canvasWrapper.addEventListener("mousedown", onMapMouseDown);
    window.addEventListener("mousemove", onMapMouseMove);
    window.addEventListener("mouseup", onMapMouseUp);
    dom.canvasWrapper.addEventListener("wheel", onMapWheel, { passive: false });

    dom.btnZoomIn.addEventListener("click", () => zoomMap(0.8));
    dom.btnZoomOut.addEventListener("click", () => zoomMap(1.25));
    dom.btnResetView.addEventListener("click", resetMapView);

    // Responsive drawer toggle (< 1100px)
    dom.btnToggleSidebar.addEventListener("click", () => {
      dom.leftRail.classList.toggle("drawer-open");
    });

    // Tabs switching
    dom.tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab");
        switchTab(tab);
      });
    });

    dom.closePaneBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        switchTab("console");
      });
    });

    // Zones table sorting
    dom.zonesTableHeaders.forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.getAttribute("data-sort");
        if (state.zoneSortColumn === col) {
          state.zoneSortAsc = !state.zoneSortAsc;
        } else {
          state.zoneSortColumn = col;
          state.zoneSortAsc = true;
        }
        renderZonesTable();
      });
    });

    // Latency History toggle
    dom.btnShowLatencyHistory.addEventListener("click", () => {
      const isVisible = dom.recordedLatencyLog.style.display === "block";
      dom.recordedLatencyLog.style.display = isVisible ? "none" : "block";
      if (!isVisible) renderLatencyEvents();
    });
  }

  function getSortedOriginIds() {
    const data = window.DEMO_DATA;
    if (!data || !data.routes) return [];
    return Object.keys(data.routes).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
  }

  function populateOriginPicker() {
    const data = window.DEMO_DATA;
    if (!data || !data.routes) return;
    const origins = getSortedOriginIds();
    dom.originCounterBadge.textContent = `${origins.length} Total`;

    dom.originPicker.innerHTML = "";
    origins.forEach((origId) => {
      const o = data.routes[origId];
      const opt = document.createElement("option");
      opt.value = origId;
      opt.textContent = `Block #${origId} (${o.assigned_zone || "N/A"})`;
      dom.originPicker.appendChild(opt);
    });

    if (origins.length > 0) {
      setOrigin(origins[0]);
    }
  }

  function setOrigin(originId) {
    state.selectedOriginId = originId;
    dom.originPicker.value = originId;

    const data = window.DEMO_DATA;
    if (data && data.routes && data.routes[originId]) {
      const o = data.routes[originId];
      dom.bannerOriginId.textContent = `ORIGIN: #${originId}`;
      dom.bannerOriginZone.textContent = o.assigned_zone || "N/A";

      const dep = o.departure_time_min || 0;
      const clock = formatClock(dep);
      dom.bannerDepartureTime.textContent = `Departure: +${dep.toFixed(1)} min (${clock} PST)`;
    }

    renderRouteCards();
    renderMap();
  }

  function setTime(minutes) {
    state.currentTimeMin = minutes;
    dom.timelineRange.value = minutes;

    const clock = formatClock(minutes);
    dom.headerClock.textContent = `T+${minutes} min (${clock} PST)`;
    dom.timelineClock.textContent = `${clock} PST`;

    const cycleNum = Math.floor(minutes / 30) + 1;
    dom.replanCycleText.textContent = `RE-PLAN #${cycleNum} ACTIVE`;

    renderMap();
  }

  function togglePlay() {
    state.isPlaying = !state.isPlaying;
    if (state.isPlaying) {
      dom.btnPlay.textContent = "⏸";
      dom.btnPlay.title = "Pause Timeline";
      state.playIntervalId = setInterval(() => {
        let nextTime = state.currentTimeMin + 10;
        if (nextTime > 360) nextTime = 0;
        setTime(nextTime);
      }, 1200);
    } else {
      dom.btnPlay.textContent = "▶";
      dom.btnPlay.title = "Play Timeline";
      if (state.playIntervalId) {
        clearInterval(state.playIntervalId);
        state.playIntervalId = null;
      }
    }
  }

  function setFieldSource(source) {
    state.fieldSource = source;
    if (source === "surrogate") {
      dom.srcBtnSurrogate.classList.add("active");
      dom.srcBtnTeacher.classList.remove("active");
      dom.surrogateWarningChip.style.display = "flex";
    } else {
      dom.srcBtnTeacher.classList.add("active");
      dom.srcBtnSurrogate.classList.remove("active");
      dom.surrogateWarningChip.style.display = "none";
    }
    renderMap();
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    dom.tabBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
    });

    dom.tabOverlays.forEach((overlay) => {
      overlay.classList.remove("active");
    });

    if (tabName === "zones") {
      document.getElementById("paneZones").classList.add("active");
      renderZonesTable();
    } else if (tabName === "latency") {
      document.getElementById("paneLatency").classList.add("active");
    } else if (tabName === "status") {
      document.getElementById("paneStatus").classList.add("active");
    }
  }

  function formatClock(elapsedMin) {
    const totalMin = Math.round(6 * 60 + 33 + elapsedMin);
    const h = Math.floor(totalMin / 60) % 24;
    const m = totalMin % 60;
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  }

  // RENDER ROUTE EVALUATION CARDS
  function renderRouteCards() {
    const data = window.DEMO_DATA;
    if (!data || !data.routes || !state.selectedOriginId) return;

    const o = data.routes[state.selectedOriginId];
    if (!o) return;

    dom.routeCardsContainer.innerHTML = "";

    const methodKeys = ["agent_a3", "agent_b_on_a3", "agent_a2_assim", "agent_a2_pure", "static"];
    methodKeys.forEach((mKey) => {
      if (!state.methods[mKey]) return;
      const mData = o[mKey];
      if (!mData) return;

      const style = METHOD_STYLES[mKey];
      const card = document.createElement("div");
      card.className = "route-card";
      card.style.borderLeftColor = style.color;

      const distKm = mData.distance_km || 0.0;
      const isTrapped = Boolean(mData.trapped);
      const isMiss = Boolean(mData.hazard_miss);
      const margin = mData.margin_min;

      let marginHtml = "";
      if (isTrapped) {
        marginHtml = `<span class="margin-chip trapped">trapped (no route)</span>`;
      } else if (margin === null || margin === undefined || mData.margin_state === "no_hazard" || margin === Infinity) {
        marginHtml = `<span class="margin-chip neutral">No hazard on route</span>`;
      } else if (margin < 0) {
        marginHtml = `<span class="margin-chip negative">${margin.toFixed(1)} min</span>`;
      } else {
        marginHtml = `<span class="margin-chip neutral">+${margin.toFixed(1)} min</span>`;
      }

      let stateText = "";
      if (isTrapped) {
        stateText = `<span style="color:var(--error); font-size:9px; font-weight:700;">🚫 TRAPPED (No route)</span>`;
      } else if (isMiss) {
        stateText = `<span style="color:var(--error); font-size:9px; font-weight:700;">⚠️ HAZARD OVERRUN</span>`;
      } else {
        stateText = `<span style="color:var(--outline); font-size:9px; font-weight:600;">No hazard on route (not a safety margin)</span>`;
      }

      const standinChip = mKey === "agent_a3" ? `<span class="chip-standin">STAND-IN (untrained surrogate)</span>` : "";

      card.innerHTML = `
        <div class="route-card-left" style="width:100%;">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
            <div style="display:flex; align-items:center;">
              <span class="route-card-method" style="color:${style.color};">${style.name}</span>
              ${standinChip}
            </div>
            <div>${stateText}</div>
          </div>
          <div style="font-family:var(--font-mono); font-size:10px; color:var(--on-surface-variant); margin-bottom:2px;">
            Route travel time: NOT AVAILABLE | Distance: ${distKm > 0 ? distKm.toFixed(2) + " km" : "NOT AVAILABLE"}
          </div>
          <div style="display:flex; align-items:center; justify-content:space-between; font-family:var(--font-mono); font-size:10px; margin-top:2px;">
            <span>Predicted margin (agent's own field):</span>
            ${marginHtml}
          </div>
          <div style="display:flex; align-items:center; justify-content:space-between; font-family:var(--font-mono); font-size:10px; color:var(--on-surface-variant); margin-top:2px;">
            <span>Margin to NIST burnovers:</span>
            <span style="color:var(--outline); font-weight:600;">NOT AVAILABLE</span>
          </div>
          <div class="route-source-cite" style="margin-top:4px; color:var(--outline);">Cite: margin_min (src/wf052/routing/router.py:174)</div>
        </div>
      `;

      dom.routeCardsContainer.appendChild(card);
    });
  }

  // RENDER ZONES SUMMARY TABLE
  function renderZonesTable() {
    const data = window.DEMO_DATA;
    if (!data || !data.zones_summary) return;

    let records = [...data.zones_summary];
    const col = state.zoneSortColumn;
    const asc = state.zoneSortAsc;

    records.sort((a, b) => {
      let va = a[col];
      let vb = b[col];
      if (typeof va === "string") {
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return asc ? va - vb : vb - va;
    });

    dom.zonesTableBody.innerHTML = "";
    records.forEach((r) => {
      const tr = document.createElement("tr");
      const isTrappedNonZero = r.trapped_count && !r.trapped_count.startsWith("0/");
      tr.innerHTML = `
        <td><strong>${r.zone_or_corridor}</strong></td>
        <td>${r.type}</td>
        <td>${r.origins_count}</td>
        <td>${r.route_changes}</td>
        <td><code>${r.margin_p10 || "NOT AVAILABLE"}</code></td>
        <td><span style="${isTrappedNonZero ? 'color:var(--error); font-weight:700;' : ''}">${r.trapped_count}</span></td>
      `;
      dom.zonesTableBody.appendChild(tr);
    });
  }

  // RENDER LATENCY EVENTS LIST
  function renderLatencyEvents() {
    const data = window.DEMO_DATA;
    if (!data || !data.updates || !data.updates.features) return;

    dom.latencyEventsList.innerHTML = "";
    data.updates.features.forEach((feat) => {
      const p = feat.properties;
      const card = document.createElement("div");
      card.style.cssText = "background:var(--surface-container-high); padding:6px; border-radius:3px; border:1px solid var(--stroke-subtle);";
      card.innerHTML = `
        <div style="color:var(--primary); font-weight:700;">Cycle @ T+${p.update_step_min}m</div>
        <div>Surrogate Latency: <strong>${p.surrogate_inf_ms.toFixed(1)} ms</strong></div>
        <div style="color:var(--on-surface-variant); font-size:9px;">Route changes: ${p.origins_route_changed} | Δ cells: ${(p.frac_cells_changed * 100).toFixed(1)}%</div>
      `;
      dom.latencyEventsList.appendChild(card);
    });
  }

  // MAP VECTOR RENDERING ENGINE
  function renderMap() {
    const data = window.DEMO_DATA;
    if (!data) return;

    renderZonesLayer(data);
    renderIsochronesLayer(data);
    renderPerimetersLayer(data);
    renderHazardsLayer(data);
    renderRoutesLayer(data);
    renderUpdatesLayer(data);
    renderOriginMarker(data);
  }

  function renderZonesLayer(data) {
    dom.layerZones.innerHTML = "";
    if (!data.zones || !data.zones.features) return;

    data.zones.features.forEach((feat) => {
      const pathD = geometryToSvgPath(feat.geometry);
      if (!pathD) return;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathD);
      path.setAttribute("fill", "rgba(39, 54, 71, 0.15)");
      path.setAttribute("stroke", "#3c494b");
      path.setAttribute("stroke-width", "0.8");
      dom.layerZones.appendChild(path);
    });
  }

  function renderIsochronesLayer(data) {
    dom.layerIsochrones.innerHTML = "";
    if (!state.layers.isochrones) return;

    const isoCollection = data.isochrones ? data.isochrones[state.fieldSource] : null;
    if (!isoCollection || !isoCollection.features) return;

    // Filter isochrones up to current timeline time
    isoCollection.features.forEach((feat) => {
      const p = feat.properties || {};
      if (p.t_low > state.currentTimeMin) return;

      const pathD = geometryToSvgPath(feat.geometry);
      if (!pathD) return;

      const color = ISOCHRONE_COLORS[p.band_label] || "#ea580c";
      const isCurrentBand = state.currentTimeMin >= p.t_low && state.currentTimeMin < p.t_high;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathD);
      path.setAttribute("fill", color);
      path.setAttribute("fill-opacity", isCurrentBand ? "0.38" : "0.22");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-width", isCurrentBand ? "1.5" : "0.8");
      if (isCurrentBand) {
        path.setAttribute("stroke-dasharray", "4,2");
      }
      dom.layerIsochrones.appendChild(path);
    });
  }

  function renderPerimetersLayer(data) {
    dom.layerPerimeters.innerHTML = "";

    // 1. Predicted perimeter
    if (state.layers.predictedPerimeter && data.perimeters && data.perimeters[state.fieldSource]) {
      const perims = data.perimeters[state.fieldSource];
      // Select closest past hour boundary (1, 2, 4, 6)
      const curHour = state.currentTimeMin / 60.0;
      let matchedHour = null;
      for (const h of [6, 4, 2, 1]) {
        if (curHour >= h) {
          matchedHour = String(h);
          break;
        }
      }
      if (matchedHour && perims[matchedHour] && perims[matchedHour].features) {
        perims[matchedHour].features.forEach((feat) => {
          const pathD = geometryToSvgPath(feat.geometry);
          if (!pathD) return;
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("d", pathD);
          path.setAttribute("fill", "none");
          path.setAttribute("stroke", state.fieldSource === "surrogate" ? "#45dfa4" : "#45def3");
          path.setAttribute("stroke-width", "2");
          dom.layerPerimeters.appendChild(path);
        });
      }
    }

    // 2. Observed truth perimeter (+11.4h)
    if (state.layers.observedPerimeter && data.perimeters && data.perimeters.observed_11_4h) {
      data.perimeters.observed_11_4h.features.forEach((feat) => {
        const pathD = geometryToSvgPath(feat.geometry);
        if (!pathD) return;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathD);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "#FFFFFF");
        path.setAttribute("stroke-width", "2");
        path.setAttribute("stroke-dasharray", "6,4");
        dom.layerPerimeters.appendChild(path);
      });
    }
  }

  function renderHazardsLayer(data) {
    dom.layerHazards.innerHTML = "";
    if (!state.layers.hazards || !data.hazards || !data.hazards.features) return;

    data.hazards.features.forEach((feat) => {
      const p = feat.properties || {};
      const onset = p.onset_elapsed_min || 0;
      // Show only active hazards that have onset by current time
      if (onset > state.currentTimeMin) return;

      const coords = feat.geometry ? feat.geometry.coordinates : null;
      if (!coords || coords.length < 2) return;

      const svgPt = PROJECTION.toSvg(coords[0], coords[1]);
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("transform", `translate(${svgPt.x.toFixed(1)}, ${svgPt.y.toFixed(1)})`);
      g.setAttribute("class", "hazard-glyph");
      g.style.cursor = "pointer";

      if (p.hazard_type === "burnover") {
        // Red circle with white X
        g.innerHTML = `
          <circle cx="0" cy="0" r="8" fill="#93000a" stroke="#ffb4ab" stroke-width="1.5"/>
          <path d="M-4,-4 L4,4 M4,-4 L-4,4" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round"/>
        `;
      } else if (p.hazard_type === "roadblock") {
        // Amber warning triangle
        g.innerHTML = `
          <polygon points="0,-9 9,6 -9,6" fill="#f59e0b" stroke="#051424" stroke-width="1.2"/>
          <text x="0" y="4" font-family="monospace" font-size="9" font-weight="900" fill="#051424" text-anchor="middle">!</text>
        `;
      } else {
        // Abandoned vehicle: hollow square (constant onset 87 min)
        g.innerHTML = `
          <rect x="-6" y="-6" width="12" height="12" fill="#1c2b3c" stroke="#859396" stroke-width="1.5"/>
          <line x1="-3" y1="0" x2="3" y2="0" stroke="#FFFFFF" stroke-width="1.2"/>
        `;
      }

      g.addEventListener("mouseenter", (e) => {
        showTooltip(e, `
          <strong style="color:var(--primary);">${p.hazard_type.toUpperCase()} HAZARD</strong><br>
          ${p.description || "Hazard Location"}<br>
          <span style="color:var(--on-surface-variant);">Onset: T+${onset.toFixed(1)}m (${formatClock(onset)} PST)</span><br>
          <span style="color:var(--outline); font-size:8px;">Ref: ${p.source_ref || "NIST TN 2252"}</span>
        `);
      });

      g.addEventListener("mouseleave", hideTooltip);
      dom.layerHazards.appendChild(g);
    });
  }

  function renderRoutesLayer(data) {
    dom.layerRoutes.innerHTML = "";
    if (!state.layers.routes || !data.routes || !state.selectedOriginId) return;

    const o = data.routes[state.selectedOriginId];
    if (!o) return;

    const methodOrder = ["static", "agent_a2_pure", "agent_a2_assim", "agent_b_on_a3", "agent_a3"];
    methodOrder.forEach((mKey) => {
      if (!state.methods[mKey]) return;
      const mData = o[mKey];
      if (!mData || !mData.path || mData.path.length < 2) return;

      const style = METHOD_STYLES[mKey];
      let pathD = "";
      mData.path.forEach((pt, idx) => {
        const svgPt = PROJECTION.toSvg(pt[0], pt[1]);
        pathD += (idx === 0 ? "M " : "L ") + svgPt.x.toFixed(1) + "," + svgPt.y.toFixed(1) + " ";
      });

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathD);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", style.color);
      path.setAttribute("stroke-width", style.strokeWidth);
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
      if (style.dashArray !== "none") {
        path.setAttribute("stroke-dasharray", style.dashArray);
      }

      dom.layerRoutes.appendChild(path);
    });
  }

  function renderUpdatesLayer(data) {
    dom.layerUpdates.innerHTML = "";
    if (!state.layers.updates || !data.updates || !data.updates.features) return;

    data.updates.features.forEach((feat) => {
      const p = feat.properties || {};
      if (p.update_step_min > state.currentTimeMin) return;

      const coords = feat.geometry ? feat.geometry.coordinates : null;
      if (!coords) return;

      const svgPt = PROJECTION.toSvg(coords[0], coords[1]);
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("transform", `translate(${svgPt.x.toFixed(1)}, ${svgPt.y.toFixed(1)})`);

      // Diamond marker
      g.innerHTML = `
        <polygon points="0,-5 5,0 0,5 -5,0" fill="#00c2d6" stroke="#051424" stroke-width="1"/>
      `;
      g.style.cursor = "pointer";

      g.addEventListener("mouseenter", (e) => {
        showTooltip(e, `
          <strong style="color:var(--primary);">30-MIN MODEL RE-PLAN CYCLE</strong><br>
          Time: T+${p.update_step_min}m (${formatClock(p.update_step_min)} PST)<br>
          Route changes: ${p.origins_route_changed} origins<br>
          Surrogate Latency: ${p.surrogate_inf_ms.toFixed(1)} ms
        `);
      });
      g.addEventListener("mouseleave", hideTooltip);
      dom.layerUpdates.appendChild(g);
    });
  }

  function renderOriginMarker(data) {
    dom.layerOriginMarker.innerHTML = "";
    if (!data.routes || !state.selectedOriginId) return;

    const o = data.routes[state.selectedOriginId];
    if (!o) return;

    // Use first coordinate of static path or first available method
    let startCoord = null;
    for (const mKey of ["static", "agent_a3", "agent_a2_assim"]) {
      if (o[mKey] && o[mKey].path && o[mKey].path.length > 0) {
        startCoord = o[mKey].path[0];
        break;
      }
    }
    if (!startCoord) return;

    const svgPt = PROJECTION.toSvg(startCoord[0], startCoord[1]);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("transform", `translate(${svgPt.x.toFixed(1)}, ${svgPt.y.toFixed(1)})`);

    g.innerHTML = `
      <circle cx="0" cy="0" r="14" fill="none" stroke="#45def3" stroke-width="1.2" opacity="0.7">
        <animate attributeName="r" values="8;20;8" dur="2.4s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.8;0.1;0.8" dur="2.4s" repeatCount="indefinite"/>
      </circle>
      <circle cx="0" cy="0" r="5" fill="#45def3" stroke="#00363d" stroke-width="1.5"/>
      <path d="M0,-12 L0,12 M-12,0 L12,0" stroke="#45def3" stroke-width="1.2"/>
    `;

    dom.layerOriginMarker.appendChild(g);
  }

  function geometryToSvgPath(geom) {
    if (!geom || !geom.coordinates) return "";
    let d = "";

    function ringToPath(ring) {
      let str = "";
      ring.forEach((pt, idx) => {
        const svgPt = PROJECTION.toSvg(pt[0], pt[1]);
        str += (idx === 0 ? "M " : "L ") + svgPt.x.toFixed(1) + "," + svgPt.y.toFixed(1) + " ";
      });
      return str + "Z ";
    }

    if (geom.type === "Polygon") {
      geom.coordinates.forEach((ring) => {
        d += ringToPath(ring);
      });
    } else if (geom.type === "MultiPolygon") {
      geom.coordinates.forEach((poly) => {
        poly.forEach((ring) => {
          d += ringToPath(ring);
        });
      });
    } else if (geom.type === "LineString") {
      geom.coordinates.forEach((pt, idx) => {
        const svgPt = PROJECTION.toSvg(pt[0], pt[1]);
        d += (idx === 0 ? "M " : "L ") + svgPt.x.toFixed(1) + "," + svgPt.y.toFixed(1) + " ";
      });
    }
    return d;
  }

  // MAP PAN AND ZOOM INTERACTIONS
  function onMapMouseDown(e) {
    state.isDragging = true;
    state.dragStart = { x: e.clientX, y: e.clientY };
    dom.canvasWrapper.classList.add("grabbing");
  }

  function onMapMouseMove(e) {
    if (!state.isDragging) return;
    const dx = e.clientX - state.dragStart.x;
    const dy = e.clientY - state.dragStart.y;
    state.dragStart = { x: e.clientX, y: e.clientY };

    // Scale dx, dy to viewBox scale
    const scaleX = state.viewBox.w / dom.gisMapSvg.clientWidth;
    const scaleY = state.viewBox.h / dom.gisMapSvg.clientHeight;

    state.viewBox.x -= dx * scaleX;
    state.viewBox.y -= dy * scaleY;
    updateViewBox();
  }

  function onMapMouseUp() {
    state.isDragging = false;
    dom.canvasWrapper.classList.remove("grabbing");
  }

  function onMapWheel(e) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.15 : 0.85;
    zoomMap(factor);
  }

  function zoomMap(factor) {
    const newW = state.viewBox.w * factor;
    const newH = state.viewBox.h * factor;
    // Keep centered
    state.viewBox.x += (state.viewBox.w - newW) / 2;
    state.viewBox.y += (state.viewBox.h - newH) / 2;
    state.viewBox.w = newW;
    state.viewBox.h = newH;
    updateViewBox();
  }

  function resetMapView() {
    state.viewBox = { ...state.initialViewBox };
    updateViewBox();
  }

  function updateViewBox() {
    dom.gisMapSvg.setAttribute(
      "viewBox",
      `${state.viewBox.x.toFixed(1)} ${state.viewBox.y.toFixed(1)} ${state.viewBox.w.toFixed(1)} ${state.viewBox.h.toFixed(1)}`
    );
  }

  function showTooltip(e, html) {
    dom.mapTooltip.innerHTML = html;
    dom.mapTooltip.style.display = "block";
    const rect = dom.canvasWrapper.getBoundingClientRect();
    dom.mapTooltip.style.left = `${e.clientX - rect.left + 12}px`;
    dom.mapTooltip.style.top = `${e.clientY - rect.top + 12}px`;
  }

  function hideTooltip() {
    dom.mapTooltip.style.display = "none";
  }

  function renderAll() {
    setTime(state.currentTimeMin);
    renderZonesTable();
    renderRouteCards();
    renderMap();
  }

  // Self-initialization
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
