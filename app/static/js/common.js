const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const fmt = {
  num: (v, d = 3) => (v == null ? "n/a" : Number(v).toFixed(d)),
  pct: (v, d = 2) => (v == null ? "n/a" : `${(Number(v) * 100).toFixed(d)}%`),
  int: (v) => (v == null ? "n/a" : Math.round(Number(v)).toLocaleString("en-US")),
  big: (v) => {
    if (v == null) return "n/a";
    const a = Math.abs(v);
    if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return Number(v).toFixed(0);
  },
  val: (v) => {
    if (v == null) return "missing";
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString("en-US");
    if (Number.isInteger(v)) return String(v);
    return Number(v).toPrecision(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  },
};

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function readJSON(id) {
  return JSON.parse(document.getElementById(id).textContent);
}

async function getJSON(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

function gradeColor(grade, count) {
  const step = count > 1 ? Math.round(((grade - 1) * 9) / (count - 1)) : 0;
  return css(`--ramp-${step}`);
}

function baseLayout() {
  const axis = {
    gridcolor: css("--grid"),
    zerolinecolor: css("--axis"),
    linecolor: css("--axis"),
    tickfont: { color: css("--ink-3") },
    title: { font: { color: css("--ink-2") } },
    automargin: true,
  };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font-sans"), color: css("--ink-2"), size: 12 },
    margin: { l: 56, r: 12, t: 8, b: 48 },
    xaxis: axis,
    yaxis: { ...axis },
    barcornerradius: 4,
    legend: { orientation: "h", y: -0.24, font: { color: css("--ink-2") } },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { color: css("--ink"), family: css("--font-sans") } },
  };
}

function plot(id, traces, layout = {}) {
  const base = baseLayout();
  const x = layout.xaxis || {};
  const y = layout.yaxis || {};
  const merged = {
    ...base,
    ...layout,
    xaxis: { ...base.xaxis, ...x, title: { ...base.xaxis.title, ...(x.title || {}) } },
    yaxis: { ...base.yaxis, ...y, title: { ...base.yaxis.title, ...(y.title || {}) } },
  };
  Plotly.react(id, traces, merged, { displayModeBar: false, responsive: true });
}

function figure(label, value, note = "") {
  return `<div class="figure"><dt>${esc(label)}</dt><dd class="value">${esc(value)}</dd><dd class="note">${esc(note)}</dd></div>`;
}

function onThemeChange(callback) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", callback);
}
