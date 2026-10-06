const data = readJSON("page-data");
const groupNames = { application: "application", bureau: "bureau", previous: "previous applications", installments: "instalments", pos_cash: "POS and cash", credit_card: "credit card", cross_table: "cross-table" };

function renderGlobal() {
  const items = data.shap_global.slice(0, 20).reverse();
  const top = Math.max(...items.map((d) => d.mean_abs));
  plot("global", [{
    type: "bar", orientation: "h", x: items.map((d) => d.mean_abs), y: items.map((d) => d.feature),
    text: items.map((d) => groupNames[d.group] || d.group), textposition: "outside", cliponaxis: false,
    textfont: { color: css("--ink-3"), size: 11 },
    marker: { color: css("--series-1") },
    hovertemplate: "%{y}<br>mean |SHAP| %{x:.4f}<br>source: %{text}<extra></extra>",
  }], { margin: { l: 8, r: 12, t: 8, b: 44 }, xaxis: { title: { text: "Mean |SHAP|, log-odds" }, range: [0, top * 1.45] }, yaxis: { automargin: true }, showlegend: false });
}

function renderBeeswarm() {
  const n = data.beeswarm.length;
  const scale = [[0, css("--risk-down")], [0.5, css("--neutral")], [1, css("--risk-up")]];
  const traces = data.beeswarm.map((f, i) => ({
    type: "scattergl", mode: "markers", x: f.shap,
    y: f.shap.map((_, k) => n - 1 - i + ((((k * 7919) % 1000) / 1000) - 0.5) * 0.6),
    showlegend: false,
    marker: {
      size: 5, opacity: 0.8, color: f.color.map((c) => (c == null ? 0.5 : c)), cmin: 0, cmax: 1, colorscale: scale, showscale: i === 0,
      colorbar: { title: { text: "Feature value", side: "right" }, tickvals: [0, 1], ticktext: ["Low", "High"], thickness: 10, len: 0.5, outlinewidth: 0, tickfont: { color: css("--ink-3") } },
    },
    hovertemplate: `${f.feature}<br>SHAP %{x:.3f}<extra></extra>`,
  }));
  plot("beeswarm", traces, {
    margin: { l: 8, r: 12, t: 8, b: 44 },
    xaxis: { title: { text: "SHAP value, log-odds" }, zeroline: true },
    yaxis: { tickvals: data.beeswarm.map((_, i) => n - 1 - i), ticktext: data.beeswarm.map((f) => f.feature), zeroline: false, showgrid: false, automargin: true },
  });
}

function renderDependence() {
  const name = document.getElementById("dependence-feature").value;
  const d = data.dependence[name];
  plot("dependence", [{
    type: "scattergl", mode: "markers", x: d.x, y: d.shap, marker: { size: 5, opacity: 0.55, color: css("--series-1") },
    hovertemplate: `${name} %{x}<br>SHAP %{y:.3f}<extra></extra>`,
  }], { xaxis: { title: { text: name } }, yaxis: { title: { text: "SHAP value, log-odds" }, zeroline: true }, showlegend: false });
}

function render() {
  renderGlobal();
  renderBeeswarm();
  renderDependence();
}

document.getElementById("dependence-feature").addEventListener("change", renderDependence);
render();
onThemeChange(render);
