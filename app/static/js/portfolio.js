const data = readJSON("page-data");
const $ = (id) => document.getElementById(id);
const gradeCount = data.master_scale.length;
let timer = null;
let last = null;

const scoreRange = (g) => {
  if (g.score_min == null) return `below ${fmt.int(g.score_max)}`;
  if (g.score_max == null) return `${fmt.int(g.score_min)} and above`;
  return `${fmt.int(g.score_min)} to ${fmt.int(g.score_max)}`;
};

function renderTotals(view) {
  const t = view.totals;
  $("totals").innerHTML = [
    figure("Applicants", fmt.int(t.count), "holdout"),
    figure("Exposure", fmt.big(t.exposure), "sum of credit amounts"),
    figure("Expected loss", fmt.big(t.el), `${fmt.pct(t.el / t.exposure)} of exposure`),
    figure("Risk-weighted assets", fmt.big(t.rwa), `at LGD ${fmt.pct(view.lgd, 0)}`),
    figure("RWA density", fmt.pct(t.rwa_density, 1), "RWA as a share of exposure"),
    figure("Capital at 8%", fmt.big(0.08 * t.rwa), "minimum total capital"),
  ].join("");
}

function renderGrades(view) {
  const live = Object.fromEntries(view.grades.map((g) => [Number(g.level), g]));
  const rows = data.master_scale.map((g) => {
    const l = live[g.grade] || {};
    const flag = g.jeffreys_p != null && g.jeffreys_p < 0.05 ? '<span class="flag-bad">PD too low</span>' : '<span class="flag-ok">pass</span>';
    return `<tr><td class="swatch"><span style="--cell: ${gradeColor(g.grade, gradeCount)}"></span></td><td>${g.grade}</td><td>${scoreRange(g)}</td>
      <td class="num">${fmt.pct(g.pd)}</td><td class="num">${fmt.int(g.holdout_count)}</td><td class="num">${fmt.pct(g.holdout_rate)}</td>
      <td class="num">${fmt.num(g.jeffreys_p, 3)}</td><td>${flag}</td><td class="num">${fmt.big(l.exposure)}</td>
      <td class="num">${fmt.big(l.el)}</td><td class="num">${fmt.big(l.rwa)}</td><td class="num">${fmt.pct(l.rwa_density, 1)}</td></tr>`;
  });
  $("grade-table").innerHTML = `<thead><tr><th></th><th>Grade</th><th>Score range</th><th class="num">Grade PD</th><th class="num">Holdout applicants</th>
    <th class="num">Holdout default rate</th><th class="num">Jeffreys p</th><th>Back-test</th><th class="num">Exposure</th><th class="num">Expected loss</th>
    <th class="num">RWA</th><th class="num">RWA density</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  const grades = data.master_scale.map((g) => `Grade ${g.grade}`);
  plot("grade-chart", [
    {
      type: "bar", name: "Grade PD", x: grades, y: data.master_scale.map((g) => g.pd),
      marker: { color: data.master_scale.map((g) => gradeColor(g.grade, gradeCount)), line: { color: css("--surface"), width: 2 } },
      hovertemplate: "%{x}<br>grade PD %{y:.2%}<extra></extra>",
    },
    {
      type: "scatter", mode: "markers", name: "Holdout default rate", x: grades, y: data.master_scale.map((g) => g.holdout_rate),
      marker: { color: css("--ink"), size: 9, symbol: "diamond", line: { color: css("--surface"), width: 2 } },
      hovertemplate: "%{x}<br>holdout default rate %{y:.2%}<extra></extra>",
    },
  ], { yaxis: { tickformat: ".0%", title: { text: "Default rate" } } });
}

function renderSegments(view) {
  const drivers = data.segment_drivers[`SEG_${view.segment}`] || {};
  const rows = [...view.segments].sort((a, b) => a.rwa_density - b.rwa_density);
  $("segment-table").innerHTML = `<thead><tr><th>Level</th><th class="num">Applicants</th><th class="num">Exposure</th><th class="num">Mean PD</th>
    <th class="num">Observed default rate</th><th class="num">Expected loss</th><th class="num">RWA</th><th class="num">RWA density</th><th>Top SHAP drivers</th></tr></thead>
    <tbody>${rows.map((s) => `<tr><td>${esc(s.level)}</td><td class="num">${fmt.int(s.count)}</td><td class="num">${fmt.big(s.exposure)}</td>
    <td class="num">${fmt.pct(s.mean_pd)}</td><td class="num">${fmt.pct(s.observed)}</td><td class="num">${fmt.big(s.el)}</td><td class="num">${fmt.big(s.rwa)}</td>
    <td class="num">${fmt.pct(s.rwa_density, 1)}</td><td class="wrap">${esc((drivers[s.level] || []).join(", "))}</td></tr>`).join("")}</tbody>`;
  plot("segment-chart", [{
    type: "bar", orientation: "h", x: rows.map((s) => s.rwa_density), y: rows.map((s) => s.level),
    text: rows.map((s) => fmt.pct(s.rwa_density, 1)), textposition: "outside", cliponaxis: false, textfont: { color: css("--ink-2"), size: 11 },
    marker: { color: css("--series-1") },
    hovertemplate: "%{y}<br>RWA density %{x:.1%}<extra></extra>",
  }], {
    margin: { l: 8, r: 12, t: 8, b: 44 },
    xaxis: { tickformat: ".0%", title: { text: "RWA as a share of exposure" }, range: [0, Math.max(...rows.map((s) => s.rwa_density)) * 1.2] },
    yaxis: { automargin: true }, showlegend: false,
  });
}

function render() {
  if (!last) return;
  renderTotals(last);
  renderGrades(last);
  renderSegments(last);
}

async function refresh() {
  const lgd = Number($("lgd").value) / 100;
  $("lgd-value").textContent = `${Math.round(lgd * 100)}%`;
  try {
    last = await getJSON(`/api/portfolio?segment=${encodeURIComponent($("segment").value)}&lgd=${lgd}`);
    $("error").textContent = "";
    render();
  } catch (err) {
    $("error").textContent = `Could not load the portfolio view: ${err.message}`;
  }
}

$("segment").addEventListener("change", refresh);
$("lgd").addEventListener("input", () => {
  $("lgd-value").textContent = `${$("lgd").value}%`;
  clearTimeout(timer);
  timer = setTimeout(refresh, 150);
});
refresh();
onThemeChange(render);
