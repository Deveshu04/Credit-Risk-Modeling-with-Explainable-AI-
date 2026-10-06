const $ = (id) => document.getElementById(id);
const labels = Object.fromEntries(readJSON("segment-labels"));
const state = { original: null, current: null };

const lgd = () => Number($("lgd").value) / 100;

function showError(message) {
  $("error").textContent = message || "";
}

function renderFigures(a) {
  const base = state.original;
  const changed = base && base !== a && base.id === a.id;
  $("result").innerHTML = [
    figure("Probability of default", fmt.pct(a.pd), changed ? `was ${fmt.pct(base.pd)} before edits` : "calibrated"),
    figure("Score", fmt.int(a.score), "600 at 50:1 odds, 20 points to double"),
    figure("Grade", String(a.grade), `grade PD ${fmt.pct(a.grade_pd)}`),
    figure("Expected loss", fmt.big(a.el), `exposure ${fmt.big(a.ead)}`),
    figure("Risk-weighted assets", fmt.big(a.rwa), `at LGD ${fmt.pct(a.lgd, 0)}`),
    figure("Actual outcome", a.target === 1 ? "Defaulted" : "Repaid", `applicant ${a.id}`),
  ].join("");
  $("segments").innerHTML = Object.entries(a.segments).map(([k, v]) => `${esc(labels[k] || k)} <b>${esc(v)}</b>`).join(". ") + ".";
  $("clipped").textContent = a.clipped.length ? `Clipped to the training range: ${a.clipped.join(", ")}.` : "";
  for (const cell of document.querySelectorAll(".scale-cell")) {
    cell.classList.toggle("is-current", Number(cell.dataset.grade) === a.grade);
  }
}

function renderWaterfall(a) {
  const names = ["Base value", ...a.waterfall.map((w) => `${w.feature} = ${w.display ?? fmt.val(w.value)}`), `${a.other.count} other features`, "This applicant"];
  const values = [a.base, ...a.waterfall.map((w) => w.contribution), a.other.contribution, 0];
  const measure = ["absolute", ...a.waterfall.map(() => "relative"), "relative", "total"];
  plot("waterfall", [{
    type: "waterfall", orientation: "h", y: names, x: values, measure,
    increasing: { marker: { color: css("--risk-up") } },
    decreasing: { marker: { color: css("--risk-down") } },
    totals: { marker: { color: css("--ink-3") } },
    connector: { line: { color: css("--axis"), width: 1 } },
    hovertemplate: "%{y}<br>%{x:+.3f} log-odds<extra></extra>",
  }], { margin: { l: 8, r: 12, t: 8, b: 44 }, yaxis: { autorange: "reversed", automargin: true }, xaxis: { title: { text: "Log-odds of default" } }, showlegend: false });
}

function renderDrivers(a) {
  const form = $("drivers");
  form.innerHTML = "";
  for (const d of a.drivers) {
    const label = document.createElement("label");
    label.className = "driver";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = d.feature;
    let input;
    if (d.labels) {
      input = document.createElement("select");
      for (let code = Math.round(d.low); code <= Math.round(d.high); code += 1) {
        const option = document.createElement("option");
        option.value = String(code);
        option.textContent = code < 0 ? "Missing" : d.labels[code] ?? String(code);
        input.append(option);
      }
      input.value = String(d.value == null ? Math.round(d.low) : Math.round(d.value));
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.min = d.low;
      input.max = d.high;
      input.placeholder = "missing";
      input.value = d.value == null ? "" : d.value;
    }
    input.dataset.feature = d.feature;
    input.dataset.original = input.value;
    const hint = document.createElement("small");
    hint.textContent = d.labels ? `${d.group}` : `${d.group}, range ${fmt.val(d.low)} to ${fmt.val(d.high)}`;
    label.append(name, input, hint);
    form.append(label);
  }
}

function show(a, fresh) {
  state.current = a;
  if (fresh) {
    state.original = a;
    $("applicant-id").value = a.id;
    renderDrivers(a);
  }
  renderFigures(a);
  renderWaterfall(a);
}

function overrides() {
  const out = {};
  for (const input of $("drivers").querySelectorAll("[data-feature]")) {
    if (input.value === "" || input.value === input.dataset.original) continue;
    out[input.dataset.feature] = Number(input.value);
  }
  return out;
}

async function load(url) {
  try {
    show(await getJSON(`${url}${url.includes("?") ? "&" : "?"}lgd=${lgd()}`), true);
    showError("");
  } catch (err) {
    showError(err.message);
  }
}

async function rescore() {
  if (!state.original) return;
  try {
    const body = JSON.stringify({ id: state.original.id, lgd: lgd(), overrides: overrides() });
    show(await getJSON("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body }), false);
    showError("");
  } catch (err) {
    showError(err.message);
  }
}

$("load").addEventListener("click", () => {
  const id = Number($("applicant-id").value);
  if (!Number.isInteger(id) || id <= 0) return showError("Enter a whole-number applicant ID, for example one shown after Random applicant.");
  load(`/api/applicant/${id}`);
});
$("applicant-id").addEventListener("keydown", (e) => { if (e.key === "Enter") $("load").click(); });
$("random").addEventListener("click", () => load("/api/applicant/random"));
$("random-grade").addEventListener("click", () => load(`/api/applicant/random?grade=${$("grade").value}`));
$("whatif").addEventListener("submit", (e) => { e.preventDefault(); rescore(); });
$("reset").addEventListener("click", () => { if (state.original) { renderDrivers(state.original); show(state.original, false); } });
$("lgd").addEventListener("change", rescore);
$("lgd").addEventListener("input", () => { $("lgd-value").textContent = `${$("lgd").value}%`; });
onThemeChange(() => state.current && show(state.current, false));
load("/api/applicant/random");
