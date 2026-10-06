const data = readJSON("page-data");
const names = { blend: "Blend", lgbm: "LightGBM", xgb: "XGBoost" };
const order = ["blend", "lgbm", "xgb"];
const seriesColor = (k) => css({ blend: "--series-1", lgbm: "--series-2", xgb: "--series-3" }[k]);
const diagonal = (top = 1) => ({ x: [0, top], y: [0, top], mode: "lines", line: { color: css("--axis"), dash: "dot", width: 1 }, hoverinfo: "skip", showlegend: false });

function render() {
  const m = data.metrics;
  plot("roc", [
    diagonal(),
    ...order.map((k) => ({
      x: data.roc[k].fpr, y: data.roc[k].tpr, mode: "lines",
      name: `${names[k]}, AUC ${fmt.num(m.holdout[k].auc, 4)}`,
      line: { color: seriesColor(k), width: 2 },
      hovertemplate: `${names[k]}<br>false positive rate %{x:.3f}<br>true positive rate %{y:.3f}<extra></extra>`,
    })),
  ], { xaxis: { title: { text: "False positive rate" }, range: [0, 1] }, yaxis: { title: { text: "True positive rate" }, range: [0, 1] } });

  plot("pr", [{
    x: data.pr.blend.recall, y: data.pr.blend.precision, mode: "lines", line: { color: seriesColor("blend"), width: 2 },
    hovertemplate: "recall %{x:.2f}<br>precision %{y:.2f}<extra></extra>",
  }], { xaxis: { title: { text: "Recall" }, range: [0, 1] }, yaxis: { title: { text: "Precision" } }, showlegend: false });

  const cal = data.calibration;
  const top = Math.max(...cal.predicted, ...cal.observed) * 1.05;
  plot("calibration", [
    diagonal(top),
    {
      x: cal.predicted, y: cal.observed, customdata: cal.count, mode: "lines+markers",
      marker: { size: 8, color: seriesColor("blend"), line: { color: css("--surface"), width: 2 } }, line: { color: seriesColor("blend"), width: 2 },
      hovertemplate: "predicted %{x:.2%}<br>observed %{y:.2%}<br>%{customdata:,} applicants<extra></extra>",
    },
  ], { xaxis: { title: { text: "Mean predicted PD" }, tickformat: ".0%" }, yaxis: { title: { text: "Observed default rate" }, tickformat: ".0%" }, showlegend: false });

  const aucs = data.folds.map((r) => r.auc);
  plot("folds", order.map((k) => {
    const rows = data.folds.filter((r) => r.model === k);
    return {
      type: "bar", name: names[k], x: rows.map((r) => `Fold ${r.fold + 1}`), y: rows.map((r) => r.auc),
      marker: { color: seriesColor(k), line: { color: css("--surface"), width: 2 } },
      hovertemplate: `${names[k]} %{x}<br>AUC %{y:.4f}<extra></extra>`,
    };
  }), { barmode: "group", bargap: 0.3, yaxis: { title: { text: "AUC" }, range: [Math.min(...aucs) - 0.01, Math.max(...aucs) + 0.005] } });
}

render();
onThemeChange(render);
