const API_BASE = (window.ACCIDENTIQ_CONFIG?.apiBaseUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
const form = document.querySelector("#prediction-form");
const button = document.querySelector("#submit-button");
const status = document.querySelector("#api-status");
const errorBox = document.querySelector("#error-message");
const resultCard = document.querySelector("#result-card");

const categoryFields = {
  weather_conditions: "Weather",
  road_surface_conditions: "Road surface",
  light_conditions: "Light conditions",
  road_type: "Road type",
  junction_detail_historic: "Junction detail",
  junction_control: "Junction control",
  urban_or_rural_area: "Area",
  first_road_class: "First road class",
};

function setStatus(message, ready = false) {
  status.textContent = message;
  status.classList.toggle("online", ready);
}

async function loadOptions() {
  try {
    const [healthResponse, optionsResponse] = await Promise.all([
      fetch(`${API_BASE}/health`), fetch(`${API_BASE}/options`),
    ]);
    if (!healthResponse.ok || !optionsResponse.ok) throw new Error("API did not respond successfully.");
    const health = await healthResponse.json();
    const options = await optionsResponse.json();
    if (!health.model_loaded) throw new Error("The model has not loaded.");
    for (const [field, title] of Object.entries(categoryFields)) {
      const select = form.elements.namedItem(field);
      select.replaceChildren(new Option(`Choose ${title.toLowerCase()}`, ""));
      for (const item of options.categories[field]) {
        select.add(new Option(item.label, String(item.value)));
      }
    }
    button.disabled = false;
    setStatus("API ready", true);
  } catch (error) {
    setStatus("API unavailable");
    errorBox.hidden = false;
    errorBox.textContent = `${error.message} Start the API and refresh this page.`;
  }
}

function renderResult(result) {
  document.querySelector("#empty-state").hidden = true;
  const panel = document.querySelector("#prediction-result");
  panel.hidden = false;
  document.querySelector("#predicted-severity").textContent = result.predicted_severity;
  document.querySelector("#max-score").textContent = `${(result.max_class_score * 100).toFixed(1)}%`;
  const classScores = document.querySelector("#class-scores");
  classScores.replaceChildren();
  for (const [label, score] of Object.entries(result.class_scores)) {
    const row = document.createElement("div");
    row.className = "score-row";
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = `${(score * 100).toFixed(1)}%`;
    row.append(name, track, value);
    classScores.append(row);
  }
  const factors = document.querySelector("#factor-list");
  factors.replaceChildren();
  for (const factor of result.important_factors) {
    const item = document.createElement("li");
    const delta = document.createElement("span");
    const change = factor.predicted_class_probability_change_pp;
    delta.className = `factor-delta ${change >= 0 ? "positive" : "negative"}`;
    delta.textContent = `${change >= 0 ? "+" : ""}${change.toFixed(1)} pp`;
    const detail = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = categoryFields[factor.feature] || factor.feature.replaceAll("_", " ");
    const values = document.createElement("small");
    values.textContent = `${factor.value} · reference: ${factor.reference}`;
    detail.append(title, values);
    item.append(delta, detail);
    factors.append(item);
  }
  document.querySelector("#explanation-caveat").textContent = result.explanation_caveat;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  button.disabled = true;
  resultCard.setAttribute("aria-busy", "true");
  button.textContent = "Estimating…";
  const data = Object.fromEntries(new FormData(form).entries());
  if (data.speed_limit === "") data.speed_limit = null;
  else data.speed_limit = Number(data.speed_limit);
  for (const field of Object.keys(categoryFields)) data[field] = Number(data[field]);
  try {
    const response = await fetch(`${API_BASE}/predict-severity`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Prediction request failed.");
    renderResult(result);
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = error.message;
  } finally {
    resultCard.setAttribute("aria-busy", "false");
    button.disabled = false;
    button.innerHTML = 'Estimate severity <span aria-hidden="true">→</span>';
  }
});

loadOptions();
