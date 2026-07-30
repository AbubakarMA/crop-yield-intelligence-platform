"use strict";

let model = null;
const formatter = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
const elements = {
  form: document.querySelector("#prediction-form"),
  country: document.querySelector("#country"),
  crop: document.querySelector("#crop"),
  year: document.querySelector("#year"),
  rainfall: document.querySelector("#rainfall"),
  pesticides: document.querySelector("#pesticides"),
  temperature: document.querySelector("#temperature"),
  message: document.querySelector("#form-message"),
  button: document.querySelector("#estimate-button"),
};

async function loadModel() {
  const response = await fetch("model.json.gz");
  if (!response.ok) throw new Error("The model file could not be loaded.");
  const bytes = new Uint8Array(await response.arrayBuffer());
  let text;
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(
      new DecompressionStream("gzip"),
    );
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(bytes);
  }
  return JSON.parse(text);
}

function populateSelect(select, values, preferred) {
  select.replaceChildren(
    ...values.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      option.selected = value === preferred;
      return option;
    }),
  );
  select.disabled = false;
}

function numericValues() {
  return {
    country: elements.country.value,
    crop: elements.crop.value,
    year: Number(elements.year.value),
    rainfall: Number(elements.rainfall.value),
    pesticides: Number(elements.pesticides.value),
    temperature: Number(elements.temperature.value),
  };
}

function inputsOutOfRange(values) {
  const ranges = model.input_ranges;
  return (
    values.year < model.supported_years[0] ||
    values.year > model.supported_years[1] ||
    values.rainfall < ranges.average_rainfall_mm_per_year[0] ||
    values.rainfall > ranges.average_rainfall_mm_per_year[1] ||
    values.pesticides < ranges.pesticides_tonnes[0] ||
    values.pesticides > ranges.pesticides_tonnes[1] ||
    values.temperature < ranges.average_temperature_c[0] ||
    values.temperature > ranges.average_temperature_c[1]
  );
}

function validateInputs() {
  if (!model) return;
  const outside = inputsOutOfRange(numericValues());
  elements.message.hidden = !outside;
  elements.message.className = "form-message warning";
  elements.message.textContent = outside
    ? "At least one numeric input is outside the training range. Move it back into range before relying on the estimate."
    : "";
  elements.button.disabled = outside;
}

function predictForest(values) {
  const featureCount = model.countries.length + model.crops.length + 4;
  const features = new Array(featureCount).fill(0);
  const countryIndex = model.countries.indexOf(values.country);
  const cropIndex = model.crops.indexOf(values.crop);
  if (countryIndex < 0 || cropIndex < 0) {
    throw new Error("Choose a country and crop observed during training.");
  }
  features[countryIndex] = 1;
  features[model.countries.length + cropIndex] = 1;

  const start = model.countries.length + model.crops.length;
  [values.year, values.rainfall, values.temperature].forEach((value, index) => {
    features[start + index] =
      (value - model.numeric_mean[index]) / model.numeric_scale[index];
  });
  features[start + 3] =
    (Math.log1p(values.pesticides) - model.pesticide_mean) /
    model.pesticide_scale;

  const predictions = model.trees.map((tree) => {
    let node = 0;
    while (tree.f[node] >= 0) {
      node =
        features[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    return tree.v[node];
  });
  const mean =
    predictions.reduce((total, value) => total + value, 0) /
    predictions.length;
  const variance =
    predictions.reduce((total, value) => total + (value - mean) ** 2, 0) /
    predictions.length;
  return {
    yieldHg: mean,
    yieldTonnes: mean / 10000,
    spreadTonnes: Math.sqrt(variance) / 10000,
  };
}

elements.form.addEventListener("input", validateInputs);
elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prediction = predictForest(numericValues());
  document.querySelector("#yield-tonnes").textContent =
    prediction.yieldTonnes.toFixed(2);
  document.querySelector("#yield-hg").textContent =
    `${formatter.format(prediction.yieldHg)} hg/ha`;
  document.querySelector("#forest-spread").textContent =
    `± ${prediction.spreadTonnes.toFixed(2)} t/ha`;
  document.querySelector("#model-version").textContent = model.version;
  document.querySelector("#result-details").hidden = false;
});

loadModel()
  .then((loaded) => {
    model = loaded;
    populateSelect(elements.country, model.countries, "Ghana");
    populateSelect(elements.crop, model.crops, "Maize");
    elements.year.min = model.supported_years[0];
    elements.year.max = model.supported_years[1];
    elements.year.value = model.supported_years[1];
    document.querySelector("#test-r2").textContent =
      model.final_test.r2.toFixed(3);
    document.querySelector("#test-mae").textContent =
      formatter.format(model.final_test.mae_hg_per_ha);
    document.querySelector("#limitations-list").replaceChildren(
      ...model.limitations.map((limitation) => {
        const item = document.createElement("li");
        item.textContent = limitation;
        return item;
      }),
    );
    elements.button.textContent = "Estimate yield";
    elements.button.disabled = false;
    validateInputs();
  })
  .catch((error) => {
    elements.message.hidden = false;
    elements.message.className = "form-message error";
    elements.message.textContent = error.message;
    elements.button.textContent = "Model unavailable";
  });
