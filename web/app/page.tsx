"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Tree = {
  l: number[];
  r: number[];
  f: number[];
  t: number[];
  v: number[];
};

type ModelData = {
  version: string;
  name: string;
  countries: string[];
  crops: string[];
  numeric_mean: number[];
  numeric_scale: number[];
  pesticide_mean: number;
  pesticide_scale: number;
  supported_years: [number, number];
  input_ranges: Record<string, [number, number]>;
  final_test: {
    mae_hg_per_ha: number;
    rmse_hg_per_ha: number;
    r2: number;
  };
  limitations: string[];
  trees: Tree[];
};

type Prediction = {
  yieldHg: number;
  yieldTonnes: number;
  spreadTonnes: number;
};

async function loadModel(): Promise<ModelData> {
  const response = await fetch("/model.json.gz");
  if (!response.ok) throw new Error("The model file could not be loaded.");
  const bytes = new Uint8Array(await response.arrayBuffer());
  const isGzip = bytes[0] === 0x1f && bytes[1] === 0x8b;
  let text: string;
  if (isGzip) {
    const stream = new Blob([bytes]).stream().pipeThrough(
      new DecompressionStream("gzip"),
    );
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(bytes);
  }
  return JSON.parse(text) as ModelData;
}

function predictForest(
  model: ModelData,
  values: {
    country: string;
    crop: string;
    year: number;
    rainfall: number;
    pesticides: number;
    temperature: number;
  },
): Prediction {
  const featureCount = model.countries.length + model.crops.length + 4;
  const features = new Array<number>(featureCount).fill(0);
  const countryIndex = model.countries.indexOf(values.country);
  const cropIndex = model.crops.indexOf(values.crop);
  if (countryIndex < 0 || cropIndex < 0) {
    throw new Error("Choose a country and crop observed during training.");
  }
  features[countryIndex] = 1;
  features[model.countries.length + cropIndex] = 1;
  const numericStart = model.countries.length + model.crops.length;
  const numericValues = [values.year, values.rainfall, values.temperature];
  numericValues.forEach((value, index) => {
    features[numericStart + index] =
      (value - model.numeric_mean[index]) / model.numeric_scale[index];
  });
  features[numericStart + 3] =
    (Math.log1p(values.pesticides) - model.pesticide_mean) /
    model.pesticide_scale;

  const treePredictions = model.trees.map((tree) => {
    let node = 0;
    while (tree.f[node] >= 0) {
      node =
        features[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    return tree.v[node];
  });
  const mean =
    treePredictions.reduce((total, value) => total + value, 0) /
    treePredictions.length;
  const variance =
    treePredictions.reduce(
      (total, value) => total + (value - mean) ** 2,
      0,
    ) / treePredictions.length;
  return {
    yieldHg: mean,
    yieldTonnes: mean / 10_000,
    spreadTonnes: Math.sqrt(variance) / 10_000,
  };
}

const formatter = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });

export default function Home() {
  const [model, setModel] = useState<ModelData | null>(null);
  const [loadError, setLoadError] = useState("");
  const [country, setCountry] = useState("Ghana");
  const [crop, setCrop] = useState("Maize");
  const [year, setYear] = useState(2013);
  const [rainfall, setRainfall] = useState(1200);
  const [pesticides, setPesticides] = useState(100);
  const [temperature, setTemperature] = useState(25);
  const [prediction, setPrediction] = useState<Prediction | null>(null);

  useEffect(() => {
    loadModel()
      .then((loaded) => {
        setModel(loaded);
        if (!loaded.countries.includes("Ghana")) {
          setCountry(loaded.countries[0]);
        }
        if (!loaded.crops.includes("Maize")) {
          setCrop(loaded.crops[0]);
        }
        setYear(loaded.supported_years[1]);
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);

  const outOfRange = useMemo(() => {
    if (!model) return false;
    const ranges = model.input_ranges;
    return (
      rainfall < ranges.average_rainfall_mm_per_year[0] ||
      rainfall > ranges.average_rainfall_mm_per_year[1] ||
      pesticides < ranges.pesticides_tonnes[0] ||
      pesticides > ranges.pesticides_tonnes[1] ||
      temperature < ranges.average_temperature_c[0] ||
      temperature > ranges.average_temperature_c[1]
    );
  }, [model, rainfall, pesticides, temperature]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!model) return;
    setPrediction(
      predictForest(model, {
        country,
        crop,
        year,
        rainfall,
        pesticides,
        temperature,
      }),
    );
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Crop Yield Intelligence">
          <span className="brand-mark">CY</span>
          <span>Crop Yield Intelligence</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#estimator">Estimator</a>
          <a href="#method">Method</a>
          <a href="#limitations">Limitations</a>
          <a
            className="github-link"
            href="https://github.com/AbubakarMA/crop-yield-intelligence-platform"
            target="_blank"
            rel="noreferrer"
          >
            View source
          </a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Agricultural machine learning · v1.0</p>
          <h1>Turn historical crop data into a defensible yield estimate.</h1>
          <p className="hero-summary">
            Explore how crop, country, rainfall, temperature, pesticide use,
            and year combine in a leakage-tested random forest trained on
            13,130 historical observations.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#estimator">
              Try the estimator
            </a>
            <a className="secondary-action" href="#method">
              See how it works
            </a>
          </div>
          <p className="scope-note">
            Historical conditional estimate—not a causal recommendation or a
            long-range forecast.
          </p>
        </div>
        <div className="hero-panel" aria-label="Final test results">
          <div className="panel-topline">
            <span>Final test · 2011–2013</span>
            <span className="status-dot">Verified</span>
          </div>
          <div className="hero-metric">
            <strong>{model ? model.final_test.r2.toFixed(3) : "0.934"}</strong>
            <span>Test R²</span>
          </div>
          <div className="metric-grid">
            <div>
              <strong>
                {model
                  ? formatter.format(model.final_test.mae_hg_per_ha)
                  : "11,815"}
              </strong>
              <span>MAE · hg/ha</span>
            </div>
            <div>
              <strong>0</strong>
              <span>Negative predictions</span>
            </div>
          </div>
          <div className="trend-visual" aria-hidden="true">
            {[28, 40, 35, 58, 48, 72, 66, 84, 78, 94].map((height, index) => (
              <span key={index} style={{ height: `${height}%` }} />
            ))}
          </div>
          <p>Chronological validation keeps future years out of training.</p>
        </div>
      </section>

      <section className="estimator-section" id="estimator">
        <div className="section-heading">
          <p className="eyebrow">Interactive model</p>
          <h2>Build one historical scenario</h2>
          <p>
            Inputs are restricted to categories and years represented in the
            source data. National pesticide use is not a farm-level dose.
          </p>
        </div>

        <div className="estimator-shell">
          <form onSubmit={handleSubmit}>
            <div className="field-grid">
              <label>
                Country
                <select
                  value={country}
                  onChange={(event) => setCountry(event.target.value)}
                  disabled={!model}
                >
                  {(model?.countries ?? ["Ghana"]).map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label>
                Crop
                <select
                  value={crop}
                  onChange={(event) => setCrop(event.target.value)}
                  disabled={!model}
                >
                  {(model?.crops ?? ["Maize"]).map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label>
                Historical year
                <input
                  type="number"
                  min={model?.supported_years[0] ?? 1990}
                  max={model?.supported_years[1] ?? 2013}
                  value={year}
                  onChange={(event) => setYear(Number(event.target.value))}
                />
              </label>
              <label>
                Annual rainfall (mm)
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={rainfall}
                  onChange={(event) =>
                    setRainfall(Number(event.target.value))
                  }
                />
              </label>
              <label>
                National pesticide use (tonnes)
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={pesticides}
                  onChange={(event) =>
                    setPesticides(Number(event.target.value))
                  }
                />
              </label>
              <label>
                Average temperature (°C)
                <input
                  type="number"
                  min="-30"
                  max="50"
                  step="0.1"
                  value={temperature}
                  onChange={(event) =>
                    setTemperature(Number(event.target.value))
                  }
                />
              </label>
            </div>
            {loadError && <p className="form-message error">{loadError}</p>}
            {outOfRange && (
              <p className="form-message warning">
                At least one numeric input is outside the training range. Move
                it back into range before relying on the estimate.
              </p>
            )}
            <button type="submit" disabled={!model || outOfRange}>
              {model ? "Estimate yield" : "Loading model…"}
            </button>
          </form>

          <aside className="prediction-card" aria-live="polite">
            <p className="result-label">Estimated yield</p>
            <strong className="prediction-value">
              {prediction ? prediction.yieldTonnes.toFixed(2) : "—"}
              <small> t/ha</small>
            </strong>
            <p className="result-subtitle">
              {prediction
                ? `${formatter.format(prediction.yieldHg)} hg/ha`
                : "Complete the scenario and run the model."}
            </p>
            {prediction && (
              <>
                <div className="result-divider" />
                <dl>
                  <div>
                    <dt>Forest spread</dt>
                    <dd>± {prediction.spreadTonnes.toFixed(2)} t/ha</dd>
                  </div>
                  <div>
                    <dt>Model version</dt>
                    <dd>{model?.version}</dd>
                  </div>
                </dl>
                <p className="result-caveat">
                  Tree spread is a model-disagreement signal, not a calibrated
                  confidence interval.
                </p>
              </>
            )}
          </aside>
        </div>
      </section>

      <section className="method-section" id="method">
        <div className="section-heading">
          <p className="eyebrow">From raw data to prediction</p>
          <h2>A production workflow with evidence at every step</h2>
        </div>
        <div className="method-grid">
          {[
            ["01", "Validate", "Resolve duplicates and enforce one country–crop–year observation."],
            ["02", "Split by time", "Train on 1990–2007, validate on 2008–2010, and seal 2011–2013."],
            ["03", "Compare", "Benchmark linear and nonlinear models using MAE, RMSE, and R²."],
            ["04", "Deploy", "Serve a portable 100-tree forest with tested input boundaries."],
          ].map(([number, title, copy]) => (
            <article key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="limitations-section" id="limitations">
        <div>
          <p className="eyebrow">Responsible use</p>
          <h2>What this model does not know</h2>
          <p>
            Strong predictive accuracy does not make the data causal or the
            model suitable for every agricultural decision.
          </p>
        </div>
        <ul>
          {(model?.limitations ?? [
            "The model estimates historical patterns; it is not a long-range climate or policy forecast.",
            "Rainfall is constant through time within each country in the source data.",
            "Pesticide use is a national total rather than a per-hectare rate.",
            "Feature importance is predictive, not causal.",
          ]).map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
      </section>

      <footer>
        <p>
          Built by <strong>Abubakar Mamudu Alutiba</strong> · Crop and Soil
          Sciences + Data Analytics
        </p>
        <a
          href="https://github.com/AbubakarMA/crop-yield-intelligence-platform"
          target="_blank"
          rel="noreferrer"
        >
          Documentation and source code
        </a>
      </footer>
    </main>
  );
}
