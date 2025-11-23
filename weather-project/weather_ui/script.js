const CLASSES = ["Cloudy", "Foggy", "Rain", "Snow", "Clear"];
let session = null;
let genAI = null;

// DEBUG: Check key
console.log("GEMINI KEY:", window.GEMINI_API_KEY ? "LOADED" : "MISSING");
if (!window.GEMINI_API_KEY || window.GEMINI_API_KEY.includes("HERE")) {
  console.error("GEMINI KEY MISSING! Get it from: https://aistudio.google.com/app/apikey");
}

// Load ONNX model
async function loadModel() {
  try {
    session = await ort.InferenceSession.create("./model/weather_model.onnx");
    console.log("ONNX Model loaded");
  } catch (e) {
    console.error("MODEL FAILED:", e);
    alert("Model not found! Check /model/weather_model.onnx");
  }
}

// Initialize Gemini with ESM wait
async function initGemini() {
  if (!window.GEMINI_API_KEY || window.GEMINI_API_KEY.includes("HERE")) {
    console.warn("Gemini disabled: No valid key");
    return;
  }

  if (genAI) return;

  try {
    console.log("Waiting for Gemini SDK (ESM)...");
    await new Promise(resolve => {
      const check = setInterval(() => {
        if (window.GoogleGenerativeAI) {
          clearInterval(check);
          resolve();
        }
      }, 100);
      setTimeout(() => { clearInterval(check); resolve(); }, 10000);
    });

    if (!window.GoogleGenerativeAI) {
      throw new Error("Gemini SDK failed to load");
    }

    genAI = new window.GoogleGenerativeAI(window.GEMINI_API_KEY);
    console.log("Gemini AI READY!");
  } catch (e) {
    console.error("Gemini init failed:", e);
  }
}

// Preprocess image
function preprocess(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 128; canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, 128, 128);
  const data = ctx.getImageData(0, 0, 128, 128).data;

  const red = [], green = [], blue = [];
  for (let i = 0; i < data.length; i += 4) {
    red.push((data[i] / 255 - 0.485) / 0.229);
    green.push((data[i + 1] / 255 - 0.456) / 0.224);
    blue.push((data[i + 2] / 255 - 0.406) / 0.225);
  }

  return new ort.Tensor("float32", new Float32Array([...red, ...green, ...blue]), [1, 3, 128, 128]);
}

// Predict
async function predict() {
  const img = document.getElementById("preview");
  if (!img.src || !session) return;

  console.log("Running prediction...");
  const input = preprocess(img);
  const results = await session.run({ input });
  const output = results.output.data;
  const probs = softmax(Array.from(output));

  await displayResults(probs);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// Generate tip
async function generateGeminiTip(weather, confidence) {
  if (!genAI) {
    console.log("Gemini not available → fallback");
    return "Make it a great day!";
  }

  const prompt = `
You are a pro weather forecaster + fashion stylist + vibe curator for a viral AI lifestyle app.

Input:

Current Weather: ${weather} (${confidence}% confidence from image)

Time: Now

Output (for next 2 hours):
Predict how the weather will trend — better, worse, or same.
Then give a mini weather forecast (e.g., “Rain picking up fast,” “Skies clearing soon,” “Storm easing off”).

Next, give a fun, practical lifestyle summary (2–3 sentences max) including:

👕 Outfit pick (comfort + aesthetic + temperature awareness)

⚠️ Precaution/advice (e.g., carry umbrella, stay hydrated, avoid roads)

🚶 Whether to go outside or chill indoors (decisive & confident)

🎧 Mood / activity / music vibe (creative & trendy — lo-fi, gym, café jazz, etc.)

Use 1–2 emojis per line for flair.

Tone:
Playful, confident, and TikTok-shareable — sounds like a  friend who knows both the sky and the style.

Example outputs:

“🌧️ Rain 85% → getting heavier! Grab your waterproof bomber + sneakers. Avoid long drives, chill indoors with chai ☕ + lo-fi beats.”

“☀️ Clear skies 93% → staying bright! Light linen shirt + shades. Step out, soak the sun, and vibe with chill summer tunes 🎶.”

“🌫️ Fog 70% → thickening ahead! Go for cozy layers + warm scarf. Stay alert if driving, best to café-hop nearby ☕.”

Output Length:
Keep it short enough for a social media caption, but long enough to feel like a mini lifestyle forecast (around 60–100 words)..
`.trim();
  try {
    console.log("Calling Gemini...");
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-liter" });
    console.log(model);
    const result = await model.generateContent(prompt);
    const tip = result.response.text().trim();
    console.log("Gemini says:", tip);
    return tip;
  } catch (e) {
    console.error("Gemini API error:", e);
    return "Weather's calling — own it!";
  }
}

// Display results
async function displayResults(probs) {
  await initGemini();

  const barsDiv = document.getElementById("bars");
  const topPred = document.getElementById("top-pred");
  const result = document.getElementById("result");
  result.classList.remove("hidden");
  barsDiv.innerHTML = "";

  let maxIdx = 0;
  probs.forEach((p, i) => {
    if (p > probs[maxIdx]) maxIdx = i;

    const div = document.createElement("div");
    div.className = "bar-container";
    div.innerHTML = `
      <div class="bar-label">
        <span>${CLASSES[i]}</span>
        <span>${(p * 100).toFixed(1)}%</span>
      </div>
      <div class="bar"><div class="bar-fill" style="width: 0%"></div></div>
    `;
    barsDiv.appendChild(div);

    setTimeout(() => {
      div.querySelector(".bar-fill").style.width = `${p * 100}%`;
    }, 100);
  });

  const topWeather = CLASSES[maxIdx];
  const confidence = (probs[maxIdx] * 100).toFixed(0);
  topPred.innerHTML = `<strong>Most likely: ${topWeather} (${confidence}%)</strong><br><em>Generating tip...</em>`;

  const tip = await generateGeminiTip(topWeather, confidence);
  topPred.innerHTML = `
    <strong>Most likely: ${topWeather} (${confidence}%)</strong><br>
    <span>${tip}</span>
  `;
}

// UI Events
document.getElementById("drop-area").addEventListener("click", () => {
  document.getElementById("file-input").click();
});

document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const prev = document.getElementById("preview");
    prev.src = ev.target.result;
    prev.style.display = "block";
    document.getElementById("predict-btn").disabled = false;
  };
  reader.readAsDataURL(file);
});

document.getElementById("predict-btn").addEventListener("click", predict);

const dropArea = document.getElementById("drop-area");
["dragenter", "dragover", "dragleave", "drop"].forEach(ev => {
  dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
});
["dragenter", "dragover"].forEach(ev => {
  dropArea.addEventListener(ev, () => dropArea.style.borderColor = "#00b894");
});
["dragleave", "drop"].forEach(ev => {
  dropArea.addEventListener(ev, () => dropArea.style.borderColor = "#74b9ff");
});
dropArea.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = ev => {
      const prev = document.getElementById("preview");
      prev.src = ev.target.result;
      prev.style.display = "block";
      document.getElementById("predict-btn").disabled = false;
    };
    reader.readAsDataURL(file);
  }
});

// Load model
loadModel();