const CLASSES = ["Cloudy", "Foggy", "Rain", "Snow", "Clear"];
let session = null;

// Load ONNX model
async function loadModel() {
  try {
    session = await ort.InferenceSession.create("./model/weather_model.onnx");
    console.log("Model loaded");
  } catch (e) {
    alert("Failed to load model. Check path.");
    console.error(e);
  }
}

// Preprocess image to 128x128
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

  const input = preprocess(img);
  const results = await session.run({ input });
  const output = results.output.data;
  const probs = softmax(Array.from(output));

  displayResults(probs);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function displayResults(probs) {
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

  topPred.textContent = `Most likely: ${CLASSES[maxIdx]} (${(probs[maxIdx] * 100).toFixed(1)}%)`;
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

// Drag & drop
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

// Load model on start
loadModel();