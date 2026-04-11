const API_BASE = "http://127.0.0.1:8000";

let predictionChart = null;
let networkGraph = null;
console.log("Script loaded");

document.addEventListener("DOMContentLoaded", () => {
  const isLoginPage = !!document.getElementById("loginForm");
  const isDashboardPage = !!document.getElementById("predictBtn");

  if (isLoginPage) {
    setupLoginPage();
  }
  if (isDashboardPage) {
    setupDashboardPage();
  }
});

function setupLoginPage() {
  const loginForm = document.getElementById("loginForm");
  const loginBtn = document.getElementById("loginBtn");
  const signupForm = document.getElementById("signupForm");
  const signupBtn = document.getElementById("signupBtn");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const signupEmailInput = document.getElementById("signupEmail");
  const signupPasswordInput = document.getElementById("signupPassword");
  const rememberMe = document.getElementById("rememberMe");
  const emailError = document.getElementById("emailError");
  const passwordError = document.getElementById("passwordError");
  const signupEmailError = document.getElementById("signupEmailError");
  const signupPasswordError = document.getElementById("signupPasswordError");
  const authMessage = document.getElementById("authMessage");

  const storedEmail = localStorage.getItem("rememberedEmail");
  if (storedEmail) {
    emailInput.value = storedEmail;
    rememberMe.checked = true;
  }

  if (loginBtn) {
    loginBtn.addEventListener("click", () => {
      console.log("Login button clicked");
    });
  }
  if (signupBtn) {
    signupBtn.addEventListener("click", () => {
      console.log("Signup button clicked");
    });
  }

  if (signupForm) {
    signupForm.addEventListener("submit", (event) => {
      event.preventDefault();
      clearAuthErrors(
        signupEmailError,
        signupPasswordError,
        emailError,
        passwordError,
        authMessage
      );

      const email = signupEmailInput.value.trim();
      const password = signupPasswordInput.value.trim();
      let isValid = true;

      if (!email) {
        signupEmailError.textContent = "Email is required.";
        isValid = false;
      } else if (!isValidEmail(email)) {
        signupEmailError.textContent = "Enter a valid email format.";
        isValid = false;
      }

      if (!password) {
        signupPasswordError.textContent = "Password is required.";
        isValid = false;
      }

      if (!isValid) {
        setAuthMessage(authMessage, "Please fill signup details correctly.", "error");
        return;
      }

      const newUser = { email, password };
      localStorage.setItem("user", JSON.stringify(newUser));
      console.log("Signup successful");
      console.log("Stored user:", newUser);
      setAuthMessage(authMessage, "Account created successfully", "success");

      // Pre-fill login form for convenience.
      emailInput.value = email;
      passwordInput.value = "";
      signupPasswordInput.value = "";
    });
  }

  loginForm.addEventListener("submit", (event) => {
    console.log("Login button clicked");
    event.preventDefault();

    emailError.textContent = "";
    passwordError.textContent = "";

    const email = emailInput.value.trim();
    const password = passwordInput.value.trim();
    let isValid = true;

    if (!email) {
      emailError.textContent = "Email is required.";
      isValid = false;
    } else if (!isValidEmail(email)) {
      emailError.textContent = "Enter a valid email format.";
      isValid = false;
    }

    if (!password) {
      passwordError.textContent = "Password is required.";
      isValid = false;
    }

    if (!isValid) {
      setAuthMessage(authMessage, "Please enter a valid email and password.", "error");
      return;
    }
    console.log("Login attempt:", { email, password });

    const storedUserRaw = localStorage.getItem("user");
    let storedUser = null;
    if (storedUserRaw) {
      try {
        storedUser = JSON.parse(storedUserRaw);
      } catch (error) {
        console.error("Could not parse stored user JSON:", error);
      }
    }
    console.log("Stored user:", storedUser);

    if (
      !storedUser ||
      storedUser.email !== email ||
      storedUser.password !== password
    ) {
      passwordError.textContent = "Invalid credentials.";
      setAuthMessage(authMessage, "Invalid credentials", "error");
      console.log("Login failed");
      return;
    }

    if (rememberMe.checked) {
      localStorage.setItem("rememberedEmail", email);
    } else {
      localStorage.removeItem("rememberedEmail");
    }

    sessionStorage.setItem("isLoggedIn", "true");
    sessionStorage.setItem("userEmail", email);
    console.log("Login success");
    console.log("Redirecting...");
    window.location.href = "dashboard.html";
  });
}

function setupDashboardPage() {
  protectDashboardRoute();
  loadMLMetrics();
  bindSliderOutput("fever", "feverVal");
  bindSliderOutput("cough", "coughVal");
  bindSliderOutput("headache", "headacheVal");
  bindSliderOutput("fatigue", "fatigueVal");

  const userEmail = sessionStorage.getItem("userEmail") || "demo@user.com";
  const userEmailDisplay = document.getElementById("userEmailDisplay");
  if (userEmailDisplay) {
    userEmailDisplay.textContent = userEmail;
  }

  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      sessionStorage.removeItem("isLoggedIn");
      sessionStorage.removeItem("userEmail");
      window.location.href = "index.html";
    });
  }

  const predictBtn = document.getElementById("predictBtn");
  predictBtn.addEventListener("click", function () {
    console.log("Predict button clicked");
    runPredictionFlow();
  });
}

function protectDashboardRoute() {
  const isLoggedIn = sessionStorage.getItem("isLoggedIn");
  if (isLoggedIn !== "true") {
    window.location.href = "index.html";
  }
}

function bindSliderOutput(inputId, outputId) {
  const input = document.getElementById(inputId);
  const output = document.getElementById(outputId);
  if (!input || !output) {
    return;
  }
  output.textContent = input.value;
  input.addEventListener("input", () => {
    output.textContent = input.value;
    // Debounce to avoid spamming the server
    if (window._predTimeout) clearTimeout(window._predTimeout);
    window._predTimeout = setTimeout(() => {
      runPredictionFlow(true);
    }, 200);
  });
}

async function runPredictionFlow(isRealtime = false) {
  console.log("Prediction flow triggered. Realtime:", isRealtime);
  if (!isRealtime) setStatus("Calling prediction service...");
  if (!isRealtime) toggleLoading(true);

  const symptoms = collectSymptoms();
  try {
    console.log("API called");
    const data = await fetch("http://127.0.0.1:8000/predict-from-tigergraph", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ symptoms, source_mode: "vertex" }),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Prediction API failed with status ${res.status}`);
        }
        return res.json();
      })
      .then((jsonData) => {
        console.log("Prediction response:", jsonData);
        console.log("Response received");
        return jsonData;
      });

    setStatus("Prediction complete.");

    const samples = data.sample_predictions || [];
    updateResultPanel(samples, data.message);
    renderPredictionChart(samples);
    renderNetworkGraph((data.graph && data.graph.nodes) || [], (data.graph && data.graph.edges) || []);

    // Pull the latest graph snapshot from backend.
    await refreshGraphData();
  } catch (error) {
    console.error("Fetch error:", error);
    alert("Failed to connect to backend");
    setStatus("Prediction failed.");
    showError(error.message || "Unknown error while predicting.");
  } finally {
    toggleLoading(false);
  }
}

async function refreshGraphData() {
  try {
    const response = await fetch(`${API_BASE}/graph-data`);
    if (!response.ok) {
      throw new Error(`Graph API failed with status ${response.status}`);
    }
    const graphData = await response.json();
    renderNetworkGraph(graphData.nodes || [], graphData.edges || []);
    console.log("Graph updated");
  } catch (error) {
    console.error("Graph refresh error:", error);
  }
}

function collectSymptoms() {
  const symptoms = {
    fever: parseInt(document.getElementById("fever").value, 10),
    cough: parseInt(document.getElementById("cough").value, 10),
    headache: parseInt(document.getElementById("headache").value, 10),
    fatigue: parseInt(document.getElementById("fatigue").value, 10),
    chest_pain: 0
  };
  console.log("Collected symptoms:", symptoms);
  return symptoms;
}

function updateResultPanel(samples, message) {
  const panel = document.getElementById("resultPanel");
  if (!panel) {
    return;
  }

  if (!samples.length) {
    panel.innerHTML = `<div class="result-empty">No predictions were returned.</div>`;
    return;
  }

  const top = samples[0];
  const confidence = top.score != null ? `${(top.score * 100).toFixed(1)}%` : "N/A";

  const rows = samples
    .slice(0, 5)
    .map((item) => {
      const scoreText = item.score != null ? (item.score * 100).toFixed(1) + "%" : "N/A";
      return `<li><span>${item.id}</span><span>${item.prediction} (${scoreText})</span></li>`;
    })
    .join("");

  panel.innerHTML = `
    <h3 class="result-headline">${top.prediction}</h3>
    <div class="confidence-chip">
      <i class="fa-solid fa-signal"></i>
      Confidence: ${confidence}
    </div>
    <p class="status-text">${escapeHtml(message || "Prediction generated successfully.")}</p>
    <ul class="prediction-list">${rows}</ul>
  `;
}

function renderPredictionChart(samples) {
  const canvas = document.getElementById("predictionChart");
  if (!canvas) {
    return;
  }

  const grouped = {};
  for (const item of samples) {
    const label = String(item.prediction || "Unknown");
    grouped[label] = (grouped[label] || 0) + 1;
  }

  const labels = Object.keys(grouped);
  const values = Object.values(grouped);
  const colors = labels.map(colorFromLabel);

  if (predictionChart) {
    predictionChart.destroy();
  }

  predictionChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Prediction Count",
          data: values,
          backgroundColor: colors,
          borderRadius: 10,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#e5eeff" },
        },
      },
      scales: {
        x: { ticks: { color: "#b8c6eb" }, grid: { color: "rgba(200,210,255,0.08)" } },
        y: {
          beginAtZero: true,
          ticks: { color: "#b8c6eb" },
          grid: { color: "rgba(200,210,255,0.08)" },
        },
      },
    },
  });
}

function renderNetworkGraph(nodes, edges) {
  const container = document.getElementById("networkGraph");
  if (!container || typeof vis === "undefined") {
    return;
  }

  const data = {
    nodes: new vis.DataSet(nodes),
    edges: new vis.DataSet(edges),
  };

  const options = {
    nodes: {
      shape: "dot",
      size: 20,
      borderWidth: 2,
      font: { color: "#f2f6ff", size: 13, face: "Inter" },
    },
    edges: {
      color: { color: "#91a1ca" },
      smooth: true,
    },
    physics: {
      enabled: false // Disable constant movement
    },
    interaction: {
      hover: true,
      navigationButtons: true,
      keyboard: true,
    },
  };

  if (networkGraph) {
    networkGraph.destroy();
  }

  networkGraph = new vis.Network(container, data, options);
}

function colorFromLabel(label) {
  const lower = String(label).toLowerCase();
  if (lower.includes("healthy")) {
    return "#2ecc71";
  }
  if (lower.includes("flu")) {
    return "#f39c12";
  }
  if (lower.includes("cold")) {
    return "#3498db";
  }
  if (lower.includes("severe") || lower.includes("critical")) {
    return "#ff5a7a";
  }
  return "#8e7dff";
}

function setStatus(text) {
  const status = document.getElementById("statusMessage");
  if (status) {
    status.textContent = text;
  }
}

function toggleLoading(isLoading) {
  const overlay = document.getElementById("loadingOverlay");
  if (!overlay) {
    return;
  }
  overlay.classList.toggle("hidden", !isLoading);
}

function showError(message) {
  const panel = document.getElementById("resultPanel");
  if (!panel) {
    return;
  }
  panel.innerHTML = `<div class="result-empty" style="color:#ff9db2;">${escapeHtml(message)}</div>`;
}

function isValidEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function clearAuthErrors(...nodes) {
  for (const node of nodes) {
    if (node) {
      node.textContent = "";
    }
  }
}

function setAuthMessage(node, message, type) {
  if (!node) {
    return;
  }
  node.textContent = message;
  node.style.color = type === "success" ? "#2ecc71" : "#ff5a7a";
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function loadMLMetrics() {
  try {
    const response = await fetch(`${API_BASE}/ml-metrics`);
    const data = await response.json();
    if (data.status === "ok") {
      document.getElementById("accuracyValue").textContent = `${data.accuracy}%`;
      document.getElementById("precisionValue").textContent = `${data.precision}%`;
    }
  } catch (error) {
    console.error("Error loading metrics:", error);
  }
}
