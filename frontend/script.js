
const API_BASE_URL = "https://fraud-detection-api-fmy7.onrender.com";

const form = document.getElementById("transaction-form");
const scoreButton = document.getElementById("score-button");
const buttonText = document.getElementById("button-text");
const errorMessage = document.getElementById("error-message");

const apiStatus = document.getElementById("api-status");
const statusDot = document.querySelector(".status-dot");

const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");
const riskBanner = document.getElementById("risk-banner");

function setApiIndicator(isOnline, message) {
    apiStatus.textContent = message;
    statusDot.style.background = isOnline ? "#4adea2" : "#ff777e";
    statusDot.style.color = isOnline ? "#4adea2" : "#ff777e";
}

async function checkHealth() {
    const healthStatus = document.getElementById("health-status");

    try {
        const response = await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new Error("Health check failed");
        }

        const data = await response.json();

        if (data.status === "healthy" && data.model_loaded === true) {
            healthStatus.textContent = "Online · Model loaded";
            healthStatus.style.color = "#4adea2";
            setApiIndicator(true, "API Connected");
        } else {
            healthStatus.textContent = "Needs attention";
            healthStatus.style.color = "#f6c96b";
            setApiIndicator(false, "Check API");
        }
    } catch (error) {
        healthStatus.textContent = "Unavailable";
        healthStatus.style.color = "#ff777e";
        setApiIndicator(false, "API unavailable");
    }
}

async function checkMonitoring() {
    const driftStatus = document.getElementById("drift-status");
    const monitorNote = document.getElementById("monitor-note");

    try {
        const response = await fetch(`${API_BASE_URL}/monitoring/status`);

        if (!response.ok) {
            throw new Error("Monitoring endpoint failed");
        }

        const data = await response.json();
        const status = String(data.status || "UNKNOWN").toUpperCase();

        driftStatus.textContent = status;
        driftStatus.style.color =
            status === "OK" || status === "NO_DRIFT"
                ? "#4adea2"
                : status === "NOT_AVAILABLE"
                    ? "#f6c96b"
                    : "#93a2b9";

        if (status === "NOT_AVAILABLE") {
            monitorNote.textContent =
                "Drift status is not available. Check the monitoring pipeline.";
        } else {
            monitorNote.textContent =
                "Status reported by the deployed monitoring endpoint. See API docs for details.";
        }
    } catch (error) {
        driftStatus.textContent = "Unavailable";
        driftStatus.style.color = "#ff777e";
        monitorNote.textContent =
            "Could not retrieve monitoring information. Try again later.";
    }
}

function displayPrediction(data) {
    const probability = Number(data.fraud_probability);
    const threshold = Number(data.threshold);

    if (
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1 ||
        !Number.isFinite(threshold)
    ) {
        throw new Error("The API returned an invalid prediction.");
    }

    const isFlagged = data.flagged === true;

    resultPlaceholder.hidden = true;
    resultContent.hidden = false;

    riskBanner.className = `risk-banner ${isFlagged ? "high-risk" : "low-risk"
        }`;

    document.getElementById("risk-symbol").textContent =
        isFlagged ? "⚠" : "✓";

    document.getElementById("risk-title").textContent =
        isFlagged ? "Flagged for review" : "Not flagged";

    document.getElementById("risk-description").textContent =
        isFlagged
            ? "The model score meets or exceeds the configured threshold."
            : "The model score is below the configured threshold.";

    document.getElementById("probability-text").textContent =
        `${(probability * 100).toFixed(2)}%`;

    document.getElementById("probability-bar").style.width =
        `${(probability * 100).toFixed(2)}%`;

    document.getElementById("threshold-text").textContent =
        `${(threshold * 100).toFixed(2)}%`;

    document.getElementById("model-version").textContent =
        data.model_version || "Not provided";

    document.getElementById("decision-note").textContent =
        `API risk level: ${data.risk_level || "Not provided"}. ` +
        "This is a model prediction, not a confirmed fraud finding.";
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    errorMessage.textContent = "";

    const amount = Number(document.getElementById("amount").value);
    const step = Number(document.getElementById("step").value);
    const type = document.getElementById("type").value;
    const nameDest = document.getElementById("destination").value.trim();

    if (!Number.isFinite(amount) || amount < 0) {
        errorMessage.textContent = "Enter a valid, non-negative amount.";
        return;
    }

    if (!Number.isSafeInteger(step) || step < 0) {
        errorMessage.textContent =
            "Transaction step must be a non-negative whole number.";
        return;
    }

    if (!nameDest) {
        errorMessage.textContent = "Enter a destination account ID.";
        return;
    }

    const transaction = {
        step,
        amount,
        type,
        nameDest
    };

    scoreButton.disabled = true;
    buttonText.textContent = "Analyzing transaction...";

    try {
        const response = await fetch(`${API_BASE_URL}/score`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(transaction)
        });

        const responseData = await response.json().catch(() => ({}));

        if (!response.ok) {
            const detail = responseData.detail;
            const message = typeof detail === "string"
                ? detail
                : "The API rejected the request. Check your inputs.";
            throw new Error(message);
        }

        displayPrediction(responseData);
    } catch (error) {
        errorMessage.textContent =
            error.name === "TypeError"
                ? "Unable to reach the API. Check your connection or try again when the service is awake."
                : error.message || "Prediction failed. Please try again.";

        console.error("Transaction scoring error:", error);
    } finally {
        scoreButton.disabled = false;
        buttonText.textContent = "Analyze transaction →";
    }
});

// Run the initial service checks when the page opens.
checkHealth();
checkMonitoring();
