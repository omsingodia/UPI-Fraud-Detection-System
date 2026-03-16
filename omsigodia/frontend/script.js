form.onsubmit = async (e) => {
    e.preventDefault();

    result.innerHTML = "⏳ Analyzing...";

    let data = Object.fromEntries(new FormData(form));

    try {
        let res = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        let out = await res.json();

        if (out.error) {
            result.innerHTML = `<h3 style="color:#ff4444">❌ API Error: ${out.error}</h3>`;
            return;
        }

        showResult(out);

    } catch (err) {
        result.innerHTML = "<h3 style='color:#ff4444'>❌ Backend not reachable</h3>";
    }
};

/* RESULT FUNCTION */
function showResult(out) {
    let risk = out.risk_score;
    let decision = out.decision;
    let bucket = out.risk_bucket || "LOW RISK";

    let color = "#00ff88";

    if (bucket === "HIGH RISK") color = "#ff4444";
    else if (bucket === "MEDIUM RISK") color = "#ff9800";

    result.innerHTML = `
    <div class="result" style="border:2px solid ${color}">
        <h2 style="color:${color}">${bucket}</h2>
        <p>Risk Score: ${risk}/100</p>
        <p><b>Decision:</b> ${decision}</p>
    </div>`;
}
