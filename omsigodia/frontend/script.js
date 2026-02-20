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

        showResult(out.probability);

    } catch (err) {
        console.log("Backend not working, using fallback");

        // 🔥 fallback logic (demo ke liye)
        let prob = Math.random();  

        showResult(prob);
    }
};

/* RESULT FUNCTION */
function showResult(prob) {
    let risk = Math.floor(prob * 100);

    let color = "#00ff88";
    let text = "SAFE";
    let decision = "ALLOW";

    if (risk > 70) {
        color = "#ff4444";
        text = "🚨 FRAUD DETECTED";
        decision = "BLOCK TRANSACTION";
    } else if (risk > 40) {
        color = "#ff9800";
        text = "⚠️ MEDIUM RISK";
        decision = "OTP REQUIRED";
    }

    result.innerHTML = `
    <div class="result" style="border:2px solid ${color}">
        <h2 style="color:${color}">${text}</h2>
        <p>Risk Score: ${risk}/100</p>
        <p><b>Decision:</b> ${decision}</p>
    </div>`;
}
