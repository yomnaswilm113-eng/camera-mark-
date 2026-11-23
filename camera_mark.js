// camera_mark.js
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();   // UI de-duplication only

// ── Start camera ──────────────────────────────────────────────────────────────
startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  recognizedIds.clear();

  // 1. Tell the backend the session is starting (camera is now ON)
  try {
    await fetch("/start_session", { method: "POST" });
  } catch (err) {
    console.error("start_session error:", err);
  }

  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    markVideo.srcObject = markStream;
    await markVideo.play();
    markStatus.innerText = "Scanning…";
    markInterval = setInterval(captureAndRecognize, 1200);
  } catch (err) {
    alert("Camera error: " + err.message);
    await endSession();           // cancel session if camera fails
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
  }
});

// ── Stop camera ───────────────────────────────────────────────────────────────
stopMarkBtn.addEventListener("click", async () => {
  await endSession();
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
});

// ── Helper: tear down camera + end backend session ────────────────────────────
async function endSession() {
  if (markInterval) { clearInterval(markInterval); markInterval = null; }
  if (markStream) { markStream.getTracks().forEach(t => t.stop()); markStream = null; }

  markStatus.innerText = "Stopping… saving attendance";

  try {
    const res = await fetch("/end_session", { method: "POST" });
    const data = await res.json();

    if (data.saved > 0) {
      markStatus.innerText =
        `✅ Session ended — ${data.saved} student(s) saved: ${data.students.join(", ")}`;
    } else {
      markStatus.innerText = "Session ended — no attendance saved (no faces recognised).";
    }
  } catch (err) {
    console.error("end_session error:", err);
    markStatus.innerText = "Session ended (server error while saving).";
  }
}

// ── Safety net: end session if the user closes / refreshes the tab ────────────
window.addEventListener("beforeunload", () => {
  // sendBeacon is fire-and-forget, safe to call during page unload
  navigator.sendBeacon("/end_session");
});

// ── Per-frame recognition ─────────────────────────────────────────────────────
async function captureAndRecognize() {
  const canvas = document.createElement("canvas");
  canvas.width = markVideo.videoWidth || 640;
  canvas.height = markVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.85));
  const fd = new FormData();
  fd.append("image", blob, "snap.jpg");

  try {
    const res = await fetch("/recognize_face", { method: "POST", body: fd });
    const j = await res.json();

    if (j.recognized) {
      markStatus.innerText =
        `👤 Recognised: ${j.name} (${Math.round(j.confidence * 100)}% conf) — will be saved on stop`;

      // Add to the live UI list only the first time we see this student
      if (!recognizedIds.has(j.student_id)) {
        recognizedIds.add(j.student_id);
        const li = document.createElement("li");
        li.className = "list-group-item";
        li.innerText = `${j.name} — first seen ${new Date().toLocaleTimeString()}`;
        recognizedList.prepend(li);
      }
    } else if (j.error === "no_active_session") {
      // Session was ended server-side (shouldn't happen in normal flow)
      clearInterval(markInterval);
      markStatus.innerText = "Session no longer active.";
    } else {
      markStatus.innerText = j.error
        ? `Not recognised: ${j.error}`
        : "No face detected";
    }
  } catch (err) {
    console.error("recognize error:", err);
  }
}
