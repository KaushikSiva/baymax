const query = new URLSearchParams(location.search);
const wsPort = query.get("wsPort") || "8770";
let host = query.get("wsHost") || location.hostname || "127.0.0.1";
const secure = location.protocol === "https:";
if (!query.has("wsHost") && /^.+-\d+\.proxy\.runpod\.net$/.test(host)) {
  host = host.replace(/-\d+\.proxy\.runpod\.net$/, `-${wsPort}.proxy.runpod.net`);
}
const proxied = /\.proxy\.runpod\.net$/.test(host);
const socketUrl = proxied
  ? `${secure ? "wss" : "ws"}://${host}`
  : `${secure ? "wss" : "ws"}://${host}:${wsPort}`;

const el = {
  connection: document.querySelector("#connection"),
  broadcast: document.querySelector("#broadcast"),
  ego: document.querySelector("#ego"),
  phase: document.querySelector("#phase"),
  rationale: document.querySelector("#rationale"),
  zone: document.querySelector("#zone"),
  incidents: document.querySelector("#incidents"),
  dispatchCount: document.querySelector("#dispatch-count"),
  transcript: document.querySelector("#transcript"),
  form: document.querySelector("#transcript-form"),
  input: document.querySelector("#transcript-input"),
  mic: document.querySelector("#mic"),
};
let socket;

const sequence = [
  "navigate_room_1",
  "inspect_room_1",
  "navigate_room_2",
  "inspect_room_2",
  "return_home",
];

function frame(image, encoded) {
  if (encoded) image.src = `data:image/jpeg;base64,${encoded}`;
}

function zoneFor(state) {
  if (state.rooms?.room_2?.atInspectionPoint) return "ROOM 202";
  if (state.rooms?.room_1?.atInspectionPoint) return "ROOM 101";
  return "CORRIDOR";
}

function renderIncidents(state) {
  const dispatches = state.dispatches || [];
  el.dispatchCount.textContent = `${dispatches.filter(item => item.ok).length} / ${state.expectedDispatches || 2}`;
  if (!dispatches.length) {
    el.incidents.innerHTML = '<li class="empty">No incidents dispatched.</li>';
    return;
  }
  el.incidents.innerHTML = dispatches.slice().reverse().map(item => `
    <li>
      <b>${String(item.incidentType || "incident").replaceAll("_", " ").toUpperCase()}</b>
      <span>${item.roomId === "room_1" ? "Room 101" : "Room 202"}</span>
      <em>${item.ok ? "API ACCEPTED" : "RETRYING"}</em>
    </li>`).join("");
}

function update(message) {
  const state = message.state || {};
  const robot = state.robot || {};
  const skill = robot.currentSkill || "wait";
  el.phase.textContent = state.result || (state.speech?.listening
    ? (state.speech?.speaking ? "LISTENING TO GRANDMA" : "WAITING FOR GRANDMA")
    : skill.replaceAll("_", " ").toUpperCase());
  el.rationale.textContent = robot.rationale || "Monitoring grounded patrol state.";
  el.zone.textContent = zoneFor(state);
  el.transcript.textContent = state.speech?.transcript || "Waiting for patient speech.";
  frame(el.broadcast, message.frames?.broadcast);
  frame(el.ego, message.frames?.ego);
  renderIncidents(state);

  let active = sequence.indexOf(skill);
  if (skill === "dispatch_incident") {
    active = state.rooms?.room_2?.visited ? 3 : 1;
  }
  document.querySelectorAll(".patrol li").forEach((item, index) => {
    const room1Done = state.rooms?.room_1?.visited;
    const room2Done = state.rooms?.room_2?.visited;
    const done = index < 2 ? room1Done : index < 4 ? room2Done : Boolean(state.result);
    item.classList.toggle("active", index === active && !state.result);
    item.classList.toggle("done", done);
  });
  document.body.classList.toggle("alerting", (state.pendingIncidents || []).length > 0);
  document.body.classList.toggle("finished", state.result === "PATROL COMPLETE");
}

function sendTranscript(value, final = true) {
  const transcript = value.trim();
  if (!transcript || !socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ type: "patient_transcript", transcript, final }));
  el.input.value = "";
}

el.form.addEventListener("submit", event => {
  event.preventDefault();
  sendTranscript(el.input.value);
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.addEventListener("start", () => {
    el.mic.classList.add("listening");
    el.mic.textContent = "LISTENING";
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "patient_speech_started" }));
    }
  });
  recognition.addEventListener("end", () => {
    el.mic.classList.remove("listening");
    el.mic.textContent = "MIC";
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "patient_speech_finished" }));
    }
  });
  recognition.addEventListener("result", event => {
    sendTranscript(event.results[0][0].transcript, event.results[0].isFinal);
  });
  el.mic.addEventListener("click", () => recognition.start());
} else {
  el.mic.disabled = true;
  el.mic.title = "Speech recognition unavailable; use typed transcript.";
}

function connect() {
  socket = new WebSocket(socketUrl);
  socket.addEventListener("open", () => { el.connection.textContent = "LIVE"; });
  socket.addEventListener("message", event => {
    try { update(JSON.parse(event.data)); } catch (_) {}
  });
  socket.addEventListener("close", () => {
    el.connection.textContent = "RECONNECTING";
    setTimeout(connect, 1000);
  });
  socket.addEventListener("error", () => socket.close());
}

connect();
