/* Ara web client. Vanilla JS: no build step, no third-party scripts, no hidden uploads. */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const state = {
    csrf: null,
    conversationId: null,
    view: "chat",
    mode: "chat",
    sessionPromise: null,
    stateData: null,
    streaming: false,
    controller: null,
    recording: null,
    recordingStarted: 0,
    recordingTimer: null,
    cameraStream: null,
    newsLoaded: false,
    topic: "",
    newsLoading: false,
    memory: [],
    correctId: null,
    confirmHandler: null,
    speechEnabled: localStorage.getItem("ara.speak") === "1",
    spoken: null,
  };

  const dom = {
    landing: $("#landing"),
    gate: $("#gate"),
    gateForm: $("#gate-form"),
    gateError: $("#gate-error"),
    gatePassword: $("#gate-password"),
    app: $("#app"),
    chatLog: $("#chat-log"),
    chatEmpty: $("#chat-empty"),
    composer: $("#composer"),
    message: $("#message"),
    send: $("#send-button"),
    stop: $("#stop-button"),
    studyBanner: $("#study-banner"),
    viewTitle: $("#view-title"),
    viewKicker: $("#view-kicker"),
    modePill: $("#mode-pill"),
    modelPill: $("#model-pill"),
    railStatus: $("#rail-status-text"),
    railDot: $("#rail-dot"),
    sensorStrip: $("#sensor-strip"),
    sensorLabel: $("#sensor-label"),
    sensorTime: $("#sensor-time"),
    sensorStop: $("#sensor-stop"),
    micButton: $("#mic-button"),
    speakToggle: $("#speak-toggle"),
    offlineChip: $("#offline-chip"),
    toastRegion: $("#toast-region"),
    newsGrid: $("#news-grid"),
    newsEmpty: $("#news-empty"),
    newsMeta: $("#news-meta"),
    newsCacheNote: $("#news-cache-note"),
    topicRow: $("#topic-row"),
    memoryList: $("#memory-list"),
    memoryEmpty: $("#memory-empty"),
    memoryCount: $("#memory-count"),
    cameraDialog: $("#camera-dialog"),
    cameraVideo: $("#camera-video"),
    cameraIdle: $("#camera-idle"),
    cameraLive: $("#camera-live"),
    cameraError: $("#camera-error"),
    cameraEnable: $("#camera-enable"),
    cameraCapture: $("#camera-capture"),
    cameraStop: $("#camera-stop"),
    confirmDialog: $("#confirm-dialog"),
    confirmText: $("#confirm-text"),
    confirmInput: $("#confirm-input"),
    confirmLabel: $("#confirm-label"),
    confirmAccept: $("#confirm-accept"),
    confirmError: $("#confirm-error"),
    correctDialog: $("#correct-dialog"),
    correctContent: $("#correct-content"),
    correctError: $("#correct-error"),
  };

  /* ------------------------------------------------------------------ utility */
  function toast(message, kind = "", timeout = 5200) {
    const node = el("div", `toast ${kind}`.trim(), message);
    dom.toastRegion.append(node);
    window.setTimeout(() => node.remove(), timeout);
  }

  function formatDate(value) {
    if (!value) return "date not provided";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "date not provided";
    try {
      return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
    } catch {
      return date.toISOString();
    }
  }

  function safeLink(value) {
    try {
      const url = new URL(value, window.location.origin);
      return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
    } catch {
      return null;
    }
  }

  function setBusy(button, busy, busyLabel) {
    if (!button) return;
    if (busy) {
      button.dataset.label = button.textContent;
      button.textContent = busyLabel;
      button.disabled = true;
    } else {
      if (button.dataset.label) button.textContent = button.dataset.label;
      button.disabled = false;
    }
  }

  async function api(path, { method = "GET", body, signal } = {}) {
    const headers = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (method !== "GET" && path !== "/api/session" && state.csrf) headers["X-Ara-CSRF"] = state.csrf;
    let response;
    try {
      response = await fetch(path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: "same-origin",
        signal,
      });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw Object.assign(new Error("Cannot reach Ara's local server. Is it still running?"), { network: true });
    }
    const type = response.headers.get("content-type") || "";
    let data = null;
    if (type.includes("application/json")) {
      data = await response.json().catch(() => ({}));
    } else {
      data = await response.text();
    }
    if (!response.ok) {
      const message = data && typeof data === "object" && data.error ? data.error : `Request failed (${response.status})`;
      throw Object.assign(new Error(message), {
        status: response.status,
        code: data && data.code,
        network: response.status >= 500,
      });
    }
    return data;
  }

  /* ---------------------------------------------------------------- session */
  async function ensureSession(password) {
    if (state.csrf) return true;
    if (state.sessionPromise) return state.sessionPromise;
    state.sessionPromise = (async () => {
      try {
        const body = password === undefined ? {} : { password };
        const data = await api("/api/session", { method: "POST", body });
        state.csrf = data.csrf;
        state.conversationId = data.conversation_id;
        return true;
      } finally {
        state.sessionPromise = null;
      }
    })();
    return state.sessionPromise;
  }

  async function openApp(targetView = "chat") {
    try {
      await ensureSession();
    } catch (error) {
      if (error.code === "bad_password") {
        showGate(error.message);
        return;
      }
      if (error.status === 401) {
        showGate("");
        return;
      }
      toast(error.message, "bad");
      return;
    }
    dom.landing.hidden = true;
    dom.gate.hidden = true;
    dom.app.hidden = false;
    document.title = "Ara — workspace";
    await Promise.all([loadState(), loadConversation()]);
    setView(targetView);
    dom.message.focus();
    const active = state.stateData && state.stateData.model ? state.stateData.model.active : "local";
    if (active === "retrieval-only") {
      toast("No language model is connected. Replies are retrieved notes, labelled retrieval-only — not generated text.", "warn", 9000);
    } else if (state.stateData && state.stateData.model && !state.stateData.model.reachable) {
      toast("The configured model is not reachable yet. Replies will fall back to labelled retrieval until it answers.", "warn", 9000);
    }
  }

  function showGate(message = "") {
    dom.landing.hidden = true;
    dom.app.hidden = true;
    dom.gate.hidden = false;
    dom.gateError.textContent = message;
    dom.gatePassword.value = "";
    dom.gatePassword.focus();
  }

  function leaveApp() {
    stopStreaming();
    stopSpeech();
    stopRecording();
    closeCamera();
    dom.app.hidden = true;
    dom.gate.hidden = true;
    dom.landing.hidden = false;
    document.title = "Ara — the AI that belongs in Nigeria";
  }

  dom.gateForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    dom.gateError.textContent = "";
    const button = dom.gateForm.querySelector('button[type="submit"]');
    setBusy(button, true, "Opening…");
    try {
      await ensureSession(dom.gatePassword.value);
      await openApp(state.view);
    } catch (error) {
      dom.gateError.textContent = error.status === 401 ? "That password was not accepted." : error.message;
    } finally {
      setBusy(button, false);
    }
  });
  $("#gate-cancel").addEventListener("click", leaveApp);

  $$("#open-app, #open-app-top, #open-app-bottom").forEach((button) => {
    button.addEventListener("click", () => openApp("chat"));
  });
  $("#open-app-news").addEventListener("click", () => openApp("news"));
  $("#leave-app").addEventListener("click", leaveApp);

  $("#logout").addEventListener("click", async () => {
    if (!state.csrf) return leaveApp();
    openConfirm("End this web session?", "Your memories stay in the local database. The session cookie is removed and this page returns to the field.", "End session", async () => {
      try {
        await api("/api/logout", { method: "POST" });
      } catch (error) {
        toast(error.message, "bad");
      }
      state.csrf = null;
      state.conversationId = null;
      leaveApp();
    });
  });

  /* ------------------------------------------------------------------- state */
  async function loadState() {
    try {
      const data = await api("/api/state");
      state.stateData = data;
      state.conversationId = data.conversation_id;
      renderState(data);
    } catch (error) {
      if (error.status === 401) {
        state.csrf = null;
        toast("Your session expired. Reload and open Ara again.", "warn");
      } else {
        toast(error.message, "bad");
      }
    }
  }

  function shortModel(name) {
    if (!name) return "";
    const base = String(name).split(/[\\/]/).pop().replace(/\.gguf$/i, "");
    return base.length > 30 ? `${base.slice(0, 29)}…` : base;
  }

  function renderState(data) {
    const model = data.model || {};
    const active = model.active || "local";
    const reachable = Boolean(model.reachable);
    const hosted = active === "hosted";
    const retrieval = active === "retrieval-only";
    const name = shortModel(model.served_model || (model.models && model.models[0]) || (model.configured_model !== "none" ? model.configured_model : null));
    dom.modelPill.textContent = retrieval
      ? "retrieval-only"
      : reachable
        ? `${hosted ? "hosted" : "local"} · ${name || "model"}`
        : `${hosted ? "hosted" : "local"} · offline`;
    dom.modelPill.classList.toggle("ok", reachable && !retrieval);
    dom.modelPill.classList.toggle("bad", !reachable || retrieval);
    dom.modelPill.title = model.message || "";
    dom.railDot.className = "status-dot " + (reachable && !retrieval ? "ok" : "bad");
    dom.railStatus.textContent = retrieval
      ? "Retrieval-only · no model connected"
      : reachable
        ? `${hosted ? "Hosted ara-ng" : "Local model"} ready · ${data.profile ? data.profile.effective : "standard"} profile`
        : "Model configured, not reachable";

    renderAbout(data);
    if (state.memory.length || state.view === "memory") renderMemory(state.memory);
  }

  function renderAbout(data) {
    const runtime = $("#about-runtime");
    const perception = $("#about-perception");
    const knowledge = $("#about-knowledge");
    const model = data.model || {};
    const profile = data.profile || {};
    runtime.replaceChildren();
    const runtimeRows = [
      ["Version", data.version],
      ["Backend", model.active || "local"],
      ["Model", shortModel(model.served_model || (model.configured_model !== "none" ? model.configured_model : null)) || "none — retrieval-only"],
      ["Reachable", model.reachable ? "yes" : "no — replies fall back to labelled retrieval"],
      ["Models served", (model.models || []).join(", ") || "none reported"],
      ["Context", model.context_length ? `${model.context_length} tokens` : "not reported by the server"],
      ["Profile", `${profile.effective || "standard"} (requested: ${profile.requested || "auto"})`],
      ["Token budget", String(profile.max_tokens || "—")],
      ["Stored messages", String((data.counts && data.counts.messages) || 0)],
      ["Stored notes", String((data.memory && data.memory.count) || 0)],
    ];
    runtimeRows.forEach(([label, value]) => {
      const row = el("div");
      row.append(el("dt", "", label), el("dd", "", value));
      runtime.append(row);
    });

    perception.replaceChildren();
    const vision = data.perception.vision;
    const transcription = data.perception.transcription;
    const perceptionRows = [
      ["Vision", vision.available ? `configured · ${vision.model}` : "not configured"],
      ["Transcription", transcription.available ? `configured · ${transcription.model}` : "not configured"],
      ["Spoken replies", data.perception.speech_synthesis.available ? "browser speech synthesis" : "unavailable on this browser"],
      ["Camera", window.isSecureContext && navigator.mediaDevices ? "browser permission required" : "needs a secure context"],
      ["Microphone", navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "browser permission required" : "not available"],
      ["Storage", data.privacy.uploads_stored ? "uploads may be stored" : "uploads are not stored"],
    ];
    perceptionRows.forEach(([label, value]) => {
      const row = el("div");
      row.append(el("dt", "", label), el("dd", "", value));
      perception.append(row);
    });

    knowledge.replaceChildren();
    const kindCounts = (data.knowledge && data.knowledge.by_kind) || {};
    const knowledgeRows = [
      ["Entries", String((data.knowledge && data.knowledge.total) || 0)],
      ["Expressions", String(kindCounts.expressions || 0)],
      ["History", String(kindCounts.history || 0)],
      ["Culture", String(kindCounts.culture || 0)],
      ["Retrieval", data.memory.retrieval.semantic ? "hybrid · local semantic vectors" : "lexical + hashed fallback (not semantic)"],
      ["Vectors", `${data.memory.retrieval.vectors} stored · ${data.memory.retrieval.missing} missing`],
    ];
    knowledgeRows.forEach(([label, value]) => {
      const row = el("div");
      row.append(el("dt", "", label), el("dd", "", value));
      knowledge.append(row);
    });

    const support = $("#sensor-support");
    const pieces = [];
    pieces.push(window.isSecureContext ? "Secure context: yes." : "Secure context: no — browser camera and microphone may be blocked except on localhost.");
    if (navigator.permissions && navigator.permissions.query) {
      pieces.push("You can revoke camera or microphone access in your browser's site settings at any time.");
    }
    if (!data.perception.vision.available) pieces.push("Image analysis is disabled until a vision endpoint is configured by the operator.");
    if (!data.perception.transcription.available) pieces.push("Voice input needs a configured local speech-to-text endpoint; typing always works.");
    support.textContent = pieces.join(" ");
  }

  /* ------------------------------------------------------------------ router */
  const viewCopy = {
    chat: { title: "Talk with Ara", kicker: "ara-ng · web chat", mode: "chat" },
    learn: { title: "Study with Ara", kicker: "ara-ng · study mode", mode: "learn" },
    news: { title: "Ara’s news plate", kicker: "Nigerian publishers", mode: null },
    memory: { title: "Memory", kicker: "You decide what stays", mode: null },
    about: { title: "About Ara", kicker: "One shared core", mode: null },
  };

  function setView(view) {
    if (!viewCopy[view]) return;
    state.view = view;
    $$(".rail-link[data-view]").forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    $$(".view").forEach((section) => { section.hidden = true; });
    if (view === "chat" || view === "learn") {
      $("#view-chat").hidden = false;
      state.mode = viewCopy[view].mode;
      dom.studyBanner.hidden = view !== "learn";
      dom.modePill.textContent = view === "learn" ? "Study" : "Chat";
      const emptyTitle = $("#empty-title");
      const emptySub = $("#empty-sub");
      if (view === "learn") {
        emptyTitle.textContent = "What should we understand today?";
        emptySub.textContent = "Ara will start from what you know, explain in small steps, and check your understanding. It can use Nigerian examples where they help.";
        dom.message.placeholder = "Ask to understand something…";
      } else {
        emptyTitle.textContent = "What are we looking at?";
        emptySub.textContent = "Ask in English or Pidgin. Ara will match your register, check its notes, and say when it is unsure.";
        dom.message.placeholder = "Talk to Ara…";
      }
      dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
      dom.message.focus();
    } else {
      $(`#view-${view}`).hidden = false;
      if (view === "news") loadNews(false);
      if (view === "memory") loadMemory();
    }
    dom.viewTitle.textContent = viewCopy[view].title;
    dom.viewKicker.textContent = viewCopy[view].kicker;
  }

  $$(".rail-link[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });

  /* -------------------------------------------------------------------- chat */
  function hideEmpty() {
    dom.chatEmpty.hidden = true;
  }

  function scrollChat() {
    dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
  }

  function addMessage(role, text, { typing = false, className = "" } = {}) {
    hideEmpty();
    const wrapper = el("div", `message message-${role} ${className}`.trim());
    const avatar = el("div", "message-avatar");
    if (role === "ara") {
      const image = document.createElement("img");
      image.src = "/frog.svg";
      image.alt = "";
      image.width = 30;
      image.height = 30;
      avatar.append(image);
    } else {
      avatar.textContent = "You";
    }
    const bubble = el("div", "bubble");
    const meta = el("div", "message-meta");
    if (typing) {
      const holder = el("div", "typing");
      const frog = document.createElement("img");
      frog.src = "/frog.svg";
      frog.alt = "";
      const dots = el("span", "typing-dots");
      dots.append(el("span"), el("span"), el("span"));
      holder.append(frog, dots);
      bubble.append(holder);
    } else {
      bubble.textContent = text;
    }
    wrapper.append(avatar, bubble);
    bubble.append(meta);
    dom.chatLog.append(wrapper);
    scrollChat();
    return { wrapper, bubble, meta };
  }

  function addMeta(metaNode, label, className = "") {
    metaNode.append(el("span", `meta-chip ${className}`.trim(), label));
  }

  function setTextareaHeight() {
    dom.message.style.height = "auto";
    dom.message.style.height = Math.min(dom.message.scrollHeight, 180) + "px";
  }

  dom.message.addEventListener("input", setTextareaHeight);
  dom.message.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      dom.composer.requestSubmit();
    }
  });
  dom.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = dom.message.value.trim();
    if (!text || state.streaming) return;
    dom.message.value = "";
    setTextareaHeight();
    sendMessage(text);
  });
  $$("#suggestions button").forEach((button) => {
    button.addEventListener("click", () => {
      const prompt = button.dataset.prompt || button.textContent || "";
      dom.message.value = prompt;
      setTextareaHeight();
      dom.composer.requestSubmit();
    });
  });

  dom.stop.addEventListener("click", stopStreaming);

  function stopStreaming() {
    if (state.controller) state.controller.abort();
  }

  function setStreaming(value) {
    state.streaming = value;
    dom.send.hidden = value;
    dom.stop.hidden = !value;
    dom.send.disabled = value;
  }

  async function sendMessage(text) {
    if (!text || state.streaming) return;
    if (!state.csrf) {
      try {
        await ensureSession();
      } catch (error) {
        showGate("");
        return;
      }
    }
    addMessage("user", text);
    const target = addMessage("ara", "", { typing: true });
    let answer = "";
    let sawDelta = false;
    let interpretation = null;
    let learning = null;
    let failure = null;
    const controller = new AbortController();
    state.controller = controller;
    setStreaming(true);
    try {
      let response;
      try {
        response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Ara-CSRF": state.csrf },
          body: JSON.stringify({ message: text, mode: state.mode }),
          signal: controller.signal,
        });
      } catch (error) {
        if (error.name === "AbortError") throw error;
        throw new Error("Cannot reach Ara's local server. Is it still running?");
      }
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Ara could not start this reply (${response.status}).`);
      }
      if (!response.body) throw new Error("This browser cannot read a streaming response.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const handle = (payload) => {
        if (!payload) return;
        if (payload.type === "delta" && payload.text) {
          if (!sawDelta) {
            target.bubble.replaceChildren();
            sawDelta = true;
          }
          answer += payload.text;
          target.bubble.textContent = answer;
          target.bubble.append(target.meta);
          scrollChat();
        } else if (payload.type === "done") {
          if (payload.text && payload.text !== answer) {
            answer = payload.text;
            target.bubble.textContent = answer;
            target.bubble.append(target.meta);
          }
          interpretation = payload.interpretation;
          learning = payload.learning;
          if (payload.answer_kind === "retrieval") {
            addMeta(target.meta, "retrieval-only · not model text", "amber");
          } else if (payload.backend === "hosted") {
            addMeta(target.meta, "hosted ara-ng", "green");
          }
          if (payload.backend_error) {
            toast(payload.backend_error, "warn", 9000);
          }
        } else if (payload.type === "error") {
          failure = payload.message || "Ara could not finish this reply.";
        }
      };
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          const raw = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const data = raw
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("");
          if (!data) continue;
          try {
            handle(JSON.parse(data));
          } catch {
            /* A malformed event must not break the rest of the stream. */
          }
        }
      }
      if (failure) throw new Error(failure);
      if (!answer.trim()) throw new Error("Ara returned an empty reply.");
      target.bubble.textContent = answer;
      target.bubble.append(target.meta);
      renderInterpretation(target.meta, interpretation);
      if (learning) {
        if (learning.saved) addMeta(target.meta, `noted as ${learning.kind || "memory"} · editable`, "green");
        if (learning.sensitive_rejected) addMeta(target.meta, "credential-looking text was not stored");
      }
      if (state.speechEnabled) speak(answer);
    } catch (error) {
      if (error.name === "AbortError") {
        target.bubble.textContent = (answer || "Reply stopped.") + "\n\n[stopped]";
        target.bubble.append(target.meta);
        addMeta(target.meta, "stopped");
      } else {
        target.wrapper.classList.add("message-error");
        target.bubble.textContent = error.message;
        target.bubble.append(target.meta);
        addMeta(target.meta, "not completed");
        toast(error.message, "bad");
      }
    } finally {
      state.controller = null;
      setStreaming(false);
      scrollChat();
    }
  }

  function renderInterpretation(meta, interpretation) {
    if (!interpretation) return;
    if (interpretation.register && interpretation.register !== "standard") {
      addMeta(meta, interpretation.register, "green");
    }
    (interpretation.detected || []).slice(0, 3).forEach((item) => {
      addMeta(meta, `${item.term}: ${item.meaning ? item.meaning.split(".")[0] : "context note"}`);
    });
    if (interpretation.clarify) addMeta(meta, "asked for clarification");
    if (interpretation.stereotype) addMeta(meta, "generalisation flagged", "green");
  }

  async function loadConversation() {
    if (!state.csrf) return;
    try {
      const data = await api("/api/conversation");
      state.conversationId = data.conversation_id;
      $$(".message", dom.chatLog).forEach((node) => node.remove());
      const messages = data.messages || [];
      if (messages.length) {
        hideEmpty();
        messages.forEach((message) => {
          addMessage(message.role === "assistant" ? "ara" : "user", message.content);
        });
        scrollChat();
      } else {
        dom.chatEmpty.hidden = false;
      }
    } catch (error) {
      if (error.status !== 401) toast(error.message, "bad");
    }
  }

  $("#new-chat").addEventListener("click", async () => {
    if (!state.csrf) return;
    if (state.streaming) stopStreaming();
    try {
      const data = await api("/api/conversation/new", { method: "POST" });
      state.conversationId = data.conversation_id;
      $$(".message", dom.chatLog).forEach((node) => node.remove());
      dom.chatEmpty.hidden = false;
      toast("Started a new conversation.");
    } catch (error) {
      toast(error.message, "bad");
    }
  });

  /* ---------------------------------------------------------------- speech */
  const synth = "speechSynthesis" in window ? window.speechSynthesis : null;
  dom.speakToggle.checked = Boolean(state.speechEnabled && synth);
  dom.speakToggle.disabled = !synth;
  if (!synth) dom.speakToggle.closest("label").title = "This browser has no speech synthesis.";
  dom.speakToggle.addEventListener("change", () => {
    state.speechEnabled = dom.speakToggle.checked && Boolean(synth);
    localStorage.setItem("ara.speak", state.speechEnabled ? "1" : "0");
    if (!state.speechEnabled) stopSpeech();
  });

  function pickVoice() {
    if (!synth) return null;
    const voices = synth.getVoices();
    const preferences = ["en-NG", "en-GB", "en-US", "en"];
    for (const prefix of preferences) {
      const found = voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith(prefix.toLowerCase()));
      if (found) return found;
    }
    return voices[0] || null;
  }

  function speak(text) {
    if (!synth || !text) return;
    stopSpeech();
    const utterance = new SpeechSynthesisUtterance(text.replace(/https?:\/\/\S+/g, "link"));
    const voice = pickVoice();
    if (voice) utterance.voice = voice;
    utterance.rate = 1;
    utterance.pitch = 1;
    state.spoken = utterance;
    utterance.onend = () => {
      if (state.spoken === utterance) {
        state.spoken = null;
        hideSensor("speech");
      }
    };
    utterance.onerror = () => {
      if (state.spoken === utterance) {
        state.spoken = null;
        hideSensor("speech");
        toast("The device could not speak this reply. Text is still available.", "warn");
      }
    };
    showSensor("speech", "Speaking reply", null);
    synth.speak(utterance);
  }

  function stopSpeech() {
    if (synth && (synth.speaking || synth.pending)) synth.cancel();
    if (state.spoken) {
      state.spoken = null;
      hideSensor("speech");
    }
  }

  /* -------------------------------------------------------------- sensor UI */
  const activeSensors = new Map();
  function renderSensors() {
    const entries = [...activeSensors.entries()];
    if (!entries.length) {
      dom.sensorStrip.hidden = true;
      dom.sensorTime.textContent = "";
      return;
    }
    dom.sensorStrip.hidden = false;
    dom.sensorLabel.textContent = entries.map(([, sensor]) => sensor.label).join(" · ");
    const timed = entries.find(([, sensor]) => sensor.time);
    dom.sensorTime.textContent = timed ? timed[1].time : "";
    dom.sensorStop.textContent = entries.length > 1 ? "Stop all" : entries[0][0] === "camera" ? "Stop camera" : entries[0][0] === "speech" ? "Stop speaking" : "Stop";
  }
  function showSensor(mode, label, time) {
    activeSensors.set(mode, { label, time: time || "" });
    renderSensors();
  }
  function hideSensor(mode) {
    if (mode) activeSensors.delete(mode);
    else activeSensors.clear();
    renderSensors();
  }
  dom.sensorStop.addEventListener("click", () => {
    // The single visible control always releases every active capture, not just one.
    if (state.recording) stopRecording();
    closeCamera();
    stopSpeech();
    hideSensor();
  });

  /* --------------------------------------------------------------------- mic */
  async function startRecording() {
    if (state.recording) return stopRecording();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("This browser cannot access a microphone here. Use HTTPS or localhost, or type instead.", "bad");
      return;
    }
    if (!window.MediaRecorder) {
      toast("This browser cannot record audio. Typing still works.", "bad");
      return;
    }
    stopSpeech();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      const reason = error && error.name === "NotAllowedError"
        ? "Microphone permission was denied. You can allow it in site settings, or type instead."
        : "No microphone is available. Typing still works.";
      toast(reason, "bad");
      return;
    }
    const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].filter((type) => {
      try { return MediaRecorder.isTypeSupported(type); } catch { return false; }
    });
    let recorder;
    try {
      recorder = new MediaRecorder(stream, options.length ? { mimeType: options[0] } : undefined);
    } catch {
      stream.getTracks().forEach((track) => track.stop());
      toast("This browser could not start audio recording. Typing still works.", "bad");
      return;
    }
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size) chunks.push(event.data);
    });
    const started = Date.now();
    const session = { recorder, stream, chunks, started, finalized: false };
    const finalize = () => {
      if (session.finalized) return;
      session.finalized = true;
      if (state.recording === session) state.recording = null;
      window.clearInterval(state.recordingTimer);
      state.recordingTimer = null;
      dom.micButton.setAttribute("aria-pressed", "false");
      finishRecording(session);
    };
    session.finalize = finalize;
    state.recording = session;
    state.recordingStarted = started;
    recorder.addEventListener("stop", finalize);
    recorder.start();
    dom.micButton.setAttribute("aria-pressed", "true");
    showSensor("mic", "Microphone active · recording", "0:00");
    state.recordingTimer = window.setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      showSensor("mic", "Microphone active · recording", `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`);
      if (seconds >= 60) stopRecording();
    }, 500);
    recorder.addEventListener("error", () => {
      toast("Recording stopped unexpectedly. Try again or type.", "bad");
      stopRecording();
    });
  }

  function stopRecording() {
    const current = state.recording;
    if (!current) return;
    if (current.recorder.state === "inactive") {
      current.finalize();
      return;
    }
    try {
      current.recorder.stop();
    } catch {
      current.finalize();
      return;
    }
    current.stream.getTracks().forEach((track) => track.stop());
    // A browser that never fires 'stop' must not leave the UI recording forever.
    window.setTimeout(() => current.finalize(), 2000);
  }

  async function finishRecording(current) {
    window.clearInterval(state.recordingTimer);
    dom.micButton.setAttribute("aria-pressed", "false");
    if (!current.chunks.length) {
      hideSensor("mic");
      return;
    }
    const blob = new Blob(current.chunks, { type: current.recorder.mimeType || "audio/webm" });
    if (!blob.size) {
      hideSensor("mic");
      return;
    }
    showSensor("mic", "Transcribing locally…", "");
    try {
      const dataUrl = await blobToDataUrl(blob);
      const result = await api("/api/transcribe", { method: "POST", body: { audio: dataUrl } });
      dom.message.value = (dom.message.value ? dom.message.value + " " : "") + result.text;
      setTextareaHeight();
      dom.message.focus();
      hideSensor("mic");
      toast("Transcribed. Check the words before sending.", "", 3800);
    } catch (error) {
      hideSensor("mic");
      toast(error.message, error.status === 503 ? "warn" : "bad", 8000);
    }
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Could not read the recording."));
      reader.readAsDataURL(blob);
    });
  }

  dom.micButton.addEventListener("click", () => {
    if (state.recording) stopRecording();
    else startRecording();
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (state.recording) stopRecording();
      if (state.cameraStream) closeCamera();
      stopSpeech();
    }
  });
  window.addEventListener("pagehide", () => {
    if (state.recording) stopRecording();
    closeCamera();
    stopSpeech();
  });

  /* ------------------------------------------------------------------ camera */
  async function enableCamera() {
    if (state.cameraStream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      dom.cameraError.textContent = "This browser cannot access a camera here. Use HTTPS or localhost.";
      return;
    }
    dom.cameraError.textContent = "";
    setBusy(dom.cameraEnable, true, "Starting…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      state.cameraStream = stream;
      dom.cameraVideo.srcObject = stream;
      await dom.cameraVideo.play().catch(() => {});
      dom.cameraIdle.hidden = true;
      dom.cameraLive.hidden = false;
      dom.cameraCapture.disabled = false;
      dom.cameraEnable.hidden = true;
      dom.cameraStop.hidden = false;
      showSensor("camera", "Camera live · frames are not stored", "");
    } catch (error) {
      const reason = error && error.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it in site settings, or upload an image instead."
        : error && error.name === "NotFoundError"
          ? "No camera was found on this device."
          : "The camera could not start. You can upload an image instead.";
      dom.cameraError.textContent = reason;
    } finally {
      setBusy(dom.cameraEnable, false);
    }
  }

  function stopCameraTracks() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((track) => track.stop());
      state.cameraStream = null;
    }
    dom.cameraVideo.srcObject = null;
    dom.cameraIdle.hidden = false;
    dom.cameraLive.hidden = true;
    dom.cameraCapture.disabled = true;
    dom.cameraEnable.hidden = false;
    dom.cameraStop.hidden = true;
    hideSensor("camera");
  }

  function closeCamera() {
    stopCameraTracks();
    if (dom.cameraDialog.open) dom.cameraDialog.close();
  }

  dom.cameraEnable.addEventListener("click", enableCamera);
  dom.cameraStop.addEventListener("click", stopCameraTracks);
  $("#camera-close").addEventListener("click", closeCamera);
  dom.cameraDialog.addEventListener("close", stopCameraTracks);
  dom.cameraDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeCamera();
  });

  dom.cameraCapture.addEventListener("click", async () => {
    if (!state.cameraStream || !dom.cameraVideo.videoWidth) {
      dom.cameraError.textContent = "The camera has no frame yet.";
      return;
    }
    dom.cameraError.textContent = "";
    setBusy(dom.cameraCapture, true, "Analysing…");
    try {
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 1280 / Math.max(dom.cameraVideo.videoWidth, dom.cameraVideo.videoHeight));
      canvas.width = Math.round(dom.cameraVideo.videoWidth * scale);
      canvas.height = Math.round(dom.cameraVideo.videoHeight * scale);
      const context = canvas.getContext("2d");
      context.drawImage(dom.cameraVideo, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", 0.85);
      const prompt = ($("#vision-prompt").value || "").trim() || "Describe this image accurately. Point out uncertainty and do not invent details.";
      const result = await api("/api/vision", { method: "POST", body: { image, prompt } });
      closeCamera();
      addVisionCard(result);
    } catch (error) {
      dom.cameraError.textContent = error.message;
      if (error.status === 503) {
        toast("Vision is not configured. Set [vision].url to a local multimodal endpoint in config.toml.", "warn", 8000);
      }
    } finally {
      setBusy(dom.cameraCapture, false);
    }
  });

  $("#camera-button").addEventListener("click", () => {
    dom.cameraError.textContent = "";
    dom.cameraDialog.showModal();
    $("#camera-close").focus();
  });

  function addVisionCard(result) {
    const message = addMessage("ara", "", { className: "vision-card" });
    message.bubble.textContent = result.text;
    message.bubble.append(message.meta);
    addMeta(message.meta, `image · ${result.image.width}×${result.image.height}`, "green");
    addMeta(message.meta, "vision model output · not stored");
    const ask = el("button", "suggestion-ask", "Ask Ara about this");
    ask.type = "button";
    ask.addEventListener("click", () => {
      dom.message.value = `I had a vision model look at an image. It reported:\n\n${result.text}\n\nHelp me think about what this means and what I should check.`;
      setTextareaHeight();
      dom.message.focus();
    });
    message.meta.append(ask);
    scrollChat();
    toast("Image analysed. The frame was not saved by Ara.");
  }

  $("#image-input").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) return;
    if (!/^image\/(jpeg|png|gif|webp)$/.test(file.type)) {
      toast("Use a JPEG, PNG, GIF, or WebP image.", "bad");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      toast("That image is larger than 8 MB.", "bad");
      return;
    }
    const button = $('label[for="image-input"]');
    button.classList.add("is-busy");
    try {
      const dataUrl = await blobToDataUrl(file);
      const prompt = dom.message.value.trim() || "Describe this image accurately. Point out uncertainty and do not invent details.";
      const result = await api("/api/vision", { method: "POST", body: { image: dataUrl, prompt } });
      addVisionCard(result);
    } catch (error) {
      toast(error.message, error.status === 503 ? "warn" : "bad", 8000);
    } finally {
      button.classList.remove("is-busy");
    }
  });

  /* -------------------------------------------------------------------- news */
  async function loadNews(refresh = false, offline = false) {
    if (state.newsLoading) return;
    state.newsLoading = true;
    setBusy($("#refresh-news"), true, refresh ? "Refreshing…" : "Checking…");
    dom.newsMeta.textContent = refresh ? "Contacting publisher feeds…" : "Reading Ara's local cache…";
    const params = new URLSearchParams();
    if (refresh) params.set("refresh", "1");
    if (offline) params.set("offline", "1");
    if (state.topic) params.set("topic", state.topic);
    try {
      const report = await api(`/api/news${params.toString() ? "?" + params.toString() : ""}`);
      state.newsLoaded = true;
      renderTopics(report.topics || []);
      renderNews(report);
    } catch (error) {
      dom.newsMeta.textContent = "Could not read the news cache.";
      toast(error.message, "bad");
    } finally {
      state.newsLoading = false;
      setBusy($("#refresh-news"), false);
    }
  }

  function renderTopics(topics) {
    const all = ["", ...topics];
    dom.topicRow.replaceChildren();
    all.forEach((topic) => {
      const button = el("button", "topic" + (topic === state.topic ? " is-active" : ""), topic || "All");
      button.type = "button";
      button.dataset.topic = topic;
      button.addEventListener("click", () => {
        state.topic = topic;
        loadNews(false);
      });
      dom.topicRow.append(button);
    });
  }

  function renderNews(report) {
    const items = report.items || [];
    dom.newsGrid.replaceChildren();
    dom.newsEmpty.hidden = items.length > 0;
    const errors = report.errors || [];
    const fetched = report.fetched_at ? formatDate(report.fetched_at) : "never";
    if (report.offline) {
      dom.newsMeta.textContent = `Cached stories only · last fetched ${fetched}`;
    } else if (report.added || report.updated) {
      dom.newsMeta.textContent = `${report.feeds_ok || 0} feeds checked · ${report.added || 0} new, ${report.updated || 0} merged · ${fetched}`;
    } else {
      dom.newsMeta.textContent = `${report.feeds_ok || 0} feeds checked · showing ${items.length} cached stories · ${fetched}`;
    }
    if (errors.length) {
      dom.newsCacheNote.hidden = false;
      dom.newsCacheNote.textContent = `${errors.length} feed${errors.length === 1 ? "" : "s"} could not be read. Cached stories are shown with their original dates. ${errors[0]}`;
    } else if (report.offline) {
      dom.newsCacheNote.hidden = false;
      dom.newsCacheNote.textContent = "Offline view: stories keep the publication time from their publisher. Nothing has been refreshed.";
    } else {
      dom.newsCacheNote.hidden = true;
    }
    items.forEach((item) => dom.newsGrid.append(newsCard(item)));
  }

  function newsCard(item) {
    const card = el("article", "news-card");
    const top = el("div", "news-card-top");
    top.append(el("span", "news-source", item.source || "Unknown source"));
    if (item.topic) top.append(el("span", "news-topic", item.topic));
    const freshnessClass = item.freshness_label === "current" ? "" : item.freshness_label === "date not provided" ? "unknown" : "older";
    top.append(el("span", `freshness ${freshnessClass}`.trim(), item.fresh ? "current" : item.freshness_label || "older"));
    card.append(top);
    const title = el("h3");
    const link = safeLink(item.link);
    if (link) {
      const anchor = el("a", "", item.title);
      anchor.href = link;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      title.append(anchor);
    } else {
      title.textContent = item.title;
    }
    card.append(title);
    if (item.summary) card.append(el("p", "", item.summary));
    if (item.historical_context) card.append(el("p", "context-note", "Context: " + item.historical_context));
    const foot = el("div", "news-card-foot");
    foot.append(el("span", "", formatDate(item.published_at)));
    if (item.related_sources && item.related_sources.length) {
      const names = item.related_sources.map((value) => String(value).split(":")[0]).slice(0, 3);
      foot.append(el("span", "", "Also: " + names.join(", ")));
    }
    card.append(foot);
    return card;
  }

  $("#refresh-news").addEventListener("click", () => loadNews(true));
  $("#offline-news").addEventListener("click", () => loadNews(false, true));

  /* ------------------------------------------------------------------ memory */
  async function loadMemory() {
    try {
      const data = await api("/api/memory");
      state.memory = data.memories || [];
      renderMemory(state.memory);
    } catch (error) {
      if (error.status !== 401) toast(error.message, "bad");
    }
  }

  function renderMemory(memories) {
    dom.memoryList.replaceChildren();
    dom.memoryEmpty.hidden = memories.length > 0;
    dom.memoryCount.textContent = `${memories.length} active note${memories.length === 1 ? "" : "s"}`;
    memories.forEach((memory) => dom.memoryList.append(memoryCard(memory)));
  }

  function memoryCard(memory) {
    const card = el("article", "memory-card");
    const body = el("div", "memory-card-body");
    body.append(el("span", "memory-kind", memory.kind));
    body.append(el("p", "memory-text", memory.content));
    const meta = el("div", "memory-meta");
    meta.append(el("span", "", `source: ${memory.source}`));
    meta.append(el("span", "", `confidence: ${Number(memory.confidence).toFixed(2)}`));
    meta.append(el("span", "", `updated: ${formatDate(memory.updated_at)}`));
    body.append(meta);
    const actions = el("div", "memory-actions");
    const correct = el("button", "", "Correct");
    correct.type = "button";
    correct.addEventListener("click", () => openCorrect(memory));
    const forget = el("button", "delete", "Forget");
    forget.type = "button";
    forget.addEventListener("click", () => {
      openConfirm("Forget this note?", `“${memory.content}” will no longer be used in replies. The record is marked forgotten, not silently kept active.`, "Forget note", async () => {
        try {
          await api(`/api/memory/${memory.id}`, { method: "DELETE" });
          await loadMemory();
          await loadState();
          toast("Note forgotten.");
        } catch (error) {
          toast(error.message, "bad");
        }
      });
    });
    actions.append(correct, forget);
    card.append(body, actions);
    return card;
  }

  $("#memory-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const content = $("#memory-content").value.trim();
    const kind = $("#memory-kind").value;
    if (!content) return;
    const button = event.submitter || $("#memory-form button[type=submit]");
    setBusy(button, true, "Saving…");
    try {
      await api("/api/memory", { method: "POST", body: { content, kind } });
      $("#memory-content").value = "";
      await loadMemory();
      await loadState();
      toast("Note saved. Àrà will use it in later conversations.");
    } catch (error) {
      toast(error.message, "bad");
    } finally {
      setBusy(button, false);
    }
  });

  $("#forget-all").addEventListener("click", () => {
    openConfirm(
      "Forget every active note?",
      "All active notes will be marked forgotten and their vectors removed. This cannot be undone without a backup.",
      "Forget all notes",
      async () => {
        try {
          await api("/api/memory/forget-all", { method: "POST", body: { confirm: "forget all" } });
          await loadMemory();
          await loadState();
          toast("All active notes were forgotten.");
        } catch (error) {
          toast(error.message, "bad");
        }
      },
      "forget all"
    );
  });

  function openCorrect(memory) {
    state.correctId = memory.id;
    dom.correctContent.value = memory.content;
    dom.correctError.textContent = "";
    dom.correctDialog.showModal();
    dom.correctContent.focus();
  }

  $("#correct-cancel").addEventListener("click", () => dom.correctDialog.close());
  $("#correct-accept").addEventListener("click", async () => {
    const content = dom.correctContent.value.trim();
    if (!content) {
      dom.correctError.textContent = "Write the corrected note first.";
      return;
    }
    const button = $("#correct-accept");
    setBusy(button, true, "Saving…");
    try {
      await api(`/api/memory/${state.correctId}/correct`, { method: "POST", body: { content } });
      dom.correctDialog.close();
      await loadMemory();
      await loadState();
      toast("Correction saved.");
    } catch (error) {
      dom.correctError.textContent = error.message;
    } finally {
      setBusy(button, false);
    }
  });

  /* ----------------------------------------------------------------- dialogs */
  function openConfirm(title, text, acceptLabel, handler, confirmWord) {
    state.confirmHandler = handler;
    $("#confirm-title").textContent = title;
    dom.confirmText.textContent = text;
    dom.confirmError.textContent = "";
    dom.confirmAccept.textContent = acceptLabel;
    dom.confirmInput.hidden = !confirmWord;
    dom.confirmLabel.hidden = !confirmWord;
    dom.confirmInput.value = "";
    if (confirmWord) {
      dom.confirmLabel.textContent = `Type “${confirmWord}” to confirm`;
      dom.confirmInput.dataset.expected = confirmWord;
    } else {
      delete dom.confirmInput.dataset.expected;
    }
    dom.confirmDialog.showModal();
    if (confirmWord) dom.confirmInput.focus();
    else $("#confirm-cancel").focus();
  }
  dom.confirmAccept.addEventListener("click", async () => {
    const expected = dom.confirmInput.dataset.expected;
    if (expected && dom.confirmInput.value.trim() !== expected) {
      dom.confirmError.textContent = `Type “${expected}” exactly.`;
      return;
    }
    const handler = state.confirmHandler;
    state.confirmHandler = null;
    dom.confirmDialog.close();
    if (handler) await handler();
  });
  $("#confirm-cancel").addEventListener("click", () => dom.confirmDialog.close());
  dom.confirmDialog.addEventListener("close", () => {
    state.confirmHandler = null;
  });

  /* --------------------------------------------------------------- keyboard */
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (dom.cameraDialog.open) {
      event.preventDefault();
      closeCamera();
    } else if (state.recording) {
      stopRecording();
    } else if (synth && synth.speaking) {
      stopSpeech();
    } else if (state.streaming) {
      stopStreaming();
    }
  });

  /* ------------------------------------------------------------------ online */
  function updateOnline() {
    dom.offlineChip.hidden = navigator.onLine;
    if (!navigator.onLine) {
      dom.offlineChip.textContent = "Offline — local server features only";
    }
    if (state.newsLoaded) loadNews(false, !navigator.onLine);
  }
  window.addEventListener("online", updateOnline);
  window.addEventListener("offline", updateOnline);
  updateOnline();

  /* -------------------------------------------------------------------- boot */
  dom.speakToggle.checked = state.speechEnabled && Boolean(synth);
  if (window.location.hash === "#app") {
    openApp("chat");
  }
})();
