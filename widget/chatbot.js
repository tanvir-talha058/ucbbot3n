/**
 * chatbot.js
 * ----------
 * UCB Bank floating chat widget — pure vanilla JavaScript, zero dependencies.
 *
 * Features:
 *  - Floating bubble fixed bottom-right
 *  - UCB Bank red (#C8102E) and white (#FFFFFF) theme
 *  - RTL support for Bangla text
 *  - Typing indicator while awaiting response
 *  - Source URL citations below each answer
 *  - Mobile responsive
 *  - Language selection persisted in sessionStorage
 *  - Session ID generated per browser session
 *
 * Embed on any page with a single tag:
 *   <script src="chatbot.js"></script>
 *
 * The widget auto-inserts the necessary CSS from chatbot.css if loaded,
 * or falls back to inline critical styles.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // CONFIGURATION — edit these values before deployment
  // ---------------------------------------------------------------------------
  var CONFIG = {
    apiUrl: "http://localhost:8000",   // FastAPI backend URL
    defaultLanguage: "english",        // "english" | "bangla" | "banglish"
    defaultTheme: "light",             // "light" | "dark"
    botName: "UCB Assistant",
    primaryColor: "#C8102E",           // UCB Bank red
    secondaryColor: "#FFFFFF",         // White
    placeholder: {
      english: "Ask about UCB Bank products...",
      bangla: "UCB ব্যাংক সম্পর্কে জিজ্ঞেস করুন...",
      banglish: "UCB Bank er bishoy jiggesh korun...",
    },
    welcomeMessage: {
      english:
        "Welcome to UCB Bank! I can help you with information about our products, services, and more. How can I assist you today?",
      bangla:
        "UCB ব্যাংকে স্বাগতম! আমি আপনাকে আমাদের পণ্য ও সেবা সম্পর্কে সাহায্য করতে পারি। আপনাকে কীভাবে সাহায্য করতে পারি?",
      banglish:
        "UCB Bank e swagotom! Ami apnake amar products o services er bishoy help korte pari. Ki help lagbe?",
    },
  };

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  var state = {
    isOpen: false,
    isTyping: false,
    language: sessionStorage.getItem("ucb_lang") || CONFIG.defaultLanguage,
    theme: localStorage.getItem("ucb_theme") || CONFIG.defaultTheme,
    sessionId: sessionStorage.getItem("ucb_session") || generateSessionId(),
  };

  // Persist session ID
  sessionStorage.setItem("ucb_session", state.sessionId);

  // ---------------------------------------------------------------------------
  // Utility functions
  // ---------------------------------------------------------------------------

  /**
   * Generate a random session ID for conversation memory.
   * @returns {string} UUID-like string
   */
  function generateSessionId() {
    return "sess_" + Math.random().toString(36).substr(2, 12) + Date.now();
  }

  /**
   * Escape HTML special characters to prevent XSS.
   * @param {string} str - Raw string to escape
   * @returns {string} HTML-escaped string
   */
  function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  /**
   * Detect if text contains Bangla characters (RTL layout hint).
   * @param {string} text
   * @returns {boolean}
   */
  function hasBangla(text) {
    return /[\u0980-\u09FF]/.test(text);
  }

  /**
   * Format source URLs as clickable links.
   * @param {string[]} sources - Array of URL strings
   * @returns {string} HTML string with anchor tags
   */
  function formatSources(sources) {
    if (!sources || sources.length === 0) return "";
    var html = '<div class="ucb-sources"><span>Sources:</span><ul>';
    sources.forEach(function (url) {
      html +=
        '<li><a href="' +
        escapeHtml(url) +
        '" target="_blank" rel="noopener">' +
        escapeHtml(url) +
        "</a></li>";
    });
    html += "</ul></div>";
    return html;
  }

  // ---------------------------------------------------------------------------
  // DOM building
  // ---------------------------------------------------------------------------

  /**
   * Inject the widget HTML into the page body.
   * Creates the bubble button and the chat panel.
   */
  function buildWidget() {
    // Container
    var container = document.createElement("div");
    container.id = "ucb-chat-container";
    if (state.theme === "dark") {
      container.classList.add("ucb-theme-dark");
    }

    // Chat panel (hidden by default)
    var panel = document.createElement("div");
    panel.id = "ucb-chat-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "UCB Bank Chat Assistant");
    panel.innerHTML =
      '<header id="ucb-chat-header">' +
      '  <div id="ucb-chat-header-info">' +
      '    <div id="ucb-chat-avatar"><img src="' + getLogoUrl() + '" alt="UCB logo" /></div>' +
      '    <div id="ucb-chat-brand">' +
      '      <div id="ucb-chat-title">' + CONFIG.botName + "</div>" +
      '      <div id="ucb-chat-status"><span class="ucb-status-dot"></span>Online</div>' +
      "    </div>" +
      "  </div>" +
      '  <div id="ucb-chat-controls">' +
      '    <button id="ucb-theme-btn" aria-label="Toggle day and night mode" title="Switch to night mode" aria-pressed="false">' +
      '      <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14" aria-hidden="true">' +
      '        <path d="M21.75 15.5A9.75 9.75 0 0 1 8.5 2.25a.75.75 0 0 0-.9-.96A10.5 10.5 0 1 0 22.71 16.4a.75.75 0 0 0-.96-.9z"/>' +
      "      </svg>" +
      "    </button>" +
      '    <div class="ucb-lang-pills" role="group" aria-label="Language">' +
      '      <button class="ucb-lang-pill active" data-language="english" type="button">EN</button>' +
      '      <button class="ucb-lang-pill" data-language="bangla" type="button">বাংলা</button>' +
      '      <button class="ucb-lang-pill" data-language="banglish" type="button">BL</button>' +
      '    </div>' +
      "  </div>" +
      "</header>" +
      '<section id="chat-area">' +
      '  <div id="welcome">' +
      '    <div class="welcome-icon" aria-hidden="true">' +
      '      <svg viewBox="0 0 24 24">' +
      '        <path d="M20 2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h3.5L12 22l4.5-4H20a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zm-9.5 10A3.5 3.5 0 1 1 14 8.5 3.5 3.5 0 0 1 10.5 12zm7 4a2.5 2.5 0 1 1 2.5-2.5 2.5 2.5 0 0 1-2.5 2.5z"/>' +
      "      </svg>" +
      "    </div>" +
      '    <h2>Welcome to UCB!</h2>' +
      '    <p>Ask me anything about accounts, loans, cards, or any banking service.</p>' +
      '    <div class="feature-grid">' +
      '      <button class="feature-card" type="button" data-query="Savings account options">' +
      '        <span class="feature-icon" aria-hidden="true">' +
      '          <svg viewBox="0 0 24 24"><path d="M12 3a7 7 0 0 0-7 7v1H3a1 1 0 0 0 0 2h1v4a3 3 0 0 0 3 3h2v1a1 1 0 1 0 2 0v-1h2v1a1 1 0 1 0 2 0v-1h2a3 3 0 0 0 3-3v-4h1a1 1 0 1 0 0-2h-2v-1a7 7 0 0 0-7-7zm-1 10V9h2v4h-2z"/></svg>' +
      "        </span>" +
      '        <span class="feature-title">Savings</span>' +
      '        <span class="feature-sub">Account options</span>' +
      "      </button>" +
      '      <button class="feature-card" type="button" data-query="Home loan financing solutions">' +
      '        <span class="feature-icon" aria-hidden="true">' +
      '          <svg viewBox="0 0 24 24"><path d="M12 3 3 10h2v10h5v-6h4v6h5V10h2z"/></svg>' +
      "        </span>" +
      '        <span class="feature-title">Home Loans</span>' +
      '        <span class="feature-sub">Financing solutions</span>' +
      "      </button>" +
      '      <button class="feature-card" type="button" data-query="Credit card benefits">' +
      '        <span class="feature-icon" aria-hidden="true">' +
      '          <svg viewBox="0 0 24 24"><path d="M21 5H3a2 2 0 0 0-2 2v2h22V7a2 2 0 0 0-2-2zm2 6H1v6a2 2 0 0 0 2 2h18a2 2 0 0 0 2-2zm-14 5H5v-2h4z"/></svg>' +
      "        </span>" +
      '        <span class="feature-title">Credit Cards</span>' +
      '        <span class="feature-sub">Cards & benefits</span>' +
      "      </button>" +
      '      <button class="feature-card" type="button" data-query="Digital banking online services">' +
      '        <span class="feature-icon" aria-hidden="true">' +
      '          <svg viewBox="0 0 24 24"><path d="M17 1H7a2 2 0 0 0-2 2v18a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2zm-5 21a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm4-5H8V4h8z"/></svg>' +
      "        </span>" +
      '        <span class="feature-title">Digital Banking</span>' +
      '        <span class="feature-sub">Online services</span>' +
      "      </button>" +
      "    </div>" +
      "  </div>" +
      '  <div id="ucb-chat-messages" role="log" aria-live="polite"></div>' +
      "</section>" +
      '<footer id="ucb-chat-input-area">' +
      '  <div id="ucb-input-shell">' +
      '    <input id="ucb-chat-input" type="text" maxlength="500" autocomplete="off" placeholder="' + CONFIG.placeholder[state.language] + '" aria-label="Type your message" />' +
      '    <span id="ucb-char-count">0/500</span>' +
      "  </div>" +
      '  <button id="ucb-voice-btn" aria-label="Voice input" title="Voice input">' +
      '    <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18" aria-hidden="true">' +
      '      <path d="M12 14.5a2.5 2.5 0 0 0 2.5-2.5V6a2.5 2.5 0 1 0-5 0v6a2.5 2.5 0 0 0 2.5 2.5Zm4.5-2.5a1 1 0 1 0-2 0 2.5 2.5 0 0 1-5 0 1 1 0 1 0-2 0 4.5 4.5 0 0 0 4 4.46V19h-2a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-2.54A4.5 4.5 0 0 0 16.5 12Z"/>' +
      "    </svg>" +
      "  </button>" +
      '  <button id="ucb-send-btn" aria-label="Send message">' +
      '    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">' +
      '      <path d="M3.6 20.5a1 1 0 0 1-1.35-1.18l2.12-6.71-2.11-6.65A1 1 0 0 1 3.6 4.9l16.8 7.02a1 1 0 0 1 0 1.84zm1.2-7.1-1.1 3.46L16.6 12 3.7 7.14l1.1 3.4H12a1 1 0 1 1 0 2Z"/>' +
      "    </svg>" +
      "  </button>" +
      "</footer>" +
      '<div id="ucb-composer-status" aria-live="polite">' +
      '  <span id="ucb-voice-status">Voice input ready.</span>' +
      '  <strong id="ucb-theme-label">Day mode</strong>' +
      "</div>";

    // Bubble button
    var bubble = document.createElement("button");
    bubble.id = "ucb-chat-bubble";
    bubble.setAttribute("aria-label", "Open UCB Bank chat");
    bubble.title = "Chat with UCB Bank";
    bubble.innerHTML =
      '<svg viewBox="0 0 24 24" fill="currentColor" width="28" height="28">' +
      '  <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>' +
      "</svg>";

    container.appendChild(panel);
    container.appendChild(bubble);
    document.body.appendChild(container);

    // Set saved language in dropdown
    document.getElementById("ucb-lang-select").value = state.language;
  }

  // ---------------------------------------------------------------------------
  // Message rendering
  // ---------------------------------------------------------------------------

  /**
   * Append a message bubble to the chat panel.
   *
   * @param {string} role - "user" or "bot"
   * @param {string} text - Message text
   * @param {string[]} [sources] - Source URLs (bot messages only)
   */
  function appendMessage(role, text, sources) {
    var messages = document.getElementById("ucb-chat-messages");
    var msgDiv = document.createElement("div");
    msgDiv.className = "ucb-message ucb-message-" + role;

    // Apply RTL direction if text contains Bangla script
    var dir = hasBangla(text) ? "rtl" : "ltr";
    msgDiv.style.direction = dir;

    var row = document.createElement("div");
    row.className = "ucb-message-row";

    var avatar = document.createElement("div");
    avatar.className = "ucb-msg-avatar ucb-msg-avatar-" + role;
    avatar.innerHTML = getMessageAvatarMarkup(role);

    var content = document.createElement("div");
    content.className = "ucb-message-content";

    var bubble = document.createElement("div");
    bubble.className = "ucb-bubble";
    // Render newlines as <br> — text is already escaped
    bubble.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");

    content.appendChild(bubble);

    // Append source citations for bot messages
    if (role === "bot" && sources && sources.length > 0) {
      var sourcesEl = document.createElement("div");
      sourcesEl.innerHTML = formatSources(sources);
      content.appendChild(sourcesEl.firstChild);
    }

    row.appendChild(avatar);
    row.appendChild(content);
    msgDiv.appendChild(row);

    messages.appendChild(msgDiv);
    // Auto-scroll to latest message
    messages.scrollTop = messages.scrollHeight;
  }

  function getMessageAvatarMarkup(role) {
    if (role === "bot") {
      return (
        '<img src="' +
        getLogoUrl() +
        '" alt="UCB logo" onerror="this.onerror=null;this.src=\'' +
        getLogoFallbackUrl() +
        '\'" />'
      );
    }

    return (
      '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
      '<circle cx="12" cy="8" r="4" fill="currentColor"></circle>' +
      '<path d="M4 20c0-3.6 3.6-6 8-6s8 2.4 8 6" fill="currentColor"></path>' +
      "</svg>"
    );
  }

  /**
   * Show or hide the animated typing indicator.
   * @param {boolean} show
   */
  function setTypingIndicator(show) {
    var existing = document.getElementById("ucb-typing");
    if (show && !existing) {
      var indicator = document.createElement("div");
      indicator.id = "ucb-typing";
      indicator.className = "ucb-message ucb-message-bot";
      indicator.innerHTML =
        '<div class="ucb-bubble ucb-typing-bubble">' +
        '<span class="ucb-dot"></span>' +
        '<span class="ucb-dot"></span>' +
        '<span class="ucb-dot"></span>' +
        "</div>";
      document.getElementById("ucb-chat-messages").appendChild(indicator);
      document.getElementById("ucb-chat-messages").scrollTop = 999999;
    } else if (!show && existing) {
      existing.remove();
    }
  }

  // ---------------------------------------------------------------------------
  // API communication
  // ---------------------------------------------------------------------------

  /**
   * Send the user query to the FastAPI /chat endpoint.
   *
   * @param {string} query - User message
   * @returns {Promise<object>} API response object
   */
  function sendToApi(query) {
    return fetch(CONFIG.apiUrl + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        language: state.language,
        session_id: state.sessionId,
      }),
    })
      .then(function (res) {
        if (!res.ok) {
          throw new Error("API error: " + res.status);
        }
        return res.json();
      });
  }

  function getSpeechRecognition() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  function getBackendOrigin() {
    try {
      return new URL(CONFIG.apiUrl, window.location.href).origin;
    } catch (err) {
      return CONFIG.apiUrl.replace(/\/$/, "").replace(/\/api$/i, "");
    }
  }

  function getLogoUrl() {
    return getBackendOrigin() + "/logo.jpg";
  }

  function getLogoFallbackUrl() {
    return getBackendOrigin() + "/favicon.svg";
  }

  function updateCharCount() {
    var input = document.getElementById("ucb-chat-input");
    var charCount = document.getElementById("ucb-char-count");
    if (!input || !charCount) return;
    charCount.textContent = input.value.length + "/500";
  }

  function setVoiceStatus(message, kind) {
    var voiceStatus = document.getElementById("ucb-voice-status");
    if (!voiceStatus) return;

    voiceStatus.textContent = message;
    voiceStatus.dataset.state = kind || "";

    if (voiceStatus._timer) {
      clearTimeout(voiceStatus._timer);
      voiceStatus._timer = null;
    }

    if (kind === "success" || kind === "info") {
      voiceStatus._timer = setTimeout(function () {
        voiceStatus.textContent = "Voice input ready.";
        voiceStatus.dataset.state = "";
        voiceStatus._timer = null;
      }, 2500);
    }
  }

  function setTheme(theme) {
    var next = theme === "dark" ? "dark" : "light";
    state.theme = next;
    localStorage.setItem("ucb_theme", next);
    var container = document.getElementById("ucb-chat-container");
    var themeBtn = document.getElementById("ucb-theme-btn");
    if (!container || !themeBtn) return;

    container.classList.toggle("ucb-theme-dark", next === "dark");
    themeBtn.setAttribute("aria-pressed", next === "dark" ? "true" : "false");
    themeBtn.title = next === "dark" ? "Switch to day mode" : "Switch to night mode";
    var themeLabel = document.getElementById("ucb-theme-label");
    if (themeLabel) {
      themeLabel.textContent = next === "dark" ? "Night mode" : "Day mode";
    }
    themeBtn.innerHTML =
      next === "dark"
        ? '<svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14" aria-hidden="true"><path d="M12 8a4 4 0 1 0 4 4 4 4 0 0 0-4-4zm0-6a1 1 0 0 1 1 1v2a1 1 0 1 1-2 0V3a1 1 0 0 1 1-1zm0 18a1 1 0 0 1 1 1v2a1 1 0 1 1-2 0v-2a1 1 0 0 1 1-1zM5.64 5.64a1 1 0 0 1 1.41 0l1.42 1.41a1 1 0 1 1-1.41 1.42L5.64 7.05a1 1 0 0 1 0-1.41zm10.88 10.88a1 1 0 0 1 1.41 0l1.42 1.41a1 1 0 0 1-1.41 1.42l-1.42-1.42a1 1 0 0 1 0-1.41zM2 13a1 1 0 1 1 0-2h2a1 1 0 1 1 0 2H2zm18 0a1 1 0 1 1 0-2h2a1 1 0 1 1 0 2h-2zM5.64 18.36a1 1 0 0 1 0-1.41l1.42-1.42a1 1 0 0 1 1.41 1.42l-1.42 1.41a1 1 0 0 1-1.41 0zm10.88-10.89a1 1 0 0 1 0-1.41l1.42-1.42a1 1 0 1 1 1.41 1.42l-1.42 1.41a1 1 0 0 1-1.41 0z"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14" aria-hidden="true"><path d="M21.75 15.5A9.75 9.75 0 0 1 8.5 2.25a.75.75 0 0 0-.9-.96A10.5 10.5 0 1 0 22.71 16.4a.75.75 0 0 0-.96-.9z"/></svg>';
  }

  function initVoiceRecognition() {
    var SpeechRecognition = getSpeechRecognition();
    var voiceBtn = document.getElementById("ucb-voice-btn");
    var input = document.getElementById("ucb-chat-input");
    var voiceStatus = document.getElementById("ucb-voice-status");
    if (!voiceBtn) return;

    if (!SpeechRecognition) {
      voiceBtn.disabled = true;
      voiceBtn.title = "Voice input is not supported in this browser";
      if (voiceStatus) {
        voiceStatus.textContent = "Voice input is not supported in this browser.";
        voiceStatus.dataset.state = "error";
      }
      return;
    }

    var recognition = new SpeechRecognition();
    var isListening = false;
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.lang = state.language === "bangla" ? "bn-BD" : "en-US";

    recognition.onstart = function () {
      isListening = true;
      voiceBtn.classList.add("ucb-listening");
      voiceBtn.title = "Listening...";
      setVoiceStatus("Listening...", "info");
    };

    recognition.onresult = function (event) {
      var transcript = "";
      for (var i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      input.value = transcript.trim();
      updateCharCount();
      setVoiceStatus("Voice captured.", "success");
    };

    recognition.onerror = function (event) {
      var reason = event && event.error ? event.error : "error";
      if (reason === "not-allowed" || reason === "service-not-allowed") {
        voiceBtn.title = "Microphone access was blocked";
        setVoiceStatus("Microphone access was blocked.", "error");
      } else if (reason === "no-speech") {
        voiceBtn.title = "No speech detected";
        setVoiceStatus("No speech detected. Try again.", "error");
      } else {
        voiceBtn.title = "Voice input unavailable";
        setVoiceStatus("Voice input unavailable.", "error");
      }
    };

    recognition.onend = function () {
      isListening = false;
      voiceBtn.classList.remove("ucb-listening");
      voiceBtn.title = "Voice input";
      if (voiceStatus && !voiceStatus.dataset.state) {
        voiceStatus.textContent = "Voice input ready.";
      }
    };

    voiceBtn.addEventListener("click", function () {
      if (isListening) {
        recognition.stop();
        return;
      }
      try {
        recognition.lang = state.language === "bangla" ? "bn-BD" : "en-US";
        recognition.start();
      } catch (err) {
        console.warn("[UCB Chatbot] Voice input failed:", err);
      }
    });

    voiceBtn._recognition = recognition;
  }

  // ---------------------------------------------------------------------------
  // Event handlers
  // ---------------------------------------------------------------------------

  /**
   * Handle user submitting a message (button click or Enter key).
   */
  function handleSend() {
    var input = document.getElementById("ucb-chat-input");
    var sendBtn = document.getElementById("ucb-send-btn");
    var query = input.value.trim();
    if (!query || state.isTyping) return;

    // Show user message
    appendMessage("user", query);
    input.value = "";
    updateCharCount();

    // Show typing indicator
    state.isTyping = true;
    setTypingIndicator(true);

    // Disable send button while waiting
    sendBtn.disabled = true;
    sendBtn.classList.add("ucb-sending");

    // Send to API
    sendToApi(query)
      .then(function (data) {
        setTypingIndicator(false);
        appendMessage("bot", data.answer, data.sources);
      })
      .catch(function (err) {
        setTypingIndicator(false);
        appendMessage(
          "bot",
          "Sorry, I couldn't connect to the server. Please try again later.",
          []
        );
        console.error("UCB Chatbot error:", err);
      })
      .finally(function () {
        state.isTyping = false;
        sendBtn.disabled = false;
        sendBtn.classList.remove("ucb-sending");
      });
  }

  function askSuggestion(el) {
    var input = document.getElementById("ucb-chat-input");
    if (!input || !el) return;
    input.value = el.getAttribute("data-query") || el.textContent || "";
    updateCharCount();
    handleSend();
  }

  /**
   * Open or close the chat panel.
   */
  function togglePanel() {
    state.isOpen = !state.isOpen;
    var panel = document.getElementById("ucb-chat-panel");
    var bubble = document.getElementById("ucb-chat-bubble");

    if (state.isOpen) {
      panel.classList.add("ucb-panel-open");
      bubble.classList.add("ucb-bubble-open");
      document.getElementById("ucb-chat-input").focus();
    } else {
      panel.classList.remove("ucb-panel-open");
      bubble.classList.remove("ucb-bubble-open");
    }
  }

  /**
   * Update UI when language is changed.
   * @param {string} lang - New language value
   */
  function setLanguage(lang) {
    state.language = lang;
    sessionStorage.setItem("ucb_lang", lang);
    var input = document.getElementById("ucb-chat-input");
    input.placeholder = CONFIG.placeholder[lang];
    // RTL for Bangla input
    input.style.direction = lang === "bangla" ? "rtl" : "ltr";
    var container = document.getElementById("ucb-chat-container");
    if (container) {
      container.classList.toggle("ucb-lang-bn", lang === "bangla");
    }
    var voiceBtn = document.getElementById("ucb-voice-btn");
    if (voiceBtn && voiceBtn._recognition) {
      voiceBtn._recognition.lang = lang === "bangla" ? "bn-BD" : "en-US";
    }

    var langButtons = document.querySelectorAll(".ucb-lang-pill");
    for (var i = 0; i < langButtons.length; i++) {
      langButtons[i].classList.toggle("active", langButtons[i].dataset.language === lang);
    }
  }

  // ---------------------------------------------------------------------------
  // Auto-resize textarea
  // ---------------------------------------------------------------------------

  /**
   * Auto-resize the textarea as user types (up to 5 lines).
   * @param {HTMLTextAreaElement} el
   */
  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  // ---------------------------------------------------------------------------
  // Event binding
  // ---------------------------------------------------------------------------

  /**
   * Attach all event listeners after DOM is built.
   */
  function bindEvents() {
    // Bubble toggle
    document.getElementById("ucb-chat-bubble").addEventListener("click", togglePanel);
    document.getElementById("ucb-theme-btn").addEventListener("click", function () {
      setTheme(state.theme === "dark" ? "light" : "dark");
    });

    document.getElementById("ucb-chat-input").addEventListener("input", updateCharCount);

    // Send button
    document.getElementById("ucb-send-btn").addEventListener("click", handleSend);

    // Enter key submits the message
    document.getElementById("ucb-chat-input").addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSend();
      }
    });

    var langButtons = document.querySelectorAll(".ucb-lang-pill");
    for (var i = 0; i < langButtons.length; i++) {
      langButtons[i].addEventListener("click", function () {
        setLanguage(this.dataset.language);
      });
    }

    var featureCards = document.querySelectorAll(".feature-card");
    for (var j = 0; j < featureCards.length; j++) {
      featureCards[j].addEventListener("click", function () {
        askSuggestion(this);
      });
    }

    initVoiceRecognition();
    updateCharCount();
  }

  // ---------------------------------------------------------------------------
  // Initialise
  // ---------------------------------------------------------------------------

  /**
   * Boot the widget: build DOM, inject CSS link, bind events.
   * Called when DOM is ready.
   */
  function init() {
    // Inject CSS link if chatbot.css is not already loaded
    if (!document.getElementById("ucb-chatbot-css")) {
      var link = document.createElement("link");
      link.id = "ucb-chatbot-css";
      link.rel = "stylesheet";
      // Derive CSS path relative to the script src
      var scripts = document.getElementsByTagName("script");
      var scriptSrc = "";
      for (var i = 0; i < scripts.length; i++) {
        if (scripts[i].src && scripts[i].src.indexOf("chatbot.js") !== -1) {
          scriptSrc = scripts[i].src;
          break;
        }
      }
      link.href = scriptSrc
        ? scriptSrc.replace("chatbot.js", "chatbot.css")
        : "chatbot.css";
      document.head.appendChild(link);
    }

    buildWidget();
    setTheme(state.theme);
    setLanguage(state.language);
    bindEvents();
    console.log("[UCB Chatbot] Widget initialised. Session:", state.sessionId);
  }

  // Wait for DOM to be ready before initialising
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
