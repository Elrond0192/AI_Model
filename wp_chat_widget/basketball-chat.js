/**
 * Basketball Performance AI – WordPress Chat Widget
 *
 * Usage
 * -----
 * The <div> rendered by the PHP shortcode carries:
 *   data-api-url  – base URL of the FastAPI server (no trailing slash)
 *   data-api-key  – optional X-API-Key header value
 *   data-title    – widget header text
 *
 * BasketballChat.init(elementId) bootstraps the widget for a given container.
 */

/* global BasketballChat */
var BasketballChat = (function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Markdown-lite renderer (bold **text**, italic _text_, code `text`)
  // -------------------------------------------------------------------------
  function renderMarkdown(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      // Bold
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      // Italic
      .replace(/_(.+?)_/g, "<em>$1</em>")
      // Inline code
      .replace(/`(.+?)`/g, "<code>$1</code>")
      // Newlines → <br>
      .replace(/\n/g, "<br>");
  }

  // -------------------------------------------------------------------------
  // Append a message bubble
  // -------------------------------------------------------------------------
  function appendMessage(messagesEl, role, text) {
    var bubble = document.createElement("div");
    bubble.className = "bball-msg bball-msg--" + role;

    var inner = document.createElement("div");
    inner.className = "bball-msg__bubble";
    inner.innerHTML = renderMarkdown(text);

    bubble.appendChild(inner);
    messagesEl.appendChild(bubble);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  // -------------------------------------------------------------------------
  // Render suggestion chips
  // -------------------------------------------------------------------------
  function renderSuggestions(suggestionsEl, suggestions, onSelect) {
    suggestionsEl.innerHTML = "";
    if (!suggestions || suggestions.length === 0) return;

    suggestions.forEach(function (s) {
      var chip = document.createElement("button");
      chip.className = "bball-chip";
      chip.textContent = s;
      chip.addEventListener("click", function () {
        suggestionsEl.innerHTML = "";
        onSelect(s);
      });
      suggestionsEl.appendChild(chip);
    });
  }

  // -------------------------------------------------------------------------
  // Send message to the API
  // -------------------------------------------------------------------------
  function sendMessage(apiUrl, apiKey, message, sessionId, onSuccess, onError) {
    var headers = { "Content-Type": "application/json" };
    if (apiKey) {
      headers["X-API-Key"] = apiKey;
    }

    fetch(apiUrl + "/api/v1/chat", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({
        message: message,
        session_id: sessionId || null,
      }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (e) {
            throw new Error(e.detail || "Request failed (" + res.status + ")");
          });
        }
        return res.json();
      })
      .then(onSuccess)
      .catch(onError);
  }

  // -------------------------------------------------------------------------
  // Bootstrap widget
  // -------------------------------------------------------------------------
  function init(elementId) {
    var container = document.getElementById(elementId);
    if (!container) return;

    var apiUrl  = container.getAttribute("data-api-url")  || "";
    var apiKey  = container.getAttribute("data-api-key")  || "";
    var title   = container.getAttribute("data-title")    || "Basketball AI";

    var sessionId = null;

    // --- Build DOM --------------------------------------------------------
    container.classList.add("basketball-chat-widget");
    container.innerHTML = [
      '<div class="bball-header">',
      '  <span class="bball-header__icon">🏀</span>',
      '  <span class="bball-header__title">' + title + "</span>",
      "</div>",
      '<div class="bball-messages" id="' + elementId + '-messages"></div>',
      '<div class="bball-suggestions" id="' + elementId + '-suggestions"></div>',
      '<div class="bball-input-row">',
      '  <input class="bball-input" id="' + elementId + '-input"',
      '         type="text" placeholder="Ask about a player or team…"',
      '         autocomplete="off" />',
      '  <button class="bball-send" id="' + elementId + '-send">Send</button>',
      "</div>",
    ].join("\n");

    var messagesEl    = document.getElementById(elementId + "-messages");
    var suggestionsEl = document.getElementById(elementId + "-suggestions");
    var inputEl       = document.getElementById(elementId + "-input");
    var sendBtn       = document.getElementById(elementId + "-send");

    // --- Welcome message --------------------------------------------------
    appendMessage(
      messagesEl,
      "bot",
      "Hi! I'm the **Basketball Performance AI** 🏀\n" +
        "Ask me anything about player performance, team fit, transfers, or peaks.\n" +
        "Type **help** for a list of things I can do."
    );

    renderSuggestions(
      suggestionsEl,
      [
        "How good is [player] at [team]?",
        "Best teams for [player]?",
        "When will [player] peak?",
      ],
      send
    );

    // --- Send handler -----------------------------------------------------
    function send(overrideText) {
      var text = (overrideText || inputEl.value).trim();
      if (!text) return;

      inputEl.value = "";
      suggestionsEl.innerHTML = "";
      appendMessage(messagesEl, "user", text);

      // Loading indicator
      var loading = appendMessage(messagesEl, "bot", "_Thinking…_");
      sendBtn.disabled = true;
      inputEl.disabled = true;

      sendMessage(
        apiUrl,
        apiKey,
        text,
        sessionId,
        function (data) {
          sessionId = data.session_id;
          loading.remove();
          sendBtn.disabled = false;
          inputEl.disabled = false;
          appendMessage(messagesEl, "bot", data.reply || "(empty response)");
          renderSuggestions(suggestionsEl, data.suggestions, send);
          inputEl.focus();
        },
        function (err) {
          loading.remove();
          sendBtn.disabled = false;
          inputEl.disabled = false;
          appendMessage(
            messagesEl,
            "bot",
            "⚠️ Error: " + (err.message || "Could not reach the AI server.")
          );
          inputEl.focus();
        }
      );
    }

    sendBtn.addEventListener("click", function () { send(); });
    inputEl.addEventListener("keydown", function (e) {
      if (e.key === "Enter") send();
    });
  }

  // -------------------------------------------------------------------------
  // Auto-init all widgets declared with data-auto-init="true"
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", function () {
    var widgets = document.querySelectorAll(
      "[data-basketball-chat][data-auto-init='true']"
    );
    widgets.forEach(function (el) {
      if (el.id) init(el.id);
    });
  });

  return { init: init };
})();
