# Basketball AI Chat Widget — WordPress Plugin

A lightweight WordPress plugin that embeds the **Basketball Performance AI**
chat interface as a shortcode anywhere on your site.

---

## Requirements

| Component | Details |
|---|---|
| WordPress | ≥ 5.8 |
| FastAPI server | Running and publicly reachable (e.g. Azure App Service) |
| Trained model | `python main.py --mode train` must have been run |

---

## Installation

1. **Copy the plugin folder** to your WordPress installation:

   ```
   wp-content/plugins/basketball-chat/
   ├── basketball-chat.php
   ├── basketball-chat.js
   └── basketball-chat.css
   ```

2. **Activate** the plugin in *WP Admin → Plugins → Basketball Performance AI Chat*.

3. **Deploy the API server** (see root `README.md`) and note the public URL,
   e.g. `https://basketball-ai.azurewebsites.net`.

4. **Add the shortcode** to any page, post, or widget:

   ```
   [basketball_chat api_url="https://basketball-ai.azurewebsites.net"]
   ```

---

## Shortcode Reference

```
[basketball_chat
  api_url="https://your-api.example.com"
  api_key=""
  title="Basketball AI"
  height="520"
]
```

| Attribute | Required | Default | Description |
|---|---|---|---|
| `api_url` | ✅ | — | Base URL of the FastAPI server (no trailing slash) |
| `api_key` | ❌ | *(empty)* | `X-API-Key` header value. Set `API_KEY` env var on the server to enable. |
| `title` | ❌ | `Basketball AI` | Header text inside the widget |
| `height` | ❌ | `520` | Widget height in pixels (minimum 300) |

---

## How it works

```
User types a message
        │
        ▼
basketball-chat.js  ──POST /api/v1/chat──▶  FastAPI server
                                                    │
                                           Intent detection
                                           Entity extraction
                                           WhatIfEngine call
                                                    │
                    ◀── JSON response ──────────────┘
        │
Widget renders reply + suggestion chips
```

The chat endpoint maintains **conversation context** via a `session_id` stored
in memory on the server. Follow-up questions like *"what about his peak?"*
correctly resolve the player mentioned in the previous turn.

---

## API endpoint

`POST /api/v1/chat`

**Request**
```json
{
  "message":    "How good is Player 1 at Team 3?",
  "session_id": "optional-uuid"
}
```

**Response**
```json
{
  "reply":       "**Player 1** at **Team 3** ...",
  "session_id":  "550e8400-...",
  "intent":      "predict",
  "data":        { "predicted_rating": 7.42, ... },
  "suggestions": ["When will Player 1 peak?", ...]
}
```

---

## Security

- Set `API_KEY` in the server's `.env` file to require `X-API-Key` on every
  request.
- Pass the same key to the shortcode via `api_key="…"`.
- Enable HTTPS on the API server before exposing it to the public internet.
- The `api_key` attribute is rendered as a plain HTML `data-*` attribute and
  will be visible in the page source — for high-security deployments, proxy
  the `/api/v1/chat` endpoint through a WordPress REST route that adds the
  header server-side.

---

## Customising the look

Override styles in your theme's `style.css` using the namespaced selectors:

```css
/* Change the header gradient */
.basketball-chat-widget .bball-header {
  background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
}

/* Adjust widget font size */
.basketball-chat-widget {
  font-size: 15px;
}
```
