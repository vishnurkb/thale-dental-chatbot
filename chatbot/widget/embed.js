/* Thale Dental chat widget - self-contained injectable script.
   Drop this on ANY page via <script src=".../chatbot-widget.js"></script>
   and it builds its own bubble + panel + styles, same-origin API by default. */
(function () {
  const API_URL = window.location.origin + "/chat";

  const style = document.createElement("style");
  style.textContent = `
    #tdc-bubble {
      position: fixed; bottom: 20px; right: 20px; width: 56px; height: 56px;
      border-radius: 50%; background: #0c6b58; color: #fff; border: none;
      font-size: 14px; font-family: system-ui, sans-serif; cursor: pointer;
      box-shadow: 0 2px 10px rgba(0,0,0,0.25); z-index: 999999;
    }
    #tdc-panel {
      position: fixed; bottom: 88px; right: 20px; width: 340px; max-height: 480px;
      background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      display: none; flex-direction: column; overflow: hidden; z-index: 999999;
      font-family: system-ui, sans-serif;
    }
    #tdc-panel.open { display: flex; }
    #tdc-header { background: #0c6b58; color: #fff; padding: 12px 16px; font-weight: 600; }
    #tdc-messages { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px; max-height: 360px; }
    .tdc-msg { padding: 8px 12px; border-radius: 10px; max-width: 85%; font-size: 14px; line-height: 1.4; }
    .tdc-msg.user { align-self: flex-end; background: #0c6b58; color: #fff; }
    .tdc-msg.bot { align-self: flex-start; background: #f0f0f0; color: #1a1a1a; }
    #tdc-input-row { display: flex; border-top: 1px solid #eee; }
    #tdc-input { flex: 1; border: none; padding: 12px; font-size: 14px; }
    #tdc-send { border: none; background: #0c6b58; color: #fff; padding: 0 16px; cursor: pointer; }
  `;
  document.head.appendChild(style);

  const bubble = document.createElement("button");
  bubble.id = "tdc-bubble";
  bubble.setAttribute("aria-label", "Open chat");
  bubble.textContent = "Chat";

  const panel = document.createElement("div");
  panel.id = "tdc-panel";
  panel.innerHTML = `
    <div id="tdc-header">Theale Dental Assistant</div>
    <div id="tdc-messages"></div>
    <div id="tdc-input-row">
      <input id="tdc-input" type="text" placeholder="Ask a question..." />
      <button id="tdc-send">Send</button>
    </div>
  `;

  document.body.appendChild(bubble);
  document.body.appendChild(panel);

  function getSessionId() {
    let id = localStorage.getItem("thale_chat_session");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("thale_chat_session", id);
    }
    return id;
  }

  const messagesEl = panel.querySelector("#tdc-messages");
  const input = panel.querySelector("#tdc-input");
  const sendBtn = panel.querySelector("#tdc-send");

  bubble.addEventListener("click", () => panel.classList.toggle("open"));

  function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = `tdc-msg ${who}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;
    addMessage(text, "user");
    input.value = "";
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: getSessionId(), message: text }),
      });
      const data = await res.json();
      addMessage(data.reply, "bot");
    } catch (e) {
      addMessage("Sorry, could not reach the chatbot. Is the server running?", "bot");
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });
})();
