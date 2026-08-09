"use strict";

const elements = {
  messages: document.getElementById("messages"),
  welcome: document.getElementById("welcomeBlock"),
  input: document.getElementById("messageInput"),
  send: document.getElementById("sendButton"),
  newChat: document.getElementById("newChatButton"),
  clear: document.getElementById("clearButton"),
  conversationList: document.getElementById("conversationList"),
  serviceStatus: document.getElementById("serviceStatus"),
  pipeline: document.getElementById("pipeline"),
  pipelineProgress: document.getElementById("pipelineProgress"),
  sidebar: document.getElementById("sidebar"),
  menu: document.getElementById("menuButton"),
  toast: document.getElementById("toast"),
};

const state = {
  socket: null,
  socketPromise: null,
  conversations: [],
  activeConversationId: null,
  messages: [],
  currentAssistant: null,
  currentAnswer: "",
  streaming: false,
  toastTimer: null,
};

const STORAGE_KEY = "finsight_conversations_v1";
const ACTIVE_CONVERSATION_KEY = "finsight_active_conversation_v1";
const MAX_CONVERSATIONS = 20;
const MAX_SAVED_MESSAGES = 50;

const pipelineOrder = ["faq", "intent", "retrieval", "rerank", "generation"];
const sourceLabels = {
  clarification: "问题补充提示",
  smalltalk: "客服固定回复",
  faq: "FAQ 知识库",
  llm_direct: "大模型直答",
  rag: "RAG 检索增强",
  demo: "演示模式",
  error: "系统提示",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(value) {
  const safe = escapeHtml(value);
  return safe
    .replace(/^### (.+)$/gm, "<strong>$1</strong>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function createConversation() {
  const id =
    window.crypto?.randomUUID?.() ||
    `conversation-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    title: "新的理财咨询",
    messages: [],
    updatedAt: Date.now(),
  };
}

function getActiveConversation() {
  return state.conversations.find((item) => item.id === state.activeConversationId) || null;
}

function normalizeStoredConversations(value) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item) => item && typeof item.id === "string" && Array.isArray(item.messages))
    .map((item) => ({
      id: item.id,
      title: typeof item.title === "string" && item.title.trim()
        ? item.title.trim().slice(0, 60)
        : "新的理财咨询",
      messages: item.messages
        .filter(
          (message) =>
            message &&
            ["user", "assistant"].includes(message.role) &&
            typeof message.content === "string",
        )
        .slice(-MAX_SAVED_MESSAGES),
      updatedAt: Number(item.updatedAt) || Date.now(),
    }))
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_CONVERSATIONS);
}

function persistConversations() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(
        [...state.conversations]
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, MAX_CONVERSATIONS),
      ),
    );
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, state.activeConversationId || "");
  } catch {
    showToast("浏览器存储不可用，本次会话可能无法长期保留");
  }
}

function syncActiveConversation({ touch = true } = {}) {
  const conversation = getActiveConversation();
  if (!conversation) return;
  conversation.messages = state.messages.slice(-MAX_SAVED_MESSAGES);
  if (touch) conversation.updatedAt = Date.now();
}

function relativeTime(timestamp) {
  const elapsed = Math.max(0, Date.now() - Number(timestamp || 0));
  if (elapsed < 60_000) return "刚刚";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}分钟前`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}小时前`;
  if (elapsed < 604_800_000) return `${Math.floor(elapsed / 86_400_000)}天前`;
  return new Date(timestamp).toLocaleDateString("zh-CN", {
    month: "numeric",
    day: "numeric",
  });
}

function renderConversationList() {
  const ordered = [...state.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  elements.conversationList.innerHTML = ordered
    .map(
      (conversation) => `
        <button
          class="conversation ${conversation.id === state.activeConversationId ? "active" : ""}"
          type="button"
          data-conversation-id="${escapeHtml(conversation.id)}"
        >
          <span class="conversation-icon">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z"></path>
            </svg>
          </span>
          <span>
            <strong>${escapeHtml(conversation.title)}</strong>
            <small>${escapeHtml(relativeTime(conversation.updatedAt))}</small>
          </span>
        </button>`,
    )
    .join("");
}

function loadConversations() {
  let stored = [];
  try {
    stored = normalizeStoredConversations(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
  } catch {
    stored = [];
  }

  if (!stored.length) stored = [createConversation()];
  state.conversations = stored;

  const storedActiveId = localStorage.getItem(ACTIVE_CONVERSATION_KEY);
  state.activeConversationId = stored.some((item) => item.id === storedActiveId)
    ? storedActiveId
    : stored[0].id;

  const active = getActiveConversation();
  state.messages = active ? active.messages.map((message) => ({ ...message })) : [];
  renderConversationList();
  renderActiveConversation();
  persistConversations();
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 2600);
}

function autoResize() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 120)}px`;
}

function ensureMessageList() {
  let list = elements.messages.querySelector(".message-list");
  if (!list) {
    elements.welcome.hidden = true;
    list = document.createElement("div");
    list.className = "message-list";
    elements.messages.appendChild(list);
  }
  return list;
}

function messageIcon(role) {
  if (role === "user") return "您";
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 4.5 7.2v9.6L12 21l7.5-4.2V7.2Z"></path>
      <path d="M8.5 12h7M12 8.5v7"></path>
    </svg>`;
}

function addMessage(role, content, options = {}) {
  const list = ensureMessageList();
  const row = document.createElement("article");
  row.className = `message-row ${role}`;
  const isPending = options.pending === true;

  row.innerHTML = `
    <div class="message-avatar">${messageIcon(role)}</div>
    <div class="message-content">
      <div class="message-label">${role === "user" ? "您" : "FINSIGHT AI"}</div>
      <div class="message-bubble ${isPending ? "stream-cursor" : ""}">
        ${
          isPending
            ? '<span class="typing-dots"><i></i><i></i><i></i></span>'
            : renderMarkdown(content)
        }
      </div>
      ${role === "assistant" ? '<div class="message-meta"></div>' : ""}
    </div>`;

  list.appendChild(row);
  if (role === "assistant" && options.metadata) {
    renderMessageMetadata(row, options.metadata);
  }
  elements.messages.scrollTo({ top: elements.messages.scrollHeight, behavior: "smooth" });
  return row;
}

function renderMessageMetadata(row, metadata = {}) {
  const meta = row.querySelector(".message-meta");
  if (!meta) return;
  const source = sourceLabels[metadata.source] || metadata.source || "智能回答";
  const latency = Number(metadata.latency_ms || 0);
  meta.innerHTML = `
    <span class="source-tag">${escapeHtml(source)}</span>
    ${latency ? `<span>${latency >= 1000 ? `${(latency / 1000).toFixed(1)}s` : `${Math.round(latency)}ms`}</span>` : ""}`;
}

function renderActiveConversation() {
  elements.messages.querySelector(".message-list")?.remove();
  elements.welcome.hidden = state.messages.length > 0;

  state.messages.forEach((message) => {
    addMessage(message.role, message.content, { metadata: message.metadata });
  });
  resetPipeline();
}

function resetPipeline() {
  elements.pipeline.querySelectorAll(".pipeline-step").forEach((step) => {
    step.classList.remove("active", "complete");
    step.querySelector(".step-status").textContent = "待命";
  });
  elements.pipelineProgress.style.height = "0";
}

function setPipelineStep(stepName, value = "处理中") {
  let index = pipelineOrder.indexOf(stepName);
  if (index < 0) {
    if (["search", "retrieve", "vector_search"].includes(stepName)) index = 2;
    else if (["ranking", "parent_retrieval"].includes(stepName)) index = 3;
    else index = 4;
  }

  pipelineOrder.forEach((name, currentIndex) => {
    const step = elements.pipeline.querySelector(`[data-step="${name}"]`);
    const status = step.querySelector(".step-status");
    step.classList.toggle("complete", currentIndex < index);
    step.classList.toggle("active", currentIndex === index);
    if (currentIndex < index) status.textContent = "完成";
    if (currentIndex === index) status.textContent = value.length > 8 ? "处理中" : value;
  });

  elements.pipelineProgress.style.height = `${Math.min(100, index * 25)}%`;
}

function completePipeline() {
  elements.pipeline.querySelectorAll(".pipeline-step").forEach((step) => {
    step.classList.remove("active");
    step.classList.add("complete");
    step.querySelector(".step-status").textContent = "完成";
  });
  elements.pipelineProgress.style.height = "100%";
}

function setStreaming(streaming) {
  state.streaming = streaming;
  elements.send.classList.toggle("streaming", streaming);
  elements.input.disabled = streaming;
  elements.send.disabled = false;
}

function updateAssistantToken(token) {
  if (!state.currentAssistant) return;
  state.currentAnswer += token;
  const bubble = state.currentAssistant.querySelector(".message-bubble");
  bubble.classList.add("stream-cursor");
  bubble.innerHTML = renderMarkdown(state.currentAnswer);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  setPipelineStep("generation", "生成中");
}

function finishAssistant(metadata = {}) {
  if (!state.currentAssistant) return;
  const bubble = state.currentAssistant.querySelector(".message-bubble");
  bubble.classList.remove("stream-cursor");
  if (!state.currentAnswer) {
    state.currentAnswer = "暂未获得有效回答，请稍后再试。";
    bubble.innerHTML = renderMarkdown(state.currentAnswer);
  }
  renderMessageMetadata(state.currentAssistant, metadata);

  state.messages.push({
    role: "assistant",
    content: state.currentAnswer,
    metadata: {
      source: metadata.source,
      intent: metadata.intent,
      latency_ms: metadata.latency_ms,
    },
  });
  state.messages = state.messages.slice(-MAX_SAVED_MESSAGES);
  syncActiveConversation();
  persistConversations();
  renderConversationList();
  completePipeline();
  setStreaming(false);
  state.currentAssistant = null;
  state.currentAnswer = "";
  elements.input.focus();
}

function failAssistant(message) {
  if (state.currentAssistant) {
    const bubble = state.currentAssistant.querySelector(".message-bubble");
    bubble.classList.remove("stream-cursor");
    bubble.innerHTML = renderMarkdown(`抱歉，${message || "服务暂时不可用，请稍后重试。"}`);
  }
  setStreaming(false);
  resetPipeline();
  state.currentAssistant = null;
  state.currentAnswer = "";
  showToast(message || "请求失败，请检查服务连接");
}

function handleSocketMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    failAssistant("收到无法解析的服务响应");
    return;
  }

  if (payload.type === "ready") {
    elements.serviceStatus.textContent = "服务在线 · WebSocket 已连接";
  } else if (payload.type === "status") {
    setPipelineStep(payload.step, payload.value || "处理中");
  } else if (payload.type === "token") {
    updateAssistantToken(payload.content || "");
  } else if (payload.type === "done") {
    finishAssistant(payload);
  } else if (payload.type === "error") {
    failAssistant(payload.message);
  }
}

function connectSocket() {
  if (state.socket?.readyState === WebSocket.OPEN) return Promise.resolve(state.socket);
  if (state.socketPromise) return state.socketPromise;

  state.socketPromise = new Promise((resolve, reject) => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
    state.socket = socket;

    const timeout = window.setTimeout(() => {
      reject(new Error("WebSocket 连接超时"));
      socket.close();
    }, 4500);

    socket.addEventListener("open", () => {
      window.clearTimeout(timeout);
      state.socketPromise = null;
      elements.serviceStatus.textContent = "服务在线 · 流式通道可用";
      resolve(socket);
    });
    socket.addEventListener("message", handleSocketMessage);
    socket.addEventListener("error", () => {
      window.clearTimeout(timeout);
      state.socketPromise = null;
      reject(new Error("WebSocket 连接失败"));
    });
    socket.addEventListener("close", () => {
      state.socketPromise = null;
      if (state.socket === socket) state.socket = null;
      elements.serviceStatus.textContent = "流式通道已断开 · 可自动重连";
    });
  });

  return state.socketPromise;
}

async function fetchFallback(message, history) {
  elements.serviceStatus.textContent = "使用 HTTP 兼容模式";
  setPipelineStep("generation", "请求中");
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  const result = await response.json();
  updateAssistantToken(result.answer || "");
  finishAssistant(result);
}

async function sendMessage(text) {
  const message = (text ?? elements.input.value).trim();
  if (!message || state.streaming) return;

  const history = state.messages.slice(-10);
  state.messages.push({ role: "user", content: message });
  state.messages = state.messages.slice(-MAX_SAVED_MESSAGES);
  addMessage("user", message);
  state.currentAssistant = addMessage("assistant", "", { pending: true });
  state.currentAnswer = "";
  setStreaming(true);
  resetPipeline();
  setPipelineStep("faq", "匹配中");

  if (state.messages.filter((item) => item.role === "user").length === 1) {
    const conversation = getActiveConversation();
    if (conversation) {
      conversation.title = message.length > 16 ? `${message.slice(0, 16)}…` : message;
    }
  }
  syncActiveConversation();
  persistConversations();
  renderConversationList();

  elements.input.value = "";
  autoResize();

  try {
    const socket = await connectSocket();
    socket.send(JSON.stringify({ message, history }));
  } catch {
    try {
      await fetchFallback(message, history);
    } catch (error) {
      failAssistant(error.message);
    }
  }
}

function stopStreaming() {
  if (!state.streaming) return;
  if (state.socket) state.socket.close(1000, "user cancelled");
  failAssistant("本次回答已停止");
}

function startNewConversation() {
  if (state.streaming) stopStreaming();
  syncActiveConversation({ touch: false });

  const current = getActiveConversation();
  if (current && current.messages.length === 0) {
    elements.input.focus();
    elements.sidebar.classList.remove("open");
    return;
  }

  const conversation = createConversation();
  state.conversations.unshift(conversation);
  state.conversations = state.conversations.slice(0, MAX_CONVERSATIONS);
  state.activeConversationId = conversation.id;
  state.messages = [];
  state.currentAssistant = null;
  state.currentAnswer = "";
  elements.input.value = "";
  renderActiveConversation();
  renderConversationList();
  persistConversations();
  autoResize();
  elements.input.focus();
  elements.sidebar.classList.remove("open");
}

function switchConversation(conversationId) {
  if (conversationId === state.activeConversationId) {
    elements.sidebar.classList.remove("open");
    return;
  }
  if (state.streaming) stopStreaming();
  syncActiveConversation({ touch: false });

  const target = state.conversations.find((item) => item.id === conversationId);
  if (!target) return;
  state.activeConversationId = target.id;
  state.messages = target.messages.map((message) => ({ ...message }));
  state.currentAssistant = null;
  state.currentAnswer = "";
  renderActiveConversation();
  renderConversationList();
  persistConversations();
  elements.input.value = "";
  autoResize();
  elements.input.focus();
  elements.sidebar.classList.remove("open");
}

function clearCurrentConversation() {
  if (state.streaming) stopStreaming();
  const conversation = getActiveConversation();
  if (!conversation) return;
  conversation.messages = [];
  conversation.title = "新的理财咨询";
  conversation.updatedAt = Date.now();
  state.messages = [];
  state.currentAssistant = null;
  state.currentAnswer = "";
  elements.input.value = "";
  renderActiveConversation();
  renderConversationList();
  persistConversations();
  autoResize();
  elements.input.focus();
  showToast("当前咨询已清空");
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("health check failed");
    const data = await response.json();
    const suffix = data.mode === "demo" ? " · 演示模式" : "";
    elements.serviceStatus.textContent = `服务在线${suffix}`;
    connectSocket().catch(() => {
      elements.serviceStatus.textContent = "HTTP 服务在线 · 流式通道重连中";
    });
  } catch {
    elements.serviceStatus.textContent = "服务连接异常";
    showToast("无法连接 FastAPI 服务，请确认后端已启动");
  }
}

elements.input.addEventListener("input", autoResize);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});
elements.send.addEventListener("click", () => {
  if (state.streaming) stopStreaming();
  else sendMessage();
});
elements.newChat.addEventListener("click", startNewConversation);
elements.clear.addEventListener("click", clearCurrentConversation);
elements.conversationList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-conversation-id]");
  if (button) switchConversation(button.dataset.conversationId);
});
elements.menu.addEventListener("click", () => elements.sidebar.classList.toggle("open"));
document.addEventListener("click", (event) => {
  if (
    window.innerWidth <= 780 &&
    elements.sidebar.classList.contains("open") &&
    !elements.sidebar.contains(event.target) &&
    !elements.menu.contains(event.target)
  ) {
    elements.sidebar.classList.remove("open");
  }
});
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.question));
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNewConversation();
  }
});

loadConversations();
autoResize();
checkHealth();
