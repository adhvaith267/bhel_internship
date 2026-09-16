/* Enterprise RAG Engine — Frontend */
"use strict";

// ── Marked config ──────────────────────────────────────────────
marked.setOptions({ breaks: true, gfm: true });

// ── State ──────────────────────────────────────────────────────
const DB_KEY = "rag_chats";
let chats = [];       // [{id, title, pinned, created, messages:[]}]
let activeChatId = null;
let streaming = false;
let abortCtrl = null;

// ── DOM refs ───────────────────────────────────────────────────
const app           = document.getElementById("app");
const viewport      = document.getElementById("viewport");
const stream        = document.getElementById("stream");
const composerInput = document.getElementById("composerInput");
const sendBtn       = document.getElementById("sendBtn");
const sendIcon      = document.getElementById("sendIcon");
const docSelect     = document.getElementById("docSelect");
const chatList      = document.getElementById("chatList");
const chatCount     = document.getElementById("chatCount");
const chatSearch    = document.getElementById("chatSearch");
const newChatBtn    = document.getElementById("newChatBtn");
const clearAllBtn   = document.getElementById("clearAllBtn");
const menuBtn       = document.getElementById("menuBtn");
const scrim         = document.getElementById("scrim");
const toast         = document.getElementById("toast");
const chatMenu      = document.getElementById("chatMenu");

// ── Persistence ────────────────────────────────────────────────
function saveChats() {
    try { localStorage.setItem(DB_KEY, JSON.stringify(chats)); } catch (_) {}
}
function loadChats() {
    try { chats = JSON.parse(localStorage.getItem(DB_KEY) || "[]"); } catch (_) { chats = []; }
}

// ── Chat management ────────────────────────────────────────────
function createChat() {
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2);
    const chat = { id, title: "New chat", pinned: false, created: Date.now(), messages: [] };
    chats.unshift(chat);
    saveChats();
    return chat;
}

function getChat(id) { return chats.find(c => c.id === id) || null; }

function switchChat(id) {
    activeChatId = id;
    const chat = getChat(id);
    stream.innerHTML = "";
    if (!chat || chat.messages.length === 0) {
        showHero();
    } else {
        chat.messages.forEach(m => renderMessage(m.role, m.content, m.sources, m.meta, false));
    }
    renderSidebar();
    scrollBottom();
}

function showHero() {
    stream.innerHTML = `
    <div class="hero" id="welcomeHero">
        <h1 class="pixel-rainbow" aria-label="Hello, let's start">
            <span class="hero-line">HELLO</span>
            <span class="hero-line">LET'S START<span class="hero-cursor"></span></span>
        </h1>
    </div>`;
}

function deleteChat(id) {
    chats = chats.filter(c => c.id !== id);
    saveChats();
    if (activeChatId === id) {
        if (chats.length > 0) switchChat(chats[0].id);
        else { activeChatId = null; stream.innerHTML = ""; showHero(); }
    }
    renderSidebar();
}

function renameChat(id, newTitle) {
    const chat = getChat(id);
    if (chat) { chat.title = newTitle.trim() || "Untitled"; saveChats(); renderSidebar(); }
}

function pinChat(id) {
    const chat = getChat(id);
    if (chat) { chat.pinned = !chat.pinned; saveChats(); renderSidebar(); }
}

// ── Sidebar rendering ──────────────────────────────────────────
function renderSidebar(filter = "") {
    const q = filter.toLowerCase();
    const filtered = chats.filter(c => !q || c.title.toLowerCase().includes(q) ||
        c.messages.some(m => m.content.toLowerCase().includes(q)));

    const pinned = filtered.filter(c => c.pinned);
    const unpinned = filtered.filter(c => !c.pinned);
    const ordered = [...pinned, ...unpinned];

    chatList.innerHTML = "";
    chatCount.textContent = chats.length;

    if (ordered.length === 0) {
        chatList.innerHTML = `<div class="empty-note">No chats yet</div>`;
        return;
    }

    ordered.forEach(chat => {
        const item = document.createElement("div");
        item.className = "chat-item" + (chat.id === activeChatId ? " active" : "");
        item.dataset.id = chat.id;

        const date = new Date(chat.created);
        const dateStr = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });

        item.innerHTML = `
            <div class="t">
                ${chat.pinned ? `<span class="pin-mark" title="Pinned">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1l-1.5 1.5L16 4 9 11H5l-1 1 4 4-4 5h2l4-4 4 4 1-1v-4l7-7 1.5 1.5L23 8z"/></svg>
                </span>` : ""}
                <span>${escHtml(chat.title)}</span>
            </div>
            <div class="d">${dateStr}</div>
            <button class="icon-btn dots-btn" title="Options">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/></svg>
            </button>`;

        item.addEventListener("click", e => {
            if (e.target.closest(".dots-btn")) return;
            if (e.target.closest(".rename-input")) return;
            switchChat(chat.id);
            if (window.innerWidth <= 820) closeSidebar();
        });

        item.querySelector(".dots-btn").addEventListener("click", e => {
            e.stopPropagation();
            showChatMenu(e, chat.id);
        });

        chatList.appendChild(item);
    });
}

// ── Context menu ───────────────────────────────────────────────
let menuTargetId = null;

function showChatMenu(e, chatId) {
    menuTargetId = chatId;
    const chat = getChat(chatId);
    chatMenu.style.display = "block";
    chatMenu.innerHTML = `
        <button data-action="rename">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            Rename
        </button>
        <button data-action="pin">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
            ${chat && chat.pinned ? "Unpin" : "Pin"}
        </button>
        <button data-action="delete" class="danger">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            Delete
        </button>`;

    const rect = e.target.closest("button").getBoundingClientRect();
    chatMenu.style.top = rect.bottom + 4 + "px";
    chatMenu.style.left = Math.min(rect.left, window.innerWidth - 160) + "px";

    chatMenu.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
            const action = btn.dataset.action;
            chatMenu.style.display = "none";
            if (action === "delete") deleteChat(menuTargetId);
            else if (action === "pin") { pinChat(menuTargetId); }
            else if (action === "rename") startRename(menuTargetId);
        });
    });
}

function startRename(id) {
    const item = chatList.querySelector(`[data-id="${id}"]`);
    if (!item) return;
    const titleEl = item.querySelector(".t > span:not(.pin-mark)") || item.querySelector(".t > span");
    if (!titleEl) return;
    const current = getChat(id)?.title || "";
    const input = document.createElement("input");
    input.className = "rename-input";
    input.value = current;
    titleEl.replaceWith(input);
    input.focus();
    input.select();
    const commit = () => { renameChat(id, input.value); };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", e => {
        if (e.key === "Enter") { e.preventDefault(); input.blur(); }
        if (e.key === "Escape") { input.value = current; input.blur(); }
    });
}

document.addEventListener("click", e => {
    if (!chatMenu.contains(e.target)) chatMenu.style.display = "none";
});

// ── Message rendering ──────────────────────────────────────────
function escHtml(s) {
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
                    .replace(/"/g,"&quot;").replace(/'/g,"&#039;");
}

function renderMessage(role, content, sources, meta, scroll = true) {
    const hero = stream.querySelector(".hero");
    if (hero) hero.remove();

    const div = document.createElement("div");
    div.className = `msg ${role}`;

    if (role === "user") {
        div.innerHTML = `<div class="user-bubble">${escHtml(content)}</div>`;
    } else {
        const htmlContent = renderMarkdown(content);
        const sourcesHtml = renderSources(sources);
        const metaHtml = renderMeta(meta);
        div.innerHTML = `
            <div class="avatar">R</div>
            <div class="a-body">
                <div class="a-text">${htmlContent}</div>
                ${sourcesHtml}
                ${metaHtml}
            </div>`;
        addCopyButtons(div);
    }

    stream.appendChild(div);
    if (scroll) scrollBottom();
    return div;
}

function renderMarkdown(text) {
    if (typeof marked === "undefined") return escHtml(text).replace(/\n/g, "<br>");
    try { return marked.parse(text); } catch (_) { return escHtml(text); }
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return "";
    const items = sources.map(s => {
        // Backend uses: document, page, score, snippet
        // Fallback to alternate field names for flexibility
        const docName = s.document || s.doc_name || s.source || "";
        const pageNum = s.page ?? s.page_number ?? "?";
        const scoreVal = s.score ?? s.relevance_score;
        const snippet = s.snippet || s.content || "";
        const score = scoreVal != null ? `<span class="src-score">${Number(scoreVal).toFixed(2)}</span>` : "";
        const quote = snippet ? `<div class="src-quote">${escHtml(snippet.slice(0, 200))}${snippet.length > 200 ? "…" : ""}</div>` : "";
        return `<div class="src-item">
            <div class="src-head">
                <span class="src-doc">${escHtml(docName)}</span>
                <span class="src-page">p.${pageNum}</span>
                ${score}
            </div>
            ${quote}
        </div>`;
    }).join("");
    return `<details class="sources">
        <summary>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            Sources
            <span class="src-count">${sources.length}</span>
        </summary>
        ${items}
    </details>`;
}

function renderMeta(meta) {
    if (!meta) return "";
    const parts = [];
    if (meta.latency_ms) parts.push(`${Math.round(meta.latency_ms)}ms`);
    if (meta.model) parts.push(escHtml(meta.model));
    if (!parts.length) return "";
    const copyBtn = `<button class="m-btn" title="Copy answer" onclick="copyAnswer(this)">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
    </button>`;
    return `<div class="msg-meta">${parts.join(" · ")}${copyBtn}</div>`;
}

function addCopyButtons(container) {
    container.querySelectorAll("pre").forEach(pre => {
        const btn = document.createElement("button");
        btn.className = "copy-code-btn";
        btn.textContent = "Copy";
        btn.addEventListener("click", () => {
            navigator.clipboard.writeText(pre.querySelector("code")?.textContent || pre.textContent)
                .then(() => { btn.textContent = "Copied!"; setTimeout(() => btn.textContent = "Copy", 1500); });
        });
        pre.style.position = "relative";
        pre.appendChild(btn);
    });
}

window.copyAnswer = function(btn) {
    const body = btn.closest(".a-body");
    const text = body?.querySelector(".a-text")?.innerText || "";
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "Copied!";
        setTimeout(() => btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`, 1500);
    });
};

// ── Typing indicator ───────────────────────────────────────────
function showTyping() {
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.id = "typingMsg";
    div.innerHTML = `<div class="avatar">R</div><div class="a-body"><div class="typing"><span></span><span></span><span></span></div></div>`;
    stream.appendChild(div);
    scrollBottom();
    return div;
}

// ── Send / stream ──────────────────────────────────────────────
async function sendMessage() {
    const question = composerInput.value.trim();
    if (!question || streaming) return;

    // Ensure active chat
    if (!activeChatId) {
        const chat = createChat();
        activeChatId = chat.id;
    }
    const chat = getChat(activeChatId);

    // Save and render user message
    chat.messages.push({ role: "user", content: question });
    saveChats();
    renderMessage("user", question, null, null);

    // Auto-title from first message
    if (chat.messages.filter(m => m.role === "user").length === 1) {
        chat.title = question.length > 45 ? question.slice(0, 45) + "…" : question;
        saveChats();
        renderSidebar();
    }

    composerInput.value = "";
    autoResize();
    setStreaming(true);
    showTyping();

    abortCtrl = new AbortController();
    let fullText = "";
    let sources = [];
    let metaData = null;
    let assistantDiv = null;
    let textDiv = null;

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                stream: true,
                doc_name: docSelect.value,
            }),
            signal: abortCtrl.signal,
        });

        if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const raw = line.slice(6).trim();
                if (raw === "[DONE]") break;

                let evt;
                try { evt = JSON.parse(raw); } catch (_) { continue; }

                if (evt.type === "token") {
                    // Remove typing indicator on first token
                    const typer = document.getElementById("typingMsg");
                    if (typer) typer.remove();

                    if (!assistantDiv) {
                        assistantDiv = document.createElement("div");
                        assistantDiv.className = "msg assistant";
                        assistantDiv.innerHTML = `<div class="avatar">R</div><div class="a-body"><div class="a-text"></div></div>`;
                        textDiv = assistantDiv.querySelector(".a-text");
                        stream.appendChild(assistantDiv);
                    }
                    // Backend sends token text in "data" field
                    fullText += evt.data || evt.text || "";
                    textDiv.innerHTML = renderMarkdown(fullText);
                    scrollBottom();

                } else if (evt.type === "sources") {
                    // Backend sends sources array in "data" field
                    sources = evt.data || evt.sources || [];

                } else if (evt.type === "done" || evt.type === "final") {
                    if (evt.data && evt.data.sources) sources = evt.data.sources;
                    if (evt.sources) sources = evt.sources;
                    metaData = {
                        latency_ms: evt.latency_ms || evt.data?.latency_ms,
                        model: evt.model || evt.data?.model,
                    };
                    if (evt.response || evt.data?.response) {
                        fullText = evt.response || evt.data.response;
                    }
                }
            }
        }
    } catch (err) {
        const typer = document.getElementById("typingMsg");
        if (typer) typer.remove();
        if (err.name !== "AbortError") {
            showToast("Error: " + (err.message || "Request failed"));
            if (!assistantDiv) {
                renderMessage("assistant", "Sorry, something went wrong. Please try again.", null, null);
            }
        }
    }

    // Finalize the assistant message
    if (assistantDiv) {
        const bodyEl = assistantDiv.querySelector(".a-body");
        if (sources.length > 0) bodyEl.insertAdjacentHTML("beforeend", renderSources(sources));
        if (metaData) bodyEl.insertAdjacentHTML("beforeend", renderMeta(metaData));
        addCopyButtons(assistantDiv);
        scrollBottom();
    } else if (typingEl) {
        typingEl.remove();
    }

    if (fullText) {
        chat.messages.push({ role: "assistant", content: fullText, sources, meta: metaData });
        saveChats();
    }

    setStreaming(false);
    renderSidebar();
    composerInput.focus();
}

function setStreaming(on) {
    streaming = on;
    sendBtn.disabled = false;
    if (on) {
        sendBtn.classList.add("stop");
        sendBtn.title = "Stop";
        sendBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>`;
    } else {
        sendBtn.classList.remove("stop");
        sendBtn.title = "Send";
        sendBtn.innerHTML = `<svg id="sendIcon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>`;
        abortCtrl = null;
    }
}

// ── Composer events ────────────────────────────────────────────
composerInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (streaming) {
            abortCtrl?.abort();
            setStreaming(false);
        } else {
            sendMessage();
        }
    }
});

composerInput.addEventListener("input", autoResize);

function autoResize() {
    composerInput.style.height = "auto";
    composerInput.style.height = Math.min(composerInput.scrollHeight, 190) + "px";
}

sendBtn.addEventListener("click", () => {
    if (streaming) {
        abortCtrl?.abort();
        setStreaming(false);
    } else {
        sendMessage();
    }
});

// ── Sidebar controls ───────────────────────────────────────────
newChatBtn.addEventListener("click", () => {
    const chat = createChat();
    switchChat(chat.id);
    composerInput.focus();
});

clearAllBtn.addEventListener("click", () => {
    if (!chats.length) return;
    if (confirm(`Delete all ${chats.length} chat(s)?`)) {
        chats = [];
        activeChatId = null;
        saveChats();
        stream.innerHTML = "";
        showHero();
        renderSidebar();
    }
});

chatSearch.addEventListener("input", () => renderSidebar(chatSearch.value));

// Keyboard shortcut Ctrl+K = new chat
document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        newChatBtn.click();
    }
});

// ── Mobile sidebar ─────────────────────────────────────────────
function openSidebar() { app.classList.add("side-open"); }
function closeSidebar() { app.classList.remove("side-open"); }
if (menuBtn) menuBtn.addEventListener("click", () => {
    app.classList.contains("side-open") ? closeSidebar() : openSidebar();
});
scrim.addEventListener("click", closeSidebar);

// ── Toast ──────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, duration = 3000) {
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), duration);
}

// ── Utilities ──────────────────────────────────────────────────
function scrollBottom() {
    viewport.scrollTo({ top: viewport.scrollHeight, behavior: "smooth" });
}

// ── Init ───────────────────────────────────────────────────────
loadChats();
if (chats.length > 0) {
    switchChat(chats[0].id);
} else {
    renderSidebar();
    showHero();
}
composerInput.focus();
