/**
 * Enterprise RAG Engine - Frontend Application
 * ChatGPT-style dark UI with streaming, citations, and chat history
 */
(function () {
    "use strict";

    /* ── Element refs ── */
    const $ = (id) => document.getElementById(id);
    const app = $("app"),
        stream = $("stream"),
        viewport = $("viewport"),
        input = $("composerInput"),
        sendBtn = $("sendBtn"),
        sendIcon = $("sendIcon"),
        chatList = $("chatList"),
        chatSearch = $("chatSearch"),
        chatCount = $("chatCount"),
        docSelect = $("docSelect"),
        toast = $("toast"),
        welcomeHero = $("welcomeHero");

    const CHATS_KEY = "bhel_internship_chats_v1";
    const DOC_KEY = "bhel_internship_docfilter";
    const SEND_SVG = sendIcon.outerHTML;
    const STOP_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

    let chats = [];
    let activeId = null;
    let generating = false;
    let aborter = null;
    let stickBottom = true;

    /* ── Helpers ── */
    function esc(s) {
        return (s || "").replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">");
    }
    function md(text) {
        if (typeof marked !== "undefined" && marked.parse) {
            try { return marked.parse(text); } catch (e) { /* fall through */ }
        }
        return esc(text).replace(/\n/g, "<br>");
    }
    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.add("show");
        clearTimeout(showToast._t);
        showToast._t = setTimeout(() => toast.classList.remove("show"), 2600);
    }
    function timeAgo(iso) {
        const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
        if (s < 60) return "just now";
        if (s < 3600) return Math.floor(s / 60) + "m ago";
        if (s < 86400) return Math.floor(s / 3600) + "h ago";
        if (s < 86400 * 7) return Math.floor(s / 86400) + "d ago";
        return new Date(iso).toLocaleDateString();
    }
    function scrollBottom(force) {
        if (stickBottom || force) viewport.scrollTop = viewport.scrollHeight;
    }
    viewport.addEventListener("scroll", () => {
        stickBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 90;
    });

    /* Client-side citation check: every [doc, page] marker must exist in sources. */
    function checkCitations(text, sources) {
        const re = /\[([^,\[\]]+?),\s*[Pp]age\s*(\d+)\]/g;
        const norm = (n) => (n || "").trim().replace(/["']/g, "").toLowerCase().replace(/\.pdf$/, "");
        const evidence = new Set((sources || []).map((s) => norm(s.document) + "|" + s.page));
        let cited = 0, ok = 0, m;
        while ((m = re.exec(text || "")) !== null) {
            cited++;
            if (evidence.has(norm(m[1]) + "|" + parseInt(m[2], 10))) ok++;
        }
        return { cited, ok };
    }

    /* ── Chat store (localStorage) ── */
    function loadChats() {
        try {
            const raw = localStorage.getItem(CHATS_KEY);
            chats = raw ? JSON.parse(raw) : [];
            if (!Array.isArray(chats)) chats = [];
        } catch (e) { chats = []; }
    }
    function saveChats() {
        try { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); } catch (e) { /* quota */ }
    }
    function getChat(id) { return chats.find((c) => c.id === id); }

    function renderList() {
        closeMenu();
        const q = (chatSearch.value || "").toLowerCase();
        const items = chats
            .slice()
            .sort((a, b) => ((b.pinned ? 1 : 0) - (a.pinned ? 1 : 0)) ||
                (new Date(b.updatedAt) - new Date(a.updatedAt)))
            .filter((c) => c.title.toLowerCase().includes(q));
        chatCount.textContent = chats.length;
        chatList.innerHTML = "";
        if (!items.length) {
            chatList.innerHTML = '<div class="empty-note">' + (chats.length ? "No matches." : "No chats yet. Ask something!") + "</div>";
            return;
        }
        const PIN_SVG = '<svg class="pin-mark" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 17v5M9 3h6l1 7 3 3H5l3-3z"/></svg>';
        const DOTS_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>';
        items.forEach((c) => {
            const el = document.createElement("div");
            el.className = "chat-item" + (c.id === activeId ? " active" : "");
            el.dataset.id = c.id;
            el.innerHTML =
                '<div class="t">' + (c.pinned ? PIN_SVG : "") +
                "<span>" + esc(c.title) + "</span></div>" +
                '<div class="d">' + timeAgo(c.updatedAt) + "</div>" +
                '<button class="icon-btn dots-btn" data-menu="' + c.id + '" title="Options">' + DOTS_SVG + "</button>";
            el.addEventListener("click", (e) => {
                const menuBtn = e.target.closest("[data-menu]");
                if (menuBtn) { e.stopPropagation(); toggleMenu(c.id, menuBtn); return; }
                openChat(c.id);
            });
            chatList.appendChild(el);
        });
    }

    /* ── Per-chat ⋯ menu: Pin / Rename / Delete ── */
    const chatMenu = document.createElement("div");
    chatMenu.id = "chatMenu";
    document.body.appendChild(chatMenu);
    let menuFor = null;

    function closeMenu() {
        if (chatMenu) chatMenu.style.display = "none";
        menuFor = null;
    }

    function toggleMenu(id, anchorBtn) {
        if (menuFor === id) { closeMenu(); return; }
        const c = getChat(id);
        if (!c) return;
        menuFor = id;
        chatMenu.innerHTML =
            '<button data-mact="pin">' + (c.pinned ? "Unpin chat" : "Pin chat") + "</button>" +
            '<button data-mact="rename">Rename</button>' +
            '<button data-mact="del" class="danger">Delete</button>';
        chatMenu.querySelectorAll("button").forEach((b) => {
            b.addEventListener("click", (e) => {
                e.stopPropagation();
                const act = b.dataset.mact;
                closeMenu();
                if (act === "pin") togglePin(id);
                else if (act === "rename") renameChat(id);
                else if (act === "del") deleteChat(id);
            });
        });
        chatMenu.style.display = "block";
        const r = anchorBtn.getBoundingClientRect();
        const mw = 160, mh = 130;
        chatMenu.style.left = Math.max(8, Math.min(r.right - mw, window.innerWidth - mw - 8)) + "px";
        chatMenu.style.top = (r.bottom + mh + 8 > window.innerHeight ? r.top - mh - 4 : r.bottom + 4) + "px";
    }

    document.addEventListener("click", (e) => {
        if (!e.target.closest("#chatMenu")) closeMenu();
    });

    function togglePin(id) {
        const c = getChat(id);
        if (!c) return;
        c.pinned = !c.pinned;
        saveChats();
        renderList();
        showToast(c.pinned ? "Chat pinned" : "Chat unpinned");
    }

    function renameChat(id) {
        const c = getChat(id);
        if (!c) return;
        const el = chatList.querySelector('[data-id="' + id + '"] .t');
        if (!el) return;
        const titleEl = el;
        const inp = document.createElement("input");
        inp.className = "rename-input";
        inp.value = c.title;
        titleEl.replaceWith(inp);
        inp.focus();
        inp.select();
        const commit = (save) => {
            if (save && inp.value.trim()) { c.title = inp.value.trim().slice(0, 80); saveChats(); }
            renderList();
        };
        inp.addEventListener("keydown", (e) => {
            if (e.key === "Enter") commit(true);
            if (e.key === "Escape") commit(false);
            e.stopPropagation();
        });
        inp.addEventListener("blur", () => commit(true));
        inp.addEventListener("click", (e) => e.stopPropagation());
    }

    function openChat(id) {
        const c = getChat(id);
        if (!c) return;
        activeId = id;
        renderStream();
        renderList();
        closeSidebarMobile();
        input.focus();
    }

    function deleteChat(id) {
        chats = chats.filter((c) => c.id !== id);
        saveChats();
        if (activeId === id) { activeId = null; renderStream(); }
        renderList();
        showToast("Chat deleted");
    }

    function newChat() {
        if (generating) stopGeneration();
        activeId = null;
        renderStream();
        renderList();
        closeSidebarMobile();
        input.focus();
    }

    function ensureChat(firstQuestion) {
        if (activeId && getChat(activeId)) return getChat(activeId);
        const c = {
            id: "c_" + Date.now().toString(36),
            title: (firstQuestion || "New chat").slice(0, 44),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            messages: []
        };
        chats.unshift(c);
        activeId = c.id;
        return c;
    }

    function touchChat(c) {
        c.updatedAt = new Date().toISOString();
        saveChats();
        renderList();
    }

    /* ── Message rendering ── */
    function uid() { return "m" + Math.random().toString(36).slice(2, 10); }

    function renderStream() {
        stream.innerHTML = "";
        const c = activeId && getChat(activeId);
        if (!c || !c.messages.length) {
            stream.appendChild(welcomeHero);
            welcomeHero.style.display = "";
            return;
        }
        welcomeHero.style.display = "none";
        c.messages.forEach((m) => {
            stream.appendChild(m.role === "user" ? userNode(m.content) : assistantNode(m));
        });
        scrollBottom(true);
    }

    function userNode(text) {
        const d = document.createElement("div");
        d.className = "msg user";
        d.innerHTML = '<div class="user-bubble">' + esc(text) + "</div>";
        return d;
    }

    function sourcesHtml(sources) {
        if (!sources || !sources.length) return "";
        const items = sources.map((s, i) =>
            '<div class="src-item"><div class="src-head">' +
            "<span class='src-doc'>#" + (i + 1) + " " + esc(s.document) + "</span>" +
            "<span class='src-page'>p." + s.page + "</span>" +
            (s.is_table ? "<span class='src-page'>table</span>" : "") +
            '<span class="src-score">' + (typeof s.score === "number" ? s.score.toFixed(2) : esc(String(s.score ?? ""))) + "</span>" +
            "</div><div class='src-quote'>" + esc(s.snippet) + "</div></div>"
        ).join("");
        return '<details class="sources"><summary><span class="src-count">' + sources.length +
            "</span>Sources & citations</summary>" + items + "</details>";
    }

    function assistantNode(m) {
        const d = document.createElement("div");
        d.className = "msg assistant";
        const id = uid();
        let notice = "";
        if (m.abstained) {
            notice = '<div class="notice amber">Not enough grounded evidence in the retrieved passages — no answer was generated.</div>';
        } else if (m.cite && m.cite.cited > 0 && m.cite.ok < m.cite.cited) {
            notice = '<div class="notice amber">' + (m.cite.cited - m.cite.ok) + " of " + m.cite.cited +
                " citation(s) could not be matched to the retrieved sources.</div>";
        } else if (m.cite && m.cite.cited > 0) {
            notice = '<div class="notice green">All ' + m.cite.cited + " citation(s) verified against sources.</div>";
        }
        const metaBits = [];
        if (m.model) metaBits.push(esc(m.model));
        if (m.latencyMs) metaBits.push((m.latencyMs / 1000).toFixed(1) + "s");
        if (m.sources && m.sources.length) metaBits.push(m.sources.length + " sources");
        d.innerHTML =
            '<div class="avatar">D</div><div class="a-body">' +
            '<div class="a-text" id="' + id + '">' + md(m.content) + "</div>" +
            notice + sourcesHtml(m.sources) +
            '<div class="msg-meta"><span>' + metaBits.join(" · ") + "</span>" +
            '<button class="m-btn" data-copy="' + id + '">Copy</button>' +
            '<button class="m-btn" data-regen="1">Regenerate</button></div></div>';
        decorateCodeBlocks(d);
        return d;
    }

    function typingNode() {
        const d = document.createElement("div");
        d.className = "msg assistant";
        d.innerHTML = '<div class="avatar">D</div><div class="a-body"><div class="typing"><span></span><span></span><span></span></div></div>';
        return d;
    }

    /* Code-block copy buttons + message actions (delegated). */
    stream.addEventListener("click", (e) => {
        const copyBtn = e.target.closest("[data-copy]");
        if (copyBtn) {
            const el = document.getElementById(copyBtn.dataset.copy);
            if (el) navigator.clipboard.writeText(el.innerText)
                .then(() => showToast("Copied to clipboard"))
                .catch(() => showToast("Copy failed"));
            return;
        }
        if (e.target.closest("[data-regen]")) { regenerate(); return; }
        const codeBtn = e.target.closest("[data-codecopy]");
        if (codeBtn) {
            const pre = codeBtn.parentElement;
            const code = pre.querySelector("code");
            navigator.clipboard.writeText(code ? code.innerText : pre.innerText)
                .then(() => showToast("Code copied"))
                .catch(() => showToast("Copy failed"));
        }
    });

    function decorateCodeBlocks(root) {
        root.querySelectorAll("pre").forEach((pre) => {
            if (pre.querySelector("[data-codecopy]")) return;
            const b = document.createElement("button");
            b.className = "copy-code-btn";
            b.setAttribute("data-codecopy", "1");
            b.textContent = "Copy";
            pre.appendChild(b);
        });
    }

    /* ── Send / stream ── */
    function setGenerating(on) {
        generating = on;
        sendBtn.classList.toggle("stop", on);
        sendBtn.innerHTML = on ? STOP_SVG : SEND_SVG;
        sendBtn.disabled = false;
        sendBtn.title = on ? "Stop" : "Send";
    }

    function stopGeneration() {
        if (aborter) aborter.abort();
    }

    async function sendQuestion(question) {
        question = (question || "").trim();
        if (!question || generating) return;
        const chat = ensureChat(question);
        if (welcomeHero.parentNode === stream) welcomeHero.style.display = "none";

        stream.appendChild(userNode(question));
        chat.messages.push({ role: "user", content: question });
        touchChat(chat);

        input.value = "";
        input.style.height = "auto";

        const typing = typingNode();
        stream.appendChild(typing);
        scrollBottom(true);
        setGenerating(true);
        aborter = new AbortController();
        const t0 = performance.now();

        const docVal = docSelect.value;
        let sources = [];
        let text = "";
        let failed = false;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question,
                    top_n: 5,
                    stream: true,
                    doc_name: docVal === "all" ? null : docVal
                }),
                signal: aborter.signal
            });
            if (!res.ok || !res.body) throw new Error("status " + res.status);

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";
            const textEl = document.createElement("div");
            let streaming = false;

            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split("\n\n");
                buf = parts.pop();
                for (const part of parts) {
                    const line = part.trim();
                    if (!line.startsWith("data:")) continue;
                    const payload = line.slice(5).trim();
                    if (payload === "[DONE]") continue;
                    let ev;
                    try { ev = JSON.parse(payload); } catch (e) { continue; }
                    if (ev.type === "sources") {
                        sources = ev.data || [];
                    } else if (ev.type === "token") {
                        if (!streaming) {
                            streaming = true;
                            typing.replaceWith(textEl);
                            textEl.className = "msg assistant";
                            textEl.innerHTML = '<div class="avatar">D</div><div class="a-body"><div class="a-text"></div></div>';
                        }
                        text += ev.data || "";
                        textEl.querySelector(".a-text").innerHTML = esc(text).replace(/\n/g, "<br>");
                        scrollBottom();
                    }
                }
            }
            if (textEl && textEl.parentNode) textEl.remove();
            else if (typing.parentNode) typing.remove();
        } catch (err) {
            if (err && err.name === "AbortError") {
                if (typing.parentNode) typing.remove();
                if (!text) {
                    showToast("Stopped");
                    setGenerating(false);
                    aborter = null;
                    return;
                }
                showToast("Stopped — partial answer kept");
            } else {
                console.error("chat error:", err);
                if (typing.parentNode) typing.remove();
                // Fallback: non-streaming endpoint (also returns grounding metadata).
                try {
                    const r2 = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question, top_n: 5, doc_name: docVal === "all" ? null : docVal })
                    });
                    if (!r2.ok) throw new Error("fallback " + r2.status);
                    const d2 = await r2.json();
                    text = d2.raw_text || "";
                    sources = d2.sources || [];
                } catch (e2) {
                    console.error("fallback error:", e2);
                    failed = true;
                }
            }
        }

        const latencyMs = Math.round(performance.now() - t0);
        setGenerating(false);
        aborter = null;

        if (failed || (!text && !sources.length)) {
            const errMsg = { role: "assistant", content: "Request failed — is the model server reachable? Try again.", sources: [], latencyMs, model: "", abstained: true, cite: { cited: 0, ok: 0 } };
            chat.messages.push(errMsg);
            stream.appendChild(assistantNode(errMsg));
            touchChat(chat);
            showToast("Request failed");
            input.focus();
            return;
        }

        const cite = checkCitations(text, sources);
        const abstained = /do not contain information|insufficient|no relevant documents/i.test(text) && cite.cited === 0;
        const msg = {
            role: "assistant", content: text, sources, latencyMs,
            model: "", abstained, cite
        };
        chat.messages.push(msg);
        touchChat(chat);
        renderStream();
        input.focus();
    }

    function regenerate() {
        if (generating) return;
        const c = activeId && getChat(activeId);
        if (!c) return;
        let lastUser = -1, lastAsst = -1;
        c.messages.forEach((m, i) => {
            if (m.role === "user") lastUser = i;
            if (m.role === "assistant") lastAsst = i;
        });
        if (lastUser < 0) return;
        const q = c.messages[lastUser].content;
        if (lastAsst > lastUser) c.messages.splice(lastAsst, 1); // replaced by fresh answer
        renderStream();
        sendQuestionAppend(q);
    }

    async function sendQuestionAppend(question) {
        // Appends a fresh assistant answer without duplicating the user bubble.
        question = (question || "").trim();
        if (!question || generating) return;
        const chat = getChat(activeId);
        if (!chat) return sendQuestion(question);
        if (welcomeHero.parentNode === stream) welcomeHero.style.display = "none";

        const typing = typingNode();
        stream.appendChild(typing);
        scrollBottom(true);
        setGenerating(true);
        aborter = new AbortController();
        const t0 = performance.now();
        const docVal = docSelect.value;
        let sources = [], text = "", failed = false;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, top_n: 5, stream: true, doc_name: docVal === "all" ? null : docVal }),
                signal: aborter.signal
            });
            if (!res.ok || !res.body) throw new Error("status " + res.status);
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split("\n\n");
                buf = parts.pop();
                for (const part of parts) {
                    const line = part.trim();
                    if (!line.startsWith("data:")) continue;
                    const payload = line.slice(5).trim();
                    if (payload === "[DONE]") continue;
                    let ev;
                    try { ev = JSON.parse(payload); } catch (e) { continue; }
                    if (ev.type === "sources") sources = ev.data || [];
                    else if (ev.type === "token") {
                        text += ev.data || "";
                        if (typing.parentNode) {
                            typing.innerHTML = '<div class="avatar">D</div><div class="a-body"><div class="a-text">' +
                                esc(text).replace(/\n/g, "<br>") + "</div></div>";
                        }
                        scrollBottom();
                    }
                }
            }
            if (typing.parentNode) typing.remove();
        } catch (err) {
            if (typing.parentNode) typing.remove();
            if (!(err && err.name === "AbortError") || !text) failed = true;
        }

        const latencyMs = Math.round(performance.now() - t0);
        setGenerating(false);
        aborter = null;

        const cite = checkCitations(text, sources);
        const msg = failed || (!text && !sources.length)
            ? { role: "assistant", content: "Request failed — is the model server reachable? Try again.", sources: [], latencyMs, model: "", abstained: true, cite: { cited: 0, ok: 0 } }
            : {
                role: "assistant", content: text, sources, latencyMs,
                model: "", abstained: false, cite
            };
        chat.messages.push(msg);
        touchChat(chat);
        renderStream();
        input.focus();
    }

    /* ── Events ── */
    function autoresize() {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 190) + "px";
    }
    input.addEventListener("input", autoresize);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
    });
    sendBtn.addEventListener("click", () => {
        if (generating) { stopGeneration(); return; }
        const q = input.value.trim();
        if (q) sendQuestion(q);
    });

    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
        if (e.key === "Escape") { closeMenu(); closeSidebarMobile(); }
    });

    $("newChatBtn").addEventListener("click", newChat);
    $("menuBtn").addEventListener("click", () => {
        if (window.innerWidth <= 820) app.classList.toggle("side-open");
        else app.classList.toggle("side-hidden");
    });
    $("scrim").addEventListener("click", closeSidebarMobile);
    function closeSidebarMobile() { app.classList.remove("side-open"); }

    chatSearch.addEventListener("input", renderList);

    docSelect.value = localStorage.getItem(DOC_KEY) || "all";
    docSelect.addEventListener("change", () => {
        localStorage.setItem(DOC_KEY, docSelect.value);
        showToast(docSelect.value === "all" ? "Searching all documents" : "Scoped to " + docSelect.value);
    });

    $("clearAllBtn").addEventListener("click", () => {
        if (!chats.length) return;
        if (!confirm("Delete all chats?")) return;
        chats = [];
        activeId = null;
        saveChats();
        renderStream();
        renderList();
        showToast("All chats cleared");
    });

    /* ── Init ── */
    loadChats();
    renderStream();
    renderList();
    input.focus();
})();