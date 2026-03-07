// AI Model Advisor chat widget — shared across all pages.
// Expects window.CHAT_IS_ADMIN (bool) set before this script loads.
(function () {
    const IS_ADMIN = !!window.CHAT_IS_ADMIN;
    const chatHistory = [];
    let chatOpen = false;

    // ---- Inject HTML ----
    const toggleBtn = document.createElement('button');
    toggleBtn.id = 'chat-toggle';
    toggleBtn.title = 'AI Model Advisor';
    toggleBtn.innerHTML = '&#x1F4AC;';
    toggleBtn.style.cssText = 'position:fixed; bottom:24px; right:24px; z-index:1000; width:52px; height:52px; border-radius:50%; border:none; background:#7c3aed; color:white; font-size:22px; cursor:pointer; box-shadow:0 4px 16px rgba(124,58,237,0.35); display:flex; align-items:center; justify-content:center;';
    toggleBtn.onclick = toggleChat;
    document.body.appendChild(toggleBtn);

    const panel = document.createElement('div');
    panel.id = 'chat-panel';
    panel.style.cssText = 'position:fixed; top:0; right:-400px; width:380px; height:100vh; z-index:999; background:white; box-shadow:-4px 0 24px rgba(0,0,0,0.12); display:flex; flex-direction:column; transition:right 0.25s ease;';
    panel.innerHTML =
        '<div style="padding:16px 18px; border-bottom:1px solid #f3f4f6; display:flex; justify-content:space-between; align-items:center; background:#faf5ff;">' +
            '<div>' +
                '<div style="font-weight:700; font-size:14px; color:#5b21b6;">AI Model Advisor</div>' +
                '<div style="font-size:11px; color:#9ca3af; margin-top:1px;">Explore hypotheses \u00b7 Update weights \u00b7 Add rules</div>' +
            '</div>' +
            '<button id="chat-close" style="background:none; border:none; font-size:18px; cursor:pointer; color:#9ca3af; padding:4px;">\u2715</button>' +
        '</div>' +
        '<div id="chat-messages" style="flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:12px;"></div>' +
        '<div style="padding:12px 14px; border-top:1px solid #f3f4f6; background:#faf5ff;">' +
            '<div style="display:flex; gap:8px; align-items:flex-end;">' +
                '<textarea id="chat-input" placeholder="Ask about hypotheses, weights, performance\u2026" rows="1" ' +
                    'style="flex:1; border:1px solid #d1d5db; border-radius:8px; padding:9px 12px; font-size:13px; resize:none; min-height:40px; max-height:120px; font-family:inherit; outline:none;"></textarea>' +
                '<button id="chat-send" style="background:#7c3aed; color:white; border:none; border-radius:8px; padding:9px 14px; font-size:13px; font-weight:600; cursor:pointer; white-space:nowrap;">Send</button>' +
            '</div>' +
        '</div>';
    document.body.appendChild(panel);

    // Wire up events after DOM is injected
    document.getElementById('chat-close').onclick = toggleChat;
    document.getElementById('chat-send').onclick = sendChatMessage;
    document.getElementById('chat-input').addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
    });

    // ---- Functions ----
    function toggleChat() {
        chatOpen = !chatOpen;
        panel.style.right = chatOpen ? '0' : '-400px';
        if (chatOpen) {
            if (chatHistory.length === 0) {
                if (IS_ADMIN) loadPendingSuggestions();
                var greeting = IS_ADMIN
                    ? 'Hi! I\'m your AI model advisor. I can help you explore hypotheses, analyze performance, and apply changes to weights or rules directly.\n\nWhat would you like to look at?'
                    : 'Hi! I\'m your AI model advisor. I can help you explore hypotheses and model performance. You can suggest changes and I\'ll send them to the admin for approval.\n\nWhat would you like to explore?';
                appendChatBubble('assistant', greeting);
            }
            setTimeout(function () { document.getElementById('chat-input').focus(); }, 260);
        }
    }

    function appendChatBubble(role, text, actions) {
        var box = document.getElementById('chat-messages');
        var wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex; flex-direction:column; align-items:' + (role === 'user' ? 'flex-end' : 'flex-start') + ';';

        var bubble = document.createElement('div');
        bubble.style.cssText = 'max-width:88%; padding:10px 13px; border-radius:12px; font-size:13px; line-height:1.55; white-space:pre-wrap; word-break:break-word; ' +
            (role === 'user'
                ? 'background:#7c3aed; color:white; border-bottom-right-radius:3px;'
                : 'background:#f3f4f6; color:#1f2937; border-bottom-left-radius:3px;');
        bubble.textContent = text;
        wrap.appendChild(bubble);

        if (actions && actions.length) {
            var actWrap = document.createElement('div');
            actWrap.style.cssText = 'display:flex; flex-direction:column; gap:6px; margin-top:6px; max-width:88%;';

            // Goal color map for bundles (priority: combined > win_rate > speed > upside)
            var goalColors = {
                combined: { bg: '#f5f3ff', text: '#4c1d95', border: '#a78bfa' },
                win_rate: { bg: '#ecfdf5', text: '#065f46', border: '#6ee7b7' },
                speed:    { bg: '#eff6ff', text: '#1e40af', border: '#93c5fd' },
                upside:   { bg: '#fef3c7', text: '#92400e', border: '#fcd34d' }
            };

            actions.forEach(function (act) {
                var btn = document.createElement('button');
                var isBundle = act.action === 'update_weights_bundle';
                var colors = (isBundle && goalColors[act.goal]) ? goalColors[act.goal] : { bg: '#ede9fe', text: '#5b21b6', border: '#c4b5fd' };
                btn.textContent = (IS_ADMIN ? '\u25B6 ' : '\uD83D\uDCA1 Suggest: ') + (act.label || act.action);
                btn.style.cssText = 'background:' + colors.bg + '; color:' + colors.text + '; border:1px solid ' + colors.border + '; border-radius:8px; padding:7px 12px; font-size:12px; font-weight:600; cursor:pointer; text-align:left;';
                btn.onclick = (function (a, b) { return function () { IS_ADMIN ? executeAction(a, b) : suggestAction(a, b); }; })(act, btn);
                actWrap.appendChild(btn);
            });
            wrap.appendChild(actWrap);
        }

        box.appendChild(wrap);
        box.scrollTop = box.scrollHeight;
    }

    function appendTypingIndicator() {
        var box = document.getElementById('chat-messages');
        var el = document.createElement('div');
        el.id = 'chat-typing';
        el.style.cssText = 'font-size:12px; color:#9ca3af; padding:4px 0;';
        el.textContent = 'AI is thinking\u2026';
        box.appendChild(el);
        box.scrollTop = box.scrollHeight;
    }

    function removeTypingIndicator() {
        var el = document.getElementById('chat-typing');
        if (el) el.remove();
    }

    async function sendChatMessage() {
        var input = document.getElementById('chat-input');
        var msg = input.value.trim();
        if (!msg) return;
        input.value = '';
        appendChatBubble('user', msg);
        chatHistory.push({ role: 'user', content: msg });
        appendTypingIndicator();
        try {
            var resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, history: chatHistory.slice(0, -1) })
            });
            var data = await resp.json();
            removeTypingIndicator();
            var reply = data.reply || data.error || 'No response.';
            appendChatBubble('assistant', reply, data.actions || []);
            chatHistory.push({ role: 'assistant', content: reply });
        } catch (e) {
            removeTypingIndicator();
            appendChatBubble('assistant', 'Network error \u2014 please try again.');
        }
    }

    async function executeAction(action, btn) {
        btn.disabled = true;
        btn.textContent = 'Applying\u2026';
        try {
            var resp = await fetch('/api/chat/execute-action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(action)
            });
            var data = await resp.json();
            btn.textContent = data.ok ? '\u2713 ' + data.message : '\u2717 ' + data.message;
            btn.style.background = data.ok ? '#d1fae5' : '#fee2e2';
            btn.style.color = data.ok ? '#065f46' : '#991b1b';
            btn.style.borderColor = data.ok ? '#6ee7b7' : '#fca5a5';
            if (data.ok) chatHistory.push({ role: 'assistant', content: '\u2705 Applied: ' + (action.label || action.action) });
        } catch (e) {
            btn.textContent = '\u2717 Network error';
            btn.style.background = '#fee2e2';
        }
    }

    async function suggestAction(action, btn) {
        btn.disabled = true;
        btn.textContent = 'Sending suggestion\u2026';
        try {
            var resp = await fetch('/api/chat/suggest-action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });
            var data = await resp.json();
            btn.textContent = data.ok ? '\u2713 Sent to admin' : '\u2717 ' + data.message;
            btn.style.background = data.ok ? '#d1fae5' : '#fee2e2';
            btn.style.color = data.ok ? '#065f46' : '#991b1b';
            btn.style.borderColor = data.ok ? '#6ee7b7' : '#fca5a5';
        } catch (e) {
            btn.textContent = '\u2717 Network error';
            btn.style.background = '#fee2e2';
        }
    }

    async function loadPendingSuggestions() {
        try {
            var resp = await fetch('/api/chat/suggestions');
            var data = await resp.json();
            if (!data.suggestions || !data.suggestions.length) return;
            var box = document.getElementById('chat-messages');
            var wrap = document.createElement('div');
            wrap.style.cssText = 'background:#fefce8; border:1px solid #fde68a; border-radius:10px; padding:12px 14px; font-size:12px; margin-bottom:4px;';
            wrap.innerHTML = '<div style="font-weight:700; color:#92400e; margin-bottom:8px;">&#x1F4EC; ' + data.suggestions.length + ' pending suggestion' + (data.suggestions.length > 1 ? 's' : '') + ' from your team</div>';
            data.suggestions.forEach(function (s) {
                var row = document.createElement('div');
                row.id = 'suggestion-row-' + s.id;
                row.style.cssText = 'display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #fde68a; gap:8px;';
                row.innerHTML =
                    '<div style="flex:1;">' +
                        '<span style="color:#78350f; font-weight:600;">' + escHtml(s.suggested_by) + '</span>' +
                        '<span style="color:#92400e;"> \u2014 ' + escHtml(s.action.label || s.action.action || '') + '</span>' +
                    '</div>' +
                    '<div style="display:flex; gap:6px; flex-shrink:0;">' +
                        '<button data-sid="' + s.id + '" class="chat-approve-btn" style="background:#7c3aed; color:white; border:none; border-radius:6px; padding:4px 10px; font-size:11px; font-weight:600; cursor:pointer;">Approve</button>' +
                        '<button data-sid="' + s.id + '" class="chat-dismiss-btn" style="background:#f3f4f6; color:#6b7280; border:none; border-radius:6px; padding:4px 10px; font-size:11px; cursor:pointer;">Dismiss</button>' +
                    '</div>';
                wrap.appendChild(row);
            });
            box.prepend(wrap);
            // Wire approve/dismiss after injecting
            wrap.querySelectorAll('.chat-approve-btn').forEach(function (b) {
                b.onclick = function () { approveSuggestion(parseInt(b.dataset.sid), b); };
            });
            wrap.querySelectorAll('.chat-dismiss-btn').forEach(function (b) {
                b.onclick = function () { dismissSuggestion(parseInt(b.dataset.sid), b); };
            });
        } catch (e) { /* not admin or no suggestions */ }
    }

    async function approveSuggestion(id, btn) {
        btn.textContent = '\u2026'; btn.disabled = true;
        var resp = await fetch('/api/chat/suggestions/' + id + '/approve', { method: 'POST' });
        var data = await resp.json();
        var row = document.getElementById('suggestion-row-' + id);
        if (row) row.remove();
        if (data.ok) appendChatBubble('assistant', '\u2705 ' + (data.message || 'Action applied.'));
    }

    async function dismissSuggestion(id, btn) {
        btn.textContent = '\u2026'; btn.disabled = true;
        await fetch('/api/chat/suggestions/' + id + '/dismiss', { method: 'POST' });
        var row = document.getElementById('suggestion-row-' + id);
        if (row) row.remove();
    }

    function escHtml(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
})();
