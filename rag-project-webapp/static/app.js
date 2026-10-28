const messages = document.querySelector('#messages');
const input = document.querySelector('#message');
const send = document.querySelector('#send');
const reset = document.querySelector('#new-chat');
const status = document.querySelector('#status');
const error = document.querySelector('#error');
let conversationId = null;
try { conversationId = localStorage.getItem('rag-conversation-id'); } catch {}
let busy = false;
function saveId(id) {
  conversationId = id;
  try { if (id) localStorage.setItem('rag-conversation-id', id); else localStorage.removeItem('rag-conversation-id'); } catch {}
}
function setBusy(value) {
  busy = value; send.disabled = value; reset.disabled = value; input.disabled = value;
  status.textContent = value ? 'Thinking and checking the paper…' : '';
}
function showError(message) { error.textContent = message; error.hidden = false; }
function bubble(role, text, sources = []) {
  const article = document.createElement('article');
  article.className = role;
  if (role === 'assistant') renderAnswer(article, text, sources);
  else article.textContent = text;
  messages.append(article); article.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  return article;
}
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = 'Request failed. Please try again.';
    try { const data = await response.json(); if (typeof data.detail === 'string') detail = data.detail; } catch {}
    if (response.status === 404) saveId(null);
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}
document.querySelector('#chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = input.value.trim(); if (busy || !text) return;
  error.hidden = true; setBusy(true);
  const userBubble = bubble('user', text);
  try {
    const data = await api('/chat', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({conversation_id: conversationId, message: text})});
    saveId(data.conversation_id); bubble('assistant', data.answer, data.sources); input.value = '';
  } catch (failure) {
    userBubble.remove(); showError(failure.message || 'Network error. Please retry.');
  } finally { setBusy(false); input.focus(); }
});
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); document.querySelector('#chat-form').requestSubmit(); }
});
reset.addEventListener('click', async () => {
  if (busy) return;
  error.hidden = true; setBusy(true);
  try {
    if (conversationId) {
      try { await api(`/conversations/${conversationId}`, {method: 'DELETE'}); }
      catch (failure) { if (conversationId) throw failure; }
    }
    saveId(null); messages.replaceChildren(); bubble('assistant', 'New chat started. What would you like to know about the paper?'); input.value = '';
  } catch (failure) { showError(failure.message); }
  finally { setBusy(false); input.focus(); }
});
if (conversationId) status.textContent = 'Continuing your saved conversation. Earlier messages are not displayed after reload.';

