let voices = [];
let selectedFile = null;
let selectedVoiceId = null;
let lastOutput = null;

const $ = (id) => document.getElementById(id);

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = 'حدث خطأ.';
    try { const data = await response.json(); message = data.detail || message; }
    catch { message = await response.text() || message; }
    throw new Error(message);
  }
  return response.json();
}

function form(data) {
  const f = new FormData();
  Object.entries(data).forEach(([k, v]) => f.append(k, v));
  return f;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
}

function toast(message, type='normal') {
  const el = $('toast');
  el.textContent = message;
  el.className = `toast show ${type}`;
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => el.className = 'toast', 3200);
}

function setBusy(button, busy, text) {
  button.disabled = busy;
  if (busy) { button.dataset.oldText = button.textContent; button.textContent = text; }
  else button.textContent = button.dataset.oldText || button.textContent;
}

async function boot() {
  const [health, languages] = await Promise.all([api('/health'), api('/api/languages')]);
  $('health').textContent = health.cuda ? `● GPU · ${health.gpu}` : '● CPU — فعّل GPU للحصول على سرعة أعلى';
  $('health').className = `health ${health.cuda ? 'online' : 'warning'}`;
  $('language').innerHTML = Object.entries(languages).map(([id, name]) => `<option value="${id}">${esc(name)}</option>`).join('');
  await loadVoices();
}

async function loadVoices() {
  voices = await api('/api/voices');
  renderVoices();
  renderVoiceSelect();
}

function renderVoiceSelect() {
  const select = $('voiceSelect');
  if (!voices.length) {
    select.innerHTML = '<option value="">ارفع صوتًا أولًا</option>';
    select.disabled = true;
    return;
  }
  select.disabled = false;
  select.innerHTML = voices.map(v => `<option value="${v.id}">${esc(v.name)}</option>`).join('');
  if (selectedVoiceId && voices.some(v => v.id === selectedVoiceId)) select.value = selectedVoiceId;
}

function renderVoices() {
  const box = $('voices');
  $('voicesEmpty').style.display = voices.length ? 'none' : 'block';
  box.innerHTML = voices.map(v => `
    <div class="voice-item ${selectedVoiceId === v.id ? 'active' : ''}">
      <div class="voice-main" onclick="chooseVoice('${v.id}')">
        <div class="voice-avatar">🎙️</div>
        <div><strong>${esc(v.name)}</strong><small>✓ تم رفعه وتجهيزه</small></div>
      </div>
      <div class="voice-actions">
        <button onclick="playReference('${v.id}')" class="small-btn">▶ سماع</button>
        <button onclick="removeVoice('${v.id}')" class="small-btn danger">حذف</button>
      </div>
    </div>`).join('');
}

window.chooseVoice = (id) => {
  selectedVoiceId = id;
  $('voiceSelect').value = id;
  renderVoices();
  toast('تم اختيار الصوت.');
};

window.playReference = (id) => {
  const v = voices.find(x => x.id === id);
  if (!v) return;
  $('referencePlayer').src = v.audio_url;
  $('voicePreview').classList.remove('hidden');
  $('referencePlayer').play().catch(() => {});
};

window.removeVoice = async (id) => {
  const v = voices.find(x => x.id === id);
  if (!v || !confirm(`حذف الصوت «${v.name}»؟`)) return;
  try {
    await api(`/api/voices/${id}`, {method:'DELETE'});
    if (selectedVoiceId === id) selectedVoiceId = null;
    await loadVoices();
    $('voicePreview').classList.add('hidden');
    toast('تم حذف الصوت.', 'success');
  } catch (e) { toast(e.message, 'error'); }
};

$('voiceFile').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  $('uploadTitle').textContent = file.name;
  $('uploadHint').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · جاهز للرفع`;
  $('uploadStatus').className = 'upload-status ready';
  $('uploadStatus').innerHTML = '✓ تم اختيار الصوت — اضغط «حفظ الصوت وتجهيزه»';
  $('voiceName').value = file.name.replace(/\.[^.]+$/, '');
  $('dropZone').classList.add('selected');
  const url = URL.createObjectURL(file);
  $('referencePlayer').src = url;
  $('voicePreview').classList.remove('hidden');
});

$('saveVoice').addEventListener('click', async () => {
  if (!selectedFile) return toast('اختر ملف صوت أولًا.', 'error');
  const button = $('saveVoice');
  setBusy(button, true, '⏳ جاري رفع وتجهيز الصوت…');
  try {
    const name = $('voiceName').value.trim() || selectedFile.name.replace(/\.[^.]+$/, '');
    const f = new FormData();
    f.append('name', name);
    f.append('audio', selectedFile);
    const voice = await api('/api/voices', {method:'POST', body:f});
    selectedVoiceId = voice.id;
    await loadVoices();
    $('voiceSelect').value = voice.id;
    $('uploadStatus').className = 'upload-status success';
    $('uploadStatus').innerHTML = `✓ تم رفع الصوت وتجهيزه بنجاح: <b>${esc(voice.name)}</b>`;
    $('referencePlayer').src = voice.audio_url;
    $('voicePreview').classList.remove('hidden');
    toast('تم رفع الصوت وتجهيزه بنجاح.', 'success');
  } catch (e) {
    $('uploadStatus').className = 'upload-status error';
    $('uploadStatus').textContent = `✕ ${e.message}`;
    toast(e.message, 'error');
  } finally {
    setBusy(button, false);
  }
});

$('deleteVoice').addEventListener('click', () => {
  selectedFile = null;
  $('voiceFile').value = '';
  $('voicePreview').classList.add('hidden');
  $('uploadStatus').className = 'upload-status hidden';
  $('dropZone').classList.remove('selected');
  $('uploadTitle').textContent = 'اضغط لاختيار ملف صوت';
  $('uploadHint').textContent = 'يمكنك أيضًا سحب الملف وإفلاته هنا';
});

$('voiceSelect').addEventListener('change', () => {
  selectedVoiceId = $('voiceSelect').value;
  renderVoices();
});

$('text').addEventListener('input', () => $('counter').textContent = `${$('text').value.length} / 3000`);

$('generate').addEventListener('click', async () => {
  const voiceId = $('voiceSelect').value;
  const text = $('text').value.trim();
  if (!voiceId) return toast('ارفع واختر صوتًا أولًا.', 'error');
  if (!text) return toast('اكتب النص أولًا.', 'error');
  const button = $('generate');
  setBusy(button, true, '⏳ جاري نسخ الصوت…');
  $('resultCard').classList.add('hidden');
  try {
    const result = await api('/api/generate', {method:'POST', body:form({voice_id:voiceId, text, language:$('language').value, fidelity:$('fidelity').value, engine:$('engine').value, reference_text:$('referenceText').value})});
    lastOutput = result;
    $('resultPlayer').src = result.url;
    $('resultInfo').textContent = `الصوت: ${result.voice} · المحرك: ${result.engine.toUpperCase()} · ${result.generation_time}s · جاهز للاستماع والتنزيل`;
    const alt = $('alternateResult');
    if (result.alternate) {
      $('alternatePlayer').src = result.alternate.url;
      $('alternateDownload').href = result.alternate.download;
      alt.classList.remove('hidden');
    } else {
      $('alternatePlayer').removeAttribute('src');
      $('alternateDownload').removeAttribute('href');
      alt.classList.add('hidden');
    }
    $('resultCard').classList.remove('hidden');
    $('resultCard').scrollIntoView({behavior:'smooth', block:'center'});
    $('resultPlayer').play().catch(() => {});
    toast('تم توليد الصوت بنجاح.', 'success');
  } catch (e) { toast(e.message, 'error'); }
  finally { setBusy(button, false); }
});

$('download').addEventListener('click', () => {
  if (!lastOutput) return toast('ولّد صوتًا أولًا.', 'error');
  const format = $('downloadFormat').value;
  window.location.href = lastOutput.formats[format] || `/api/download/${lastOutput.id}?format=${format}`;
});

// دعم السحب والإفلات بشكل واضح.
const zone = $('dropZone');
['dragenter','dragover'].forEach(event => zone.addEventListener(event, e => { e.preventDefault(); zone.classList.add('drag'); }));
['dragleave','drop'].forEach(event => zone.addEventListener(event, e => { e.preventDefault(); zone.classList.remove('drag'); }));
zone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  selectedFile = file;
  $('voiceFile').files = e.dataTransfer.files;
  $('voiceFile').dispatchEvent(new Event('change'));
});

$('engine').addEventListener('change', () => {
  const isF5 = $('engine').value === 'f5';
  const isAuto = $('engine').value === 'auto';
  const isEnsemble = $('engine').value === 'ensemble';
  $('referenceTextWrap').classList.toggle('hidden', !isF5 && !isAuto && !isEnsemble);
  $('referenceText').required = isF5;
  if (isF5) $('referenceText').focus();
});

boot().catch(e => {
  $('health').textContent = 'تعذر الاتصال بالمحرك';
  $('health').className = 'health warning';
  toast(e.message, 'error');
});
