// =====================================================================
// GLOBAL APP STATE
// =====================================================================
const state = {
    apiKey: '',
    apiKeys: [],
    apiKeyStatuses: {},
    analysisApiKeys: [],
    apiVerified: false,
    groqApiKey: '',
    selectedVideo: '',
    selectedBgm: '',
    bgmVolume: 12,
    modelName: 'gemini-3.1-flash-lite',
    thinkingEffort: 'low',
    script: '',
    sentences: [],
    voiceName: 'vi-VN-HoaiMyNeural',
    voiceRate: '+20%',
    narrationBudget: null,
    projectId: '',
    scriptProjectId: '',
    scriptVideo: '',
    splitScriptSnapshot: '',
    highestStepReached: 1
};

let apiVerificationTimer = null;
let apiVerificationController = null;
let apiVerificationRevision = 0;

function narrationSentencesPerMinute() {
    const rate = Number(dom.voiceRateSlider?.value ?? 20);
    return 13.5 * (1 + rate / 100) / 1.2;
}

function displayActualNarrationTiming(taskData) {
    const timing = taskData?.narration_timing;
    const element = document.getElementById('stat-est-duration');
    if (element && Number.isFinite(timing?.actual_minutes)) {
        element.textContent = `${timing.actual_minutes.toFixed(1)}p (TTS thực tế)`;
    }
}

// =====================================================================
// DOM ELEMENTS REFERENCE
// =====================================================================
const dom = {
    // Stepper
    steps: document.querySelectorAll('.step'),
    stepLines: document.querySelectorAll('.step-line'),
    panels: document.querySelectorAll('.wizard-panel'),
    
    // Step 1
    apiKeyInput: document.getElementById('api-key-input'),
    apiKeyInput2: document.getElementById('api-key-input-2'),
    apiKeyInput3: document.getElementById('api-key-input-3'),
    btnToggleApiKey: document.getElementById('btn-toggle-api-key'),
    btnAddApiSlot: document.getElementById('btn-add-api-slot'),
    additionalApiKeys: document.getElementById('additional-api-keys'),
    apiKeyRows: document.querySelectorAll('[data-api-row]'),
    apiPriorityButtons: document.querySelectorAll('.btn-api-priority'),
    apiRemoveButtons: document.querySelectorAll('.btn-remove-api-slot'),
    apiKeysInlineError: document.getElementById('api-keys-inline-error'),
    apiKeysGroup: document.getElementById('api-keys-group'),
    groqApiKeyInput: document.getElementById('groq-api-key-input'),
    modelSelect: document.getElementById('model-select'),
    thinkingEffortSelect: document.getElementById('thinking-effort-select'),
    recapAiSettingsSlot: document.getElementById('recap-ai-settings-slot'),
    btnVerifyApi: document.getElementById('btn-verify-api'),
    apiStatusBadge: document.getElementById('api-status-badge'),
    videoSelect: document.getElementById('video-select'),
    videoDropZone: document.getElementById('video-drop-zone'),
    videoFileInput: document.getElementById('video-file-input'),
    videoUploadProgress: document.getElementById('video-upload-progress'),
    btnToStep2: document.getElementById('btn-to-step2'),
    
    // Step 2
    styleSelect: document.getElementById('style-select'),
    reviewMinutes: document.getElementById('review-minutes'),
    reviewMinutesHint: document.getElementById('review-minutes-hint'),
    customPrompt: document.getElementById('custom-prompt'),
    subtitleFileInput: document.getElementById('subtitle-file-input'),
    btnGenerateScript: document.getElementById('btn-generate-script'),
    btnQuickVideo: document.getElementById('btn-quick-video'),
    scriptEditor: document.getElementById('script-editor'),
    scriptCharCount: document.getElementById('script-char-count'),
    scriptGenLoading: document.getElementById('script-gen-loading'),
    scriptLoadingTitle: document.getElementById('script-loading-title'),
    scriptProgressFill: document.getElementById('script-progress-fill'),
    scriptLoadingMsg: document.getElementById('script-loading-msg'),
    btnToStep3: document.getElementById('btn-to-step3'),
    
    // Step 3
    voiceSelect: document.getElementById('voice-select'),
    voiceRateSlider: document.getElementById('voice-rate-slider'),
    voiceRateVal: document.getElementById('voice-rate-val'),
    btnReSplit: document.getElementById('btn-re-split'),
    btnAddSentence: document.getElementById('btn-add-sentence'),
    sentencesList: document.getElementById('sentences-list'),
    btnToStep4: document.getElementById('btn-to-step4'),
    
    // Step 4
    bgmSelect: document.getElementById('bgm-select'),
    bgmVolumeSlider: document.getElementById('bgm-volume-slider'),
    bgmVolumeVal: document.getElementById('bgm-volume-val'),
    bgmDropZone: document.getElementById('bgm-drop-zone'),
    bgmFileInput: document.getElementById('bgm-file-input'),
    bgmUploadProgress: document.getElementById('bgm-upload-progress'),
    filterMirror: document.getElementById('filter-mirror'),
    filterZoom: document.getElementById('filter-zoom'),
    filterKenBurns: document.getElementById('filter-ken-burns'),
    filterContrast: document.getElementById('filter-contrast'),
    filterSpeed: document.getElementById('filter-speed'),
    subSize: document.getElementById('sub-size'),
    subColor: document.getElementById('sub-color'),
    subOpacity: document.getElementById('sub-opacity'),
    btnToStep5: document.getElementById('btn-to-step5'),
    
    // Step 5
    summaryVideo: document.getElementById('summary-video'),
    summarySentences: document.getElementById('summary-sentences'),
    summaryVoice: document.getElementById('summary-voice'),
    summaryBgm: document.getElementById('summary-bgm'),
    summaryFilters: document.getElementById('summary-filters'),
    btnStartRender: document.getElementById('btn-start-render'),
    renderProgressBox: document.getElementById('render-progress-box'),
    renderStatusLbl: document.getElementById('render-status-lbl'),
    renderStatusPercent: document.getElementById('render-status-percent'),
    renderProgressFill: document.getElementById('render-progress-fill'),
    renderLogTerminal: document.getElementById('render-log-terminal'),
    videoPreviewPlaceholder: document.getElementById('video-preview-placeholder'),
    videoPreviewWrapper: document.getElementById('video-preview-wrapper'),
    finalVideoPlayer: document.getElementById('final-video-player'),
    btnDownloadVideo: document.getElementById('btn-download-video'),
    btnRenderBack: document.getElementById('btn-render-back'),
    exportDirectoryDisplay: document.getElementById('export-directory-display'),
    btnChooseExportDirectory: document.getElementById('btn-choose-export-directory'),
    
    // Commons
    backButtons: document.querySelectorAll('.btn-back'),
    toastContainer: document.getElementById('toast-container')
};

// =====================================================================
// HELPER FUNCTIONS
// =====================================================================
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fa-solid ${type === 'success' ? 'fa-circle-check text-green' : 'fa-circle-exclamation text-red'}"></i>
        <span>${message}</span>
    `;
    dom.toastContainer.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s reverse forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function createProjectId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
    }
    return `project-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function resetProjectData(nextVideo = '') {
    state.projectId = createProjectId();
    state.selectedVideo = nextVideo;
    state.script = '';
    state.sentences = [];
    state.scriptProjectId = '';
    state.scriptVideo = '';
    state.splitScriptSnapshot = '';
    state.highestStepReached = Math.min(state.highestStepReached, 2);

    if (dom.scriptEditor) dom.scriptEditor.value = '';
    if (dom.scriptCharCount) dom.scriptCharCount.textContent = '0 từ';
    if (dom.sentencesList) dom.sentencesList.innerHTML = '';
    if (dom.renderProgressBox) dom.renderProgressBox.classList.add('hidden');
    if (dom.renderLogTerminal) dom.renderLogTerminal.innerHTML = '';
    if (dom.videoPreviewWrapper) dom.videoPreviewWrapper.classList.add('hidden');
    if (dom.videoPreviewPlaceholder) dom.videoPreviewPlaceholder.classList.remove('hidden');
    if (dom.finalVideoPlayer) {
        dom.finalVideoPlayer.pause();
        dom.finalVideoPlayer.removeAttribute('src');
        dom.finalVideoPlayer.load();
    }
    if (dom.btnDownloadVideo) dom.btnDownloadVideo.removeAttribute('href');
    if (dom.btnStartRender) {
        dom.btnStartRender.disabled = false;
        dom.btnStartRender.innerHTML = '<i class="fa-solid fa-circle-play"></i> BẮT ĐẦU XUẤT VIDEO REVIEW';
    }
    if (dom.btnRenderBack) dom.btnRenderBack.disabled = false;
    stopRenderTimer();
}

async function loadExportDirectory() {
    try {
        const res = await fetch('/api/export-directory');
        const data = await res.json();
        if (dom.exportDirectoryDisplay) dom.exportDirectoryDisplay.value = data.directory || '';
        if (data.directory && data.valid === false && data.message) {
            showToast(data.message, 'error');
        }
    } catch (_) {
        showToast('Không kiểm tra được thư mục xuất video.', 'error');
    }
}

async function chooseExportDirectory() {
    if (!dom.btnChooseExportDirectory) return false;
    dom.btnChooseExportDirectory.disabled = true;
    try {
        const res = await fetch('/api/export-directory', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            dom.exportDirectoryDisplay.value = data.directory;
            showToast('Đã lưu thư mục xuất. Video sau khi render sẽ tự chép vào đây.', 'success');
            return true;
        } else if (!data.cancelled) {
            showToast(data.message || 'Không thể chọn thư mục xuất.', 'error');
        }
    } catch (_) {
        showToast('Không thể mở hộp chọn thư mục.', 'error');
    } finally {
        dom.btnChooseExportDirectory.disabled = false;
    }
    return false;
}

async function ensureExportDirectory() {
    try {
        const res = await fetch('/api/export-directory');
        const data = await res.json();
        if (data.valid && data.directory) {
            dom.exportDirectoryDisplay.value = data.directory;
            return true;
        }
    } catch (_) {
        showToast('Không kiểm tra được thư mục xuất video.', 'error');
        return false;
    }
    showToast('Hãy chọn thư mục lưu video trước khi bắt đầu render.', 'error');
    return chooseExportDirectory();
}

function updateAPIIndicator(isValid, message) {
    state.apiVerified = Boolean(isValid);
    const indicator = dom.apiStatusBadge.querySelector('.indicator');
    const statusText = dom.apiStatusBadge.querySelector('.status-text');
    
    if (isValid) {
        indicator.className = 'indicator green';
        statusText.textContent = 'API Đã Kết Nối';
        dom.apiStatusBadge.style.background = 'rgba(16, 185, 129, 0.08)';
        dom.apiStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else {
        indicator.className = 'indicator red';
        statusText.textContent = message || 'Chưa xác thực API';
        dom.apiStatusBadge.style.background = 'rgba(239, 68, 68, 0.08)';
        dom.apiStatusBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
    }
    if (dom.btnToStep2) checkNextStep1Availability();
}

function updateCharCount() {
    const text = dom.scriptEditor.value.trim();
    const wordCount = text ? text.split(/\s+/).filter(w => w.length > 0).length : 0;
    const estMinutes = (wordCount / (20 * narrationSentencesPerMinute())).toFixed(1);
    dom.scriptCharCount.textContent = `${wordCount} từ (~${estMinutes} phút)`;
}

// =====================================================================
// STEP NAVIGATION LOGIC
// =====================================================================
function switchPanel(stepNum) {
    // Mở Panel tương ứng
    dom.panels.forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${stepNum}`).classList.remove('hidden');
    document.getElementById(`panel-${stepNum}`).classList.add('active');
    
    // Cập nhật Stepper CSS
    dom.steps.forEach(step => {
        const sNum = parseInt(step.getAttribute('data-step'));
        step.classList.remove('active', 'completed');
        if (sNum === stepNum) {
            step.classList.add('active');
        } else if (sNum < state.highestStepReached) {
            step.classList.add('completed');
        }
    });
    
    dom.stepLines.forEach((line, idx) => {
        line.classList.remove('completed');
        if (idx + 1 < state.highestStepReached) {
            line.classList.add('completed');
        }
    });
    
    // Một số logic chuẩn bị khi vào bước cụ thể
    if (stepNum === 3) {
        // Tự động phân tách kịch bản thành câu nếu câu đang trống
        if (state.sentences.length === 0) {
            autoSplitScript();
        }
    } else if (stepNum === 5) {
        updateRenderSummary();
    }
}

function enableStep(stepNum) {
    if (stepNum > state.highestStepReached) {
        state.highestStepReached = stepNum;
    }
    // Kích hoạt click sự kiện
    const stepEl = document.querySelector(`.step[data-step="${stepNum}"]`);
    if (stepEl) {
        stepEl.style.cursor = 'pointer';
    }
}

// =====================================================================
// INITIAL LOADING & API TASKS
// =====================================================================
function apiKeySlotInputs() {
    return [dom.apiKeyInput, dom.apiKeyInput2, dom.apiKeyInput3];
}

function setApiKeySlotValues(keys) {
    const padded = [...keys.slice(0, 3), '', '', ''].slice(0, 3);
    apiKeySlotInputs().forEach((input, index) => {
        if (input) input.value = padded[index];
    });
    refreshApiSlotVisibility(Math.max(1, keys.length));
}

function paintApiKeyStatus(key, status = '', message = '') {
    apiKeySlotInputs().forEach(input => {
        if (!input || input.value.trim() !== key) return;
        input.classList.remove(
            'api-key-usable', 'api-key-exhausted',
            'api-key-invalid', 'api-key-rate-limited', 'api-key-active',
            'api-key-error', 'api-key-network-error', 'api-key-temporary-error',
            'api-key-probe-error', 'api-key-connection-error'
        );
        // Successful keys stay visually neutral. Only failures need attention.
        if (status && !['usable', 'active', 'pending'].includes(status)) {
            input.classList.add(`api-key-${status.replace('_', '-')}`);
        }
        input.title = message || '';
        const row = input.closest('.input-with-icon');
        if (row) row.dataset.apiStatus = status || '';
    });
}

function repaintAllApiKeyStatuses() {
    apiKeySlotInputs().forEach(input => {
        if (!input) return;
        const key = input.value.trim();
        const info = state.apiKeyStatuses[key] || {};
        paintApiKeyStatus(key, info.status || '', info.message || '');
    });
}

function usableApiKeys() {
    const validation = validateApiKeySlots();
    if (!validation.valid) return null;
    return validation.keys.filter(key => {
        const status = (state.apiKeyStatuses[key] || {}).status;
        return status === 'usable' || status === 'active';
    });
}

function syncTaskApiKeyStates(taskData) {
    const rows = Array.isArray(taskData && taskData.api_key_states)
        ? taskData.api_key_states : [];
    if (!rows.length || !state.analysisApiKeys.length) return;
    rows.forEach(row => {
        const priority = Number(row.priority || 0);
        const key = state.analysisApiKeys[priority - 1];
        if (!key) return;
        const status = String(row.status || 'pending');
        const messages = {
            active: 'Đang được sử dụng', usable: 'API dùng được',
            exhausted: 'Đã hết quota ngày', error: 'API gặp lỗi', pending: 'Đang chờ',
        };
        state.apiKeyStatuses[key] = { status, message: messages[status] || status };
    });
    const currentKeys = apiKeySlotInputs()
        .slice(0, visibleApiSlotCount())
        .map(input => input ? input.value.trim() : '')
        .filter(Boolean);
    repaintAllApiKeyStatuses();
    // Slot order is user-owned: runtime status must never move a key.
    state.apiKeys = currentKeys;
    state.apiKey = state.apiKeys[0] || '';
    persistOrderedApiKeys(currentKeys);
}

function validateApiKeySlots() {
    const values = apiKeySlotInputs().map(input => input ? input.value.trim() : '');
    apiKeySlotInputs().forEach(input => input && input.classList.remove('api-key-duplicate'));
    let error = '';
    const visibleCount = visibleApiSlotCount();
    for (let index = 0; index < visibleCount - 1; index++) {
        if (!values[index] && values.slice(index + 1, visibleCount).some(Boolean)) {
            error = `Hãy nhập API ${index + 1} trước API tiếp theo.`;
            break;
        }
    }
    const firstIndexByKey = new Map();
    values.slice(0, visibleCount).forEach((key, index) => {
        if (!key) return;
        if (firstIndexByKey.has(key)) {
            const firstIndex = firstIndexByKey.get(key);
            apiKeySlotInputs()[firstIndex].classList.add('api-key-duplicate');
            apiKeySlotInputs()[index].classList.add('api-key-duplicate');
            if (!error) error = `API ${index + 1} bị trùng với API ${firstIndex + 1}.`;
        } else {
            firstIndexByKey.set(key, index);
        }
    });
    if (dom.apiKeysInlineError) {
        dom.apiKeysInlineError.textContent = error;
        dom.apiKeysInlineError.classList.toggle('hidden', !error);
    }
    return { valid: !error, keys: values.slice(0, visibleCount).filter(Boolean), error };
}

function collectApiKeys() {
    const result = validateApiKeySlots();
    if (!result.valid) return null;
    return result.keys;
}

function scheduleAutomaticApiVerification(delay = 700) {
    window.clearTimeout(apiVerificationTimer);
    const revision = ++apiVerificationRevision;
    if (apiVerificationController) apiVerificationController.abort();
    apiVerificationTimer = window.setTimeout(() => {
        const apiKeys = collectApiKeys();
        const groqApiKey = dom.groqApiKeyInput ? dom.groqApiKeyInput.value.trim() : '';
        if (apiKeys === null) {
            updateAPIIndicator(false, dom.apiKeysInlineError.textContent || 'Danh sách Gemini API không hợp lệ.');
            return;
        }
        if (!apiKeys.length && !groqApiKey) {
            updateAPIIndicator(false, 'Chưa nhập Gemini API và Groq API');
            return;
        }
        verifyApiKey(false, revision);
    }, delay);
}

function persistOrderedApiKeys(keys) {
    localStorage.setItem('gemini_api_keys', JSON.stringify(keys.slice(0, 3)));
    if (keys[0]) localStorage.setItem('gemini_api_key', keys[0]);
    else localStorage.removeItem('gemini_api_key');
}

function visibleApiSlotCount() {
    return 1 + Array.from(dom.apiKeyRows || []).filter(row => !row.classList.contains('hidden')).length;
}

function refreshApiSlotVisibility(count) {
    const visibleCount = Math.max(1, Math.min(3, Number(count) || 1));
    Array.from(dom.apiKeyRows || []).forEach((row, index) => {
        row.classList.toggle('hidden', index + 2 > visibleCount);
    });
    if (dom.additionalApiKeys) dom.additionalApiKeys.classList.toggle('hidden', visibleCount === 1);
    if (dom.btnAddApiSlot) dom.btnAddApiSlot.disabled = visibleCount >= 3;
    Array.from(dom.apiRemoveButtons || []).forEach(button => {
        const slotIndex = Number(button.dataset.apiSlot);
        button.classList.toggle('hidden', slotIndex !== visibleCount - 1);
    });
}

function addApiSlot() {
    const count = visibleApiSlotCount();
    if (count >= 3) return;
    refreshApiSlotVisibility(count + 1);
    apiKeySlotInputs()[count].focus();
}

function removeApiSlot(slotIndex) {
    const visibleCount = visibleApiSlotCount();
    if (slotIndex !== visibleCount - 1 || slotIndex < 1) return;
    const input = apiKeySlotInputs()[slotIndex];
    if (input) input.value = '';
    refreshApiSlotVisibility(visibleCount - 1);
    updateAPIIndicator(false);
    validateApiKeySlots();
    checkNextStep1Availability();
    scheduleAutomaticApiVerification();
}

function prioritizeApiSlot(slotIndex) {
    const inputs = apiKeySlotInputs();
    if (!Number.isInteger(slotIndex) || slotIndex <= 0 || slotIndex >= inputs.length) return;
    const values = inputs.map(input => input ? input.value.trim() : '');
    const selected = values[slotIndex];
    if (!selected) return;

    // Move the chosen key to slot 1 and shift the earlier keys down. This is
    // easier to understand than swapping slot 1 with slot 3 (1,2,3 -> 3,2,1).
    const reordered = [selected, ...values.filter((_, index) => index !== slotIndex)];
    inputs.forEach((input, index) => {
        if (input) input.value = reordered[index] || '';
    });

    const orderedKeys = reordered.slice(0, visibleApiSlotCount()).filter(Boolean);
    state.apiKeys = orderedKeys;
    state.apiKey = orderedKeys[0] || '';
    persistOrderedApiKeys(orderedKeys);
    repaintAllApiKeyStatuses();
    validateApiKeySlots();
    updateAPIIndicator(false, `Đã đưa API ${slotIndex + 1} lên dòng đầu; đang xác thực lại...`);
    showToast(`Đã đưa API ${slotIndex + 1} lên dòng đầu.`, 'success');
    checkNextStep1Availability();
    scheduleAutomaticApiVerification();
}

async function loadInitialData() {
    if (dom.modelSelect && dom.recapAiSettingsSlot) {
        const modelGroup = dom.modelSelect.closest('.form-group');
        if (modelGroup && modelGroup.parentElement !== dom.recapAiSettingsSlot) {
            dom.recapAiSettingsSlot.insertBefore(modelGroup, dom.recapAiSettingsSlot.firstChild);
        }
    }
    // Load API key tu localStorage neu co
    let savedKeys = [];
    try {
        const parsed = JSON.parse(localStorage.getItem('gemini_api_keys') || '[]');
        if (Array.isArray(parsed)) savedKeys = parsed.slice(0, 3);
    } catch (_) {}
    if (!savedKeys.length) {
        const savedKey = localStorage.getItem('gemini_api_key');
        if (savedKey) savedKeys = [savedKey];
    }
    if (savedKeys.length) setApiKeySlotValues(savedKeys);
    else refreshApiSlotVisibility(1);
    state.apiKeys = collectApiKeys() || [];
    state.apiKey = state.apiKeys[0] || '';
    state.groqApiKey = localStorage.getItem('groq_api_key') || '';
    if (dom.groqApiKeyInput) dom.groqApiKeyInput.value = state.groqApiKey;
    state.thinkingEffort = localStorage.getItem('gemini_thinking_effort') || 'low';
    if (dom.thinkingEffortSelect) dom.thinkingEffortSelect.value = state.thinkingEffort;
    
    // Tai danh sach video va nhac nen co san
    await loadVideosList();
    await loadBgmsList();
    
    // Luôn kiểm tra khi tool vừa mở. Gemini và Groq được xác thực độc lập,
    // nên một bên bị thiếu/lỗi không ngăn bên còn lại được kiểm tra.
    scheduleAutomaticApiVerification(0);
}

async function readApiValidationResponse(response, providerName) {
    const rawBody = await response.text();
    try {
        return JSON.parse(rawBody);
    } catch (_) {
        const detail = rawBody.trim().slice(0, 120);
        throw new Error(
            `${providerName} trả HTTP ${response.status} nhưng dữ liệu phản hồi không hợp lệ` +
            (detail ? `: ${detail}` : '.')
        );
    }
}

function populateGeminiModelOptions(primaryData) {
    if (!primaryData || !primaryData.models || !primaryData.models.length || !dom.modelSelect) return;
    const modelSelect = dom.modelSelect;
    const currentVal = modelSelect.value;
    modelSelect.innerHTML = '';
    const modelRows = (primaryData.all_models || primaryData.models.map(item => (
        typeof item === 'string' ? { id: item, selectable: true } : item
    )))
        .filter(row => row && row.id);

    const preferred = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-3-flash-preview', 'gemini-3.5-flash-lite', 'gemini-3.1-flash-lite'];
    const sorted = modelRows.sort((a, b) => {
        const aIdx = preferred.indexOf(a.id);
        const bIdx = preferred.indexOf(b.id);
        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
        if (aIdx !== -1) return -1;
        if (bIdx !== -1) return 1;
        if (a.selectable !== b.selectable) return a.selectable ? -1 : 1;
        return a.id.localeCompare(b.id);
    });

    sorted.forEach((row, i) => {
        const m = row.id;
        const opt = document.createElement('option');
        opt.value = m;
        opt.disabled = !row.selectable;
        let label = m;
        if (m.includes('flash-lite')) label += ' (Nhẹ, nhanh)';
        else if (m.includes('flash')) label += ' (Nhanh)';
        else if (m.includes('pro')) label += ' (Thông minh)';
        if (i === 0) label += ' ⭐ Khuyên dùng';
        if (!row.selectable) label += ' — không dùng để viết recap';
        opt.textContent = label;
        if (m === currentVal) opt.selected = true;
        modelSelect.appendChild(opt);
    });

    if (!currentVal || !sorted.some(row => row.id === currentVal && row.selectable)) {
        const firstSelectable = Array.from(modelSelect.options).findIndex(opt => !opt.disabled);
        modelSelect.selectedIndex = Math.max(0, firstSelectable);
    }
    console.log(`[+] Loaded ${sorted.length} models from API`);
}

async function verifyApiKey(showSuccessToast = true, expectedRevision = null) {
    const revision = expectedRevision === null ? ++apiVerificationRevision : expectedRevision;
    if (revision !== apiVerificationRevision) return false;
    const apiKeys = collectApiKeys();
    if (apiKeys === null) {
        const message = dom.apiKeysInlineError.textContent || 'Danh sách API không hợp lệ.';
        updateAPIIndicator(false, message);
        if (showSuccessToast) showToast(message, 'error');
        return false;
    }
    const groqApiKey = dom.groqApiKeyInput ? dom.groqApiKeyInput.value.trim() : '';
    if (!apiKeys.length && !groqApiKey) {
        const message = 'Chưa nhập Gemini API và Groq API';
        updateAPIIndicator(false, message);
        if (showSuccessToast) showToast(message, 'error');
        return false;
    }
    if (apiVerificationController) apiVerificationController.abort();
    const controller = new AbortController();
    apiVerificationController = controller;
    dom.btnVerifyApi.disabled = true;
    dom.btnVerifyApi.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang xác thực...';
    
    try {
        let primaryData = null;
        const usableEntries = [];
        const geminiErrors = [];
        for (let index = 0; index < apiKeys.length; index++) {
            dom.btnVerifyApi.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Xác thực API ${index + 1}/${apiKeys.length}...`;
            const res = await fetch('/api/check-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKeys[index] }),
                signal: controller.signal
            });
            const data = await readApiValidationResponse(res, `Gemini API ${index + 1}`);
            if (revision !== apiVerificationRevision) return false;
            const status = data.status || (data.valid && data.usable !== false ? 'usable' : 'invalid');
            state.apiKeyStatuses[apiKeys[index]] = {
                status,
                message: data.message || ''
            };
            paintApiKeyStatus(apiKeys[index], status, data.message || '');
            if (data.valid && data.usable !== false) {
                usableEntries.push({ key: apiKeys[index], data });
                if (!primaryData) primaryData = data;
            } else {
                const message = data.message || `Xác thực thất bại (HTTP ${res.status}).`;
                geminiErrors.push(`API ${index + 1}: ${message}`);
                showToast(`API ${index + 1}: ${message}`, 'error');
            }
        }
        // Keep the user's fixed slot order. Verification may mark a key, but
        // it must never move API 2/3 above API 1 automatically.
        repaintAllApiKeyStatuses();
        state.apiKeys = usableEntries.map(entry => entry.key);
        state.apiKey = state.apiKeys[0] || '';
        if (primaryData) {
            persistOrderedApiKeys(apiKeys);
            populateGeminiModelOptions(primaryData);
        }

        let groqData = null;
        if (groqApiKey) {
            const groqRes = await fetch('/api/check-groq-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: groqApiKey }),
                signal: controller.signal
            });
            groqData = await readApiValidationResponse(groqRes, 'Groq API');
            if (revision !== apiVerificationRevision) return false;
            if (!groqData.valid) {
                state.groqApiKey = '';
                showToast(`Groq: ${groqData.message || `Xác thực thất bại (HTTP ${groqRes.status}).`}`, 'error');
            } else {
                state.groqApiKey = groqApiKey;
                localStorage.setItem('groq_api_key', groqApiKey);
            }
        } else {
            state.groqApiKey = '';
        }

        const geminiValid = Boolean(primaryData);
        const groqValid = Boolean(groqData && groqData.valid);
        if (geminiValid && groqValid) {
            updateAPIIndicator(true);
            if (showSuccessToast) {
                showToast(
                    `Đã xác thực ${usableEntries.length}/${apiKeys.length} Gemini API và Groq API.`,
                    'success'
                );
            }
            checkNextStep1Availability();
            return true;
        }

        let statusMessage = '';
        if (!apiKeys.length) statusMessage = 'Chưa nhập Gemini API';
        else if (!geminiValid) statusMessage = geminiErrors[0] || 'Không có Gemini API dùng được';
        else if (!groqApiKey) statusMessage = 'Gemini hợp lệ; chưa nhập Groq API';
        else statusMessage = `Groq: ${(groqData && groqData.message) || 'API không hợp lệ'}`;
        updateAPIIndicator(false, statusMessage);
        if (showSuccessToast && (!apiKeys.length || !groqApiKey)) showToast(statusMessage, 'error');
        checkNextStep1Availability();
        return false;
    } catch (err) {
        if (err && err.name === 'AbortError') return false;
        const message = (err && err.message) || 'Không thể kết nối đến máy chủ backend.';
        updateAPIIndicator(false, message);
        showToast(message, 'error');
        return false;
    } finally {
        if (apiVerificationController === controller) {
            apiVerificationController = null;
            dom.btnVerifyApi.disabled = false;
            dom.btnVerifyApi.innerHTML = '<i class="fa-solid fa-rotate"></i> Xác Thực Lại API';
        }
    }
}

async function loadVideosList() {
    try {
        const previousVideo = state.selectedVideo || dom.videoSelect.value;
        const res = await fetch('/api/list-videos');
        const list = await res.json();
        
        dom.videoSelect.innerHTML = '<option value="" disabled selected>-- Chọn video của bạn --</option>';
        list.forEach(vid => {
            const opt = document.createElement('option');
            opt.value = vid;
            opt.textContent = vid;
            dom.videoSelect.appendChild(opt);
        });
        
        const automaticVideo = list.includes(previousVideo)
            ? previousVideo
            : (list.length === 1 ? list[0] : (list.includes('video_goc.mp4') ? 'video_goc.mp4' : ''));
        if (automaticVideo) {
            dom.videoSelect.value = automaticVideo;
            if (automaticVideo !== state.selectedVideo) resetProjectData(automaticVideo);
            checkNextStep1Availability();
            await loadVideoDurationSuggestion(automaticVideo);
        }
    } catch (err) {
        showToast('Không thể tải danh sách video.', 'error');
    }
}

async function loadVideoDurationSuggestion(videoName) {
    if (!videoName || !dom.reviewMinutes) return;
    try {
        const res = await fetch(`/api/video-info?video_name=${encodeURIComponent(videoName)}`);
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.message || 'Không đọc được video');
        dom.reviewMinutes.value = Number(data.suggested_review_minutes).toFixed(1);
        dom.reviewMinutes.dataset.sourceMinutes = String(data.source_duration_minutes);
        updateReviewMinutesHint();
    } catch (err) {
        dom.reviewMinutes.value = '';
        if (dom.reviewMinutesHint) {
            dom.reviewMinutesHint.textContent = 'Không đọc được thời lượng; chọn lại video để thử lại.';
        }
    }
}

function updateReviewMinutesHint() {
    if (!dom.reviewMinutes || !dom.reviewMinutesHint) return;
    const sourceMinutes = Number(dom.reviewMinutes.dataset.sourceMinutes);
    const selectedMinutes = Number(dom.reviewMinutes.value);
    if (!Number.isFinite(sourceMinutes) || !Number.isFinite(selectedMinutes) || selectedMinutes <= 0) return;
    const baseline = sourceMinutes * 0.2;
    const ratio = selectedMinutes / Math.max(0.1, baseline);
    const mode = ratio < 0.8 ? 'cô đọng' : (ratio <= 1.2 ? 'cân bằng' : 'mở rộng');
    const rate = Number(dom.voiceRateSlider?.value ?? 20);
    const sentenceCount = Math.max(3, Math.round(Math.min(selectedMinutes, sourceMinutes) * narrationSentencesPerMinute()));
    dom.reviewMinutesHint.textContent = `Video gốc ${sourceMinutes} phút; chế độ ${mode}. Khóa ${sentenceCount} câu ở ${rate >= 0 ? '+' : ''}${rate}%. Số phút là ước tính, đối chiếu sau khi tạo giọng.`;
}

async function loadBgmsList() {
    try {
        const res = await fetch('/api/list-bgm');
        const list = await res.json();
        
        dom.bgmSelect.innerHTML = '<option value="">Không dùng nhạc nền (Chỉ giữ giọng thuyết minh)</option>';
        list.forEach(bgm => {
            const opt = document.createElement('option');
            opt.value = bgm;
            opt.textContent = bgm;
            dom.bgmSelect.appendChild(opt);
        });
    } catch (err) {
        showToast('Không thể tải danh sách nhạc nền BGM.', 'error');
    }
}

async function loadSfxList() {
    try {
        const res = await fetch('/api/list-sfx');
        const list = await res.json();
        const container = document.querySelector('.prompt-templates:has(.sfx-chip)');
        if (container) {
            container.innerHTML = '';
            const maxVisible = 6;
            list.forEach((sfx, index) => {
                const btn = document.createElement('button');
                btn.className = 'prompt-chip sfx-chip';
                btn.setAttribute('data-prompt', `[SFX=${sfx}]`);
                // Format ten dep hon
                let displayName = sfx.replace(/\.(mp3|wav|m4a|aac|ogg)$/i, '');
                // Gioi han 15 ky tu cho dep
                if (displayName.length > 20) displayName = displayName.substring(0, 18) + '...';
                
                btn.innerHTML = `<i class="fa-solid fa-music"></i> ${displayName}`;
                
                if (index >= maxVisible) {
                    btn.style.display = 'none';
                    btn.classList.add('sfx-hidden');
                }
                
                btn.addEventListener('click', () => {
                    if (dom.scriptEditor) {
                        const promptText = `[SFX=${sfx}]`;
                        const startPos = dom.scriptEditor.selectionStart;
                        const endPos = dom.scriptEditor.selectionEnd;
                        const val = dom.scriptEditor.value;
                        
                        if (startPos !== undefined && startPos !== null) {
                            dom.scriptEditor.value = val.substring(0, startPos) + " " + promptText + " " + val.substring(endPos);
                            dom.scriptEditor.focus();
                            const newPos = startPos + promptText.length + 2;
                            dom.scriptEditor.selectionStart = newPos;
                            dom.scriptEditor.selectionEnd = newPos;
                        } else {
                            dom.scriptEditor.value += " " + promptText;
                        }
                        updateCharCount();
                        
                        btn.style.borderColor = 'var(--accent-purple)';
                        btn.style.color = 'var(--accent-purple)';
                        setTimeout(() => {
                            btn.style.borderColor = '';
                            btn.style.color = '';
                        }, 800);
                        showToast('Đã chèn SFX vào kịch bản!', 'success');
                    }
                });
                container.appendChild(btn);
            });
            
            if (list.length > maxVisible) {
                const toggleBtn = document.createElement('button');
                toggleBtn.className = 'prompt-chip sfx-toggle-btn';
                toggleBtn.style.background = 'rgba(255,255,255,0.05)';
                toggleBtn.style.borderStyle = 'dashed';
                toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-down"></i> Xem thêm (${list.length - maxVisible})`;
                
                toggleBtn.addEventListener('click', () => {
                    const hiddenBtns = container.querySelectorAll('.sfx-hidden');
                    const isExpanded = toggleBtn.classList.contains('expanded');
                    
                    if (isExpanded) {
                        hiddenBtns.forEach(b => b.style.display = 'none');
                        toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-down"></i> Xem thêm (${list.length - maxVisible})`;
                        toggleBtn.classList.remove('expanded');
                    } else {
                        hiddenBtns.forEach(b => b.style.display = 'inline-flex');
                        toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-up"></i> Thu gọn`;
                        toggleBtn.classList.add('expanded');
                    }
                });
                container.appendChild(toggleBtn);
            }
        }
    } catch (err) {
        console.error("Loi load SFX: ", err);
    }
}

function checkNextStep1Availability() {
    const hasUsableApi = Array.isArray(state.apiKeys) && state.apiKeys.length > 0;
    const isReady = state.apiVerified && hasUsableApi && Boolean(dom.videoSelect.value);
    dom.btnToStep2.disabled = !isReady;
    dom.btnToStep2.setAttribute('aria-disabled', isReady ? 'false' : 'true');
    dom.btnToStep2.classList.toggle('is-ready', isReady);
    if (isReady) {
        enableStep(2);
    }
}

async function continueToScriptStep() {
    const selectedVideo = dom.videoSelect.value;
    if (!selectedVideo) {
        showToast('Hãy chọn video trong danh sách hoặc tải video mới lên trước.', 'error');
        dom.videoSelect.focus();
        return;
    }
    if (!dom.apiKeyInput.value.trim()) {
        showToast('Hãy nhập API Key Gemini trước.', 'error');
        dom.apiKeyInput.focus();
        return;
    }
    if (!state.apiVerified) {
        const verified = await verifyApiKey(false);
        if (!verified) return;
    }
    if (selectedVideo !== state.selectedVideo) resetProjectData(selectedVideo);
    state.selectedVideo = selectedVideo;
    enableStep(2);
    switchPanel(2);
}

// =====================================================================
// FILE UPLOAD HANDLING (DRAG & DROP)
// =====================================================================
function setupDragAndDrop(dropZone, fileInput, progressBox, type, callback) {
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0], progressBox, type, callback);
        }
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0], progressBox, type, callback);
        }
    });
}

function handleFileUpload(file, progressBox, type, callback) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);
    
    progressBox.classList.remove('hidden');
    const fill = progressBox.querySelector('.progress-fill');
    const percent = progressBox.querySelector('.progress-percent');
    
    fill.style.width = '0%';
    percent.textContent = '0%';
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload-file', true);
    
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            fill.style.width = `${pct}%`;
            percent.textContent = `${pct}%`;
        }
    };
    
    xhr.onload = () => {
        progressBox.classList.add('hidden');
        if (xhr.status === 200) {
            const res = JSON.parse(xhr.responseText);
            if (res.success) {
                const renameNote = res.renamed ? ` (dã dổi tên: ${res.filename})` : '';
                showToast(`Tải lên tệp '${file.name}' thành công!${renameNote}`, 'success');
                callback(res.filename);
            } else {
                showToast(res.message, 'error');
            }
        } else {
            showToast('Lỗi khi tải file lên máy chủ.', 'error');
        }
    };
    
    xhr.onerror = () => {
        progressBox.classList.add('hidden');
        showToast('Kết nối tải tệp bị gián đoạn.', 'error');
    };
    
    xhr.send(formData);
}

// =====================================================================
// STEP 2: AI SCRIPT GENERATOR
// =====================================================================
let analysisInProgress = false;

function setAnalysisButtonsBusy(busy) {
    analysisInProgress = busy;
    if (dom.btnGenerateScript) dom.btnGenerateScript.disabled = busy;
    if (dom.btnQuickVideo) dom.btnQuickVideo.disabled = busy;
}

async function generateScript(options = {}) {
    if (analysisInProgress) {
        showToast('Gemini đang tạo kịch bản, vui lòng chờ hoàn tất.', 'error');
        return false;
    }
    const onSuccess = typeof options.onSuccess === 'function' ? options.onSuccess : null;
    const onError = typeof options.onError === 'function' ? options.onError : null;
    const allApiKeys = collectApiKeys();
    if (allApiKeys === null) {
        showToast(dom.apiKeysInlineError.textContent || 'Danh sách API không hợp lệ.', 'error');
        if (onError) onError();
        return false;
    }
    const apiKeys = usableApiKeys() || [];
    if (!state.apiVerified || !apiKeys.length) {
        showToast('Không có Gemini API đã xác thực và còn quota.', 'error');
        if (onError) onError();
        return false;
    }
    state.apiKeys = apiKeys;
    state.apiKey = state.apiKeys[0] || '';
    state.analysisApiKeys = [...state.apiKeys];
    const requestedVideo = dom.videoSelect.value;
    // Every Gemini generation is a new isolated project, even for the same file.
    resetProjectData(requestedVideo);
    state.selectedVideo = requestedVideo;
    state.modelName = dom.modelSelect.value;
    state.groqApiKey = dom.groqApiKeyInput ? dom.groqApiKeyInput.value.trim() : '';
    state.thinkingEffort = dom.thinkingEffortSelect ? dom.thinkingEffortSelect.value : 'low';
    localStorage.setItem('gemini_thinking_effort', state.thinkingEffort);
    const requestedProjectId = state.projectId;
    
    const style = dom.styleSelect.value;
    const customPromptText = dom.customPrompt.value;
    let subtitleText = '';
    const subtitleFile = dom.subtitleFileInput && dom.subtitleFileInput.files
        ? dom.subtitleFileInput.files[0]
        : null;
    if (subtitleFile) {
        if (subtitleFile.size > 5 * 1024 * 1024) {
            showToast('File phụ đề vượt quá 5 MB.', 'error');
            if (onError) onError();
            return false;
        }
        try {
            subtitleText = await subtitleFile.text();
        } catch (error) {
            showToast('Không đọc được file phụ đề.', 'error');
            if (onError) onError(error);
            return false;
        }
    }
    const reviewMinutesValue = dom.reviewMinutes ? Number(dom.reviewMinutes.value) : NaN;
    
    if (!state.selectedVideo) {
        showToast('Vui lòng chọn video trước!', 'error');
        if (onError) onError();
        return false;
    }
    if (dom.reviewMinutes && (!Number.isFinite(reviewMinutesValue) || reviewMinutesValue <= 0)) {
        showToast('Thời lượng recap phải là số lớn hơn 0.', 'error');
        dom.reviewMinutes.focus();
        if (onError) onError();
        return false;
    }

    setAnalysisButtonsBusy(true);
    dom.scriptGenLoading.classList.remove('hidden');
    // Invalidate every derived value from the previous Gemini result now.
    state.script = '';
    state.sentences = [];
    state.scriptProjectId = '';
    state.scriptVideo = '';
    state.splitScriptSnapshot = '';
    dom.scriptEditor.value = '';
    dom.sentencesList.innerHTML = '';
    updateCharCount();
    updateScriptLoading(0, 'Đang gửi yêu cầu khởi tạo đến máy chủ...', 'Kết nối Gemini...');
    
    try {
        const res = await fetch('/api/analyze-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_key: state.apiKey,
                api_keys: state.apiKeys,
                model_name: state.modelName,
                groq_api_key: state.groqApiKey,
                thinking_effort: state.thinkingEffort,
                video_name: state.selectedVideo,
                project_id: requestedProjectId,
                style: style,
                custom_prompt: customPromptText,
                review_minutes: reviewMinutesValue,
                voice_rate: `${Number(dom.voiceRateSlider?.value ?? 20) >= 0 ? '+' : ''}${dom.voiceRateSlider?.value ?? 20}%`,
                subtitle_text: subtitleText,
                enable_multi_voice: document.getElementById('enable-multi-voice') ? document.getElementById('enable-multi-voice').checked : false
            })
        });
        const data = await res.json();
        
        if (data.success) {
            // Bắt đầu Polling tiến trình AI
            if (data.resumed) {
                showToast('Đang tiếp tục từ checkpoint; giữ lại các nhóm câu đã hoàn thành.', 'success');
            }
            pollTaskStatus(data.task_id, (statusData) => {
                if (state.projectId !== requestedProjectId || state.selectedVideo !== requestedVideo) return;
                updateScriptLoading(statusData.progress, statusData.message, 'AI Phân Tích Khung Hình');
            }, async (scriptResult, taskData) => {
                if (state.projectId !== requestedProjectId || state.selectedVideo !== requestedVideo) {
                    setAnalysisButtonsBusy(false);
                    if (onError) onError('Dự án đã thay đổi trong khi Gemini đang xử lý.');
                    return;
                }
                dom.scriptGenLoading.classList.add('hidden');
                dom.scriptEditor.value = scriptResult;
                state.script = scriptResult;
                state.sentences = [];
                state.splitScriptSnapshot = '';
                state.scriptProjectId = taskData.project_id || requestedProjectId;
                localStorage.setItem(`analysis_for_render_${state.scriptProjectId}`, data.task_id);
                state.scriptVideo = taskData.video_name || requestedVideo;
                state.narrationBudget = taskData.narration_budget || null;
                updateCharCount();
                if (taskData.sentence_tolerance_met === false) {
                    showToast(
                        `Đã viết ${taskData.actual_scenes} câu từ cảnh thật; ngắn hơn mục tiêu ${taskData.requested_target_scenes} câu vì thiếu sự kiện hình ảnh.`,
                        'warning'
                    );
                } else {
                    const sentenceCount = taskData.actual_sentences;
                    const sentenceTarget = taskData.target_sentence_count;
                    showToast(sentenceCount != null && sentenceTarget != null
                        ? `Hoàn tất: ${sentenceCount}/${sentenceTarget} câu, ${taskData.actual_words} từ.`
                        : 'AI viết kịch bản hoàn tất!', 'success');
                }
                enableStep(3);
                setAnalysisButtonsBusy(false);
                if (onSuccess) await onSuccess(scriptResult, taskData);
            }, (errMessage, taskData) => {
                if (state.projectId !== requestedProjectId || state.selectedVideo !== requestedVideo) {
                    setAnalysisButtonsBusy(false);
                    if (onError) onError(errMessage);
                    return;
                }
                dom.scriptGenLoading.classList.add('hidden');
                setAnalysisButtonsBusy(false);
                if (taskData && typeof taskData.result === 'string' && taskData.result.trim()) {
                    dom.scriptEditor.value = taskData.result;
                    state.script = taskData.result;
                    state.sentences = [];
                    state.splitScriptSnapshot = '';
                    updateCharCount();
                }
                showToast(errMessage, 'error');
                if (onError) onError(errMessage);
            });
        } else {
            dom.scriptGenLoading.classList.add('hidden');
            setAnalysisButtonsBusy(false);
            showToast(data.message, 'error');
            if (onError) onError(data.message);
            return false;
        }
    } catch (err) {
        dom.scriptGenLoading.classList.add('hidden');
        setAnalysisButtonsBusy(false);
        showToast('Lỗi gửi yêu cầu viết kịch bản.', 'error');
        if (onError) onError(err);
        return false;
    }
    return true;
}

function updateScriptLoading(progress, message, title) {
    dom.scriptLoadingTitle.textContent = title;
    dom.scriptProgressFill.style.width = `${progress}%`;
    dom.scriptLoadingMsg.textContent = message;
}

function updateCharCount() {
    const text = dom.scriptEditor.value.trim();
    const count = text ? text.split(/\s+/).length : 0;
    const minutes = (count / (20 * narrationSentencesPerMinute())).toFixed(1);
    dom.scriptCharCount.textContent = `${count} từ (~${minutes} phút, ước tính)`;
    state.script = text;
}

// =====================================================================
// STEP 3: SENTENCES SPLITTING & TTS PREVIEW
// =====================================================================
async function autoSplitScript() {
    state.script = dom.scriptEditor.value.trim();
    if (!state.script) return false;
    if (state.scriptProjectId !== state.projectId || state.scriptVideo !== state.selectedVideo) {
        showToast('Kịch bản hiện tại không thuộc video đang chọn. Hãy tạo lại bằng Gemini.', 'error');
        return false;
    }
    const scriptToSplit = state.script;
    
    dom.sentencesList.innerHTML = '<div class="text-muted p-3"><i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích cấu trúc kịch bản...</div>';
    
    try {
        const res = await fetch('/api/split-sentences', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: scriptToSplit })
        });
        const payload = await res.json();
        if (!res.ok) {
            throw new Error(payload.message || `HTTP ${res.status}`);
        }
        if (!Array.isArray(payload)) {
            throw new Error('Máy chủ trả dữ liệu phân đoạn không đúng định dạng.');
        }
        const sentences = payload.filter(item => typeof item === 'string' && item.trim());
        if (sentences.length === 0) {
            throw new Error('Kịch bản không tạo được phân đoạn hợp lệ.');
        }
        if (state.projectId !== state.scriptProjectId || dom.scriptEditor.value.trim() !== scriptToSplit) return false;
        state.sentences = sentences;
        state.splitScriptSnapshot = scriptToSplit;
        renderSentences();
        updateSentenceStats();
        enableStep(4);
        return true;
    } catch (err) {
        const message = err && err.message ? err.message : 'Lỗi không xác định';
        dom.sentencesList.innerHTML = '';
        const errorBox = document.createElement('div');
        errorBox.className = 'text-red p-3';
        errorBox.textContent = `Lỗi khi phân đoạn câu: ${message}`;
        dom.sentencesList.appendChild(errorBox);
        return false;
    }
}

async function createQuickVideo() {
    if (analysisInProgress) {
        showToast('Gemini đang tạo kịch bản, vui lòng chờ hoàn tất.', 'error');
        return;
    }
    if (dom.btnQuickVideo) {
        dom.btnQuickVideo.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ĐANG TẠO KỊCH BẢN...';
    }

    const restoreQuickButton = () => {
        if (!dom.btnQuickVideo) return;
        dom.btnQuickVideo.disabled = false;
        dom.btnQuickVideo.innerHTML = '<i class="fa-solid fa-bolt"></i> TẠO VIDEO NHANH';
    };

    const started = await generateScript({
        onSuccess: async () => {
            if (dom.btnQuickVideo) {
                dom.btnQuickVideo.disabled = true;
                dom.btnQuickVideo.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ĐANG PHÂN CÂU...';
            }
            const splitOk = await autoSplitScript();
            if (!splitOk) {
                restoreQuickButton();
                showToast('Tạo nhanh đã dừng vì không phân câu được.', 'error');
                return;
            }
            enableStep(5);
            switchPanel(5);
            showToast('Đã tạo kịch bản, đang tự động render video.', 'success');
            await startRenderVideo();
            restoreQuickButton();
        },
        onError: restoreQuickButton,
    });
    if (!started) restoreQuickButton();
}

function spokenScriptStats(rows) {
    const text = rows.join('\n').replace(/\[(?:TIME|BEAT|SHOT|VOICE|SFX)=[^\]]*\]|<(?:VOICE|SFX)=[^>]*>/gi, ' ');
    const count = text.split(/(?<!\d)[.!?]+(?:["”’)\]]*)\s*|(?<=\d)[!?]+\s*|(?<=\d)\.(?!\d)\s*/u)
        .filter(part => /[\p{L}\p{N}_]/u.test(part)).length;
    const wordCount = (text.match(/[\p{L}\p{N}_]+/gu) || []).length;
    return { count, wordCount };
}

function updateSentenceStats() {
    const { count, wordCount } = spokenScriptStats(state.sentences);
    const estMinutes = (count / narrationSentencesPerMinute()).toFixed(1);
    
    const elCount = document.getElementById('stat-sentence-count');
    const elWords = document.getElementById('stat-word-count');
    const elDuration = document.getElementById('stat-est-duration');
    if (elCount) elCount.textContent = state.narrationBudget?.target_sentence_count
        ? `${count}/${state.narrationBudget.target_sentence_count}` : count;
    if (elWords) elWords.textContent = wordCount;
    if (elDuration) elDuration.textContent = `~${estMinutes}p`;
}

function renderSentences() {
    dom.sentencesList.innerHTML = '';
    
    if (state.sentences.length === 0) {
        dom.sentencesList.innerHTML = '<div class="text-muted p-3">Kịch bản trống hoặc chưa được tách câu.</div>';
        return;
    }
    
    state.sentences.forEach((sentence, idx) => {
        const item = document.createElement('div');
        item.className = 'sentence-item';
        item.innerHTML = `
            <div class="sentence-index">${idx + 1}</div>
            <div class="sentence-text-wrapper">
                <textarea class="sentence-textarea" rows="2">${sentence}</textarea>
            </div>
            <div class="sentence-actions">
                <button type="button" class="btn-preview-item" title="Nghe thử giọng đọc">
                    <i class="fa-solid fa-play"></i>
                </button>
                <button type="button" class="btn-delete-item" title="Xóa câu này">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `;
        
        // Sự kiện nghe thử TTS từng câu
        const btnPlay = item.querySelector('.btn-preview-item');
        const textarea = item.querySelector('.sentence-textarea');
        
        // Dong bo nguoc lai state khi sua cau
        textarea.addEventListener('input', () => {
            state.sentences[idx] = textarea.value.trim();
            updateSentenceStats();
        });
        
        // Nut xoa cau
        const btnDelete = item.querySelector('.btn-delete-item');
        btnDelete.addEventListener('click', () => {
            state.sentences.splice(idx, 1);
            renderSentences();
            updateSentenceStats();
        });
        
        let audioPlayer = null;
        btnPlay.addEventListener('click', async () => {
            if (btnPlay.classList.contains('playing')) {
                // Đang chơi nhạc thì dừng lại
                if (audioPlayer) {
                    audioPlayer.pause();
                }
                return;
            }
            
            const txt = textarea.value.trim();
            if (!txt) return;
            
            // Đặt trạng thái loading
            btnPlay.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
            btnPlay.disabled = true;
            
            try {
                const previewRes = await fetch('/api/preview-tts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: txt,
                        voice: dom.voiceSelect.value,
                        rate: `${parseInt(dom.voiceRateSlider.value) >= 0 ? '+' : ''}${dom.voiceRateSlider.value}%`
                    })
                });
                const previewData = await previewRes.json();
                
                if (previewData.success) {
                    btnPlay.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
                    btnPlay.classList.add('playing');
                    btnPlay.disabled = false;
                    
                    audioPlayer = new Audio(previewData.url);
                    audioPlayer.play();
                    
                    audioPlayer.onended = () => {
                        btnPlay.classList.remove('playing');
                        btnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
                        audioPlayer = null;
                    };
                    
                    audioPlayer.onerror = () => {
                        btnPlay.classList.remove('playing');
                        btnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
                        showToast('Lỗi phát âm thanh.', 'error');
                        audioPlayer = null;
                    };
                } else {
                    btnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
                    btnPlay.disabled = false;
                    showToast(previewData.message, 'error');
                }
            } catch (err) {
                btnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
                btnPlay.disabled = false;
                showToast('Lỗi gọi API tổng hợp giọng nói.', 'error');
            }
        });
        
        dom.sentencesList.appendChild(item);
    });
}

// =====================================================================
// STEP 5: RENDER SUMMARY & VIDEO GENERATION
// =====================================================================
function updateRenderSummary() {
    dom.summaryVideo.textContent = state.selectedVideo;
    const spokenCount = spokenScriptStats(state.sentences).count;
    const sentenceTarget = state.narrationBudget?.target_sentence_count;
    dom.summarySentences.textContent = `${spokenCount}${sentenceTarget ? '/' + sentenceTarget : ''} câu (${state.sentences.length} đoạn)`;
    
    const voiceText = dom.voiceSelect.options[dom.voiceSelect.selectedIndex].text;
    dom.summaryVoice.textContent = `${voiceText} (${dom.voiceRateSlider.value >= 0 ? '+' : ''}${dom.voiceRateSlider.value}%)`;
    
    dom.summaryBgm.textContent = dom.bgmSelect.value ? `${dom.bgmSelect.value} (Âm lượng: ${dom.bgmVolumeSlider.value}%)` : 'Không dùng';
    
    // Thu thap bo loc lach ban quyen
    const filters = [];
    if (dom.filterMirror.checked) filters.push('Lật ngang');
    if (dom.filterZoom.checked) filters.push('Zoom-in 8%');
    if (dom.filterContrast.value !== '1.0') filters.push(`Tương phản ${parseInt(parseFloat(dom.filterContrast.value)*100)}%`);
    if (dom.filterSpeed.value !== '1.0') filters.push(`Tốc độ chạy hình ${dom.filterSpeed.value}x`);
    
    dom.summaryFilters.textContent = filters.length > 0 ? filters.join(', ') : 'Không sử dụng bộ lọc';
    
    // Subtitle info
    const subSummary = document.getElementById('summary-subtitle');
    if (subSummary) {
        subSummary.textContent = `${dom.subSize.value}px, ${dom.subColor.value}, Nền mờ: ${dom.subOpacity.value}`;
    }
}

// Render timer
let renderTimerInterval = null;
let renderStartTime = null;

function startRenderTimer() {
    renderStartTime = Date.now();
    const timerEl = document.getElementById('render-timer');
    renderTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - renderStartTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        if (timerEl) timerEl.innerHTML = `<i class="fa-regular fa-clock"></i> ${mins}:${secs}`;
    }, 1000);
}

function stopRenderTimer() {
    if (renderTimerInterval) {
        clearInterval(renderTimerInterval);
        renderTimerInterval = null;
    }
}

async function startRenderVideo() {
    // Claim synchronously, before awaiting the directory dialog/network.
    if (startRenderVideo.submitting || dom.btnStartRender.disabled) return;
    startRenderVideo.submitting = true;
    try {
        return await submitRenderVideo();
    } finally {
        startRenderVideo.submitting = false;
    }
}

async function submitRenderVideo() {
    let renderedBackendLogCount = 0;
    // Không bắt đầu tác vụ tốn thời gian khi chưa có nơi lưu chắc chắn.
    // Luồng "Tạo video nhanh" cũng đi qua đây nên không thể bỏ sót bước chọn.
    if (!await ensureExportDirectory()) return;
    if (!state.projectId || state.scriptProjectId !== state.projectId || state.scriptVideo !== state.selectedVideo) {
        showToast('Dữ liệu kịch bản không thuộc video đang chọn. Hãy tạo lại kịch bản Gemini.', 'error');
        return;
    }
    if (state.splitScriptSnapshot !== dom.scriptEditor.value.trim()) {
        showToast('Kịch bản đã thay đổi nhưng chưa tách lại câu. Hãy quay lại bước 3 và bấm Tách lại.', 'error');
        return;
    }
    const renderingProjectId = state.projectId;
    const renderingVideo = state.selectedVideo;
    // Thu thap cac cau da sua doi tu giao dien
    const textareas = dom.sentencesList.querySelectorAll('.sentence-textarea');
    const finalSentences = [];
    textareas.forEach(ta => {
        if (ta.value.trim()) finalSentences.push(ta.value.trim());
    });
    
    if (finalSentences.length === 0) {
        showToast('Kịch bản thuyết minh trống!', 'error');
        return;
    }
    
    state.sentences = finalSentences;
    
    dom.btnStartRender.disabled = true;
    dom.btnStartRender.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ĐANG XỬ LÝ...';
    dom.btnRenderBack.disabled = true;
    dom.renderProgressBox.classList.remove('hidden');
    dom.renderLogTerminal.innerHTML = '<div class="log-line text-cyan">[*] Bắt đầu gửi lệnh xuất video sang GPU NVENC render...</div>';
    
    // Bat dau dem thoi gian
    startRenderTimer();
    
    const payload = {
        video_name: renderingVideo,
        project_id: renderingProjectId,
        render_engine: document.getElementById('render-engine')?.value || 'legacy_v1',
        analysis_task_id: localStorage.getItem(`analysis_for_render_${renderingProjectId}`) || '',
        bgm_name: dom.bgmSelect.value,
        bgm_volume: parseInt(dom.bgmVolumeSlider.value),
        voice_name: dom.voiceSelect.value,
        voice_rate: `${dom.voiceRateSlider.value >= 0 ? '+' : ''}${dom.voiceRateSlider.value}%`,
        sentences: state.sentences,
        mirror: dom.filterMirror.checked,
        zoom_factor: dom.filterZoom.checked ? 1.08 : 1.0,
        enable_zoom_crop: dom.filterZoom.checked,
        enable_ken_burns: dom.filterKenBurns ? dom.filterKenBurns.checked : false,
        speed_factor: parseFloat(dom.filterSpeed.value),
        contrast: parseFloat(dom.filterContrast.value),
        brightness: 1.0,
        aspect_ratio: document.querySelector('input[name="aspect_ratio"]:checked')?.value || "9:16",
        enable_subtitles: document.getElementById('enable-subtitles') ? document.getElementById('enable-subtitles').checked : true,
        enable_karaoke_sub: document.getElementById('enable-karaoke-sub') ? document.getElementById('enable-karaoke-sub').checked : true,
        font_size: parseInt(dom.subSize.value),
        font_color: dom.subColor.value,
        bg_opacity: parseInt(dom.subOpacity.value),
        export_mode: document.getElementById('export-mode').value,
        review_minutes: state.narrationBudget?.target_minutes ?? dom.reviewMinutes?.value ?? 'auto'
    };
    
    try {
        const res = await fetch('/api/render-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (state.projectId !== renderingProjectId || state.selectedVideo !== renderingVideo) return;
        
        if (data.success) {
            localStorage.setItem('active_render_task_id', data.task_id);
            pollTaskStatus(data.task_id, (statusData) => {
                if (state.projectId !== renderingProjectId || state.selectedVideo !== renderingVideo) return;
                // Cap nhat tien do render
                dom.renderStatusPercent.textContent = `${statusData.progress}%`;
                dom.renderProgressFill.style.width = `${statusData.progress}%`;
                dom.renderStatusLbl.textContent = statusData.message;
                
                // Hien thi nhat ky that tu backend, khong lap lai status moi 2 giay.
                const backendLogs = Array.isArray(statusData.logs) ? statusData.logs : [];
                backendLogs.slice(renderedBackendLogCount).forEach(entry => {
                    const logLine = document.createElement('div');
                    const level = entry.level || 'INFO';
                    logLine.className = `log-line ${level === 'ERROR' ? 'text-red' : (level === 'SUCCESS' ? 'text-green' : '')}`;
                    logLine.textContent = `[${entry.time}] [${level}] ${entry.message}`;
                    dom.renderLogTerminal.appendChild(logLine);
                    if (entry.details) {
                        const detailLine = document.createElement('pre');
                        detailLine.className = 'log-line text-red';
                        detailLine.textContent = entry.details;
                        dom.renderLogTerminal.appendChild(detailLine);
                    }
                });
                renderedBackendLogCount = backendLogs.length;
                dom.renderLogTerminal.scrollTop = dom.renderLogTerminal.scrollHeight;
            }, (videoUrl, taskData) => {
                displayActualNarrationTiming(taskData);
                localStorage.removeItem('active_render_task_id');
                if (state.projectId !== renderingProjectId || state.selectedVideo !== renderingVideo) return;
                // Render thanh cong!
                stopRenderTimer();
                dom.btnStartRender.disabled = false;
                dom.btnStartRender.innerHTML = '<i class="fa-solid fa-circle-play"></i> BẮT ĐẦU XUẤT VIDEO REVIEW';
                dom.btnRenderBack.disabled = false;
                
                dom.videoPreviewPlaceholder.classList.add('hidden');
                dom.videoPreviewWrapper.classList.remove('hidden');
                
                if (videoUrl.endsWith('.zip')) {
                    // Che do xuat CapCut (ZIP)
                    dom.finalVideoPlayer.style.display = 'none';
                    dom.btnDownloadVideo.href = videoUrl;
                    dom.btnDownloadVideo.innerHTML = '<i class="fa-solid fa-file-zipper"></i> TẢI GÓI NGUYÊN LIỆU CAPCUT (ZIP)';
                    showToast('Đã đóng gói nguyên liệu CapCut thành công!', 'success');
                } else {
                    // Che do xuat Full Video (MP4)
                    dom.finalVideoPlayer.style.display = 'block';
                    dom.finalVideoPlayer.src = videoUrl;
                    dom.finalVideoPlayer.load();
                    dom.btnDownloadVideo.href = videoUrl;
                    dom.btnDownloadVideo.innerHTML = '<i class="fa-solid fa-circle-down"></i> TẢI VIDEO REVIEW NGAY (MP4)';
                    showToast('Render và xuất video MP4 hoàn thành!', 'success');
                }
                if (taskData.export_path) {
                    showToast('Đã lưu thêm vào: ' + taskData.export_path, 'success');
                } else if (taskData.export_error) {
                    showToast('Video đã render nhưng chưa chép được: ' + taskData.export_error, 'error');
                }
            }, (errMessage) => {
                localStorage.removeItem('active_render_task_id');
                if (state.projectId !== renderingProjectId || state.selectedVideo !== renderingVideo) return;
                stopRenderTimer();
                dom.btnStartRender.disabled = false;
                dom.btnStartRender.innerHTML = '<i class="fa-solid fa-rotate-right"></i> THỬ LẠI / TIẾP TỤC XUẤT VIDEO';
                dom.btnRenderBack.disabled = false;
                showToast(errMessage, 'error');
            });
        } else {
            stopRenderTimer();
            dom.btnStartRender.disabled = false;
            dom.btnStartRender.innerHTML = '<i class="fa-solid fa-circle-play"></i> BẮT ĐẦU XUẤT VIDEO REVIEW';
            dom.btnRenderBack.disabled = false;
            showToast(data.message, 'error');
        }
    } catch (err) {
        if (state.projectId !== renderingProjectId || state.selectedVideo !== renderingVideo) return;
        stopRenderTimer();
        dom.btnStartRender.disabled = false;
        dom.btnStartRender.innerHTML = '<i class="fa-solid fa-circle-play"></i> BẮT ĐẦU XUẤT VIDEO REVIEW';
        dom.btnRenderBack.disabled = false;
        showToast('Lỗi hệ thống khi gửi tác vụ render.', 'error');
    }
}

// =====================================================================
// POLLING GENERAL SYSTEM TASK
// =====================================================================
async function resumePersistedRender() {
    const taskId = localStorage.getItem('active_render_task_id');
    if (!taskId) return;

    try {
        const response = await fetch(`/api/task-status/${taskId}`);
        const task = await response.json();
        if (task.status === 'not_found') {
            localStorage.removeItem('active_render_task_id');
            return;
        }

        state.projectId = task.project_id || state.projectId || createProjectId();
        state.selectedVideo = task.video_name || state.selectedVideo;
        state.highestStepReached = 5;
        switchPanel(5);
        dom.renderProgressBox.classList.remove('hidden');
        dom.btnStartRender.disabled = true;
        dom.btnRenderBack.disabled = true;
        dom.btnStartRender.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ĐANG TIẾP TỤC XỬ LÝ...';
        dom.renderLogTerminal.innerHTML = '';
        startRenderTimer();
        let shownLogCount = 0;

        const showProgress = (statusData) => {
            dom.renderStatusPercent.textContent = `${statusData.progress || 0}%`;
            dom.renderProgressFill.style.width = `${statusData.progress || 0}%`;
            dom.renderStatusLbl.textContent = statusData.message || 'Worker đang xử lý...';
            const logs = Array.isArray(statusData.logs) ? statusData.logs : [];
            logs.slice(shownLogCount).forEach(entry => {
                const line = document.createElement('div');
                line.className = `log-line ${entry.level === 'ERROR' ? 'text-red' : (entry.level === 'SUCCESS' ? 'text-green' : '')}`;
                line.textContent = `[${entry.time}] [${entry.level || 'INFO'}] ${entry.message}`;
                dom.renderLogTerminal.appendChild(line);
            });
            shownLogCount = logs.length;
            dom.renderLogTerminal.scrollTop = dom.renderLogTerminal.scrollHeight;
        };

        pollTaskStatus(taskId, showProgress, (videoUrl, taskData) => {
            displayActualNarrationTiming(taskData);
            localStorage.removeItem('active_render_task_id');
            stopRenderTimer();
            dom.btnStartRender.disabled = false;
            dom.btnRenderBack.disabled = false;
            dom.btnStartRender.innerHTML = '<i class="fa-solid fa-circle-play"></i> BẮT ĐẦU XUẤT VIDEO REVIEW';
            dom.videoPreviewPlaceholder.classList.add('hidden');
            dom.videoPreviewWrapper.classList.remove('hidden');
            dom.btnDownloadVideo.href = videoUrl;
            if (videoUrl.endsWith('.zip')) {
                dom.finalVideoPlayer.style.display = 'none';
                dom.btnDownloadVideo.innerHTML = '<i class="fa-solid fa-file-zipper"></i> TẢI GÓI NGUYÊN LIỆU CAPCUT (ZIP)';
            } else {
                dom.finalVideoPlayer.style.display = 'block';
                dom.finalVideoPlayer.src = videoUrl;
                dom.finalVideoPlayer.load();
                dom.btnDownloadVideo.innerHTML = '<i class="fa-solid fa-circle-down"></i> TẢI VIDEO REVIEW NGAY (MP4)';
            }
            showToast('Đã khôi phục và hoàn thành tác vụ xuất video.', 'success');
        }, (message) => {
            localStorage.removeItem('active_render_task_id');
            stopRenderTimer();
            dom.btnStartRender.disabled = false;
            dom.btnRenderBack.disabled = false;
            dom.btnStartRender.innerHTML = '<i class="fa-solid fa-rotate-right"></i> THỬ LẠI / TIẾP TỤC XUẤT VIDEO';
            showToast(message, 'error');
        });
    } catch (error) {
        // Keep the id: the web server may still be restarting while the worker runs.
        showToast('Chưa kết nối lại được giao diện. Tác vụ render vẫn được giữ để thử lại.', 'error');
    }
}

function pollTaskStatus(taskId, onProgress, onSuccess, onError) {
    let errorCount = 0;
    let polling = false;
    let finished = false;
    const timer = setInterval(async () => {
        // A slow status response used to allow many overlapping polls. All
        // pending responses could report success and launch quick-render again.
        if (polling || finished) return;
        polling = true;
        try {
            const res = await fetch(`/api/task-status/${taskId}`);
            const data = await res.json();
            if (finished) return;
            syncTaskApiKeyStates(data);
            
            const recoveredAfterDisconnect = errorCount >= 3;
            errorCount = 0;
            if (recoveredAfterDisconnect) {
                showToast('Đã kết nối lại máy chủ. Worker vẫn tiếp tục công việc.', 'success');
            }
            
            if (data.status === 'success') {
                finished = true;
                clearInterval(timer);
                // Cap nhat progress len 100% truoc khi goi onSuccess
                onProgress({ ...data, progress: 100, message: 'Hoan thanh!' });
                onSuccess(data.result, data);
            } else if (data.status === 'failed' || data.status === 'not_found') {
                finished = true;
                clearInterval(timer);
                // Đẩy log lỗi cuối cùng lên terminal trước khi hiện thông báo.
                onProgress(data);
                onError(data.message || 'Lỗi tác vụ không xác định.', data);
            } else {
                // Vẫn đang chạy
                onProgress(data);
            }
        } catch (err) {
            errorCount++;
            // Do not declare the render failed just because Flask is temporarily
            // unavailable. The detached worker may still be rendering normally.
            if (errorCount === 3) {
                showToast('Tạm mất kết nối giao diện; worker vẫn tiếp tục chạy. Đang tự kết nối lại…', 'error');
            }
        } finally {
            polling = false;
        }
    }, 2000);
}

// =====================================================================
// EVENT LISTENERS BINDING
// =====================================================================
function setupEventListeners() {
    // Stepper click chuyển tab
    dom.steps.forEach(step => {
        step.addEventListener('click', () => {
            const stepNum = parseInt(step.getAttribute('data-step'));
            switchPanel(stepNum);
        });
    });
    
    // Quay lại
    dom.backButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = parseInt(btn.getAttribute('data-target'));
            switchPanel(target);
        });
    });
    
    // Toggle hiển thị API Key
    dom.btnToggleApiKey.addEventListener('click', () => {
        const type = dom.apiKeyInput.type === 'password' ? 'text' : 'password';
        [dom.apiKeyInput, dom.apiKeyInput2, dom.apiKeyInput3].forEach(input => {
            if (input) input.type = type;
        });
        dom.btnToggleApiKey.querySelector('i').className = type === 'password' ? 'fa-solid fa-eye' : 'fa-solid fa-eye-slash';
    });

    dom.btnAddApiSlot.addEventListener('click', addApiSlot);

    dom.apiPriorityButtons.forEach(button => {
        button.addEventListener('click', () => {
            prioritizeApiSlot(Number(button.dataset.apiSlot));
        });
    });

    dom.apiRemoveButtons.forEach(button => {
        button.addEventListener('click', () => {
            removeApiSlot(Number(button.dataset.apiSlot));
        });
    });

    // API verification button
    dom.btnVerifyApi.addEventListener('click', () => verifyApiKey(true));
    
    // Check key on text input change
    [dom.apiKeyInput, dom.apiKeyInput2, dom.apiKeyInput3].forEach(input => {
        if (!input) return;
        input.addEventListener('input', () => {
            input.classList.remove(
                'api-key-usable', 'api-key-exhausted',
                'api-key-invalid', 'api-key-rate-limited', 'api-key-active',
                'api-key-error', 'api-key-network-error', 'api-key-temporary-error',
                'api-key-probe-error', 'api-key-connection-error'
            );
            const editedKey = input.value.trim();
            if (editedKey) {
                state.apiKeyStatuses[editedKey] = {
                    status: 'pending', message: 'Chưa xác thực'
                };
            }
            const currentKeys = collectApiKeys();
            if (!currentKeys || JSON.stringify(currentKeys) !== JSON.stringify(state.apiKeys)) {
                updateAPIIndicator(false);
            }
            checkNextStep1Availability();
            scheduleAutomaticApiVerification();
        });
    });
    if (dom.groqApiKeyInput) {
        dom.groqApiKeyInput.addEventListener('input', () => {
            if (dom.groqApiKeyInput.value.trim() !== state.groqApiKey) {
                updateAPIIndicator(false);
            }
            checkNextStep1Availability();
            scheduleAutomaticApiVerification();
        });
    }
    if (dom.thinkingEffortSelect) {
        dom.thinkingEffortSelect.addEventListener('change', () => {
            state.thinkingEffort = dom.thinkingEffortSelect.value;
            localStorage.setItem('gemini_thinking_effort', state.thinkingEffort);
        });
    }
    
    // Select video input change
    dom.videoSelect.addEventListener('change', () => {
        const nextVideo = dom.videoSelect.value;
        if (nextVideo !== state.selectedVideo) resetProjectData(nextVideo);
        state.selectedVideo = nextVideo;
        checkNextStep1Availability();
        loadVideoDurationSuggestion(nextVideo);
    });
    if (dom.reviewMinutes) dom.reviewMinutes.addEventListener('input', updateReviewMinutesHint);
    if (dom.styleSelect) dom.styleSelect.addEventListener('change', updateReviewMinutesHint);
    
    // Drag/Drop Video Upload Setup
    setupDragAndDrop(dom.videoDropZone, dom.videoFileInput, dom.videoUploadProgress, 'video', (filename) => {
        // Tải lại list và chọn video mới upload
        loadVideosList().then(() => {
            dom.videoSelect.value = filename;
            resetProjectData(filename);
            checkNextStep1Availability();
            loadVideoDurationSuggestion(filename);
        });
    });
    
    // Button Step 1 -> Step 2
    dom.btnToStep2.addEventListener('click', continueToScriptStep);
    
    // AI Scriptwriter Button
    dom.btnGenerateScript.addEventListener('click', generateScript);
    if (dom.btnQuickVideo) dom.btnQuickVideo.addEventListener('click', createQuickVideo);
    
    // Script Editor changes
    dom.scriptEditor.addEventListener('input', updateCharCount);
    
    // Button Step 2 -> Step 3
    dom.btnToStep3.addEventListener('click', () => {
        // Lưu giá trị kịch bản cuối cùng và chuyển bước
        state.script = dom.scriptEditor.value.trim();
        if (!state.script) {
            showToast('Nội dung kịch bản trống!', 'error');
            return;
        }
        if (state.script !== state.splitScriptSnapshot) {
            state.sentences = [];
        }
        switchPanel(3);
    });
    
    // Re-split Sentences Button
    dom.btnReSplit.addEventListener('click', autoSplitScript);
    
    // Add new sentence Button
    dom.btnAddSentence.addEventListener('click', () => {
        state.sentences.push('(Nhập câu mới tại đây)');
        renderSentences();
        updateSentenceStats();
        // Scroll to bottom
        dom.sentencesList.scrollTop = dom.sentencesList.scrollHeight;
        showToast('Đã thêm 1 câu mới!', 'success');
    });
    
    // Voice Sliders Display Sync
    dom.voiceRateSlider.addEventListener('input', () => {
        const val = parseInt(dom.voiceRateSlider.value);
        dom.voiceRateVal.textContent = `${val >= 0 ? '+' : ''}${val}%`;
        updateReviewMinutesHint();
        updateSentenceStats();
        updateCharCount();
    });
    
    // Button Step 3 -> Step 4
    dom.btnToStep4.addEventListener('click', () => {
        // Đảm bảo lấy danh sách câu đã chỉnh sửa
        const textareas = dom.sentencesList.querySelectorAll('.sentence-textarea');
        const updated = [];
        textareas.forEach(ta => {
            if (ta.value.trim()) updated.push(ta.value.trim());
        });
        
        if (updated.length === 0) {
            showToast('Vui lòng tạo kịch bản và phân câu!', 'error');
            return;
        }
        state.sentences = updated;
        switchPanel(4);
        updateSubtitlePreviewBackground(state.selectedVideo);
    });
    
    // Su kien toggle hien thi phu de
    const toggleSubtitles = document.getElementById('enable-subtitles');
    if (toggleSubtitles) {
        toggleSubtitles.addEventListener('change', (e) => {
            const previewBox = document.getElementById('subtitle-preview-box');
            if(e.target.checked) {
                if(previewBox) previewBox.style.opacity = '1';
                dom.subSize.disabled = false;
                dom.subColor.disabled = false;
            } else {
                if(previewBox) previewBox.style.opacity = '0.3';
                dom.subSize.disabled = true;
                dom.subColor.disabled = true;
            }
        });
    }
    
    // Drag/Drop BGM Upload Setup
    setupDragAndDrop(dom.bgmDropZone, dom.bgmFileInput, dom.bgmUploadProgress, 'bgm', (filename) => {
        loadBgmsList().then(() => {
            dom.bgmSelect.value = filename;
        });
    });
    
    // BGM Volume Slider Sync
    dom.bgmVolumeSlider.addEventListener('input', () => {
        dom.bgmVolumeVal.textContent = `${dom.bgmVolumeSlider.value}%`;
    });
    
    // ===== Subtitle Live Preview =====
    function updateSubtitlePreview() {
        const previewText = document.getElementById('preview-subtitle-text');
        if (!previewText) return;
        
        const size = parseInt(dom.subSize.value) || 42;
        const color = dom.subColor.value || '#FFFFFF';
        const opacity = parseInt(dom.subOpacity.value) || 150;
        
        // Tinh toan nhu backend: ASS Fontsize = user_px * 384 / 1080
        // Preview scale: preview box ~200px cao, video that 1080px => ty le ~0.185
        // => preview_px = user_px * (384/1080) * (200/384) = user_px * 200/1080 ≈ user_px * 0.185
        const scaledSize = Math.max(8, Math.round(size * 0.185));
        const bgAlpha = (opacity / 255).toFixed(2);
        
        previewText.style.fontSize = `${scaledSize}px`;
        previewText.style.color = color;
        previewText.style.background = `rgba(0, 0, 0, ${bgAlpha})`;
    }
    
    dom.subSize.addEventListener('input', updateSubtitlePreview);
    dom.subColor.addEventListener('input', updateSubtitlePreview);
    dom.subOpacity.addEventListener('input', updateSubtitlePreview);
    
    // Subtitle Preset Chips
    document.querySelectorAll('.preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const size = chip.getAttribute('data-size');
            const color = chip.getAttribute('data-color');
            const opacity = chip.getAttribute('data-opacity');
            
            dom.subSize.value = size;
            dom.subColor.value = color;
            dom.subOpacity.value = opacity;
            
            // Cap nhat active state
            document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            
            updateSubtitlePreview();
            showToast('Đã áp dụng mẫu phụ đề!', 'success');
        });
    });
    
    // Initial preview
    updateSubtitlePreview();
    
    // Button Step 4 -> Step 5
    dom.btnToStep5.addEventListener('click', () => {
        enableStep(5);
        switchPanel(5);
    });
    
    // Start Rendering Button
    dom.btnStartRender.addEventListener('click', startRenderVideo);
    if (dom.btnChooseExportDirectory) {
        dom.btnChooseExportDirectory.addEventListener('click', chooseExportDirectory);
    }
}

// =====================================================================
// DOCUMENT ON READY
// =====================================================================
document.addEventListener('DOMContentLoaded', () => {
    loadInitialData().then(resumePersistedRender);
    loadSfxList();
    loadExportDirectory();
    setupEventListeners();
    
    // Prompt Template Chips - nhan de dien vao o "Yeu Cau Chi Tiet"
    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const promptText = chip.getAttribute('data-prompt');
            if (promptText && dom.customPrompt) {
                if (dom.customPrompt.value.trim() !== '') {
                    dom.customPrompt.value += '\n- ' + promptText;
                } else {
                    dom.customPrompt.value = promptText;
                }
                dom.customPrompt.focus();
                // Hieu ung visual: chip sang len
                chip.style.borderColor = 'var(--accent-green)';
                chip.style.color = 'var(--accent-green)';
                setTimeout(() => {
                    chip.style.borderColor = '';
                    chip.style.color = '';
                }, 800);
                showToast('Đã điền mẫu prompt!', 'success');
            }
        });
    });
    
    // Bắt sự kiện cho các thẻ Cảm xúc
    document.querySelectorAll('.emotion-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const insertText = tag.getAttribute('data-insert');
            if (insertText && dom.customPrompt) {
                if (dom.customPrompt.value.trim() !== '') {
                    dom.customPrompt.value += '\n- [Cảm xúc]: ' + insertText;
                } else {
                    dom.customPrompt.value = '- [Cảm xúc]: ' + insertText;
                }
                dom.customPrompt.focus();
                showToast('Đã chèn lệnh cảm xúc!', 'success');
            }
        });
    });

    // Bat su kien Goi Y Nhac Nen
    const btnSuggestBgm = document.getElementById('btn-suggest-bgm');
    if (btnSuggestBgm) {
        btnSuggestBgm.addEventListener('click', () => {
            const style = dom.styleSelect ? dom.styleSelect.value : "";
            const options = Array.from(dom.bgmSelect.options);
            
            let targetKeywords = [];
            if (style === "dramatic") targetKeywords = ["epic", "battle", "action", "cinematic", "dramatic"];
            else if (style === "humorous") targetKeywords = ["funny", "sneaky", "clown", "quirky", "comedy"];
            else if (style === "shorts") targetKeywords = ["epic", "action", "suspense", "cyberpunk"];
            else if (style === "news") targetKeywords = ["news", "corporate", "upbeat", "electronic"];
            else if (style === "tutorial") targetKeywords = ["lofi", "chill", "tutorial", "ambient"];
            
            let found = false;
            for (let i = 0; i < options.length; i++) {
                let val = options[i].value.toLowerCase();
                if (val && targetKeywords.some(kw => val.includes(kw))) {
                    dom.bgmSelect.selectedIndex = i;
                    found = true;
                    // Kich hoat event change
                    dom.bgmSelect.dispatchEvent(new Event('change'));
                    break;
                }
            }
            
            if (found) {
                showToast(`Đã tự động chọn nhạc nền cho phong cách: ${style}`, 'success');
            } else {
                showToast('Không tìm thấy nhạc phù hợp! Hãy tải các file nhạc có tên như epic, funny, lofi...', 'warning');
            }
        });
    }
});


// Canvas Subtitle Preview (Step 4)
function updateSubtitlePreviewBackground(videoFilename) {
    const previewBox = document.getElementById('subtitle-preview-box');
    if (!previewBox || !videoFilename) return;

    // Tạo một video tag ẩn trong bộ nhớ để load video gốc
    const tempVideo = document.createElement('video');
    tempVideo.crossOrigin = "anonymous";
    tempVideo.src = `/static/output/${videoFilename.replace('.mp4', '_gemini_temp.mp4')}`;
    
    // Fallback neu compressed chua co
    tempVideo.onerror = function() {
        tempVideo.onerror = null;
        tempVideo.src = `/source-video/${encodeURIComponent(videoFilename)}`;
    };

    tempVideo.currentTime = 2.0; // Seek đến giây thứ 2 để lấy hình

    tempVideo.addEventListener('loadeddata', () => {
        // Dùng Canvas chụp lại frame hình ở giây thứ 2
        const canvas = document.createElement('canvas');
        canvas.width = tempVideo.videoWidth || 1280;
        canvas.height = tempVideo.videoHeight || 720;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(tempVideo, 0, 0, canvas.width, canvas.height);

        // Biến frame hình thành ảnh nền cho hộp Preview Phụ Đề
        const frameDataUrl = canvas.toDataURL('image/jpeg');
        previewBox.style.backgroundImage = `url(${frameDataUrl})`;
        previewBox.style.backgroundSize = "cover";
        previewBox.style.backgroundPosition = "center";
        previewBox.style.boxShadow = "inset 0 0 100px rgba(0,0,0,0.6)";
        
        // Gỡ icon film đi
        const icon = previewBox.querySelector('.preview-bg-icon');
        if(icon) icon.style.opacity = "0.1";
    });
}

// =====================================================================
// VOICE PREVIEW - Nghe thu giong doc
// =====================================================================
(function() {
    const btn = document.getElementById('btn-preview-voice');
    if (!btn) return;
    
    const audio = document.getElementById('voice-preview-audio');
    const status = document.getElementById('voice-preview-status');
    const voiceSelect = document.getElementById('voice-select');
    const rateSlider = document.getElementById('voice-rate-slider');
    
    const sampleText = "Trong thế giới đầy bí ẩn này, một cuộc phiêu lưu mới đang bắt đầu. Liệu nhân vật chính có thể vượt qua mọi thử thách và tìm ra sự thật?";
    
    btn.onclick = async () => {
        const voice = voiceSelect.value;
        const rateVal = parseInt(rateSlider?.value || 20);
        const rate = (rateVal >= 0 ? '+' : '') + rateVal + '%';
        
        // UI loading
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tạo...';
        status.style.display = 'inline';
        status.textContent = '';
        audio.style.display = 'none';
        audio.pause();
        
        try {
            const res = await fetch('/api/preview-tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: sampleText, voice, rate })
            });
            const data = await res.json();
            
            if (data.success) {
                audio.src = data.url + '?t=' + Date.now();
                audio.style.display = 'block';
                audio.play();
                status.textContent = '✅ Đang phát';
                status.style.color = '#4ade80';
            } else {
                status.textContent = '❌ ' + (data.message || 'Lỗi');
                status.style.color = '#f87171';
            }
        } catch(e) {
            status.textContent = '❌ Lỗi kết nối';
            status.style.color = '#f87171';
        }
        
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-headphones"></i> Nghe thử giọng';
        
        setTimeout(() => { status.style.display = 'none'; }, 5000);
    };
})();
