// State Management
const state = {
    chunkCount: 0,
    documents: [],
    geminiActive: false,
    apiKeyConfigured: false,
    activeSources: [],
    isUploading: false,
    isQuerying: false,
    activeJobId: null,       // tracks current background upload job
    jobPollInterval: null    // reference to polling interval so we can clear it
};

// UI Elements
const els = {
    dbChunkCount: document.getElementById('db-chunk-count'),
    apiStatus: document.getElementById('api-status'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    uploadProgressBar: document.getElementById('upload-progress-bar'),
    progressFill: document.getElementById('progress-fill'),
    progressStatus: document.getElementById('progress-status'),
    docsList: document.getElementById('docs-list'),
    btnClearDocs: document.getElementById('btn-clear-docs'),
    btnClearChroma: document.getElementById('btn-clear-chroma'),
    chatMessages: document.getElementById('chat-messages'),
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    activeDocBadge: document.getElementById('active-doc-badge'),
    sourceDrawer: document.getElementById('source-drawer'),
    closeDrawerBtn: document.getElementById('close-drawer-btn'),
    toggleSourcesBtn: document.getElementById('toggle-sources-btn'),
    sourceListContent: document.getElementById('source-list-content'),

    // Modal Elements
    alertModal: document.getElementById('alert-modal'),
    modalTitle: document.getElementById('modal-title'),
    modalMessage: document.getElementById('modal-message'),
    modalCancel: document.getElementById('modal-btn-cancel'),
    modalConfirm: document.getElementById('modal-btn-confirm'),
    modalClose: document.querySelector('.modal-close')
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    fetchStatus();
    setupEventListeners();
    autoResizeTextarea();
});

// --- Event Listeners Setup ---
function setupEventListeners() {
    els.dropZone.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', handleFileSelect);

    els.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.dropZone.classList.add('dragover');
    });

    els.dropZone.addEventListener('dragleave', () => {
        els.dropZone.classList.remove('dragover');
    });

    els.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    els.chatForm.addEventListener('submit', handleChatSubmit);
    els.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleChatSubmit(e);
        }
    });

    els.btnClearDocs.addEventListener('click', () => {
        showConfirmModal(
            'Clear Local Files',
            'Are you sure you want to delete all uploaded PDFs and their generated Markdown files from disk? This will not clear Chroma DB.',
            async () => {
                try {
                    const res = await fetch('/api/clear-docs', { method: 'POST' });
                    const data = await res.json();
                    appendSystemMessage(`📁 Local storage cleared: ${data.message || 'Success'}`);
                    fetchStatus();
                } catch (err) {
                    appendSystemMessage(`❌ Error clearing files: ${err.message}`);
                }
            }
        );
    });

    els.btnClearChroma.addEventListener('click', () => {
        showConfirmModal(
            'Reset Chroma DB & Cache',
            'Warning: This will delete all indexed document chunks from Chroma DB and clear the query cache. This action cannot be undone. Proceed?',
            async () => {
                try {
                    const res = await fetch('/api/clear-chroma', { method: 'POST' });
                    const data = await res.json();
                    appendSystemMessage(`🧹 Database Reset: ${data.message || 'Success'}`);
                    fetchStatus();
                } catch (err) {
                    appendSystemMessage(`❌ Error resetting database: ${err.message}`);
                }
            }
        );
    });

    els.toggleSourcesBtn.addEventListener('click', () => {
        els.sourceDrawer.classList.toggle('closed');
        els.toggleSourcesBtn.classList.toggle('active');
    });

    els.closeDrawerBtn.addEventListener('click', () => {
        els.sourceDrawer.classList.add('closed');
        els.toggleSourcesBtn.classList.remove('active');
    });

    els.modalCancel.addEventListener('click', hideModal);
    els.modalClose.addEventListener('click', hideModal);
    els.alertModal.addEventListener('click', (e) => {
        if (e.target === els.alertModal) hideModal();
    });
}

// --- Fetch Status ---
async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error('Status endpoint failed');
        const data = await res.json();

        state.chunkCount = data.chunk_count;
        state.documents = data.documents;
        state.geminiActive = data.gemini_active;
        state.apiKeyConfigured = data.api_key_configured;

        updateStatusUI();
    } catch (err) {
        console.error('Error fetching system status:', err);
        els.apiStatus.className = 'status-badge error';
        els.apiStatus.innerHTML = '<span class="pulse-ring"></span>Server Off';
    }
}

function updateStatusUI() {
    els.dbChunkCount.textContent = state.chunkCount;

    if (state.geminiActive) {
        els.apiStatus.className = 'status-badge success';
        els.apiStatus.innerHTML = '<span class="pulse-ring"></span>Gemini Active';
    } else {
        els.apiStatus.className = 'status-badge error';
        els.apiStatus.innerHTML = `<span class="pulse-ring"></span>${state.apiKeyConfigured ? 'Error (Check logs)' : 'API Key Missing'}`;
    }

    if (state.documents.length === 0) {
        els.docsList.innerHTML = `
            <div class="empty-docs-state">
                <i data-lucide="file-warning"></i>
                <p>No documents uploaded yet</p>
            </div>
        `;
        els.activeDocBadge.textContent = 'No Documents Active';
        els.activeDocBadge.style.background = 'rgba(239, 68, 68, 0.15)';
        els.activeDocBadge.style.color = 'var(--error-color)';
        els.activeDocBadge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
    } else {
        els.activeDocBadge.textContent = `${state.documents.length} PDF(s) Indexed`;
        els.activeDocBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        els.activeDocBadge.style.color = 'var(--success-color)';
        els.activeDocBadge.style.borderColor = 'rgba(16, 185, 129, 0.2)';

        els.docsList.innerHTML = state.documents.map(doc => `
            <div class="doc-item">
                <div class="doc-info">
                    <i data-lucide="file-text"></i>
                    <span class="doc-name" title="${doc}">${doc}</span>
                </div>
            </div>
        `).join('');
    }
    lucide.createIcons();
}

// --- Upload Handler ---
function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
    }
}

async function uploadFile(file) {
    if (state.isUploading) return;
    if (!file.name.endsWith('.pdf')) {
        appendSystemMessage('❌ Error: Only PDF files are allowed.');
        return;
    }

    state.isUploading = true;
    showUploadProgress(0, `Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        // ── Step 1: Send file to server ──────────────────────────────────
        // Server now returns IMMEDIATELY with a job_id
        // It does NOT wait for Docling to finish
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Upload failed');
        }

        const result = await response.json();

        // ── Step 2: Check if server rejected the file immediately ────────
        // e.g. too many pages, wrong type
        if (result.status !== 'processing') {
            throw new Error(result.message || 'Unexpected server response');
        }

        // ── Step 3: File accepted — start polling for background job ─────
        state.activeJobId = result.job_id;
        const fileSizeMb = result.file_size_mb || '?';
        const pageCount = result.page_count ? `${result.page_count} pages` : '';
        const mode = result.processing_mode || '';

        appendSystemMessage(
            `📄 <strong>${result.filename}</strong> received (${fileSizeMb} MB${pageCount ? ', ' + pageCount : ''}).` +
            `<br>⚙️ Processing mode: <em>${mode}</em>. Indexing in background...`
        );

        showUploadProgress(10, 'File received. Starting Docling processing...');

        // ── Step 4: Poll /api/job/{job_id} every 3 seconds ──────────────
        startJobPolling(result.job_id, result.filename);

    } catch (err) {
        // Upload itself failed (network error, page limit, wrong file type)
        appendSystemMessage(`❌ Upload failed: ${err.message}`);
        resetUploadProgress('Upload failed');
        state.isUploading = false;
    }
}

// ── Job Polling ──────────────────────────────────────────────────────────────
// Polls /api/job/{job_id} every 3 seconds
// Updates the progress bar with live status from the background thread
// Stops when job is "complete" or "failed"

function startJobPolling(jobId, filename) {
    // Map progress stage strings to approximate % values for the progress bar
    const progressStages = {
        'Starting Docling conversion...':                      15,
        'Docling conversion complete. Exporting Markdown...':  40,
        'Markdown saved. Extracting image contexts...':        50,
        'Image extraction skipped (large file mode).':         55,
        'Chunking Markdown text...':                           65,
        'Indexing':                                            80,   // prefix match
        'Processing complete.':                               100,
        'Processing failed.':                                 100,
    };

    // Clear any existing poll interval (safety measure)
    if (state.jobPollInterval) {
        clearInterval(state.jobPollInterval);
    }

    state.jobPollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/job/${jobId}`);
            if (!res.ok) {
                // Job endpoint not found — stop polling
                clearInterval(state.jobPollInterval);
                state.jobPollInterval = null;
                appendSystemMessage(`⚠️ Lost track of job ${jobId}. Check server logs.`);
                resetUploadProgress('Tracking lost');
                state.isUploading = false;
                return;
            }

            const job = await res.json();

            // ── Update progress bar text ──────────────────────────────
            let progressPct = 10;
            const progressText = job.progress || '';

            // Find matching stage for progress %
            for (const [stage, pct] of Object.entries(progressStages)) {
                if (progressText.startsWith(stage)) {
                    progressPct = pct;
                    break;
                }
            }

            // Show extracting images progress with count if available
            if (progressText.startsWith('Extracting images')) {
                progressPct = 55;
            }

            showUploadProgress(progressPct, progressText);

            // ── Job complete ──────────────────────────────────────────
            if (job.status === 'complete') {
                clearInterval(state.jobPollInterval);
                state.jobPollInterval = null;
                state.activeJobId = null;
                state.isUploading = false;

                showUploadProgress(100, 'Processing complete!');
                appendSystemMessage(
                    `✅ <strong>${filename}</strong> fully indexed into ChromaDB.<br>` +
                    `📦 ${job.chunks_added} chunks added. Total in DB: ${job.total_chunks}.`
                );

                // Refresh sidebar document list and chunk count
                fetchStatus();

                // Auto-hide progress bar after 3 seconds
                setTimeout(() => {
                    resetUploadProgress('');
                }, 3000);
            }

            // ── Job failed ────────────────────────────────────────────
            if (job.status === 'failed') {
                clearInterval(state.jobPollInterval);
                state.jobPollInterval = null;
                state.activeJobId = null;
                state.isUploading = false;

                appendSystemMessage(
                    `❌ Processing failed for <strong>${filename}</strong>.<br>` +
                    `Error: ${job.error || 'Unknown error. Check server logs.'}`
                );

                resetUploadProgress('Processing failed');
            }

        } catch (pollErr) {
            // Network error during polling — don't stop, keep trying
            console.warn('Polling error (will retry):', pollErr.message);
            showUploadProgress(null, 'Connection interrupted, retrying...');
        }

    }, 3000); // poll every 3 seconds
}

// ── Progress Bar Helpers ─────────────────────────────────────────────────────

function showUploadProgress(percent, statusText) {
    els.uploadProgressBar.style.display = 'block';

    // Only update fill width if percent provided (null = keep current)
    if (percent !== null) {
        els.progressFill.style.width = `${percent}%`;
    }

    if (statusText) {
        els.progressStatus.textContent = statusText;
    }

    // Turn bar red on failure
    if (statusText && (statusText.toLowerCase().includes('fail') || statusText.toLowerCase().includes('error'))) {
        els.progressFill.style.backgroundColor = 'var(--error-color)';
    } else {
        els.progressFill.style.backgroundColor = '';
    }
}

function resetUploadProgress(statusText) {
    setTimeout(() => {
        els.uploadProgressBar.style.display = 'none';
        els.progressFill.style.width = '0%';
        els.progressFill.style.backgroundColor = '';
        if (statusText !== undefined) {
            els.progressStatus.textContent = statusText;
        }
    }, 3000);
}

// --- Chat Queries Handler ---
async function handleChatSubmit(e) {
    e.preventDefault();
    const queryText = els.chatInput.value.trim();
    if (!queryText || state.isQuerying) return;

    // Warn user if an upload is still in progress
    if (state.isUploading) {
        appendSystemMessage('⏳ A document is still being processed. You can still query existing documents, but the new one may not be ready yet.');
    }

    state.isQuerying = true;
    appendUserMessage(queryText);
    els.chatInput.value = '';
    els.chatInput.style.height = 'auto';

    const typingIndicatorEl = appendTypingIndicator();
    scrollToBottom();

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: queryText })
        });

        removeTypingIndicator(typingIndicatorEl);

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'RAG Query Failed');
        }

        const data = await response.json();

        appendAssistantMessage(data.answer, data.sources, data.cached, data.image_paths);

        if (data.sources && data.sources.length > 0) {
            state.activeSources = data.sources;
            populateSourcesDrawer(data.sources);
            if (els.sourceDrawer.classList.contains('closed')) {
                els.sourceDrawer.classList.remove('closed');
                els.toggleSourcesBtn.classList.add('active');
            }
        }

    } catch (err) {
        removeTypingIndicator(typingIndicatorEl);
        appendSystemMessage(`❌ Query failed: ${err.message}`);
    } finally {
        state.isQuerying = false;
    }
}

// --- Message Rendering Helpers ---
function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message user-message';
    msg.innerHTML = `
        <div class="avatar"><i data-lucide="user"></i></div>
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
        </div>
    `;
    els.chatMessages.appendChild(msg);
    lucide.createIcons();
    scrollToBottom();
}

function appendAssistantMessage(text, sources, cached, imagePaths) {
    const msg = document.createElement('div');
    msg.className = 'message assistant-message';

    let metaHtml = '';
    if (sources && sources.length > 0) {
        metaHtml += `<div class="message-meta">`;
        if (cached) {
            metaHtml += `<span class="source-badge" style="color: var(--success-color); border-color: rgba(16, 185, 129, 0.2);" title="Retrieved from Local Request Cache"><i data-lucide="zap"></i> Cached</span>`;
        }
        metaHtml += `<span class="source-badge" onclick="openSourcesDrawer()"><i data-lucide="book-open"></i> ${sources.length} Sources</span>`;
        metaHtml += `</div>`;
    }

    let imagesHtml = '';
    if (imagePaths && imagePaths.length > 0) {
        imagesHtml += `<div class="message-images-gallery">`;
        imagePaths.forEach(path => {
            let url = path;
            if (!url.startsWith('/') && !url.startsWith('http')) {
                url = '/' + url;
            }
            imagesHtml += `
                <div class="message-image-card" onclick="openImageModal('${url.replace(/'/g, "\\'")}')">
                    <img src="${escapeHtml(url)}" alt="Retrieved diagram/image">
                    <span class="image-zoom-icon"><i data-lucide="zoom-in"></i></span>
                </div>
            `;
        });
        imagesHtml += `</div>`;
    }

    msg.innerHTML = `
        <div class="avatar"><i data-lucide="bot"></i></div>
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
            ${imagesHtml}
            ${metaHtml}
        </div>
    `;

    els.chatMessages.appendChild(msg);
    lucide.createIcons();
    scrollToBottom();
}

function appendSystemMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message system-message';
    msg.innerHTML = `
        <div class="avatar" style="background: var(--bg-sidebar); border: 1px solid var(--border-color); box-shadow: none;"><i data-lucide="info" style="color: var(--text-secondary)"></i></div>
        <div class="message-content" style="background: rgba(15, 23, 42, 0.3);">
            <p>${text}</p>
        </div>
    `;
    els.chatMessages.appendChild(msg);
    lucide.createIcons();
    scrollToBottom();
}

function appendTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message assistant-message typing-container';
    indicator.innerHTML = `
        <div class="avatar"><i data-lucide="bot"></i></div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    els.chatMessages.appendChild(indicator);
    lucide.createIcons();
    return indicator;
}

function removeTypingIndicator(indicatorEl) {
    if (indicatorEl && indicatorEl.parentNode) {
        indicatorEl.parentNode.removeChild(indicatorEl);
    }
}

// --- Context Drawer Management ---
function populateSourcesDrawer(sources) {
    if (!sources || sources.length === 0) {
        els.sourceListContent.innerHTML = `
            <div class="empty-sources-state">
                <i data-lucide="help-circle"></i>
                <p>No query context retrieved yet</p>
            </div>
        `;
        return;
    }

    els.sourceListContent.innerHTML = sources.map((src) => {
        const isImage = src.type === 'image' && src.image_path;
        return `
            <div class="source-card">
                <div class="source-card-header">
                    <span class="source-card-title" title="${src.source}">${src.source}</span>
                    <span class="source-card-meta">${isImage ? 'Image' : `Chunk ${src.chunk_id}`}</span>
                </div>
                ${isImage ? `
                    <div class="source-image-container" onclick="openImageModal('${src.image_path.replace(/'/g, "\\'")}')" style="cursor: pointer;">
                        <img src="${escapeHtml(src.image_path)}" class="source-image-preview" alt="Retrieved source image">
                    </div>
                ` : ''}
                <div class="source-snippet">${escapeHtml(src.snippet)}...</div>
            </div>
        `;
    }).join('');
    lucide.createIcons();
}

window.openSourcesDrawer = function () {
    els.sourceDrawer.classList.remove('closed');
    els.toggleSourcesBtn.classList.add('active');
};

// --- Modal Utilities ---
let activeConfirmCallback = null;

function showConfirmModal(title, message, onConfirm) {
    els.modalTitle.textContent = title;
    els.modalMessage.textContent = message;
    activeConfirmCallback = onConfirm;
    els.alertModal.style.display = 'flex';
    els.modalConfirm.focus();
}

function hideModal() {
    els.alertModal.style.display = 'none';
    activeConfirmCallback = null;
}

els.modalConfirm.addEventListener('click', () => {
    if (activeConfirmCallback) activeConfirmCallback();
    hideModal();
});

// --- General UI Helpers ---
function scrollToBottom() {
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, function (m) { return map[m]; });
}

function autoResizeTextarea() {
    els.chatInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight - 6) + 'px';
    });
}

window.openImageModal = function (url) {
    const overlay = document.createElement('div');
    overlay.className = 'image-lightbox-overlay';
    overlay.innerHTML = `
        <div class="lightbox-content">
            <img src="${url}" alt="Enlarged image">
            <button class="lightbox-close"><i data-lucide="x"></i></button>
        </div>
    `;
    document.body.appendChild(overlay);
    lucide.createIcons();

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay || e.target.closest('.lightbox-close')) {
            overlay.remove();
        }
    });
};












































