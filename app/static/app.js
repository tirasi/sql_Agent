// S4: every value that comes from the DB, the LLM or the server is escaped
// before it goes into innerHTML, so a stored value like
// <img src=x onerror=...> is shown as text instead of running as code.
// C8: one id per page load = one conversation, so follow-up questions
// ("what about Mumbai?") and clarification answers ("2") work.
const SESSION_ID = (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : 'sess-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);

function esc(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

document.addEventListener('DOMContentLoaded', () => {
    // Icons initialization
    lucide.createIcons();

    // DOM Elements
    const schemaSidebar = document.getElementById('schemaSidebar');
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    
    const tablesList = document.getElementById('tablesList');
    const synonymsList = document.getElementById('synonymsList');
    
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    const executionStatus = document.getElementById('executionStatus');
    
    const answerBody = document.getElementById('answerBody');
    const sqlResultContainer = document.getElementById('sqlResultContainer');
    const sqlCode = document.getElementById('sqlCode');
    const copySqlBtn = document.getElementById('copySqlBtn');
    
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const resultsTable = document.getElementById('resultsTable');
    
    // Sidebar toggle
    toggleSidebarBtn.addEventListener('click', () => {
        schemaSidebar.classList.toggle('hidden');
    });

    // Preset suggested query chips
    document.querySelectorAll('.suggest-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            queryInput.value = chip.getAttribute('data-query');
            queryForm.dispatchEvent(new Event('submit'));
        });
    });

    // Copy SQL to clipboard
    copySqlBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(sqlCode.textContent)
            .then(() => {
                const icon = copySqlBtn.querySelector('i');
                copySqlBtn.innerHTML = '<i data-lucide="check" class="text-green"></i>';
                lucide.createIcons();
                setTimeout(() => {
                    copySqlBtn.innerHTML = '<i data-lucide="copy"></i>';
                    lucide.createIcons();
                }, 2000);
            })
            .catch(err => console.error('Failed to copy: ', err));
    });

    // Tab Switch Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.add('hidden'));
            
            btn.classList.add('active');
            const tabId = `tab-${btn.getAttribute('data-tab')}`;
            document.getElementById(tabId).classList.remove('hidden');
        });
    });

    // Load Schema Information
    fetchSchemaInfo();

    // Query Submission
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = queryInput.value.trim();
        if (!question) return;

        // Reset UI States
        submitBtn.disabled = true;
        submitBtn.querySelector('span').textContent = 'Processing...';
        executionStatus.textContent = 'Preparing...';
        executionStatus.className = 'status-badge running';
        
        answerBody.innerHTML = `
            <div class="empty-state">
                <i data-lucide="loader" class="animate-spin text-blue"></i>
                <p>The AI Agent is executing the LangGraph nodes. Please follow the execution trace on the left panel...</p>
            </div>
        `;
        lucide.createIcons();
        sqlResultContainer.classList.add('hidden');
        
        // Reset all steps UI
        const stepIds = ['context_resolution', 'schema_retrieval', 'sql_generation', 'validation', 'execution', 'response_generation'];
        stepIds.forEach(id => {
            const card = document.getElementById(`step-${id}`);
            card.className = 'step-card';
            card.querySelector('.step-status').innerHTML = '<i data-lucide="circle-dashed"></i>';
            document.getElementById(`data-${id}`).innerHTML = '';
        });
        lucide.createIcons();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question, session_id: SESSION_ID })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let lastNode = null;
            let finalState = {};

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // save incomplete line to buffer

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    
                    const payload = JSON.parse(trimmed.substring(6));
                    
                    if (payload.event === 'started') {
                        executionStatus.textContent = 'Running Trace';
                    } else if (payload.event === 'update') {
                        const node = payload.node;
                        const state = payload.state;
                        
                        // Merge into final state
                        Object.assign(finalState, state);
                        
                        // Handle transition: Mark previous node as completed
                        if (lastNode && lastNode !== node) {
                            markStepCompleted(lastNode, finalState);
                        }
                        
                        // Mark current node as active
                        markStepActive(node, state);
                        lastNode = node;
                    } else if (payload.event === 'completed') {
                        // Mark the last node as completed
                        if (lastNode) {
                            markStepCompleted(lastNode, finalState);
                        }
                        
                        executionStatus.textContent = 'Success';
                        executionStatus.className = 'status-badge completed';
                        renderFinalResponse(finalState);
                    } else if (payload.event === 'error') {
                        throw new Error(payload.error);
                    }
                }
            }

        } catch (error) {
            console.error(error);
            executionStatus.textContent = 'Failed';
            executionStatus.className = 'status-badge failed';
            
            // Mark active card as failed
            const activeCard = document.querySelector('.step-card.active');
            if (activeCard) {
                activeCard.className = 'step-card error';
                activeCard.querySelector('.step-status').innerHTML = '<i data-lucide="x-circle" class="text-red"></i>';
            }
            
            answerBody.innerHTML = `
                <div class="error-card">
                    <h3><i data-lucide="alert-triangle"></i> Execution Error</h3>
                    <p style="margin-top: 10px;">${esc(error.message)}</p>
                </div>
            `;
            lucide.createIcons();
        } finally {
            submitBtn.disabled = false;
            submitBtn.querySelector('span').textContent = 'Execute';
            lucide.createIcons();
        }
    });

    // Helper: Mark step as active
    function markStepActive(node, state) {
        const card = document.getElementById(`step-${node}`);
        if (!card) return;
        
        card.className = 'step-card active';
        card.querySelector('.step-status').innerHTML = '<i data-lucide="refresh-cw" class="animate-spin text-blue"></i>';
        lucide.createIcons();
        
        // Render step data dynamically as it executes
        const dataDiv = document.getElementById(`data-${node}`);
        renderStepData(node, state, dataDiv);
    }

    // Helper: Mark step as completed
    function markStepCompleted(node, state) {
        const card = document.getElementById(`step-${node}`);
        if (!card) return;
        
        card.className = 'step-card completed';
        card.querySelector('.step-status').innerHTML = '<i data-lucide="check-circle-2" class="text-green"></i>';
        lucide.createIcons();
        
        // Finalize step data view
        const dataDiv = document.getElementById(`data-${node}`);
        renderStepData(node, state, dataDiv);
    }

    // Helper: Render individual step metadata in the trace
    function renderStepData(node, state, container) {
        if (!state) return;
        let html = '';
        
        switch (node) {
            case 'context_resolution':
                if (state.resolved_input) {
                    html += `<div class="detail-line"><span class="detail-key">Resolved Input:</span> <span class="detail-val-highlight">${esc(state.resolved_input)}</span></div>`;
                }
                if (state.detected_synonyms && Object.keys(state.detected_synonyms).length > 0) {
                    html += `<div class="detail-line"><span class="detail-key">Synonyms Mapped:</span> <span class="detail-val-code">${esc(JSON.stringify(state.detected_synonyms))}</span></div>`;
                }
                break;
                
            case 'schema_retrieval':
                if (state.relevant_tables) {
                    html += `<div class="detail-line"><span class="detail-key">Target Tables:</span> <span class="detail-val-highlight">${esc(state.relevant_tables.join(', '))}</span></div>`;
                }
                if (state.schema_context) {
                    html += `<div class="detail-line"><span class="detail-key">Context length:</span> <span class="detail-val">${state.schema_context.length} chars</span></div>`;
                }
                break;
                
            case 'sql_generation':
                if (state.sql_query) {
                    html += `<div class="detail-line"><span class="detail-key">Generated Query:</span></div>`;
                    html += `<div class="detail-line" style="margin-top: 4px;"><span class="detail-val-code" style="word-break: break-all;">${esc(state.sql_query)}</span></div>`;
                }
                if (state.attempts !== undefined) {
                    html += `<div class="detail-line"><span class="detail-key">Attempt:</span> <span class="detail-val">${state.attempts}</span></div>`;
                }
                break;
                
            case 'validation':
                if (state.is_valid !== null && state.is_valid !== undefined) {
                    const statusClass = state.is_valid ? 'text-green' : 'text-red';
                    html += `<div class="detail-line"><span class="detail-key">Syntax Status:</span> <span class="${statusClass} font-semibold">${state.is_valid ? 'VALID' : 'INVALID'}</span></div>`;
                }
                if (state.validation_error) {
                    html += `<div class="detail-line" style="color: var(--accent-red);"><span class="detail-key">Error:</span> ${esc(state.validation_error)}</div>`;
                }
                break;
                
            case 'execution':
                if (state.execution_error) {
                    html += `<div class="detail-line" style="color: var(--accent-red);"><span class="detail-key">DB Error:</span> ${esc(state.execution_error)}</div>`;
                } else if (state.rows) {
                    html += `<div class="detail-line"><span class="detail-key">Query Result:</span> <span class="detail-val">${state.rows.length} rows, ${state.columns ? state.columns.length : 0} columns</span></div>`;
                    if (state.confidence !== undefined && state.confidence !== null) {
                        html += `<div class="detail-line"><span class="detail-key">Agreement:</span> <span class="detail-val">${Math.round(state.confidence * 100)}% of candidates, ${esc(state.candidate_groups)} distinct result(s)</span></div>`;
                    }
                    if (state.needs_clarification) {
                        html += `<div class="detail-line"><span class="detail-key">Ambiguous:</span> <span class="detail-val">asking which reading you meant</span></div>`;
                    }
                }
                break;
                
            case 'response_generation':
                if (state.final_answer) {
                    html += `<div class="detail-line"><span class="detail-key">Summary Answer:</span> <span class="detail-val">Generated successfully</span></div>`;
                }
                break;
        }
        
        container.innerHTML = html;
    }

    // Helper: Render Final Answer response and data tables
    function renderFinalResponse(state) {
        if (!state.final_answer) {
            answerBody.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="alert-circle" class="text-orange"></i>
                    <p>Execution finished, but no final answer was produced.</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        // Render natural language answer
        answerBody.innerHTML = `
            <div class="answer-card">
                ${esc(state.final_answer).replace(/\n/g, '<br>')}
            </div>
        `;
        
        // Show SQL query container if query is present
        if (state.sql_query) {
            sqlResultContainer.classList.remove('hidden');
            sqlCode.textContent = state.sql_query;
            
            // Render table headers and rows
            if (state.columns && state.rows) {
                // Populate Headers
                let headersHtml = '<tr>';
                state.columns.forEach(col => {
                    headersHtml += `<th>${esc(col)}</th>`;
                });
                headersHtml += '</tr>';
                resultsTable.querySelector('thead').innerHTML = headersHtml;

                // Populate Rows
                let rowsHtml = '';
                if (state.rows.length === 0) {
                    rowsHtml = `<tr><td colspan="${state.columns.length}" style="text-align: center; color: var(--text-muted);">No records found.</td></tr>`;
                } else {
                    state.rows.forEach(row => {
                        rowsHtml += '<tr>';
                        // Row value can be simple values or nested lists
                        if (Array.isArray(row)) {
                            row.forEach(val => {
                                rowsHtml += `<td>${val !== null ? esc(val) : 'NULL'}</td>`;
                            });
                        } else {
                            rowsHtml += `<td>${row !== null ? esc(row) : 'NULL'}</td>`;
                        }
                        rowsHtml += '</tr>';
                    });
                }
                resultsTable.querySelector('tbody').innerHTML = rowsHtml;
            } else {
                resultsTable.querySelector('thead').innerHTML = '';
                resultsTable.querySelector('tbody').innerHTML = `<tr><td style="text-align: center; color: var(--text-muted);">No tabular data returned.</td></tr>`;
            }
        }
        lucide.createIcons();
    }

    // Helper: Load database schemas for sidebar catalog
    async function fetchSchemaInfo() {
        try {
            const res = await fetch('/api/schema');
            if (!res.ok) throw new Error("Failed to load schema");
            const data = await res.json();
            
            // Render Tables List
            if (data.tables && Object.keys(data.tables).length > 0) {
                let html = '';
                Object.entries(data.tables).forEach(([tableName, meta]) => {
                    html += `
                        <div class="table-item">
                            <div class="table-name" onclick="toggleColumnList('${tableName}')">
                                <i data-lucide="chevron-right" class="chevron-icon" id="chevron-${tableName}"></i>
                                <span>${esc(tableName)}</span>
                            </div>
                            <div class="column-list hidden" id="cols-${tableName}">
                                <div class="column-name" style="font-style: italic; color: var(--text-muted); margin-bottom: 4px;">
                                    "${esc(meta.description)}"
                                </div>
                                ${Object.entries(meta.columns).map(([colName, colDesc]) => `
                                    <div class="column-name">
                                        <strong>${esc(colName)}</strong>: <span style="font-size: 0.75rem; opacity: 0.85;">${esc(colDesc)}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                });
                tablesList.innerHTML = html;
            } else {
                tablesList.innerHTML = '<div class="loading-placeholder">No tables found.</div>';
            }



            // Render Synonyms Mapping
            if (data.synonyms && Object.keys(data.synonyms).length > 0) {
                synonymsList.innerHTML = Object.entries(data.synonyms).map(([term, dbMapping]) => `
                    <div class="synonym-item">
                        <span class="synonym-from">${esc(term)}</span>
                        <i data-lucide="arrow-right" style="width: 12px; color: var(--text-muted);"></i>
                        <span class="synonym-to">${esc(dbMapping)}</span>
                    </div>
                `).join('');
            } else {
                synonymsList.innerHTML = '<div class="loading-placeholder">No synonyms found.</div>';
            }

            lucide.createIcons();

        } catch (error) {
            console.error("Schema loading error:", error);
            tablesList.innerHTML = '<div class="loading-placeholder text-red">Failed to load tables.</div>';
            synonymsList.innerHTML = '<div class="loading-placeholder text-red">Failed to load synonyms.</div>';
        }
    }
});

// Global function to toggle column lists in schema panel
window.toggleColumnList = function(tableName) {
    const list = document.getElementById(`cols-${tableName}`);
    const chevron = document.getElementById(`chevron-${tableName}`);
    if (list && chevron) {
        list.classList.toggle('hidden');
        if (list.classList.contains('hidden')) {
            chevron.style.transform = 'rotate(0deg)';
        } else {
            chevron.style.transform = 'rotate(90deg)';
        }
    }
}
