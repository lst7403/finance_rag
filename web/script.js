let allData = [];
let currentView = 'table';
let sortConfig = { key: 'ticker', direction: 'asc' };
let currentModalTicker = null;

const tableTooltip = document.getElementById('tableTooltip');
const TOOLTIP_DEFS = window.TOOLTIP_DEFS || {};

// API Base (points to FastAPI backend on port 8000 if opened on different port or file system)
const API_BASE = (window.location.port === '8000') ? '' : 'http://localhost:8000';

function buildTooltipHtml(definition) {
    if (!definition) return '';
    const tone = definition.tone || 'neutral';

    return `
        <div class='tooltip-title'>${definition.title}</div>
        <div class='tooltip-section'>
            <span class='tooltip-label'>Definition</span>
            ${definition.definition}
        </div>
        <div class='tooltip-section'>
            <span class='tooltip-label'>Evaluation</span>
            <span class='${tone}'>${definition.evaluation}</span>
        </div>
        <div class='tooltip-section'>
            <span class='tooltip-label'>Filter Guide</span>
            ${definition.filterGuide}
        </div>
        ${definition.note ? `<div class='tooltip-note'>${definition.note}</div>` : ''}
    `;
}

function applyHeaderTooltips() {
    document.querySelectorAll('.metric-header').forEach(header => {
        const key = header.dataset.sort;
        const definition = TOOLTIP_DEFS[key];
        if (!definition) return;
        header.dataset.tooltip = buildTooltipHtml(definition);
    });
}

function showHeaderTooltip(event) {
    const header = event.currentTarget;
    const text = header.dataset.tooltip;
    if (!text || !tableTooltip) return;

    tableTooltip.innerHTML = text;
    tableTooltip.classList.add('visible');
    positionTooltip(event);
}

function moveHeaderTooltip(event) {
    if (!tableTooltip || !tableTooltip.classList.contains('visible')) return;
    positionTooltip(event);
}

function positionTooltip(event) {
    const padding = 15;
    let x = event.clientX + padding;
    let y = event.clientY + padding;

    const tooltipRect = tableTooltip.getBoundingClientRect();
    if (x + tooltipRect.width > window.innerWidth - 10) {
        x = event.clientX - tooltipRect.width - padding;
    }
    if (y + tooltipRect.height > window.innerHeight - 10) {
        y = event.clientY - tooltipRect.height - padding;
    }

    tableTooltip.style.left = `${Math.max(10, x)}px`;
    tableTooltip.style.top = `${Math.max(10, y)}px`;
}

function hideHeaderTooltip() {
    if (!tableTooltip) return;
    tableTooltip.classList.remove('visible');
}

function populateSectorIndustryDropdowns() {
    const sectorSelect = document.getElementById('sectorFilter');
    if (!sectorSelect) return;

    const currentSector = sectorSelect.value;
    const uniqueSectors = Array.from(
        new Set(allData.map(c => c.sector).filter(Boolean))
    ).sort((a, b) => a.localeCompare(b));

    sectorSelect.innerHTML = '<option value="">All Sectors</option>' +
        uniqueSectors.map(sec => `<option value="${sec}">${sec}</option>`).join('');

    if (currentSector && uniqueSectors.includes(currentSector)) {
        sectorSelect.value = currentSector;
    }

    updateIndustryOptions();
}

function updateIndustryOptions() {
    const sectorSelect = document.getElementById('sectorFilter');
    const industrySelect = document.getElementById('industryFilter');
    if (!industrySelect) return;

    const selectedSector = sectorSelect ? sectorSelect.value : '';
    const currentIndustry = industrySelect.value;

    let relevantData = allData;
    if (selectedSector) {
        relevantData = allData.filter(c => c.sector === selectedSector);
    }

    const uniqueIndustries = Array.from(
        new Set(relevantData.map(c => c.industry).filter(Boolean))
    ).sort((a, b) => a.localeCompare(b));

    industrySelect.innerHTML = '<option value="">All Industries</option>' +
        uniqueIndustries.map(ind => `<option value="${ind}">${ind}</option>`).join('');

    if (currentIndustry && uniqueIndustries.includes(currentIndustry)) {
        industrySelect.value = currentIndustry;
    } else {
        industrySelect.value = '';
    }
}

async function loadData() {
    try {
        // Fetch enriched stocks from API querying the local SQLite database
        let response = await fetch(`${API_BASE}/api/stocks`).catch(() => null);
        if (response && response.ok) {
            const resData = await response.json();
            const stocks = resData.stocks || [];
            allData = stocks.map(s => ({
                ...s,
                ...(s.fundamentals || {})
            }));
            console.log(`[WebUI] Loaded ${allData.length} stocks from SQLite (${resData.database || 'foundamental_sql.db'})`);
            
            const dbText = document.getElementById('dbStatusText');
            if (dbText) {
                dbText.textContent = `SQLite: ${resData.database || 'foundamental_sql.db'} (${allData.length} stocks)`;
            }
        } else {
            // Direct SQL endpoint attempt
            let sqlRes = await fetch(`${API_BASE}/api/sql/stocks`).catch(() => null);
            if (sqlRes && sqlRes.ok) {
                const resData = await sqlRes.json();
                const stocks = resData.stocks || [];
                allData = stocks.map(s => ({
                    ...s,
                    ...(s.fundamentals || {})
                }));
                const dbText = document.getElementById('dbStatusText');
                if (dbText) dbText.textContent = `SQLite: foundamental_sql.db (${allData.length} stocks)`;
            } else {
                // Inform user to start server for SQLite connection
                document.getElementById('tableBody').innerHTML =
                    '<tr><td colspan="20" class="no-data"><strong>SQLite Backend Disconnected.</strong><br>Please start the server with <code>python app.py</code> to load data from <code>data/database/foundamental_sql.db</code>.</td></tr>';
                const dbText = document.getElementById('dbStatusText');
                if (dbText) dbText.textContent = 'SQLite: Disconnected';
                return;
            }
        }

        populateSectorIndustryDropdowns();
        renderData();
        updateStats();
        checkSystemStatus();
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('tableBody').innerHTML =
            '<tr><td colspan="20" class="no-data">Error loading financial metrics from SQLite database. Ensure backend server is running.</td></tr>';
    }
}

async function checkSystemStatus() {
    const badge = document.getElementById('systemStatusBadge');
    const statusText = document.getElementById('statusText');
    const dbText = document.getElementById('dbStatusText');
    if (!badge || !statusText) return;

    try {
        const res = await fetch(`${API_BASE}/api/system/status`, { timeout: 3000 });
        if (res.ok) {
            const data = await res.json();
            const ollama = data.ollama || {};
            const cuda = data.cuda || {};
            const cudaLabel = cuda.cuda_available 
                ? `⚡ CUDA (${(cuda.device_name || 'GPU').replace('NVIDIA ', '')})` 
                : '🖥️ CPU Mode';

            badge.title = `Hardware: ${cuda.device_name || 'CPU'}\nVRAM: ${cuda.vram_allocated_mb || 0}MB / ${cuda.vram_total_gb || 0}GB\nDatabase: ${data.fundamentals_database || 'SQLite'}\nEmbedding: ${data.embedding_model || 'Local'}\nRouter: LLM RouterRetriever (Fundamentals / News / Dossiers)\nReranker: ${data.reranker_model || 'Cross-Encoder'}\nVector DB: ${data.vector_store || 'ChromaDB'}`;

            if (dbText && data.fundamentals_database) {
                dbText.textContent = `SQLite: foundamental_sql.db (${data.stocks_count} stocks)`;
            }

            if (ollama.status === 'online') {
                badge.className = 'status-badge status-online';
                statusText.textContent = `Qwen 2.5 (7B) Online • ${cudaLabel} • Router + Reranker Active • ${data.stocks_count} Stocks`;
            } else {
                badge.className = 'status-badge status-offline';
                statusText.textContent = `Ollama Offline (${cudaLabel})`;
            }
            return;
        }
    } catch (e) {
        // Ignored, fallback offline
    }

    badge.className = 'status-badge status-offline';
    statusText.textContent = 'AI Server Disconnected';
}

function formatPercent(value) {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return (value * 100).toFixed(2) + '%';
}

function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return parseFloat(value).toFixed(decimals);
}

function getValueClass(value, type = 'normal') {
    if (value === null || value === undefined || isNaN(value)) return 'neutral';
    if (type === 'net_debt_to_ebitda') {
        if (value < 0) return 'positive'; // Net Cash is excellent
        if (value > 3.0) return 'negative';
        return 'neutral';
    }
    if (type === 'pe_ratio') {
        if (value > 0 && value < 20) return 'positive';
        if (value > 50) return 'negative';
        return 'neutral';
    }
    return value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral';
}

function getFilteredData() {
    let filtered = [...allData];

    // Global Search
    const searchVal = document.getElementById('searchInput').value.toLowerCase().trim();
    if (searchVal) {
        filtered = filtered.filter(item =>
            (item.ticker && item.ticker.toLowerCase().includes(searchVal)) ||
            (item.company_name && item.company_name.toLowerCase().includes(searchVal)) ||
            (item.sector && item.sector.toLowerCase().includes(searchVal)) ||
            (item.industry && item.industry.toLowerCase().includes(searchVal))
        );
    }

    // Preset Filter
    const metricFilter = document.getElementById('metricFilter').value;
    if (metricFilter === 'high-growth') {
        filtered = filtered.filter(c => c.eps_growth_1y > 0.20);
    } else if (metricFilter === 'high-profitability') {
        filtered = filtered.filter(c => c.operating_margin > 0.30);
    } else if (metricFilter === 'strong-cashflow') {
        filtered = filtered.filter(c => c.fcf_margin > 0.20);
    } else if (metricFilter === 'low-debt') {
        filtered = filtered.filter(c => c.net_debt_to_ebitda !== null && c.net_debt_to_ebitda < 2.0);
    } else if (metricFilter === 'undervalued') {
        filtered = filtered.filter(c => c.pe_ratio !== null && c.pe_ratio > 0 && c.pe_ratio < 20);
    } else if (metricFilter === 'expanding-margins') {
        filtered = filtered.filter(c => c.operating_margin_trend > 0);
    } else if (metricFilter === 'has-news') {
        filtered = filtered.filter(c => (c.news_count || 0) > 0);
    }

    // Column Filters
    document.querySelectorAll('.column-filter-input').forEach(input => {
        const key = input.dataset.filterKey;
        const val = input.value.trim();
        if (!val) return;

        if (key === 'ticker' || key === 'company_name') {
            filtered = filtered.filter(c => c[key] && c[key].toLowerCase().includes(val.toLowerCase()));
        } else {
            const numVal = parseFloat(val);
            if (isNaN(numVal)) return;

            const wrap = input.closest('.column-filter-wrap');
            const op = wrap ? wrap.querySelector('.column-filter-op')?.value || '>' : '>';
            
            filtered = filtered.filter(c => {
                let cellVal = c[key];
                if (cellVal === null || cellVal === undefined || isNaN(cellVal)) return false;

                // Adjust for percentage fields
                if (key.includes('growth') || key.includes('margin') || key.includes('roe') || key.includes('roic')) {
                    cellVal = cellVal * 100;
                }

                switch (op) {
                    case '>': return cellVal > numVal;
                    case '<': return cellVal < numVal;
                    case '>=': return cellVal >= numVal;
                    case '<=': return cellVal <= numVal;
                    case '=': return Math.abs(cellVal - numVal) < 0.001;
                    default: return true;
                }
            });
        }
    });

    // Sector & Industry Select Filters
    const sectorFilter = document.getElementById('sectorFilter');
    if (sectorFilter && sectorFilter.value) {
        filtered = filtered.filter(c => c.sector === sectorFilter.value);
    }
    const industryFilter = document.getElementById('industryFilter');
    if (industryFilter && industryFilter.value) {
        filtered = filtered.filter(c => c.industry === industryFilter.value);
    }

    // Sorting
    filtered.sort((a, b) => {
        let valA = a[sortConfig.key];
        let valB = b[sortConfig.key];

        if (valA === null || valA === undefined || isNaN(valA)) valA = sortConfig.direction === 'asc' ? Infinity : -Infinity;
        if (valB === null || valB === undefined || isNaN(valB)) valB = sortConfig.direction === 'asc' ? Infinity : -Infinity;

        if (typeof valA === 'string') {
            return sortConfig.direction === 'asc'
                ? valA.localeCompare(valB)
                : valB.localeCompare(valA);
        }

        return sortConfig.direction === 'asc' ? valA - valB : valB - valA;
    });

    return filtered;
}

function renderTable() {
    const filtered = getFilteredData();
    const tbody = document.getElementById('tableBody');

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="20" class="no-data">No matching companies found.</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(c => {
        const newsCount = c.news_count || 0;
        const newsBadgeHtml = newsCount > 0
            ? `<span class="news-badge has-news" title="${c.latest_headline || 'Has recent news'}">📰 ${newsCount}</span>`
            : `<span class="news-badge no-news">-</span>`;

        return `
            <tr onclick="openCompanyModal('${c.ticker}')">
                <td class="sticky-col col-ticker">${c.ticker}</td>
                <td class="col-company-name">${c.company_name || c.ticker}</td>
                <td class="col-news-badge">${newsBadgeHtml}</td>
                <td class="col-sector">${c.sector || 'N/A'}</td>
                <td class="col-industry">${c.industry || 'N/A'}</td>

                <td class="col-rev-growth ${getValueClass(c.revenue_growth_1y)}">${formatPercent(c.revenue_growth_1y)}</td>
                <td class="col-rev-cagr ${getValueClass(c.revenue_cagr_3y)}">${formatPercent(c.revenue_cagr_3y)}</td>
                <td class="col-eps-growth ${getValueClass(c.eps_growth_1y)}">${formatPercent(c.eps_growth_1y)}</td>
                <td class="col-eps-cagr ${getValueClass(c.eps_cagr_3y)}">${formatPercent(c.eps_cagr_3y)}</td>

                <td class="col-gross-margin ${getValueClass(c.gross_margin)}">${formatPercent(c.gross_margin)}</td>
                <td class="col-op-margin ${getValueClass(c.operating_margin)}">${formatPercent(c.operating_margin)}</td>
                <td class="col-net-margin ${getValueClass(c.net_margin)}">${formatPercent(c.net_margin)}</td>
                <td class="col-op-trend ${getValueClass(c.operating_margin_trend)}">${formatPercent(c.operating_margin_trend)}</td>
                <td class="col-roe ${getValueClass(c.roe)}">${formatPercent(c.roe)}</td>
                <td class="col-roic ${getValueClass(c.roic)}">${formatPercent(c.roic)}</td>

                <td class="col-fcf-ni">${formatNumber(c.fcf_to_net_income)}</td>
                <td class="col-fcf-margin ${getValueClass(c.fcf_margin)}">${formatPercent(c.fcf_margin)}</td>
                <td class="col-fcf-growth ${getValueClass(c.fcf_growth_1y)}">${formatPercent(c.fcf_growth_1y)}</td>

                <td class="col-net-debt ${getValueClass(c.net_debt_to_ebitda, 'net_debt_to_ebitda')}">${formatNumber(c.net_debt_to_ebitda)}</td>
                <td class="col-pe ${getValueClass(c.pe_ratio, 'pe_ratio')}">${formatNumber(c.pe_ratio)}</td>
            </tr>
        `;
    }).join('');
}

function renderCards() {
    const filtered = getFilteredData();
    const cardView = document.getElementById('cardView');

    if (filtered.length === 0) {
        cardView.innerHTML = '<div class="no-data" style="grid-column: 1/-1; text-align: center; padding: 40px;">No matching companies found.</div>';
        return;
    }

    cardView.innerHTML = filtered.map(c => `
        <div class="stock-card" onclick="openCompanyModal('${c.ticker}')">
            <div class="stock-card-header">
                <div class="stock-card-title">
                    <h3>${c.ticker} • ${c.company_name || c.ticker}</h3>
                    <p>${c.sector || 'N/A'} • ${c.industry || 'N/A'}</p>
                </div>
                ${c.news_count ? `<span class="news-badge has-news">📰 ${c.news_count}</span>` : ''}
            </div>
            <div class="card-metrics-grid">
                <div class="card-metric-row">
                    <span class="card-metric-label">P/E Ratio</span>
                    <span class="card-metric-value ${getValueClass(c.pe_ratio, 'pe_ratio')}">${formatNumber(c.pe_ratio)}</span>
                </div>
                <div class="card-metric-row">
                    <span class="card-metric-label">ROIC</span>
                    <span class="card-metric-value ${getValueClass(c.roic)}">${formatPercent(c.roic)}</span>
                </div>
                <div class="card-metric-row">
                    <span class="card-metric-label">Rev Growth 1Y</span>
                    <span class="card-metric-value ${getValueClass(c.revenue_growth_1y)}">${formatPercent(c.revenue_growth_1y)}</span>
                </div>
                <div class="card-metric-row">
                    <span class="card-metric-label">Gross Margin</span>
                    <span class="card-metric-value ${getValueClass(c.gross_margin)}">${formatPercent(c.gross_margin)}</span>
                </div>
                <div class="card-metric-row">
                    <span class="card-metric-label">FCF Margin</span>
                    <span class="card-metric-value ${getValueClass(c.fcf_margin)}">${formatPercent(c.fcf_margin)}</span>
                </div>
                <div class="card-metric-row">
                    <span class="card-metric-label">Net Debt/EBITDA</span>
                    <span class="card-metric-value ${getValueClass(c.net_debt_to_ebitda, 'net_debt_to_ebitda')}">${formatNumber(c.net_debt_to_ebitda)}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function updateStats() {
    const filtered = getFilteredData();
    document.getElementById('totalCompanies').textContent = filtered.length;

    const peValues = filtered.map(c => c.pe_ratio).filter(v => v !== null && !isNaN(v) && v > 0);
    document.getElementById('avgPE').textContent = peValues.length > 0
        ? (peValues.reduce((a, b) => a + b) / peValues.length).toFixed(2)
        : 'N/A';

    const roeValues = filtered.map(c => c.roe).filter(v => v !== null && !isNaN(v));
    document.getElementById('avgROE').textContent = roeValues.length > 0
        ? ((roeValues.reduce((a, b) => a + b) / roeValues.length) * 100).toFixed(1) + '%'
        : 'N/A';

    const growthValues = filtered.map(c => c.eps_growth_1y).filter(v => v !== null && !isNaN(v));
    document.getElementById('avgGrowth').textContent = growthValues.length > 0
        ? ((growthValues.reduce((a, b) => a + b) / growthValues.length) * 100).toFixed(1) + '%'
        : 'N/A';

    const fcfValues = filtered.map(c => c.fcf_margin).filter(v => v !== null && !isNaN(v));
    document.getElementById('avgFCFMargin').textContent = fcfValues.length > 0
        ? ((fcfValues.reduce((a, b) => a + b) / fcfValues.length) * 100).toFixed(1) + '%'
        : 'N/A';
}

function renderData() {
    if (currentView === 'table') {
        renderTable();
    } else {
        renderCards();
    }
    updateStats();
}

// Modal Functions & Tabs
async function openCompanyModal(ticker) {
    const company = allData.find(c => c.ticker === ticker);
    if (!company) return;

    currentModalTicker = ticker;
    document.getElementById('modalTicker').textContent = `${company.ticker} - ${company.company_name || ticker}`;
    document.getElementById('modalSubtitle').textContent = `${company.sector || 'N/A'} • ${company.industry || 'N/A'}`;

    // Reset Tabs
    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.modal-tab-pane').forEach(p => p.classList.remove('active'));
    document.querySelector('.modal-tab[data-tab="tab-fundamentals"]').classList.add('active');
    document.getElementById('tab-fundamentals').classList.add('active');

    // Reset Thesis Pane
    document.getElementById('thesisOutput').innerHTML = `
        <div class="thesis-placeholder">
            Click the button above to generate a full Bull/Bear investment report for ${ticker} using Qwen 2.5.
        </div>
    `;

    // Render Fundamentals Tab
    document.getElementById('modalBody').innerHTML = `
        <div class="modal-sql-source">
            <span>💾</span> <span>Database: <strong>foundamental_sql.db</strong> (Table: <code>stock_fundamentals</code> • Ticker: <strong>${ticker}</strong>)</span>
        </div>
        <div class="modal-card">
            <h3>📈 Growth Dynamics</h3>
            <div class="metric"><span class="metric-label">Revenue Growth 1Y</span><span class="metric-value ${getValueClass(company.revenue_growth_1y)}">${formatPercent(company.revenue_growth_1y)}</span></div>
            <div class="metric"><span class="metric-label">Revenue CAGR 3Y</span><span class="metric-value ${getValueClass(company.revenue_cagr_3y)}">${formatPercent(company.revenue_cagr_3y)}</span></div>
            <div class="metric"><span class="metric-label">EPS Growth 1Y</span><span class="metric-value ${getValueClass(company.eps_growth_1y)}">${formatPercent(company.eps_growth_1y)}</span></div>
            <div class="metric"><span class="metric-label">EPS CAGR 3Y</span><span class="metric-value ${getValueClass(company.eps_cagr_3y)}">${formatPercent(company.eps_cagr_3y)}</span></div>
        </div>
        <div class="modal-card">
            <h3>💰 Profitability & Returns</h3>
            <div class="metric"><span class="metric-label">Gross Margin</span><span class="metric-value ${getValueClass(company.gross_margin)}">${formatPercent(company.gross_margin)}</span></div>
            <div class="metric"><span class="metric-label">Operating Margin</span><span class="metric-value ${getValueClass(company.operating_margin)}">${formatPercent(company.operating_margin)}</span></div>
            <div class="metric"><span class="metric-label">Net Margin</span><span class="metric-value ${getValueClass(company.net_margin)}">${formatPercent(company.net_margin)}</span></div>
            <div class="metric"><span class="metric-label">ROE</span><span class="metric-value ${getValueClass(company.roe)}">${formatPercent(company.roe)}</span></div>
            <div class="metric"><span class="metric-label">ROIC</span><span class="metric-value ${getValueClass(company.roic)}">${formatPercent(company.roic)}</span></div>
        </div>
        <div class="modal-card">
            <h3>💵 Cash Flow & Solvency</h3>
            <div class="metric"><span class="metric-label">FCF Margin</span><span class="metric-value ${getValueClass(company.fcf_margin)}">${formatPercent(company.fcf_margin)}</span></div>
            <div class="metric"><span class="metric-label">FCF / Net Income</span><span class="metric-value">${formatNumber(company.fcf_to_net_income)}</span></div>
            <div class="metric"><span class="metric-label">Net Debt / EBITDA</span><span class="metric-value ${getValueClass(company.net_debt_to_ebitda, 'net_debt_to_ebitda')}">${formatNumber(company.net_debt_to_ebitda)}</span></div>
            <div class="metric"><span class="metric-label">P/E Ratio</span><span class="metric-value ${getValueClass(company.pe_ratio, 'pe_ratio')}">${formatNumber(company.pe_ratio)}</span></div>
        </div>
    `;

    document.getElementById('detailModal').classList.add('active');

    // Fetch News for this ticker
    loadModalNews(ticker);
}

async function loadModalNews(ticker) {
    const newsListEl = document.getElementById('modalNewsList');
    const newsCountEl = document.getElementById('modalNewsCount');
    newsListEl.innerHTML = '<div class="loading-spinner">Fetching latest news catalysts...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/news/${ticker}`);
        if (res.ok) {
            const data = await res.json();
            const newsItems = data.news || [];
            newsCountEl.textContent = newsItems.length;

            if (newsItems.length === 0) {
                newsListEl.innerHTML = '<div class="no-data">No recent news catalysts recorded for this ticker.</div>';
                return;
            }

            newsListEl.innerHTML = newsItems.map(item => `
                <div class="modal-news-card">
                    <div class="news-headline">${item.title}</div>
                    <div class="news-meta">Source: <strong>${item.publisher}</strong> • ${item.publish_time}</div>
                    <div class="news-summary">${item.summary || 'No summary text available.'}</div>
                    ${item.link ? `<a href="${item.link}" target="_blank" rel="noopener" class="news-link">Read Full Article ↗</a>` : ''}
                </div>
            `).join('');
            return;
        }
    } catch (e) {
        console.error('Error fetching modal news:', e);
    }

    newsListEl.innerHTML = '<div class="no-data">Failed to load news. Ensure backend API is active.</div>';
}

async function generateModalThesis() {
    if (!currentModalTicker) return;

    const outputEl = document.getElementById('thesisOutput');
    const btn = document.getElementById('generateThesisBtn');

    btn.disabled = true;
    btn.textContent = '⏳ Analyzing Fundamentals & News with Qwen 2.5...';
    outputEl.innerHTML = '<div class="loading-spinner">Generating institutional investment thesis via 3-Stage RAG...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/rag/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                ticker: currentModalTicker,
                use_reranker: true 
            })
        });

        if (res.ok) {
            const data = await res.json();
            const formattedThesis = formatMarkdownText(data.thesis);
            
            outputEl.innerHTML = `
                <div class="ai-response-meta" style="margin-bottom: 14px;">
                    <span class="meta-tag ticker-tag">🎯 ${data.ticker} Research Memo</span>
                    ${data.reranked ? '<span class="meta-tag rerank-tag">Cross-Encoder Reranked</span>' : ''}
                    <span class="meta-tag model-tag">🧠 ${data.model || 'qwen2.5:7b'}</span>
                    ${data.latency_seconds ? `<span class="meta-tag latency-tag">⏱️ ${data.latency_seconds}s</span>` : ''}
                </div>
                <div class="thesis-content">
                    ${formattedThesis}
                </div>
                ${data.citations && data.citations.length ? `
                    <div class="citations-tray">
                        <div class="citations-title">Sources & Catalysts Evaluated (${data.citations.length}):</div>
                        ${data.citations.map(c => `
                            <a href="${c.link || '#'}" target="_blank" rel="noopener" class="citation-pill" title="${c.excerpt || ''}">
                                [${c.ticker}] ${c.title}
                                ${c.score !== null && c.score !== undefined ? `<span class="citation-score">★ ${(c.score * 100).toFixed(0)}%</span>` : ''}
                            </a>
                        `).join('')}
                    </div>
                ` : ''}
            `;
        } else {
            const err = await res.json().catch(() => ({}));
            outputEl.innerHTML = `<div class="ai-error-box"><div class="ai-error-title">Thesis Generation Failed</div><div class="ai-error-desc">${err.detail || 'Ensure Ollama is running (qwen2.5:7b) and server is active.'}</div></div>`;
        }
    } catch (e) {
        outputEl.innerHTML = `<div class="ai-error-box"><div class="ai-error-title">Connection Error</div><div class="ai-error-desc">${e.message}</div></div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '⚡ Generate Thesis with Qwen 2.5';
    }
}

// AI Chat Drawer Functionality
function toggleAiDrawer(open = null) {
    const drawer = document.getElementById('aiDrawer');
    const backdrop = document.getElementById('aiDrawerBackdrop');
    if (!drawer) return;

    const shouldOpen = open !== null ? open : !drawer.classList.contains('active');

    if (shouldOpen) {
        drawer.classList.add('active');
        if (backdrop) backdrop.classList.add('active');
        const input = document.getElementById('aiChatInput');
        if (input) setTimeout(() => input.focus(), 150);
    } else {
        drawer.classList.remove('active');
        if (backdrop) backdrop.classList.remove('active');
    }
}

function appendChatMessage(role, content, citations = [], metadata = {}) {
    const thread = document.getElementById('aiChatMessages');
    if (!thread) return null;

    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;

    const avatar = role === 'user' ? '👤' : '🤖';

    let metaHtml = '';
    if (metadata && (metadata.routed || metadata.reranked || metadata.latency)) {
        metaHtml = `
            <div class="ai-response-meta">
                ${metadata.routed ? '<span class="meta-tag router-tag" title="Query routed via RouterRetriever">⚡ Router Active</span>' : ''}
                ${metadata.reranked ? '<span class="meta-tag rerank-tag" title="Cross-Encoder Reranked">🎯 Reranked</span>' : ''}
                ${metadata.model ? `<span class="meta-tag model-tag">🧠 ${metadata.model}</span>` : ''}
                ${metadata.latency ? `<span class="meta-tag latency-tag">⏱️ ${metadata.latency}</span>` : ''}
            </div>
        `;
    }

    let citationsHtml = '';
    if (citations && citations.length > 0) {
        citationsHtml = `
            <div class="citations-tray">
                <div class="citations-title">Data & News Sources Evaluated (${citations.length}):</div>
                ${citations.map(c => `
                    <a href="${c.link || '#'}" target="_blank" rel="noopener" class="citation-pill" title="${c.excerpt || ''}">
                        <strong class="citation-ticker">[${c.ticker}]</strong> ${c.title}
                        ${c.score !== null && c.score !== undefined ? `<span class="citation-score">★ ${(c.score * 100).toFixed(0)}%</span>` : ''}
                    </a>
                `).join('')}
            </div>
        `;
    }

    msgDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${metaHtml}
            <div class="message-text">${content}</div>
            ${citationsHtml}
        </div>
    `;

    thread.appendChild(msgDiv);
    thread.scrollTop = thread.scrollHeight;
    return msgDiv;
}

function appendTerminalLog(bodyEl, msg, tag = 'INFO') {
    if (!bodyEl) return;
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const line = document.createElement('div');
    line.className = `terminal-line tag-${tag.toLowerCase()}`;
    line.innerHTML = `<span class="t-time">[${timeStr}]</span> <span class="t-msg">${escapeHtml(msg)}</span>`;
    bodyEl.appendChild(line);
    bodyEl.scrollTop = bodyEl.scrollHeight;

    const container = bodyEl.closest('.rag-terminal-container');
    if (container) {
        const countSpan = container.querySelector('.terminal-count');
        if (countSpan) countSpan.textContent = bodyEl.children.length;
    }
}

function toggleTerminalLogs(btn) {
    const container = btn.closest('.rag-terminal-container');
    if (!container) return;
    const body = container.querySelector('.terminal-logs-body');
    if (!body) return;
    const isCollapsed = body.classList.toggle('collapsed');
    btn.innerHTML = isCollapsed 
        ? `▸ View Logs (<span class="terminal-count">${body.children.length}</span>)` 
        : `▾ Hide Logs (<span class="terminal-count">${body.children.length}</span>)`;
}

async function sendAiQuery(queryText) {
    if (!queryText || !queryText.trim()) return;
    queryText = queryText.trim();

    const input = document.getElementById('aiChatInput');
    const sendBtn = document.getElementById('sendChatBtn');
    
    // 1. Append User Message
    appendChatMessage('user', `<p>${escapeHtml(queryText)}</p>`);
    if (input) input.value = '';
    
    // 2. Disable controls while processing
    if (input) input.disabled = true;
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<span>Streaming...</span>';
    }

    // 3. Append Assistant Placeholder with Live Terminal Console
    const assistantMsg = appendChatMessage(
        'assistant', 
        `
        <div class="rag-terminal-container">
            <div class="terminal-header">
                <div class="terminal-dots">
                    <span class="dot red"></span>
                    <span class="dot yellow"></span>
                    <span class="dot green"></span>
                    <span class="terminal-title">🖥️ Terminal Execution Logs</span>
                </div>
                <div class="terminal-actions">
                    <span class="terminal-status-badge running"><span class="spinner-inline-small"></span> Running</span>
                    <button type="button" class="terminal-toggle-btn" onclick="toggleTerminalLogs(this)">▾ Hide Logs (<span class="terminal-count">1</span>)</button>
                </div>
            </div>
            <div class="terminal-logs-body">
                <div class="terminal-line tag-init"><span class="t-time">[${new Date().toTimeString().split(' ')[0]}]</span> <span class="t-msg">🚀 Initializing 3-Stage Financial RAG Pipeline for query: "${escapeHtml(queryText)}"...</span></div>
            </div>
        </div>
        <div class="message-text streaming-text"></div>
        `
    );
    const contentDiv = assistantMsg ? assistantMsg.querySelector('.message-content') : null;
    const textDiv = contentDiv ? contentDiv.querySelector('.streaming-text') : null;
    const terminalBody = contentDiv ? contentDiv.querySelector('.terminal-logs-body') : null;
    const statusBadge = contentDiv ? contentDiv.querySelector('.terminal-status-badge') : null;

    let streamedTokens = '';
    let finalCitations = [];
    let finalMetadata = {};

    try {
        const streamUrl = `${API_BASE}/api/rag/stream?query=${encodeURIComponent(queryText)}&use_reranker=true&use_router=true`;
        const response = await fetch(streamUrl);

        if (!response.ok) {
            throw new Error(`Stream connection failed with status ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const eventBlocks = buffer.split('\n\n');
            buffer = eventBlocks.pop(); // keep partial block in buffer

            for (const block of eventBlocks) {
                const lines = block.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const rawJson = line.slice(6).trim();
                        if (!rawJson) continue;

                        try {
                            const event = JSON.parse(rawJson);

                            // Handle Step / Terminal Log Events
                            if (event.type === 'step') {
                                const logMsg = event.log_line || `[${(event.stage || 'step').toUpperCase()}] ${event.message || event.title}`;
                                appendTerminalLog(terminalBody, logMsg, event.stage || 'step');
                            }

                            // Handle Word-by-Word Streaming Tokens
                            else if (event.type === 'token') {
                                streamedTokens += event.content || '';
                                if (textDiv) {
                                    textDiv.innerHTML = formatMarkdownText(streamedTokens) + '<span class="streaming-cursor"></span>';
                                }
                                const thread = document.getElementById('aiChatMessages');
                                if (thread) thread.scrollTop = thread.scrollHeight;
                            }

                            // Handle Stream Completion
                            else if (event.type === 'done') {
                                finalCitations = event.citations || [];
                                finalMetadata = event.metadata || {};
                                
                                if (event.log_line) {
                                    appendTerminalLog(terminalBody, event.log_line, 'done');
                                }
                                if (statusBadge) {
                                    statusBadge.className = 'terminal-status-badge completed';
                                    statusBadge.textContent = `✅ Completed (${finalMetadata.latency_seconds || '1.2'}s)`;
                                }
                            }

                            // Handle Stream Errors
                            else if (event.type === 'error') {
                                if (terminalBody) {
                                    appendTerminalLog(terminalBody, `❌ Error: ${event.message}`, 'error');
                                }
                                throw new Error(event.message || 'Stream processing error');
                            }
                        } catch (parseErr) {
                            console.error('Error parsing SSE event:', parseErr, rawJson);
                        }
                    }
                }
            }
        }

        // Final UI Polish after streaming completes
        if (textDiv) {
            textDiv.innerHTML = formatMarkdownText(streamedTokens);
            textDiv.classList.remove('streaming-text');
        }

        // Render Citations Tray
        let citationsHtml = '';
        if (finalCitations && finalCitations.length > 0) {
            citationsHtml = `
                <div class="citations-tray">
                    <div class="citations-title">Data & News Sources Evaluated (${finalCitations.length}):</div>
                    ${finalCitations.map(c => `
                        <a href="${c.link || '#'}" target="_blank" rel="noopener" class="citation-pill" title="${escapeHtml(c.excerpt || '')}">
                            <strong class="citation-ticker">[${c.ticker}]</strong> ${escapeHtml(c.title || '')}
                            ${c.score !== null && c.score !== undefined ? `<span class="citation-score">★ ${(c.score * 100).toFixed(0)}%</span>` : ''}
                        </a>
                    `).join('')}
                </div>
            `;
        }

        // Render Execution Summary Badge under the message
        if (contentDiv) {
            const summaryHtml = `
                <div class="ai-response-meta" style="margin-top: 10px;">
                    <span class="meta-tag router-tag" title="Routed via RouterRetriever">⚡ ${finalMetadata.tool || 'RouterRetriever'}</span>
                    ${finalMetadata.reranked ? `<span class="meta-tag rerank-tag" title="Cross-Encoder Reranked">🎯 Reranked (${finalMetadata.top_k || 6} Chunks)</span>` : ''}
                    <span class="meta-tag model-tag">🧠 ${finalMetadata.model || 'qwen2.5:7b'}</span>
                    <span class="meta-tag latency-tag">⏱️ ${finalMetadata.latency_seconds || '1.0'}s</span>
                </div>
            `;
            const textWrap = contentDiv.querySelector('.message-text');
            if (textWrap) {
                textWrap.innerHTML = formatMarkdownText(streamedTokens);
            }
            contentDiv.insertAdjacentHTML('beforeend', summaryHtml + citationsHtml);
        }

    } catch (e) {
        console.error('Streaming Query Error:', e);
        if (contentDiv) {
            contentDiv.innerHTML = `
                <div class="ai-error-box">
                    <div class="ai-error-title">⚠️ AI Assistant Error</div>
                    <div class="ai-error-desc">${escapeHtml(e.message)}</div>
                    <div class="ai-error-hint">
                        <strong>Troubleshooting Tips:</strong>
                        <ul>
                            <li>Make sure Ollama is active: <code>ollama run qwen2.5:7b</code></li>
                            <li>Ensure FastAPI backend is running on port 8000: <code>python app.py</code></li>
                        </ul>
                    </div>
                </div>
            `;
        }
    } finally {
        if (input) {
            input.disabled = false;
            input.focus();
        }
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.innerHTML = '<span>Send</span> ➔';
        }
        const thread = document.getElementById('aiChatMessages');
        if (thread) thread.scrollTop = thread.scrollHeight;
    }
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatMarkdownText(text) {
    if (!text) return '';

    let html = text.trim();

    // 1. Escape HTML
    html = escapeHtml(html);

    // 2. Headings
    html = html.replace(/^#### (.*?)$/gm, '<h5 class="md-heading md-h5">$1</h5>');
    html = html.replace(/^### (.*?)$/gm, '<h4 class="md-heading md-h4">$1</h4>');
    html = html.replace(/^## (.*?)$/gm, '<h3 class="md-heading md-h3">$1</h3>');
    html = html.replace(/^# (.*?)$/gm, '<h2 class="md-heading md-h2">$1</h2>');

    // 3. Dividers
    html = html.replace(/^---$/gm, '<hr class="md-divider">');

    // 4. Bold & Italics
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // 5. Financial Verdicts Highlighting
    html = html.replace(/\b(BULLISH|STRONG BUY)\b/g, '<span class="verdict-pill verdict-bullish">$1</span>');
    html = html.replace(/\b(BEARISH|STRONG SELL)\b/g, '<span class="verdict-pill verdict-bearish">$1</span>');
    html = html.replace(/\b(NEUTRAL|HOLD)\b/g, '<span class="verdict-pill verdict-neutral">$1</span>');

    // 6. Code tags
    html = html.replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>');

    // 7. Bullet Lists & Numbered Lists
    html = html.replace(/^\s*[-*•]\s+(.*)$/gm, '<li class="md-list-item">$1</li>');
    html = html.replace(/^\s*(\d+)\.\s+(.*)$/gm, '<li class="md-list-item"><span class="md-list-num">$1.</span> $2</li>');

    // 8. Wrap lines in paragraphs or lists
    const lines = html.split('\n');
    let inList = false;
    let result = '';

    for (let rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            if (inList) {
                result += '</ul>';
                inList = false;
            }
        } else if (line.startsWith('<li')) {
            if (!inList) {
                result += '<ul class="md-list">';
                inList = true;
            }
            result += line;
        } else if (line.startsWith('<h') || line.startsWith('<hr')) {
            if (inList) {
                result += '</ul>';
                inList = false;
            }
            result += line;
        } else {
            if (inList) {
                result += '</ul>';
                inList = false;
            }
            result += `<p class="md-paragraph">${line}</p>`;
        }
    }
    if (inList) {
        result += '</ul>';
    }

    return result;
}

// Event Listeners Setup
document.addEventListener('DOMContentLoaded', () => {
    applyHeaderTooltips();

    document.getElementById('searchInput').addEventListener('input', renderData);
    document.getElementById('metricFilter').addEventListener('change', renderData);

    const sectorFilter = document.getElementById('sectorFilter');
    if (sectorFilter) {
        sectorFilter.addEventListener('change', () => {
            updateIndustryOptions();
            renderData();
        });
    }

    const industryFilter = document.getElementById('industryFilter');
    if (industryFilter) {
        industryFilter.addEventListener('change', renderData);
    }

    document.querySelectorAll('.column-filter-input, .column-filter-op').forEach(control => {
        control.addEventListener('input', renderData);
        control.addEventListener('change', renderData);
    });

    document.querySelectorAll('table th.sortable').forEach(th => {
        th.addEventListener('mouseenter', showHeaderTooltip);
        th.addEventListener('mousemove', moveHeaderTooltip);
        th.addEventListener('mouseleave', hideHeaderTooltip);

        th.addEventListener('click', (e) => {
            if (e.target.closest('.column-filter-wrap')) return;

            const targetTh = e.target.closest('th.sortable');
            if (!targetTh) return;

            const key = targetTh.dataset.sort;
            if (sortConfig.key === key) {
                sortConfig.direction = sortConfig.direction === 'asc' ? 'desc' : 'asc';
            } else {
                sortConfig.key = key;
                sortConfig.direction = 'asc';
            }
            renderTable();
        });
    });

    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentView = e.target.dataset.view;

            document.getElementById('tableView').style.display = currentView === 'table' ? '' : 'none';
            document.getElementById('cardView').style.display = currentView === 'card' ? '' : 'none';
            renderData();
        });
    });

    // Modal Tabs & Close
    document.querySelectorAll('.modal-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.modal-tab-pane').forEach(p => p.classList.remove('active'));

            e.target.classList.add('active');
            const targetPaneId = e.target.dataset.tab;
            document.getElementById(targetPaneId).classList.add('active');
        });
    });

    document.querySelector('.modal-close').addEventListener('click', () => {
        document.getElementById('detailModal').classList.remove('active');
    });

    document.getElementById('detailModal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('detailModal')) {
            document.getElementById('detailModal').classList.remove('active');
        }
    });

    document.getElementById('generateThesisBtn').addEventListener('click', generateModalThesis);

    // AI Drawer Events
    document.getElementById('openAiDrawerBtn').addEventListener('click', () => toggleAiDrawer(true));
    document.getElementById('floatingAiBtn').addEventListener('click', () => toggleAiDrawer(true));
    document.getElementById('closeAiDrawerBtn').addEventListener('click', () => toggleAiDrawer(false));
    document.getElementById('aiDrawerBackdrop').addEventListener('click', () => toggleAiDrawer(false));

    // Suggested Prompt Chips
    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.dataset.prompt;
            toggleAiDrawer(true);
            sendAiQuery(prompt);
        });
    });

    // Chat Form Submit
    document.getElementById('aiChatForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const input = document.getElementById('aiChatInput');
        sendAiQuery(input.value);
    });

    document.getElementById('aiChatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('aiChatForm').requestSubmit();
        }
    });

    document.getElementById('clearChatBtn').addEventListener('click', () => {
        document.getElementById('aiChatMessages').innerHTML = `
            <div class="chat-message assistant">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <p>Chat cleared. Ready for your next financial question!</p>
                </div>
            </div>
        `;
    });

    document.getElementById('refreshPipelineBtn').addEventListener('click', async () => {
        const btn = document.getElementById('refreshPipelineBtn');
        btn.disabled = true;
        btn.textContent = '🔄 Syncing All Data...';
        try {
            await fetch(`${API_BASE}/api/pipeline/refresh`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_fundamentals: true, refresh_news: true })
            });
            alert('Full pipeline sync started in background (Fundamentals + News + Vector DB)!');
        } catch (e) {
            alert('Failed to trigger refresh: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = '🔄 Sync All Data';
        }
    });

    document.getElementById('metricsToggle').addEventListener('click', () => {
        const content = document.getElementById('metricsContent');
        content.style.display = content.style.display === 'none' ? 'block' : 'none';
    });

    // Initial Load
    loadData();
});
