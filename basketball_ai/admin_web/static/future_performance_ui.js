(() => {
  function esc(value) {
    return String(value ?? '').replace(/[&<>'"]/g, ch => ({
      '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'
    }[ch]));
  }
  function cookie(name) {
    const part = document.cookie.split('; ').find(x => x.startsWith(name + '='));
    return part ? decodeURIComponent(part.split('=').slice(1).join('=')) : '';
  }
  async function request(path, options = {}) {
    const headers = { Accept: 'application/json', ...(options.headers || {}) };
    if ((options.method || 'GET').toUpperCase() !== 'GET') {
      const csrf = cookie('hm_ai_admin_csrf');
      if (csrf) headers['X-CSRF-Token'] = csrf;
    }
    const response = await fetch(path, { credentials: 'same-origin', ...options, headers });
    let body = {};
    try { body = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(body.detail || body.message || ('HTTP ' + response.status));
    return body;
  }
  function fmt(value, digits = 3) {
    return value === null || value === undefined || value === '' || Number.isNaN(Number(value))
      ? '—' : Number(value).toFixed(digits);
  }
  function fmtInt(value) {
    if (value === null || value === undefined) return '—';
    return new Intl.NumberFormat('it-IT', { maximumFractionDigits: 0 }).format(Number(value));
  }
  function pct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return (Number(value) * 100).toFixed(1) + '%';
  }
  function kpi(label, value, sub) {
    return '<div class="kpi-card"><div class="kpi-label">' + esc(label) +
      '</div><div class="kpi-value">' + esc(value) +
      '</div><div class="kpi-sub">' + sub + '</div></div>';
  }
  function installTemplate() {
    if (document.querySelector('#page-future-performance')) return;
    const holder = document.createElement('div');
    holder.innerHTML = "<template id=\"page-future-performance\"><section class=\"page\">\n  <div class=\"page-heading action-heading\">\n    <div><h1>Player Future Performance</h1>\n    <p>Modello predittivo multivariato season-ahead. Prevede le performance del giocatore nella stagione successiva, separato da Prediction Model e BB-Rating.</p></div>\n  </div>\n  <div id=\"future-performance-kpis\" class=\"kpi-grid four\"></div>\n  <div class=\"split-grid\">\n    <section class=\"panel\"><div class=\"panel-heading\"><h2>Stato modello</h2></div><div id=\"future-performance-status\"></div></section>\n    <section class=\"panel\"><div class=\"panel-heading\"><h2>Dataset &amp; finestra temporale</h2></div><div id=\"future-performance-dataset\"></div></section>\n  </div>\n  <section class=\"panel\"><div class=\"panel-heading\"><h2>Target predetti</h2><span class=\"muted-inline\">valori t+1 + incertezza OOS</span></div>\n    <div id=\"future-performance-targets\" class=\"table-shell\"></div>\n  </section>\n  <section class=\"panel\"><div class=\"panel-heading\"><h2>Walk-forward validation</h2><span class=\"muted-inline\">expanding OOS</span></div>\n    <div id=\"future-performance-backtest\" class=\"table-shell\"></div>\n  </section>\n  <section class=\"panel\"><div class=\"panel-heading\"><h2>Training job</h2></div><div id=\"future-performance-job\"></div></section>\n  <div id=\"future-performance-error\" class=\"microcopy\"></div>\n</section></template>";
    document.body.appendChild(holder.firstElementChild);
  }

  function mountPage() {
    const root = document.querySelector('#page-root');
    if (!root) return null;
    const existing = root.querySelector('#future-performance-kpis');
    if (existing) return existing.closest('.page');

    const page = document.createElement('section');
    page.className = 'page';
    page.innerHTML = \
`<div class="page-heading action-heading">
      <div><h1>Player Future Performance</h1>
      <p>Modello predittivo multivariato season-ahead. Prevede le performance del giocatore nella stagione successiva, separato da Prediction Model e BB-Rating.</p></div>
      <div class="button-row">
        <button id="future-performance-refresh" class="btn btn-secondary btn-small">Aggiorna</button>
        <button id="future-performance-train" class="btn btn-primary btn-small">Genera / aggiorna modello</button>
      </div>
    </div>
    <div id="future-performance-kpis" class="kpi-grid four"></div>
    <div class="split-grid">
      <section class="panel"><div class="panel-heading"><h2>Stato modello</h2></div><div id="future-performance-status"></div></section>
      <section class="panel"><div class="panel-heading"><h2>Dataset &amp; finestra temporale</h2></div><div id="future-performance-dataset"></div></section>
    </div>
    <section class="panel"><div class="panel-heading"><h2>Target predetti</h2><span class="muted-inline">valori t+1 + incertezza OOS</span></div>
      <div id="future-performance-targets" class="table-shell"></div>
    </section>
    <section class="panel"><div class="panel-heading"><h2>Walk-forward validation</h2><span class="muted-inline">expanding OOS</span></div>
      <div id="future-performance-backtest" class="table-shell"></div>
    </section>
    <section class="panel"><div class="panel-heading"><h2>Training job</h2></div><div id="future-performance-job"></div></section>
    <div id="future-performance-error" class="microcopy"></div>
  </div>\`;
    root.replaceChildren(page);
    return page;
  }

  let renderGeneration = 0;
  let activeController = null;

  async function render() {
    const generation = ++renderGeneration;
    if (activeController) activeController.abort();
    const controller = new AbortController();
    activeController = controller;
    mountPage();

    const pageRoot = document.querySelector('#page-root');
    const page = pageRoot ? pageRoot.querySelector('#future-performance-kpis')?.closest('.page') : null;
    if (!page) return;
    const q = selector => page.querySelector(selector);
    const isLive = () => generation === renderGeneration && page.isConnected;
    const setHtml = (selector, html) => {
      if (!isLive()) return;
      const node = q(selector);
      if (node) node.innerHTML = html;
    };
    const setText = (selector, value) => {
      if (!isLive()) return;
      const node = q(selector);
      if (node) node.textContent = value;
    };

    try {
      const [status, health] = await Promise.all([
        request('/admin-api/future-performance', { signal: controller.signal }),
        request('/admin-api/api-health', { signal: controller.signal })
      ]);
      if (!isLive()) return;

      const ds = status.dataset || {};
      const job = status.job || {};
      const runtime = health.future_performance || {};
      const ready = !!status.ready;
      const targets = status.targets || {};
      const folds = status.backtest?.folds || [];

      setHtml('#future-performance-kpis', [
        kpi('Versione', status.future_performance_version || '—', '<span>Future Performance</span>'),
        kpi('Feature contract', status.feature_version || '—', '<span>separate</span>'),
        kpi('Training pairs', fmtInt(ds.n_pairs), '<span>' + fmtInt(ds.n_players) + ' giocatori</span>'),
        kpi('Serving API', runtime.loaded ? 'Loaded' : 'Not loaded', runtime.loaded ? '<span class="dot green"></span>artifact caricato' : '<span class="dot yellow"></span>riavvio API necessario')
      ].join(''));

      setHtml('#future-performance-status',
        '<div class="model-state ' + (ready ? 'ok' : 'warning') + '">' +
          '<div class="model-state-mark">' + (ready ? '✓' : '△') + '</div><div><strong>' +
          (ready ? 'Future Performance serving ready' : 'Modello non ancora pronto') +
          '</strong><p>' +
          (ready ? 'Artifact fitted, target disponibili e walk-forward report presente.' : 'Esegui il training dal Control Center per creare l’artifact.') +
          '</p></div></div>' +
          '<div class="summary-list"><div class="summary-row"><span>API runtime version</span><strong>' + esc(runtime.version || '—') + '</strong></div>' +
          '<div class="summary-row"><span>API feature version</span><strong>' + esc(runtime.feature_version || '—') + '</strong></div></div>'
      );

      setHtml('#future-performance-dataset',
        '<div class="summary-list">' +
        '<div class="summary-row"><span>Pairs</span><strong>' + esc(fmtInt(ds.n_pairs)) + '</strong></div>' +
        '<div class="summary-row"><span>Players</span><strong>' + esc(fmtInt(ds.n_players)) + '</strong></div>' +
        '<div class="summary-row"><span>Target seasons</span><strong>' + esc((ds.training_target_seasons || []).join(', ') || '—') + '</strong></div>' +
        '<div class="summary-row"><span>Leagues</span><strong>' + esc((ds.leagues || []).join(', ') || '—') + '</strong></div>' +
        '<div class="summary-row"><span>Competitions</span><strong>' + esc((ds.competitions || []).join(', ') || '—') + '</strong></div>' +
        '</div>'
      );

      const targetRows = Object.entries(targets).map(([key, item]) => {
        const oos = item.oos || {};
        const uncertainty = item.uncertainty || {};
        return '<tr><td><strong>' + esc(item.label || key) + '</strong><br><span class="muted-inline">' + esc(key) + '</span></td>' +
          '<td>' + esc(fmtInt(item.n_training)) + '</td>' +
          '<td>' + esc(fmt(oos.rmse_mean, 4)) + '</td>' +
          '<td>' + esc(fmt(oos.mae_mean, 4)) + '</td>' +
          '<td>' + esc(fmt(uncertainty.p50, 4)) + '</td>' +
          '<td>' + esc(fmt(uncertainty.p90, 4)) + '</td></tr>';
      }).join('');
      setHtml('#future-performance-targets',
        '<table><thead><tr><th>Target</th><th>Train N</th><th>OOS RMSE</th><th>OOS MAE</th><th>P50 abs err</th><th>P90 abs err</th></tr></thead><tbody>' +
        (targetRows || '<tr><td colspan="6" class="muted-inline">Nessun target disponibile.</td></tr>') +
        '</tbody></table>'
      );

      const foldRows = folds.map(fold =>
        '<tr><td><strong>' + esc(fold.target_season) + '</strong></td><td>' +
        esc(fold.train_through_target) + '</td><td>' + esc(fmtInt(fold.n_test_rows)) +
        '</td><td>' + esc(Object.keys(fold.targets || {}).length) + '</td></tr>'
      ).join('');
      setHtml('#future-performance-backtest',
        '<table><thead><tr><th>Target season</th><th>Train through</th><th>OOS rows</th><th>Targets validated</th></tr></thead><tbody>' +
        (foldRows || '<tr><td colspan="4" class="muted-inline">Nessun fold OOS disponibile.</td></tr>') +
        '</tbody></table>'
      );

      const running = job.status === 'running' || job.status === 'queued';
      setHtml('#future-performance-job',
        '<div class="job-head"><div><div class="job-status">' + esc(job.stage || 'Ready') + '</div>' +
        '<div class="job-message">' + esc(job.error || job.message || 'Nessun training in corso.') + '</div></div>' +
        '<span class="registry-badge ' + (running ? 'candidate' : '') + '">' + esc(job.status || 'idle') + '</span></div>' +
        '<div class="progress-track"><div class="progress-bar" style="width:' +
        Math.max(0, Math.min(100, Number(job.progress) || 0)) + '%"></div></div>'
      );

      const trainButton = q('#future-performance-train');
      const refreshButton = q('#future-performance-refresh');
      if (!trainButton || !refreshButton) return;
      trainButton.disabled = running || !status.active_profile;
      trainButton.textContent = running ? 'Training in corso…' : 'Genera / aggiorna modello';
      trainButton.onclick = async () => {
        trainButton.disabled = true;
        try {
          await request('/admin-api/future-performance/train', { method:'POST', body: JSON.stringify({}), signal: controller.signal });
          if (isLive()) await render();
        } catch (error) {
          if (error?.name === 'AbortError') return;
          setText('#future-performance-error', error.message);
          if (isLive()) trainButton.disabled = false;
        }
      };
      refreshButton.onclick = () => render().catch(error => {
        if (error?.name !== 'AbortError') setText('#future-performance-error', error.message);
      });
      if (running && isLive()) {
        clearTimeout(App.futurePerformanceTimer);
        App.futurePerformanceTimer = setTimeout(() => {
          if (isLive()) render().catch(() => {});
        }, 2200);
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      setText('#future-performance-error', error.message);
    }
  }

  window.renderFuturePerformancePage = render;

  function activateNavButton(button) {
    document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item === button));
  }
  function open() {
    const button = document.querySelector('[data-page="future-performance"]');
    if (!button) return;
    activateNavButton(button);
    render().catch(error => {
      const root = document.querySelector('#page-root');
      if (root) root.innerHTML = '<div class="empty-state"><strong>Future Performance</strong><p>' + esc(error.message) + '</p></div>';
    });
  }
  function install() {
    if (window.__hmFuturePerformanceInstalled) return;
    window.__hmFuturePerformanceInstalled = true;
    installTemplate();
    const navigation = document.querySelector('.sidebar-nav');
    if (!navigation) return;
    if (!document.querySelector('[data-page="future-performance"]')) {
      const bb = document.querySelector('[data-page="bb-rating"]');
      const button = document.createElement('button');
      button.className = 'nav-item';
      button.dataset.page = 'future-performance';
      button.innerHTML = '<span class="nav-icon"><i class="ph ph-crystal-ball"></i></span><span>Future Performance</span>';
      if (bb) bb.after(button); else navigation.appendChild(button);
      button.addEventListener('click', event => { event.preventDefault(); open(); });
    }
    document.addEventListener('click', event => {
      const target = event.target.closest?.('[data-page="future-performance"]');
      if (target && !target.matches('.nav-item')) {
        event.preventDefault();
        open();
      }
    });
    const labelObserver = new MutationObserver(() => {
      document.querySelectorAll('[data-page="future-performance"]').forEach(item => {
        if (!item.classList.contains('nav-item')) item.textContent = 'Apri Future Performance';
      });
    });
    labelObserver.observe(document.body, { childList: true, subtree: true });
    document.querySelectorAll('[data-page="future-performance"]').forEach(item => {
      if (!item.classList.contains('nav-item')) item.textContent = 'Apri Future Performance';
    });
  }
  window.addEventListener('DOMContentLoaded', install);
  install();
})();