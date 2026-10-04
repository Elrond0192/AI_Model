const App={user:null,page:'overview',overview:null,trainingTimer:null,bbRatingTimer:null,futurePerformanceTimer:null,controllers:new Set()};
const $=(s,r=document)=>r.querySelector(s);const $$=(s,r=document)=>[...r.querySelectorAll(s)];
const root=$('#page-root');

function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function cookie(name){const p=document.cookie.split('; ').find(x=>x.startsWith(name+'='));return p?decodeURIComponent(p.split('=').slice(1).join('=')):'';}
function fmt(v,d=3){if(v===null||v===undefined||v===''||Number.isNaN(Number(v)))return '—';return Number(v).toFixed(d);}
function fmtInt(v){if(v===null||v===undefined)return '—';return new Intl.NumberFormat('it-IT',{notation:Number(v)>=1000000?'compact':'standard',maximumFractionDigits:1}).format(Number(v));}
function pct(v){if(v===null||v===undefined||Number.isNaN(Number(v)))return '—';return `${(Number(v)*100).toFixed(1)}%`;}
function shortId(v){const s=String(v||'');return s.length>18?`${s.slice(0,12)}…${s.slice(-5)}`:(s||'Not promoted');}
function toast(message,type='success'){const el=$('#toast');el.textContent=message;el.className=`toast show ${type}`;clearTimeout(el._timer);el._timer=setTimeout(()=>el.className='toast',3600);}
function loading(){root.innerHTML='<div class="loading"><div><div class="spinner"></div>Caricamento…</div></div>';}
function empty(icon,title,copy,button='',page=''){return `<div class="empty-state"><div class="empty-icon">${icon}</div><strong>${esc(title)}</strong><p>${esc(copy)}</p>${button?`<button class="btn btn-primary btn-small jump" data-page="${esc(page)}">${esc(button)}</button>`:''}</div>`;}
function summaryRows(rows){return `<div class="summary-list">${rows.map(([a,b])=>`<div class="summary-row"><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`).join('')}</div>`;}

async function api(path,options={}){
  const headers={'Accept':'application/json',...(options.headers||{})};
  if(options.body!==undefined){headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
  if((options.method||'GET').toUpperCase()!=='GET'){const token=cookie('hm_ai_admin_csrf');if(token)headers['X-CSRF-Token']=token;}
  const controller=new AbortController();
  App.controllers.add(controller);
  const fetchOptions={credentials:'same-origin',...options,headers};
  if(!options.signal)fetchOptions.signal=controller.signal;
  try{
    const res=await fetch(path,fetchOptions);
    let data={};try{data=await res.json();}catch(_){data={};}
    if(res.status===401){showLogin();throw new Error('Sessione scaduta');}
    if(!res.ok)throw new Error(data.detail||data.message||`HTTP ${res.status}`);
    return data;
  }finally{
    App.controllers.delete(controller);
  }
}

function showLogin(){App.controllers.forEach(controller=>controller.abort());App.controllers.clear();App.user=null;$('#app-shell').classList.add('hidden');$('#login-screen').classList.remove('hidden');}
function showApp(user){App.user=user;$('#login-screen').classList.add('hidden');$('#app-shell').classList.remove('hidden');$('#user-name').textContent=user.username;$('#user-role').textContent=user.role==='admin'?'Administrator':user.role;$('#user-avatar').textContent=(user.username||'A')[0].toUpperCase();$$('.admin-only').forEach(el=>el.classList.toggle('hidden',user.role!=='admin'));}
function updateClock(){const d=new Date();$('#top-utc').textContent=d.toISOString().replace('T',' ').slice(0,19);}
setInterval(updateClock,1000);updateClock();

function activateNav(page){$$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.page===page));}
function mountTemplate(page){const t=$(`#page-${page}`);if(!t){root.innerHTML='<div class="loading">Pagina non disponibile</div>';return;}root.replaceChildren(t.content.cloneNode(true));$$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));}
async function navigate(page){
  App.controllers.forEach(controller=>controller.abort());
  App.controllers.clear();
  clearTimeout(App.trainingTimer);
  clearTimeout(App.bbRatingTimer);
  clearTimeout(App.futurePerformanceTimer);
  App.page=page;
  activateNav(page);
  mountTemplate(page);
  try{
    if(page==='overview')await renderOverview();
    if(page==='data')await renderData();
    if(page==='training')await renderTraining();
    if(page==='backtests')await renderBacktests();
    if(page==='bb-rating')await renderBBRating();
    if(page==='future-performance')await renderFuturePerformancePage();
    if(page==='scenarios')await renderScenarios();
    if(page==='registry')await renderRegistry();
    if(page==='health')await renderHealth();
    if(page==='diagnostics')await renderDiagnostics();
    if(page==='audit')await renderAudit();
    if(page==='settings')await renderSettings();
  }catch(e){if(e?.name!=='AbortError')toast(e.message,'error');}
}
function kpi(icon,label,value,sub='',tone=''){return `<div class="kpi-card"><div class="kpi-icon ${tone}">${icon}</div><div class="kpi-label">${esc(label)}</div><div class="kpi-value ${tone==='green'?'online':''}">${esc(value)}</div><div class="kpi-sub">${sub}</div></div>`;}
function updateTop(o){$('#top-db').textContent=o.database||'not selected';$('#top-model').textContent=shortId(o.production?.run_id);const bb=o.bb_rating||{};$('#top-bb-rating').textContent=bb.bb_rating_version?'v'+esc(bb.bb_rating_version):'—';$('#top-api').textContent=o.api?.online?'● V2':'● OFFLINE';$('#top-api').className=o.api?.online?'meta-ok':'';}

async function renderOverview(){
  const o=await api('/admin-api/overview');App.overview=o;updateTop(o);
  const s=o.summary||{},bb=o.bb_rating||{},prod=o.production||{},cand=o.candidate||{},q=o.candidate_quality||{};
  const allReady=(o.api?.online&&o.data_loaded&&bb.ready&&!!prod.run_id);
  const dot=$('#overview-system-dot');const status=$('#overview-system-status');
  if(dot)dot.className='hero-status-dot '+(allReady?'online':'warning');
  if(status)status.textContent=allReady?'SYSTEM OPERATIONAL':'ACTION REQUIRED';
  const systemDot=$('#system-dot'),systemLabel=$('#system-label');
  if(systemDot)systemDot.className='system-dot '+(allReady?'online':'warning');
  if(systemLabel)systemLabel.textContent=allReady?'SYSTEM OPERATIONAL':'SYSTEM ATTENTION';

  $('#overview-models').innerHTML=[
    ['Prediction Model',prod.run_id?'Production':'Not promoted',prod.run_id?shortId(prod.run_id):'No active candidate','registry','blue'],
    ['BB-Rating',bb.ready?'Serving ready':'Needs verification',bb.bb_rating_version?'v'+bb.bb_rating_version:'—','bb-rating','violet'],
    ['Future Performance','Independent model','Season-ahead forecasts','future-performance','cyan'],
    ['Metric Rating','Analytical layer','Metric-level strengths','diagnostics','green']
  ].map(([name,state,meta,page,tone])=>`<button class="model-strip-card ${tone} jump" data-page="${page}"><span class="model-strip-icon"></span><span class="model-strip-copy"><strong>${esc(name)}</strong><small>${esc(state)}</small><em>${esc(meta)}</em></span><i class="ph ph-arrow-up-right"></i></button>`).join('');

  $('#overview-kpis').innerHTML=[
    kpi('▤','Database',o.database||'Not selected',`<span class="dot ${o.data_loaded?'green':'yellow'}"></span>${o.data_loaded?'Connected and loaded':'Needs setup'}`),
    kpi('▥','Player observations',fmtInt(s.player_rows),'<span>player / competition rows</span>','green'),
    kpi('◉','Competitions',String((s.competitions||[]).length||'—'),'<span>loaded</span>','violet'),
    kpi('⬡','Production',prod.run_id?'Active':'Not promoted',prod.run_id?`<span class="dot green"></span>${esc(shortId(prod.run_id))}`:'<span class="dot red"></span>Promotion required','orange'),
    kpi('◆','BB-Rating',bb.bb_rating_version?'v'+bb.bb_rating_version:'—',bb.ready?'<span class="dot green"></span>Serving ready':'<span class="dot yellow"></span>Verify artifact','violet'),
    kpi('⌘','API',o.api?.online?'Online':'Offline',`<span class="dot ${o.api?.online?'green':'red'}"></span>V2 inference`,'green')
  ].join('');

  $('#readiness-body').innerHTML=(o.readiness||[]).map(r=>{
    const page=r.action==='data'?'data':r.action;
    const labels={registry:'Open registry',training:'Open training','bb-rating':'Open BB-Rating','future-performance':'Open Future Performance',data:'Open data'};
    return `<tr><td><strong>${esc(r.component)}</strong></td><td><span class="status ${r.status}"><span class="status-dot">${r.status==='ok'?'✓':r.status==='danger'?'×':'△'}</span>${esc(r.label)}</span></td><td>${esc(r.detail)}</td><td class="right"><button class="btn btn-small btn-secondary readiness-action" data-page="${esc(page)}">${esc(labels[page]||'Open')}</button></td></tr>`;
  }).join('');
  $$('.readiness-action').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));

  $('#overview-registry').innerHTML=(prod.run_id||cand.run_id)?summaryRows([
    ['Production',prod.run_id?shortId(prod.run_id):'—'],
    ['Candidate',cand.run_id?shortId(cand.run_id):'—'],
    ['Candidate RMSE',cand.overall_rmse!==undefined?fmt(cand.overall_rmse):'—']
  ]):empty('⬡','No model registered','Run training to generate the first candidate.','Open training','training');

  $('#overview-backtest').innerHTML=q.rmse!==null&&q.rmse!==undefined?summaryRows([
    ['RMSE',fmt(q.rmse)],['MAE',fmt(q.mae)],['Coverage',pct(q.coverage)],['OOS folds',String(q.seasons||0)]
  ]):empty('▤','No backtest available','Run training to generate a candidate and its walk-forward report.','Open training','training');

  const quality=[['RMSE',fmt(q.rmse)],['MAE',fmt(q.mae)],['R²',fmt(q.r2)],['Coverage',pct(q.coverage)],['Seasons',q.seasons??'—'],['Competitions',q.competitions??'—']];
  $('#candidate-quality').innerHTML=quality.map(([a,b])=>`<div class="quality-card"><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`).join('');
  $$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));
}
async function renderData(){
  async function loadSnapshots(){const data=await api('/admin-api/snapshots'),active=data.active?.snapshot_id,items=data.snapshots||[];$('#snapshots-list').innerHTML=items.length?items.map(s=>`<div class="profile-card ${active===s.snapshot_id?'active':''}"><div class="profile-top"><div class="profile-name">${esc(s.snapshot_id)}</div>${active===s.snapshot_id?'<span class="registry-badge">attivo</span>':''}</div><div class="profile-meta"><span>${esc(s.profile||'—')}</span><span>${esc((s.seasons||[]).length?`${s.seasons[0]}–${s.seasons.at(-1)}`:'—')}</span><span>${fmtInt(s.player_rows)} righe giocatore</span><span>SHA ${esc(String(s.sha256||'').slice(0,12))}</span></div><div class="button-row"><button class="btn btn-primary btn-small snapshot-activate" data-id="${esc(s.snapshot_id)}" ${active===s.snapshot_id?'disabled':''}>Usa per il training</button></div></div>`).join(''):empty('◫','Nessuno storico pronto','Nel riquadro del profilo, verifica la connessione e scegli “Prepara storico modello”.');$$('.snapshot-activate').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;try{const result=await api(`/admin-api/snapshots/${encodeURIComponent(b.dataset.id)}/activate`,{method:'POST',body:{}});toast(`Storico ${result.snapshot?.snapshot_id||b.dataset.id} attivato`);await loadSnapshots();}catch(e){toast(e.message,'error');b.disabled=false;}}));}
  async function loadProfiles(){
    const data=await api('/admin-api/profiles');$('#top-db').textContent=data.active||'not selected';
    const entries=Object.entries(data.profiles||{});$('#profiles-list').innerHTML=entries.length?entries.map(([name,p])=>`<div class="profile-card ${data.active===name?'active':''}"><div class="profile-top"><div class="profile-name">${esc(name)}</div>${data.active===name?'<span class="registry-badge">attivo</span>':''}</div><div class="profile-meta"><span>${esc(p.host||'—')}:${esc(p.port||5432)}</span><span>${esc(p.database||'—')}</span><span>${esc(p.user||'—')}</span><span>${esc(p.source_schema||'AI_Source')} → ${esc(p.ai_schema||'AI')}</span></div><p class="microcopy">Il primo pulsante controlla PostgreSQL. Il secondo salva lo storico core per il training: non esegue training e non legge PBP/lineup.</p><div class="button-row"><button class="btn btn-secondary btn-small profile-test" data-name="${esc(name)}">1. Verifica database</button><button class="btn btn-primary btn-small profile-load" data-name="${esc(name)}">2. Prepara storico modello</button></div></div>`).join(''):empty('▤','Nessun profilo configurato','Crea il profilo PostgreSQL production nel pannello a destra.');
    $$('.profile-test').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;try{const r=await api(`/admin-api/profiles/${encodeURIComponent(b.dataset.name)}/test`,{method:'POST',body:{}});toast(r.message);}catch(e){toast(e.message,'error');}finally{b.disabled=false;}}));
    $$('.profile-load').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;b.textContent='Preparazione storico…';try{const r=await api(`/admin-api/profiles/${encodeURIComponent(b.dataset.name)}/load`,{method:'POST',body:{}});toast(`Storico pronto: ${fmtInt(r.summary.player_rows)} righe core`);await Promise.all([loadProfiles(),loadSnapshots()]);}catch(e){toast(e.message,'error');}finally{b.disabled=false;b.textContent='2. Prepara storico modello';}}));
  }
  async function loadLifecycle(){const data=await api('/admin-api/season-lifecycle'),items=data.seasons||[],labels={unclassified:'Da classificare',in_progress:'In corso',complete:'Completa',locked:'Bloccata'};$('#season-lifecycle-list').innerHTML=items.length?items.map(item=>`<div class="profile-card"><div class="profile-top"><div class="profile-name">Stagione ${esc(item.season)}</div><span class="registry-badge ${item.status==='complete'?'':'candidate'}">${esc(labels[item.status]||item.status)}</span></div><div class="button-row"><select class="season-status" data-season="${esc(item.season)}"><option value="in_progress" ${item.status==='in_progress'?'selected':''}>In corso</option><option value="complete" ${item.status==='complete'?'selected':''}>Completa</option><option value="locked" ${item.status==='locked'?'selected':''}>Bloccata</option></select><button class="btn btn-secondary btn-small season-status-save" data-season="${esc(item.season)}">Salva stato</button></div></div>`).join(''):empty('◫','Prepara prima uno snapshot','Le stagioni disponibili appariranno dopo la preparazione dello storico.');$$('.season-status-save').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;try{const status=$(`.season-status[data-season="${b.dataset.season}"]`).value;await api('/admin-api/season-lifecycle',{method:'POST',body:{season:Number(b.dataset.season),status}});toast(`Stagione ${b.dataset.season}: stato salvato`);await loadLifecycle();}catch(e){toast(e.message,'error');b.disabled=false;}}));}
  $('#refresh-profiles').addEventListener('click',loadProfiles);$('#refresh-snapshots').addEventListener('click',loadSnapshots);$('#refresh-lifecycle').addEventListener('click',loadLifecycle);$('#profile-form').addEventListener('submit',async ev=>{ev.preventDefault();const form=new FormData(ev.currentTarget),body=Object.fromEntries(form.entries());body.port=Number(body.port);const btn=$('button[type="submit"]',ev.currentTarget);btn.disabled=true;try{await api('/admin-api/profiles',{method:'POST',body});toast('Profilo salvato');ev.currentTarget.querySelector('[name=password]').value='';await loadProfiles();}catch(e){toast(e.message,'error');}finally{btn.disabled=false;}});await Promise.all([loadProfiles(),loadSnapshots(),loadLifecycle()]);
}

async function renderTraining(){
  const d=await api('/admin-api/training'),s=d.summary||{},ts=d.training_summary||s,j=d.job||{};
  const datasetLoaded=s.player_rows!==undefined&&s.player_rows!==null;
  const excluded=d.excluded_in_progress_seasons||[];
  const blocked=d.blocking_seasons||[];
  const seasonSub=excluded.length?`<span>${esc(excluded.join(', '))} In corso escluse</span>`:'<span>solo stagioni Complete</span>';
  $('#training-kpis').innerHTML=[
    kpi('▤','Snapshot training',d.snapshot?.snapshot_id?shortId(d.snapshot.snapshot_id):'—',`<span class="dot ${datasetLoaded?'green':'yellow'}"></span>${datasetLoaded?'Storico pronto':'Prepara lo storico in Dati e snapshot'}`),
    kpi('◫','Training seasons',ts.season_min?`${ts.season_min}–${ts.season_max}`:'—',seasonSub,'green'),
    kpi('◉','Competitions',String((ts.competitions||[]).length||'—'),'<span>observed</span>','violet'),
    kpi('▥','Training observations',fmtInt(ts.player_rows),blocked.length?`<span class="dot red"></span>Bloccato: ${esc(blocked.join(', '))}`:'<span>player rows dopo filtro lifecycle</span>','orange')
  ].join('');
  $('#training-contract').innerHTML=(d.contract||[]).map(r=>`<tr><td><strong>${esc(r.stage)}</strong></td><td>${esc(r.seasons)}</td><td>${esc(r.purpose)}</td></tr>`).join('');
  $('#training-config').innerHTML=Object.entries(d.configuration||{}).map(([a,b])=>`<div class="config-item"><span>${esc(a.replaceAll('_',' '))}</span><strong>${esc(Array.isArray(b)?b.join(', '):b)}</strong></div>`).join('');
  const job=$('#training-job');
  const idleMessage=!datasetLoaded&&j.status==='idle'?'Prima prepara e attiva uno snapshot nella pagina Dati e snapshot. Il training non carica dati automaticamente.':(j.error||j.message||'');
  job.innerHTML=`<div class="job-head"><div><div class="job-status">${esc(j.stage||'Ready')}</div><div class="job-message">${esc(idleMessage)}</div></div><span class="registry-badge ${j.status==='running'?'candidate':''}">${esc(j.status||'idle')}</span></div><div class="progress-track"><div class="progress-bar" style="width:${Math.max(0,Math.min(100,Number(j.progress)||0))}%"></div></div>${j.run_id?`<div class="job-message">Run ID: ${esc(j.run_id)}</div>`:''}`;
  const start=$('#start-training');
  start.disabled=!d.can_start;
  start.addEventListener('click',async()=>{
    const originalText=start.textContent;
    start.disabled=true;
    try{
      if(!datasetLoaded||!d.snapshot?.snapshot_id)throw new Error('Prepara e attiva prima uno snapshot in Dati e snapshot');
      start.textContent='Controllo snapshot…';
      const loaded={summary:s,snapshot:d.snapshot};
      const seasons=d.training_seasons||[];
      const blocked=d.blocking_seasons||[];
      if(blocked.length)throw new Error(`Stagioni bloccanti: ${blocked.join(', ')}.`);
      if(seasons.length<5)throw new Error(`Servono almeno cinque stagioni Complete per il training. Disponibili: ${seasons.length}.`);
      start.textContent='Avvio training…';
      await api('/admin-api/training/start',{method:'POST',body:{}});
      toast(`Training avviato sullo snapshot ${loaded.snapshot?.snapshot_id||'attivo'}`);
      await renderTraining();
    }catch(e){
      toast(e.message,'error');
      start.disabled=false;
      start.textContent=originalText;
    }
  });
  if(j.status==='running'&&App.page==='training')App.trainingTimer=setTimeout(()=>renderTraining().catch(e=>toast(e.message,'error')),2200);
}

async function renderBBRating(){
  const data=await api('/admin-api/bb-rating');
  const health=await api('/admin-api/api-health');
  const u=data.uncertainty||{},ds=data.dataset||{},v=data.validation||{},f=data.files||{},j=data.job||{},apiBB=health.bb_rating||{};
  $('#top-bb-rating').textContent=data.bb_rating_version?'v'+esc(data.bb_rating_version):'—';

  $('#bb-rating-kpis').innerHTML=[
    kpi('◆','BB-Rating',data.bb_rating_version?'v'+data.bb_rating_version:'—',data.ready?'<span class="dot green"></span>Ready':'<span class="dot yellow"></span>Artifact da verificare','violet'),
    kpi('◎','Calibration',data.calibration_version?'v'+data.calibration_version:'—',data.ready?'<span class="dot green"></span>Report disponibile':'<span class="dot yellow"></span>Non disponibile','orange'),
    kpi('▥','Dataset',fmtInt(ds.rows),ds.players!=null?'<span>'+fmtInt(ds.players)+' giocatori</span>':'<span>righe calibration</span>','green'),
    kpi('◌','Uncertainty',u.status||'—',apiBB.uncertainty_loaded?'<span class="dot green"></span>Loaded by API':'<span class="dot red"></span>Not loaded','orange')
  ].join('');

  const readyClass=data.ready?'ok':'warning';
  $('#bb-rating-status').innerHTML='<div class="model-state '+readyClass+'"><div class="model-state-mark">'+(data.ready?'✓':'△')+'</div><div><strong>'+
    (data.ready?'BB-Rating serving ready':'BB-Rating artifact da verificare')+
    '</strong><p>'+(data.ready?'Report di calibrazione e serving artifact presenti e coerenti.':'Il Control Center non trova una coppia completa report + serving artifact.')+
    '</p></div></div>';

  const runtimeCal=apiBB.runtime_calibration_version||'—';
  const runtimeBb=apiBB.runtime_bb_rating_version||'—';
  $('#bb-rating-runtime').innerHTML=summaryRows([
    ['BB-Rating version',data.bb_rating_version||'—'],
    ['Calibration version',data.calibration_version||'—'],
    ['API engine',apiBB.loaded?'Loaded':'Not loaded'],
    ['API uncertainty',apiBB.uncertainty_loaded?'Loaded':'Not loaded'],
    ['Runtime BB-Rating',runtimeBb],
    ['Runtime calibration',runtimeCal],
    ['Source contract',ds.source_contract||'—'],
    ['Active DB profile',data.active_profile||'—']
  ]);

  $('#bb-rating-dataset').innerHTML=summaryRows([
    ['Rows',fmtInt(ds.rows)],
    ['Players',fmtInt(ds.players)],
    ['Seasons',ds.season_min!=null&&ds.season_max!=null?ds.season_min+'–'+ds.season_max:'—'],
    ['Competitions',(ds.competitions||[]).join(', ')||'—'],
    ['Leagues',String((ds.leagues||[]).length||'—')],
    ['Score available',pct(v.score_available_rate)],
    ['Primary context',pct(v.primary_context_share)],
    ['Explainable metrics',v.explainable_metric_count??'—']
  ]);

  const support=u.support||{},selected=u.selected_structure||{},oos=u.oos_models||[];
  $('#bb-rating-uncertainty').innerHTML=summaryRows([
    ['Status',u.status||'—'],
    ['Training target seasons',(u.training_target_seasons||[]).join(', ')||'—'],
    ['Excluded target seasons',(u.excluded_target_seasons||[]).join(', ')||'—'],
    ['Training rows',fmtInt(u.training_rows)],
    ['Exact league+exposure cells',support.available_exact_cells??'—'],
    ['Primary exact share',pct(support.primary_exact_share)],
    ['Mean P90 interval width',fmt(selected.mean_interval_width_p90,2)],
    ['Fallback order',(u.fallback_order||[]).join(' → ')]
  ])+'<div class="bb-rating-oos-table table-shell"><table><thead><tr><th>Model</th><th>OOS N</th><th>P50</th><th>P75</th><th>P90</th></tr></thead><tbody>'+
  (oos.length?oos.map(m=>'<tr><td><strong>'+esc(m.model||'—')+'</strong></td><td>'+esc(m.n_oos??'—')+'</td><td>'+pct(m.coverage?.p50)+'</td><td>'+pct(m.coverage?.p75)+'</td><td>'+pct(m.coverage?.p90)+'</td></tr>').join(''):'<tr><td colspan="5" class="muted-inline">Nessun modello OOS disponibile.</td></tr>')+
  '</tbody></table></div>';

  $('#bb-rating-files').innerHTML=summaryRows([
    ['Calibration report',f.report||'—'],
    ['Serving artifact',f.uncertainty||'—'],
    ['Report updated',f.report_modified_at||'—'],
    ['Artifact updated',f.uncertainty_modified_at||'—']
  ]);

  const warnings=[];
  if(data.status!=='ready')warnings.push(['Artifact','Verifica report e serving artifact']);
  if(apiBB.loaded===false)warnings.push(['API engine','BB-Rating engine non caricato nel servizio inference']);
  if(apiBB.uncertainty_loaded===false)warnings.push(['API uncertainty','Artifact uncertainty non caricato nel servizio inference']);
  if(data.calibration_version&&apiBB.runtime_calibration_version&&data.calibration_version!==apiBB.runtime_calibration_version)
    warnings.push(['Serving version','Artifact calibration '+data.calibration_version+' presente; API runtime '+apiBB.runtime_calibration_version+'. Riavvia/redeploy l’API per caricare il nuovo artifact.']);
  $('#bb-rating-warnings').innerHTML=warnings.length?warnings.map(x=>'<div class="health-item"><div class="health-copy"><strong>'+esc(x[0])+'</strong><span>'+esc(x[1])+'</span></div><span class="status warning">△ Verifica</span></div>').join(''):'<div class="empty-state compact"><strong>Nessuna anomalia</strong><p>Artifact e serving risultano disponibili.</p></div>';

  const refresh=$('#bb-rating-refresh');
  refresh.onclick=()=>renderBBRating().catch(e=>toast(e.message,'error'));
  const calibrate=$('#bb-rating-calibrate');
  calibrate.disabled=j.status==='running'||!data.active_profile;
  calibrate.textContent=j.status==='running'?'Calibrazione in corso…':'Ricalibra BB-Rating';
  calibrate.onclick=async()=>{
    calibrate.disabled=true;
    try{
      await api('/admin-api/bb-rating/calibrate',{method:'POST',body:{}});
      toast('Calibrazione BB-Rating avviata');
      await renderBBRating();
    }catch(e){toast(e.message,'error');calibrate.disabled=false;}
  };

  $('#bb-rating-job').innerHTML=j.status&&j.status!=='idle'?'<div class="job-head"><div><div class="job-status">'+esc(j.stage||'Calibration')+'</div><div class="job-message">'+esc(j.error||j.message||'')+'</div></div><span class="registry-badge '+((j.status==='running'||j.status==='queued')?'candidate':'')+'">'+esc(j.status)+'</span></div><div class="progress-track"><div class="progress-bar" style="width:'+Math.max(0,Math.min(100,Number(j.progress)||0))+'%"></div></div>':'<div class="job-message">Nessuna calibrazione eseguita dal Control Center in questa sessione.</div>';
  if((j.status==='running'||j.status==='queued')&&App.page==='bb-rating')App.bbRatingTimer=setTimeout(()=>renderBBRating().catch(e=>toast(e.message,'error')),2200);
}

function chartSvg(points){if(!points||points.length<1)return empty('⌁','Dati non disponibili','I fold walk-forward appariranno dopo il training.');const vals=points.map(p=>Number(p.rmse)).filter(Number.isFinite);if(!vals.length)return '';const min=Math.min(...vals),max=Math.max(...vals),range=Math.max(max-min,.001),w=640,h=210,pad=34;const coords=points.map((p,i)=>{const x=pad+(i*(w-pad*2)/Math.max(1,points.length-1));const y=h-pad-((Number(p.rmse)-min)/range)*(h-pad*2);return{x,y,p};});return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><line class="chart-grid-line" x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}"/><line class="chart-grid-line" x1="${pad}" y1="${pad}" x2="${w-pad}" y2="${pad}"/><polyline class="chart-line" points="${coords.map(c=>`${c.x},${c.y}`).join(' ')}"/>${coords.map(c=>`<circle class="chart-dot" cx="${c.x}" cy="${c.y}" r="5"/><text class="chart-label" x="${c.x}" y="${h-9}" text-anchor="middle">${esc(c.p.target_season??'')}</text>`).join('')}</svg>`;}
async function renderBacktests(){const d=await api('/admin-api/backtests'),r=d.report||{},o=r.overall||{},q=d.quality||{};$('#backtest-kpis').innerHTML=[kpi('⌁','Ensemble RMSE',fmt(o.rmse),'<span>OOT</span>'),kpi('↗','Base RMSE',fmt(r.base_rmse),'<span>XGBoost base</span>','green'),kpi('◫','Persistence RMSE',fmt(r.persistence_rmse),'<span>previous season</span>','violet'),kpi('◎','Interval coverage',pct(o.interval_coverage),'<span>conformal</span>','orange')].join('');$('#backtest-chart').innerHTML=chartSvg(q.folds||[]);$('#backtest-summary').innerHTML=d.candidate?.run_id?summaryRows([['Run ID',shortId(d.candidate.run_id)],['Samples',o.n??'—'],['Competitions',q.competitions??'—'],['Valid',r.valid?'Yes':'No']]):empty('▤','Nessun candidate','Completa un training run per ispezionare il report.','Avvia training','training');const comps=Object.entries(r.by_competition||{});$('#competition-body').innerHTML=comps.length?comps.map(([name,m])=>`<tr><td><strong>${esc(name)}</strong></td><td>${esc(m.n??'—')}</td><td>${esc(fmt(m.rmse))}</td><td>${esc(fmt(m.mae))}</td><td>${esc(fmt(m.bias))}</td><td>${esc(pct(m.interval_coverage))}</td></tr>`).join(''):'<tr><td colspan="6" class="muted-inline">Nessuna metrica per competition disponibile.</td></tr>';$$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));}

function csvValues(value){return String(value||'').split(',').map(v=>v.trim()).filter(Boolean);}
const scenarioSelections={player:[],team:[]};

function renderScenarioChips(picker){
  const entity=picker.dataset.entity;
  const chips=picker.querySelector('[data-chips]');
  chips.innerHTML=scenarioSelections[entity].map(item=>
    `<span class="entity-chip"><span class="entity-chip-copy"><span class="entity-chip-name">${esc(item.name)}</span><span class="entity-chip-subtitle">${esc(item.subtitle||'')}</span></span><button type="button" class="entity-chip-remove" data-selection-id="${esc(item.selection_id)}" aria-label="Rimuovi">×</button></span>`
  ).join('');
  chips.querySelectorAll('.entity-chip-remove').forEach(button=>button.addEventListener('click',()=>{
    scenarioSelections[entity]=scenarioSelections[entity].filter(item=>item.selection_id!==button.dataset.selectionId);
    renderScenarioChips(picker);
  }));
}

async function searchScenarioEntities(picker){
  const entity=picker.dataset.entity;
  const input=picker.querySelector('[data-search]');
  const menu=picker.querySelector('[data-results]');
  const q=input.value.trim();
  menu.classList.remove('hidden');
  menu.innerHTML='<div class="entity-picker-empty">Ricerca…</div>';
  try{
    const data=await api(`/admin-api/scenario-entities?entity=${encodeURIComponent(entity)}&q=${encodeURIComponent(q)}&limit=12`);
    const selected=new Set(scenarioSelections[entity].map(item=>item.selection_id));
    const items=(data.items||[]).filter(item=>!selected.has(item.selection_id));
    menu.innerHTML=items.length?items.map(item=>
      `<button type="button" class="entity-result" data-selection-id="${esc(item.selection_id)}"><span class="entity-result-copy"><span class="entity-result-name">${esc(item.name)}</span><span class="entity-result-subtitle">${esc(item.subtitle||'')}</span></span><span class="entity-result-status ${item.identity_status==='canonical'?'':'fallback'}">${esc(item.identity_status||'canonical')}</span></button>`
    ).join(''):'<div class="entity-picker-empty">Nessun risultato.</div>';
    menu.querySelectorAll('.entity-result').forEach(button=>button.addEventListener('click',()=>{
      const item=items.find(value=>value.selection_id===button.dataset.selectionId);
      if(!item)return;
      scenarioSelections[entity].push(item);
      input.value='';
      renderScenarioChips(picker);
      menu.classList.add('hidden');
      input.focus();
    }));
  }catch(e){menu.innerHTML=`<div class="entity-picker-empty">${esc(e.message)}</div>`;}
}

function setupScenarioEntityPicker(picker){
  const input=picker.querySelector('[data-search]');
  const menu=picker.querySelector('[data-results]');
  let timer=null;
  input.addEventListener('input',()=>{
    clearTimeout(timer);
    timer=setTimeout(()=>searchScenarioEntities(picker),220);
  });
  input.addEventListener('focus',()=>{
    if(!menu.children.length)searchScenarioEntities(picker);
    else menu.classList.remove('hidden');
  });
  document.addEventListener('click',event=>{
    if(!picker.contains(event.target))menu.classList.add('hidden');
  },{once:false});
  renderScenarioChips(picker);
}

async function renderScenarios(){
  const form=$('#scenario-form'),result=$('#scenario-result'),submit=$('#scenario-submit');
  scenarioSelections.player=[];scenarioSelections.team=[];
  $$('.entity-picker',form).forEach(setupScenarioEntityPicker);
  const overview=App.overview||await api('/admin-api/overview');
  form.season.value=overview.summary?.season_max||new Date().getFullYear();
  form.addEventListener('submit',async ev=>{
    ev.preventDefault();
    let parameters={};
    try{const raw=form.parameters.value.trim();parameters=raw?JSON.parse(raw):{};}
    catch(_){toast('Parameters JSON non valido','error');return;}
    const payload={
      scenario:form.scenario.value,
      player_selection_ids:scenarioSelections.player.map(item=>item.selection_id),
      team_selection_ids:scenarioSelections.team.map(item=>item.selection_id),
      source_league:form.source_league.value.trim()||null,
      target_league:form.target_league.value.trim()||null,
      season:Number(form.season.value),
      competition:form.competition.value.trim()||'RS',
      top_n:Number(form.top_n.value||5),
      parameters:{...parameters,simulations:Number(form.simulations.value||5000)}
    };
    submit.disabled=true;submit.textContent='Valutazione…';result.textContent='Il motore scenari sta elaborando la richiesta…';
    try{const data=await api('/admin-api/scenarios/evaluate',{method:'POST',body:payload});result.textContent=JSON.stringify(data,null,2);}
    catch(e){result.textContent='';toast(e.message,'error');}
    finally{submit.disabled=false;submit.textContent='Valuta scenario';}
  });
}

function registryCard(title,entry,tone=''){if(!entry||!entry.run_id)return `<div class="registry-card">${empty('⬡',`Nessun ${title.toLowerCase()}`,`Non è presente un run ${title.toLowerCase()} nel registry.`)}</div>`;return `<div class="registry-card"><span class="registry-badge ${tone}">${esc(title)}</span><div class="registry-run">${esc(entry.run_id)}</div><div class="registry-meta">Version: ${esc(entry.model_version||'—')}<br>Data cutoff: ${esc(entry.data_cutoff||'—')}<br>OOT RMSE: ${esc(fmt(entry.overall_rmse))}<br>Registered: ${esc(entry.registered_at||entry.promoted_at||'—')}</div></div>`;}
async function renderRegistry(){const d=await api('/admin-api/registry');$('#top-model').textContent=shortId(d.production?.run_id);$('#registry-cards').innerHTML=registryCard('Production',d.production)+registryCard('Candidate',d.candidate,'candidate');const hist=d.history||[];$('#registry-history').innerHTML=hist.length?[...hist].reverse().map(h=>`<tr><td>${esc(h.at||'—')}</td><td>${esc(h.event||'—')}</td><td>${esc(h.run_id||'—')}</td></tr>`).join(''):'<tr><td colspan="3" class="muted-inline">Nessun evento nel registry.</td></tr>';const check=$('#registry-confirm'),prom=$('#promote-btn'),roll=$('#rollback-btn');check.addEventListener('change',()=>{prom.disabled=!check.checked||!d.candidate;roll.disabled=!check.checked||!d.previous;});prom.addEventListener('click',async()=>{prom.disabled=true;try{const r=await api('/admin-api/registry/promote',{method:'POST',body:{confirm:true}});toast(r.reason||'Candidate promoted');await renderRegistry();}catch(e){toast(e.message,'error');prom.disabled=false;}});roll.addEventListener('click',async()=>{roll.disabled=true;try{const r=await api('/admin-api/registry/rollback',{method:'POST',body:{confirm:true}});toast(r.reason||'Rollback completed');await renderRegistry();}catch(e){toast(e.message,'error');roll.disabled=false;}});}


async function renderDiagnostics(){
  const results=$('#diagnostics-results'), summary=$('#diagnostics-summary');
  let running=0, passed=0, failed=0;
  const playerContexts=new Map();
  let teamSelectionManual=false;
  function escJson(v){try{return esc(JSON.stringify(v,null,2));}catch(_){return esc(String(v));}}
  function renderResult(data){
    const ok=data.overall_ok;
    if(ok)passed++;else failed++;
    const checks=(data.checks||[]).map(c=>'<div class="diagnostic-check '+(c.ok?'ok':'fail')+'"><span>'+(c.ok?'✓':'×')+'</span>'+esc(c.label)+'</div>').join('');
    const r=data.result||{};
    return '<article class="diagnostic-card '+(ok?'ok':'fail')+'"><div class="diagnostic-card-head"><div><strong>'+esc(data.model||'Health')+'</strong><span>'+esc(data.endpoint||'')+'</span></div><div class="diagnostic-badges"><span class="status '+(ok?'ok':'danger')+'">'+(ok?'PASS':'FAIL')+'</span><span class="diagnostic-latency">'+esc(r.latency_ms??'—')+' ms</span></div></div><div class="diagnostic-checks">'+checks+'</div><details><summary>Request / response</summary><div class="diagnostic-json"><div><small>Request</small><pre>'+escJson(data.request||{})+'</pre></div><div><small>Response</small><pre>'+escJson(r.body||r)+'</pre></div></div></details></article>';
  }
  async function run(model){
    const player=$('#diag-player').value;
    if(!player){toast('Seleziona prima un giocatore','error');return null;}
    running++; $$('.diag-run').forEach(b=>b.disabled=true); $('#diagnostics-all').disabled=true;
    try{
      const body={model,player_global_id:player,team_global_id:$('#diag-team').value,league:$('#diag-league').value.trim(),season:Number($('#diag-season').value),competition:$('#diag-competition').value,metrics:$('#diag-metrics').value.split(',').map(x=>x.trim()).filter(Boolean)};
      const data=await api('/admin-api/diagnostics/test',{method:'POST',body});
      results.insertAdjacentHTML('afterbegin',renderResult(data));
      summary.textContent='Ultimi test: '+passed+' PASS · '+failed+' FAIL';
      return data;
    }catch(e){toast(e.message,'error');return null;}
    finally{running--;if(!running){$$('.diag-run').forEach(b=>b.disabled=false);$('#diagnostics-all').disabled=false;}}
  }
  async function searchDiagnosticEntities(entity, inputSelector, selectSelector, placeholder){
    const q=$(inputSelector).value.trim();
    if(q.length<2)return;
    try{
      const data=await api('/admin-api/scenario-entities?entity='+encodeURIComponent(entity)+'&q='+encodeURIComponent(q)+'&limit=12');
      if(entity==='player'){
        (data.items||[]).forEach(item=>playerContexts.set(String(item.global_id||item.selection_id),item.contexts||[]));
      }
      $(selectSelector).innerHTML='<option value="">'+esc(placeholder)+'</option>'+(data.items||[]).map(item=>{
        const leagues=(item.league_keys||[]).join(' / ');
        const label=[item.name||item.global_id||item.selection_id,leagues].filter(Boolean).join(' · ');
        return '<option value="'+esc(item.global_id||'')+'" data-league="'+esc(item.league_key||'')+'" data-leagues="'+esc(leagues)+'" data-current-team="'+esc(item.current_team_global_id||'')+'" data-current-team-name="'+esc(item.current_team_name||'')+'">'+esc(label)+'</option>';
      }).join('');
    }catch(e){toast(e.message,'error');}
  }
  function applyPlayerContext(leagueOverride){
    const selected=$('#diag-player').selectedOptions[0];
    const playerId=$('#diag-player').value;
    if(!selected||!playerId)return;
    const contexts=playerContexts.get(playerId)||[];
    const league=String(leagueOverride||$('#diag-league').value||selected.dataset.league||'').trim().toUpperCase();
    if(league)$('#diag-league').value=league;
    const context=contexts.find(item=>String(item.league_key||'').toUpperCase()===league)
      ||contexts.find(item=>String(item.league_key||'').toUpperCase()===String(selected.dataset.league||'').toUpperCase())
      ||contexts[0];
    if(!context?.team_global_id)return;
    const teamSelect=$('#diag-team');
    const existing=[...teamSelect.options].find(option=>option.value===context.team_global_id);
    if(existing){
      teamSelect.value=context.team_global_id;
      if(context.league_key)teamSelect.dataset.contextLeague=String(context.league_key).toUpperCase();
      return;
    }
    const option=document.createElement('option');
    option.value=context.team_global_id;
    option.textContent=context.team_name||'Squadra del contesto selezionato';
    option.dataset.league=String(context.league_key||league).toUpperCase();
    teamSelect.appendChild(option);
    teamSelect.value=context.team_global_id;
  }
  let searchTimer;
  $('#diag-player-search').addEventListener('input',()=>{
    clearTimeout(searchTimer);
    searchTimer=setTimeout(()=>searchDiagnosticEntities('player','#diag-player-search','#diag-player','Seleziona giocatore'),300);
  });
  $('#diag-player').addEventListener('change',event=>{
    teamSelectionManual=false;
    const selected=event.target.selectedOptions[0];
    const league=selected?.dataset?.league||'';
    if(league)$('#diag-league').value=league;
    applyPlayerContext(league);
  });
  $('#diag-league').addEventListener('change',()=>{
    if(!teamSelectionManual)applyPlayerContext($('#diag-league').value);
  });
  let teamSearchTimer;
  $('#diag-team-search').addEventListener('input',()=>{
    clearTimeout(teamSearchTimer);
    teamSearchTimer=setTimeout(()=>searchDiagnosticEntities('team','#diag-team-search','#diag-team','Seleziona squadra'),300);
  });
  $('#diag-team').addEventListener('change',()=>{teamSelectionManual=!!$('#diag-team').value;});
  $$('.diag-run').forEach(b=>b.addEventListener('click',()=>run(b.dataset.model)));
  $('#diagnostics-all').addEventListener('click',async()=>{
    results.innerHTML='';
    const models=['prediction','bb_rating','future_performance','metric_rating'];
    for(const model of models)await run(model);
  });
  $('#diagnostics-health').addEventListener('click',async()=>{
    try{
      const data=await api('/admin-api/diagnostics/health-check',{method:'POST',body:{}});
      data.checks.forEach(item=>results.insertAdjacentHTML('afterbegin',renderResult({model:'Health',endpoint:item.endpoint,result:item.result,checks:[{label:item.label,ok:item.overall_ok}],overall_ok:item.overall_ok})));
      summary.textContent='Health Check: '+(data.overall_ok?'PASS':'FAIL');
    }catch(e){toast(e.message,'error');}
  });
}

async function renderHealth(){async function refresh(){const d=await api('/admin-api/api-health');const live=!!d.live?.ok,ready=!!d.ready?.ok,bb=d.bb_rating||{};$('#top-api').textContent=live?'● V2':'● OFFLINE';$('#top-api').className=live?'meta-ok':'';$('#health-kpis').innerHTML=[kpi('⌘','API process',live?'Online':'Offline',`<span class="dot ${live?'green':'red'}"></span>/health/live`,'green'),kpi('✓','Inference ready',ready?'Ready':'Not ready',`<span class="dot ${ready?'green':'yellow'}"></span>/health/ready`),kpi('◆','BB-Rating',bb.loaded?'Loaded':'Not loaded',bb.uncertainty_loaded?'<span class="dot green"></span>uncertainty loaded':'<span class="dot red"></span>uncertainty missing','violet'),kpi('↗','Internal base',d.base_url||'—','<span>Docker network</span>','violet'),kpi('⬡','API contract','V2','<span>server-to-server</span>','orange')].join('');$('#health-checks').innerHTML=[['Live endpoint',d.live],['Readiness endpoint',d.ready]].map(([name,x])=>`<div class="health-item"><div class="health-copy"><strong>${esc(name)}</strong><span>${esc(x?.error||JSON.stringify(x?.body||{}).slice(0,160))}</span></div><span class="status ${x?.ok?'ok':'danger'}">${x?.ok?'✓ OK':'× KO'}</span></div>`).join('');$('#health-endpoints').innerHTML=(d.endpoints||[]).map(e=>`<div class="endpoint-item"><code>${esc(e)}</code><span class="registry-badge">V2</span></div>`).join('');}$('#refresh-health').addEventListener('click',()=>refresh().catch(e=>toast(e.message,'error')));await refresh();}

async function renderAudit(){async function refresh(){const d=await api('/admin-api/audit?limit=200');const rows=d.entries||[];$('#audit-body').innerHTML=rows.length?rows.map(r=>`<tr><td>${esc(r.ts||'—')}</td><td><strong>${esc(r.action||'—')}</strong></td><td>${esc(r.actor||'—')}</td><td>${esc(r.target||'—')}</td><td>${esc(r.details||'')}</td></tr>`).join(''):'<tr><td colspan="5" class="muted-inline">Nessun evento audit disponibile.</td></tr>';}$('#refresh-audit').addEventListener('click',()=>refresh().catch(e=>toast(e.message,'error')));await refresh();}

async function renderSettings(){const d=await api('/admin-api/settings');$('#runtime-settings').innerHTML=Object.entries(d.runtime||{}).map(([k,v])=>`<div class="settings-row"><span>${esc(k.replaceAll('_',' '))}</span><strong>${esc(v??'—')}</strong></div>`).join('');$('#promotion-settings').innerHTML=Object.entries(d.promotion_gates||{}).map(([k,v])=>`<div class="settings-row"><span>${esc(k.replaceAll('_',' '))}</span><strong>${esc(v)}</strong></div>`).join('');$('#password-form').addEventListener('submit',async ev=>{ev.preventDefault();const body=Object.fromEntries(new FormData(ev.currentTarget).entries());const btn=$('button',ev.currentTarget);btn.disabled=true;try{await api('/admin-api/settings/password',{method:'POST',body});toast('Password aggiornata. Effettua nuovamente il login.');showLogin();}catch(e){toast(e.message,'error');btn.disabled=false;}});}

$('#login-form').addEventListener('submit',async ev=>{ev.preventDefault();const error=$('#login-error'),button=$('button',ev.currentTarget);error.textContent='';button.disabled=true;try{const d=await api('/admin-api/login',{method:'POST',body:{username:$('#login-username').value,password:$('#login-password').value}});showApp(d.user);await navigate('overview');}catch(e){error.textContent=e.message;}finally{button.disabled=false;}});
$('#logout-btn').addEventListener('click',async()=>{try{await api('/admin-api/logout',{method:'POST',body:{}});}catch(_){}showLogin();});
$$('.nav-item').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));

(async function boot(){try{const s=await api('/admin-api/session');showApp(s.user);await navigate('overview');}catch(_){showLogin();}})();
