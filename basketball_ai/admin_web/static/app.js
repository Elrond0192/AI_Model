const App={user:null,page:'overview',overview:null,trainingTimer:null,bbRatingTimer:null,futurePerformanceTimer:null,bbRatingPredictionTimer:null,controllers:new Set()};
const $=(s,r=document)=>r.querySelector(s);const $$=(s,r=document)=>[...r.querySelectorAll(s)];
const root=$('#page-root');

// The browser-native POST fallback redirects invalid credentials to a clean URL.
if(new URLSearchParams(window.location.search).get('login')==='failed'){
  const loginRedirectError=$('#login-error');
  if(loginRedirectError)loginRedirectError.textContent='Credenziali non valide';
  window.history.replaceState(null,'',window.location.pathname);
}

function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function cookie(name){const p=document.cookie.split('; ').find(x=>x.startsWith(name+'='));return p?decodeURIComponent(p.split('=').slice(1).join('=')):'';}
function fmt(v,d=3){if(v===null||v===undefined||v===''||Number.isNaN(Number(v)))return '—';return Number(v).toFixed(d);}
function fmtInt(v){if(v===null||v===undefined)return '—';return new Intl.NumberFormat('it-IT',{notation:Number(v)>=1000000?'compact':'standard',maximumFractionDigits:1}).format(Number(v));}
function pct(v){if(v===null||v===undefined||Number.isNaN(Number(v)))return '—';return `${(Number(v)*100).toFixed(1)}%`;}

function progressMarkup(job){
  const status=String(job?.status||'idle').toLowerCase();
  if(!status||status==='idle')return '';
  const raw=job?.progress;
  const numeric=raw===null||raw===undefined||raw===''?NaN:Number(raw);
  const progress=Number.isFinite(numeric)?Math.round(Math.max(0,Math.min(100,numeric))):null;
  const active=status==='running'||status==='queued';
  const indeterminate=active&&(status==='queued'||progress===null||progress===0);
  const label=status==='queued'?'In coda':status==='running'?'Avanzamento':status==='complete'?'Completato':status==='failed'?'Esecuzione terminata con errore':status==='cancelled'?'Annullato':'Stato avanzamento';
  const displayed=progress===null?'—':`${progress}%`;
  const width=indeterminate?38:(progress??0);
  const valueAttribute=progress===null?'':` aria-valuenow="${progress}"`;
  const valueText=status==='queued'?' aria-valuetext="In coda"':'';
  return `<div class="job-progress ${active?'is-active':''} ${status==='queued'?'is-queued':''}">
    <div class="job-progress-meta"><span>${esc(label)}</span><strong>${displayed}</strong></div>
    <div class="progress-track" role="progressbar" aria-label="Avanzamento del job" aria-valuemin="0" aria-valuemax="100"${valueAttribute}${valueText}>
      <div class="progress-bar ${indeterminate?'is-indeterminate':''}" style="width:${width}%"></div>
    </div>
  </div>`;
}
function shortId(v){const s=String(v||'');return s.length>18?`${s.slice(0,12)}…${s.slice(-5)}`:(s||'Not promoted');}
function toast(message,type='success'){const el=$('#toast');el.textContent=message;el.className=`toast show ${type}`;clearTimeout(el._timer);el._timer=setTimeout(()=>el.className='toast',3600);}
function loading(){root.innerHTML='<div class="loading"><div><div class="spinner"></div>Caricamento…</div></div>';}
function empty(icon,title,copy,button='',page=''){return `<div class="empty-state"><div class="empty-icon">${icon}</div><strong>${esc(title)}</strong><p>${esc(copy)}</p>${button?`<button class="btn btn-primary btn-small jump" data-page="${esc(page)}">${esc(button)}</button>`:''}</div>`;}
function summaryRows(rows){return `<div class="summary-list">${rows.map(([a,b])=>`<div class="summary-row"><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`).join('')}</div>`;}

async function api(path,options={}){
  const {keepAlive=false,...requestOptions}=options;
  const headers={'Accept':'application/json',...(requestOptions.headers||{})};
  if(requestOptions.body!==undefined){headers['Content-Type']='application/json';requestOptions.body=JSON.stringify(requestOptions.body);}
  if((requestOptions.method||'GET').toUpperCase()!=='GET'){const token=cookie('hm_ai_admin_csrf');if(token)headers['X-CSRF-Token']=token;}
  const controller=new AbortController();
  if(!keepAlive)App.controllers.add(controller);
  const fetchOptions={credentials:'same-origin',...requestOptions,headers};
  if(!requestOptions.signal&&!keepAlive)fetchOptions.signal=controller.signal;
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

async function enterBackOffice(user){
  showApp(user);
  // Finish the first render before auto-load starts: this avoids two concurrent
  // overview renders racing over the same DOM after login/session restoration.
  await navigate('overview');
  let result=null,error=null;
  try{result=await api('/admin-api/auto-load',{method:'POST',body:{},keepAlive:true});}
  catch(e){error=e.message;}
  // Auto-load updates the backend state/snapshot. Refresh only if the same
  // authenticated session is still active and the operator stayed on overview.
  if(App.user===user&&App.page==='overview')await navigate('overview');
  if(result?.loaded)toast(`Dati aggiornati da ${result.profile}: ${fmtInt(result.summary?.player_rows)} righe giocatore`);
  else if(result?.message)toast(result.message,'error');
  else if(error)toast('Caricamento automatico dal DB non riuscito: '+error,'error');
}
function updateClock(){const d=new Date();$('#top-utc').textContent=d.toISOString().replace('T',' ').slice(0,19);}
setInterval(updateClock,1000);updateClock();

function activateNav(page){$$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.page===page));}
function scrollPageToTop(){
  const containers=[document.scrollingElement,document.documentElement,document.body,$('.main-area'),root];
  containers.forEach(element=>{
    if(element){element.scrollTop=0;element.scrollLeft=0;}
  });
  if(typeof window.scrollTo==='function')window.scrollTo(0,0);
}
function mountTemplate(page){const t=$(`#page-${page}`);if(!t){root.innerHTML='<div class="loading">Pagina non disponibile</div>';return;}root.replaceChildren(t.content.cloneNode(true));$$('.admin-only',root).forEach(el=>el.classList.toggle('hidden',App.user?.role!=='admin'));$$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));}
async function navigate(page){
  if(['training','backtests','registry'].includes(page))page='model-releases';
  if(!document.getElementById('page-'+page))page='overview';
  App.controllers.forEach(controller=>controller.abort());
  App.controllers.clear();
  clearTimeout(App.trainingTimer);
  clearTimeout(App.bbRatingTimer);
  clearTimeout(App.futurePerformanceTimer);
  clearTimeout(App.bbRatingPredictionTimer);
  App.page=page;
  activateNav(page);
  mountTemplate(page);
  scrollPageToTop();
  try{
    if(page==='overview')await renderOverview();
    if(page==='data')await renderData();
    if(page==='ai-jobs')await renderAiJobs();
    if(page==='bb-rating')await renderBBRating();
    if(page==='future-performance')await renderFuturePerformancePage();
    if(page==='scenarios')await renderScenarios();
    if(page==='model-releases')await renderModelReleases();
    if(page==='health')await renderHealth();
    if(page==='diagnostics')await renderDiagnostics();
    if(page==='audit')await renderAudit();
    if(page==='settings')await renderSettings();
  }catch(e){if(e?.name!=='AbortError')toast(e.message,'error');}
}
function kpi(icon,label,value,sub='',tone=''){return `<div class="kpi-card"><div class="kpi-icon ${tone}">${icon}</div><div class="kpi-label">${esc(label)}</div><div class="kpi-value ${tone==='green'?'online':''}">${esc(value)}</div><div class="kpi-sub">${sub}</div></div>`;}
function updateTop(o){$('#top-db').textContent=o.database||'not selected';$('#top-model').textContent=o.bb_rating_prediction?.loaded?shortId(o.bb_rating_prediction?.version||'BB Prediction'): 'BB Prediction pending';const bb=o.bb_rating||{};$('#top-bb-rating').textContent=bb.bb_rating_version?'v'+esc(bb.bb_rating_version):'—';$('#top-api').textContent=o.api?.online?'● V2':'● OFFLINE';$('#top-api').className=o.api?.online?'meta-ok':'';}

async function renderOverview(){
  const [overviewResult, healthResult, predictionResult, bbRatingResult, futureResult] = await Promise.all([
    api('/admin-api/overview'),
    api('/admin-api/api-health').catch(()=>({})),
    api('/admin-api/bb-rating-prediction').catch(()=>({})),
    api('/admin-api/bb-rating').catch(()=>({})),
    api('/admin-api/future-performance').catch(()=>({}))
  ]);
  const o=overviewResult;
  const health=healthResult||{};
  const prediction=predictionResult||{};
  const bbStatus=bbRatingResult||{};
  const futureStatus=futureResult||{};
  o.bb_rating_prediction=health.bb_rating_prediction||{};
  App.overview=o;
  updateTop(o);

  const bb=o.bb_rating||{};
  const fp=o.future_performance||{};
  const modelPrediction=o.bb_rating_prediction||{};
  const predictionRelease=prediction.release||{};
  const predictionJob=prediction.job||{};
  const bbJob=bbStatus.job||{};
  const futureJob=futureStatus.job||{};
  const readyCount=[!!bb.ready,!!fp.ready,!!modelPrediction.loaded].filter(Boolean).length;
  const allReady=!!(o.api?.online&&o.data_loaded&&bb.ready&&modelPrediction.loaded&&fp.ready);
  const status=$('#overview-system-status');
  if(status)status.textContent=allReady?'TUTTI I SISTEMI OPERATIVI':'ATTENZIONE: VERIFICA RICHIESTA';
  const detail=$('#overview-system-detail');
  if(detail)detail.textContent=allReady
    ?'API pronta · Dati disponibili · BB-Rating e BB Prediction Model caricati'
    :(!o.api?.online?'Inference API non raggiungibile':(!o.data_loaded?'Seleziona e carica un profilo dati':'Verifica release e stato dei modelli'));
  const meta=$('#overview-system-meta');
  if(meta)meta.textContent=o.database||'Profilo non selezionato';
  const systemIcon=$('#overview-system-icon');
  if(systemIcon)systemIcon.className=allReady?'ph ph-check-circle':'ph ph-warning-circle';
  const systemDot=$('#system-dot'),systemLabel=$('#system-label');
  if(systemDot)systemDot.className='system-dot '+(allReady?'online':'warning');
  if(systemLabel)systemLabel.textContent=allReady?'SYSTEM OPERATIONAL':'SYSTEM ATTENTION';

  const s=o.summary||{};
  $('#overview-kpis').innerHTML=[
    kpi('⬡','Modelli pronti',readyCount+' / 3','<span>release caricate e disponibili</span>',readyCount===3?'green':'orange'),
    kpi('▥','Osservazioni giocatore',fmtInt(s.player_rows),'<span>righe giocatore / competizione</span>','green'),
    kpi('◉','Competizioni',String((s.competitions||[]).length||'—'),'<span>presenti nello snapshot caricato</span>','violet'),
    kpi('⌘','API inference',o.api?.online?'READY':'OFFLINE',o.api?.online?'<span class="dot green"></span>API raggiungibile':'<span class="dot red"></span>Verifica connessione',o.api?.online?'green':'orange')
  ].join('');

  const jobRecords=[
    {name:'BB Prediction Model',page:'model-releases',job:predictionJob},
    {name:'BB-Rating',page:'bb-rating',job:bbJob},
    {name:'Future Performance',page:'future-performance',job:futureJob}
  ].filter(item=>item.job&&item.job.status&&item.job.status!=='idle');
  const activeJobs=jobRecords.filter(item=>['running','queued'].includes(item.job.status));
  const sortByTime=(a,b)=>{
    const ta=Date.parse(a.job.started_at||a.job.finished_at||'')||0;
    const tb=Date.parse(b.job.started_at||b.job.finished_at||'')||0;
    return tb-ta;
  };
  const currentRecord=(activeJobs.length?activeJobs.slice().sort(sortByTime):jobRecords.slice().sort((a,b)=>(Date.parse(b.job.finished_at||b.job.started_at||'')||0)-(Date.parse(a.job.finished_at||a.job.started_at||'')||0)))[0];
  const jobPanel=$('#overview-job-panel');
  const jobState=$('#overview-job-state');
  if(currentRecord){
    const job=currentRecord.job;
    const active=['running','queued'].includes(job.status);
    const statusLabels={queued:'IN CODA',running:'IN ESECUZIONE',complete:'COMPLETATO',failed:'ERRORE',cancelled:'ANNULLATO'};
    const className=job.status==='failed'?'is-failed':job.status==='complete'?'is-complete':job.status==='cancelled'?'is-cancelled':active?'is-active':'is-idle';
    if(jobState){jobState.className='dashboard-state-pill '+className;jobState.textContent=statusLabels[job.status]||String(job.status).toUpperCase();}
    const message=job.error||job.message||'Nessun dettaglio aggiuntivo.';
    const progress=['failed','cancelled'].includes(job.status)?'':progressMarkup(job);
    jobPanel.innerHTML='<div class="dashboard-job-top"><div class="dashboard-job-identity"><span class="dashboard-model-icon"><i class="ph ph-activity"></i></span><div><strong>'+esc(currentRecord.name)+'</strong><span>'+esc(job.stage||'Preparazione')+'</span></div></div><button class="btn btn-secondary btn-small" id="overview-job-open">Apri dettagli <i class="ph ph-arrow-up-right"></i></button></div>'+
      '<div class="dashboard-job-message">'+esc(message)+'</div>'+
      progress+
      '<div class="dashboard-job-foot"><span><i class="ph ph-git-branch"></i> Fase: '+esc(job.stage||'—')+'</span><span>'+esc(job.run_id?'Run ID: '+job.run_id:(job.candidate_id?'Candidate: '+job.candidate_id:'Produzione invariata durante il training'))+'</span></div>';
    const openJob=$('#overview-job-open');
    if(openJob)openJob.onclick=()=>navigate(currentRecord.page);
  }else{
    if(jobState){jobState.className='dashboard-state-pill is-idle';jobState.textContent='NESSUN JOB ATTIVO';}
    jobPanel.innerHTML='<div class="dashboard-empty"><span class="dashboard-empty-icon"><i class="ph ph-check-circle"></i></span><div><strong>Nessun job in esecuzione</strong><span>Avvia una calibrazione o un training dalle rispettive pagine. Quando un job partirà, qui appariranno fase e avanzamento effettivo.</span></div></div><div class="dashboard-empty-actions"><button class="btn btn-secondary btn-small" id="overview-open-models">Apri modelli <i class="ph ph-arrow-up-right"></i></button><button class="btn btn-ghost btn-small" id="overview-open-bb-rating">BB-Rating <i class="ph ph-arrow-up-right"></i></button></div>';
    $('#overview-open-models').onclick=()=>navigate('model-releases');
    $('#overview-open-bb-rating').onclick=()=>navigate('bb-rating');
  }

  const readiness=o.readiness||[];
  $('#readiness-body').innerHTML=readiness.map(r=>{
    const page=r.action==='data'?'data':r.action;
    const icon=r.component==='PostgreSQL source'?'ph-database':r.component==='Competition context'?'ph-chart-bar':r.component==='BB Prediction Model'?'ph-trend-up':r.component==='Player Future Performance'?'ph-crystal-ball':'ph-star';
    const stateClass=r.status==='ok'?'is-ok':r.status==='danger'?'is-danger':'is-warning';
    return '<div class="dashboard-health-row"><span class="dashboard-health-icon"><i class="ph '+icon+'"></i></span><div class="dashboard-health-copy"><strong>'+esc(r.component)+'</strong><span>'+esc(r.detail)+'</span></div><span class="dashboard-health-status '+stateClass+'">'+esc(r.label)+'</span><button class="dashboard-health-open readiness-action" data-page="'+esc(page)+'" aria-label="Apri '+esc(r.component)+'"><i class="ph ph-arrow-up-right"></i></button></div>';
  }).join('');
  $$('.readiness-action').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));

  const versionOrDash=(v)=>v?'v'+v:'—';
  const predProd=predictionRelease.production||{};
  const predCandidate=predictionRelease.candidate||{};
  const rows=[
    {name:'BB-Rating',description:'Scoring osservato · scala ufficiale 1–100',state:bb.ready?'READY':(['running','queued'].includes(bbJob.status)?'IN TRAINING':'DA VERIFICARE'),version:versionOrDash(bb.bb_rating_version),validation:bb.ready?'Calibrazione disponibile':'Artifact da verificare',page:'bb-rating'},
    {name:'BB Prediction Model',description:'Forecast futuro del BB-Rating · scala 1–100',state:modelPrediction.loaded?'READY':(['running','queued'].includes(predictionJob.status)?'IN TRAINING':(predCandidate.valid?'CANDIDATE':'DA VERIFICARE')),version:versionOrDash(predProd.version||predCandidate.version),validation:predCandidate.valid?'Candidate validata':(['running','queued'].includes(predictionJob.status)?'Walk-forward in corso':'Gate di validazione da verificare'),page:'model-releases'},
    {name:'Future Performance',description:'Previsione statistica della stagione successiva',state:fp.ready?'READY':(['running','queued'].includes(futureJob.status)?'IN TRAINING':'DA VERIFICARE'),version:versionOrDash(futureStatus.future_performance_version||fp.future_performance_version),validation:fp.ready?'Artifact disponibile':'Artifact da verificare',page:'future-performance'}
  ];
  $('#overview-model-table').innerHTML=rows.map(row=>{
    const klass=row.state==='READY'?'is-ok':row.state==='IN TRAINING'?'is-running':row.state==='CANDIDATE'?'is-candidate':'is-warning';
    return '<tr><td><div class="dashboard-table-model"><span class="dashboard-table-model-icon"><i class="ph '+(row.name==='BB-Rating'?'ph-star':row.name==='BB Prediction Model'?'ph-trend-up':'ph-crystal-ball')+'"></i></span><div><strong>'+esc(row.name)+'</strong><small>'+esc(row.description)+'</small></div></div></td><td><span class="dashboard-health-status '+klass+'">'+esc(row.state)+'</span></td><td class="dashboard-version">'+esc(row.version)+'</td><td>'+esc(row.validation)+'</td><td class="right"><button class="dashboard-table-open" data-page="'+esc(row.page)+'">Apri <i class="ph ph-arrow-up-right"></i></button></td></tr>';
  }).join('');
  $$('.dashboard-table-open').forEach(b=>b.onclick=()=>navigate(b.dataset.page));

  $('#overview-registry').innerHTML=summaryRows([
    ['Stato runtime',modelPrediction.loaded?'Caricato':(predCandidate.valid?'Candidate valida':'Da validare/promuovere')],
    ['Versione produzione',predProd.version||'—'],
    ['Scala target','BB-Rating ufficiale 1–100']
  ]);
  $('#overview-backtest').innerHTML=summaryRows([
    ['Stato',fp.ready?'Disponibile':'Da verificare'],
    ['Versione',futureStatus.future_performance_version||fp.future_performance_version||'—'],
    ['Obiettivo','Statistiche della stagione successiva']
  ]);
  $('#candidate-quality').innerHTML='<div class="quality-card"><span>Validazione</span><strong>Walk-forward per scenario</strong></div><div class="quality-card"><span>Promozione</span><strong>Solo se batte il baseline</strong></div><div class="quality-card"><span>Scala target</span><strong>BB-Rating ufficiale 1–100</strong></div>';

  clearTimeout(App.overviewTimer);
  if(activeJobs.length&&App.page==='overview')App.overviewTimer=setTimeout(()=>renderOverview().catch(e=>{if(e?.name!=='AbortError')toast(e.message,'error')}),2500);
  const refresh=$('#overview-refresh');
  if(refresh)refresh.onclick=()=>renderOverview().catch(e=>toast(e.message,'error'));
}
async function renderData(){
  async function loadSnapshots(){const data=await api('/admin-api/snapshots'),active=data.active?.snapshot_id,items=data.snapshots||[];$('#snapshots-list').innerHTML=items.length?items.map(s=>`<div class="profile-card ${active===s.snapshot_id?'active':''}"><div class="profile-top"><div class="profile-name">${esc(s.snapshot_id)}</div>${active===s.snapshot_id?'<span class="registry-badge">attivo</span>':''}</div><div class="profile-meta"><span>${esc(s.profile||'—')}</span><span>${esc((s.seasons||[]).length?`${s.seasons[0]}–${s.seasons.at(-1)}`:'—')}</span><span>${fmtInt(s.player_rows)} righe giocatore</span><span>SHA ${esc(String(s.sha256||'').slice(0,12))}</span></div><div class="button-row"><button class="btn btn-primary btn-small snapshot-activate" data-id="${esc(s.snapshot_id)}" ${active===s.snapshot_id?'disabled':''}>Usa per il training</button></div></div>`).join(''):empty('◫','Nessuno storico pronto','Apri il BO per il caricamento automatico oppure usa “Ricarica da PostgreSQL” per aggiornare i dati.');$$('.snapshot-activate').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;try{const result=await api(`/admin-api/snapshots/${encodeURIComponent(b.dataset.id)}/activate`,{method:'POST',body:{}});toast(`Storico ${result.snapshot?.snapshot_id||b.dataset.id} attivato`);await loadSnapshots();}catch(e){toast(e.message,'error');b.disabled=false;}}));}
  async function loadProfiles(){
    const data=await api('/admin-api/profiles');$('#top-db').textContent=data.active||'not selected';
    const entries=Object.entries(data.profiles||{});$('#profiles-list').innerHTML=entries.length?entries.map(([name,p])=>`<div class="profile-card ${data.active===name?'active':''}"><div class="profile-top"><div class="profile-name">${esc(name)}</div>${data.active===name?'<span class="registry-badge">attivo</span>':''}</div><div class="profile-meta"><span>${esc(p.host||'—')}:${esc(p.port||5432)}</span><span>${esc(p.database||'—')}</span><span>${esc(p.user||'—')}</span><span>${esc(p.source_schema||'AI_Source')} → ${esc(p.ai_schema||'AI')}</span></div><p class="microcopy">All’apertura del BO i dati vengono caricati automaticamente. Il secondo pulsante ricarica il profilo e aggiorna lo snapshot core: non esegue training e non legge PBP/lineup.</p><div class="button-row"><button class="btn btn-secondary btn-small profile-test" data-name="${esc(name)}">1. Verifica database</button><button class="btn btn-primary btn-small profile-load" data-name="${esc(name)}">2. Ricarica da PostgreSQL</button></div></div>`).join(''):empty('▤','Nessun profilo configurato','Crea il profilo PostgreSQL production nel pannello a destra.');
    $$('.profile-test').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;try{const r=await api(`/admin-api/profiles/${encodeURIComponent(b.dataset.name)}/test`,{method:'POST',body:{}});toast(r.message);}catch(e){toast(e.message,'error');}finally{b.disabled=false;}}));
    $$('.profile-load').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;b.textContent='Preparazione storico…';try{const r=await api(`/admin-api/profiles/${encodeURIComponent(b.dataset.name)}/load`,{method:'POST',body:{}});toast(`Storico pronto: ${fmtInt(r.summary.player_rows)} righe core`);await Promise.all([loadProfiles(),loadSnapshots()]);}catch(e){toast(e.message,'error');}finally{b.disabled=false;b.textContent='2. Ricarica da PostgreSQL';}}));
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
  const trainingActive=['running','queued'].includes(j.status);
  job.innerHTML=`<div class="job-head"><div><div class="job-status">${esc(j.stage||'Ready')}</div><div class="job-message">${esc(idleMessage)}</div></div><span class="registry-badge ${trainingActive?'candidate':''}">${esc(j.status||'idle')}</span></div>${progressMarkup(j)}${j.run_id?`<div class="job-message">Run ID: ${esc(j.run_id)}</div>`:''}${trainingActive?'<div class="button-row"><button id="training-cancel" class="btn btn-danger btn-small" '+(j.cancel_requested?'disabled':'')+'>'+(j.cancel_requested?'Interruzione richiesta…':'Interrompi training')+'</button></div>':''}`;
  const cancelTraining=$('#training-cancel');
  if(cancelTraining)cancelTraining.onclick=async()=>{cancelTraining.disabled=true;try{await api('/admin-api/jobs/training/cancel',{method:'POST',body:{}});toast('Interruzione training richiesta');await renderTraining();}catch(e){toast(e.message,'error');cancelTraining.disabled=false;}};
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
    warnings.push(['Serving version','Artifact calibration '+data.calibration_version+' presente; API runtime '+apiBB.runtime_calibration_version+'. verifica il riavvio automatico e la readiness dell’API.']);
  $('#bb-rating-warnings').innerHTML=warnings.length?warnings.map(x=>'<div class="health-item"><div class="health-copy"><strong>'+esc(x[0])+'</strong><span>'+esc(x[1])+'</span></div><span class="status warning">△ Verifica</span></div>').join(''):'<div class="empty-state compact"><strong>Nessuna anomalia</strong><p>Artifact e serving risultano disponibili.</p></div>';

  const refresh=$('#bb-rating-refresh');
  refresh.onclick=()=>renderBBRating().catch(e=>toast(e.message,'error'));
  const calibrate=$('#bb-rating-calibrate');
  const calibrationBusy=['running','queued'].includes(j.status);
  calibrate.disabled=calibrationBusy||!data.active_profile;
  calibrate.textContent=calibrationBusy?(j.status==='queued'?'Calibrazione in coda…':'Calibrazione in corso…'):'Ricalibra BB-Rating';
  calibrate.onclick=async()=>{
    calibrate.disabled=true;
    try{
      await api('/admin-api/bb-rating/calibrate',{method:'POST',body:{}});
      toast('Calibrazione BB-Rating avviata');
      await renderBBRating();
    }catch(e){toast(e.message,'error');calibrate.disabled=false;}
  };

  $('#bb-rating-job').innerHTML=j.status&&j.status!=='idle'?'<div class="job-head"><div><div class="job-status">'+esc(j.stage||'Calibration')+'</div><div class="job-message">'+esc(j.error||j.message||'')+'</div></div><span class="registry-badge '+((j.status==='running'||j.status==='queued')?'candidate':'')+'">'+esc(j.status)+'</span></div>'+progressMarkup(j):'<div class="job-message">Nessuna calibrazione eseguita dal Control Center in questa sessione.</div>';
  if(['running','queued'].includes(j.status)) {
    $('#bb-rating-job').insertAdjacentHTML('beforeend','<div class="button-row"><button id="bb-rating-cancel" class="btn btn-danger btn-small" '+(j.cancel_requested?'disabled':'')+'>'+(j.cancel_requested?'Interruzione richiesta…':'Interrompi calibrazione')+'</button></div>');
    const cancel=$('#bb-rating-cancel');
    cancel.onclick=async()=>{cancel.disabled=true;try{await api('/admin-api/jobs/bb-rating/cancel',{method:'POST',body:{}});toast('Interruzione BB-Rating richiesta');await renderBBRating();}catch(e){toast(e.message,'error');cancel.disabled=false;}};
  }
  if((j.status==='running'||j.status==='queued')&&App.page==='bb-rating')App.bbRatingTimer=setTimeout(()=>renderBBRating().catch(e=>toast(e.message,'error')),2200);
}

async function renderModelReleases(){
  const data=await api('/admin-api/model-releases');
  const prediction=await api('/admin-api/bb-rating-prediction');
  let health={};
  try{health=await api('/admin-api/api-health');}catch(_){health={};}
  const models=data.models||{};
  const job=prediction.job||{};
  const release=prediction.release||{};
  const fmtRelease=(m)=>{
    const prod=m.production||{},cand=m.candidate||{},prev=m.previous||{};
    return '<div class="summary-list">'+
      '<div class="summary-row"><span>Production</span><strong>'+esc(prod.valid?'Validata':'Non valida / assente')+'</strong></div>'+
      '<div class="summary-row"><span>Versione produzione</span><strong>'+esc(prod.version||'—')+'</strong></div>'+
      '<div class="summary-row"><span>Candidate</span><strong>'+esc(cand.id||'Nessuna')+'</strong></div>'+
      '<div class="summary-row"><span>Validazione candidate</span><strong>'+esc(cand.valid?'OK':(cand.reason||'Non disponibile'))+'</strong></div>'+
      '<div class="summary-row"><span>Versione candidate</span><strong>'+esc(cand.version||'—')+'</strong></div>'+
      '<div class="summary-row"><span>OOF samples</span><strong>'+esc(cand.oof_samples??'—')+'</strong></div>'+
      '<div class="summary-row"><span>MAE walk-forward</span><strong>'+esc(fmt(cand.oof_mae_after_calibration??cand.validation?.oof_mae_after_calibration??cand.mae))+'</strong></div>'+
      '<div class="summary-row"><span>Precedente disponibile</span><strong>'+esc(prev.valid?'Sì':'No')+'</strong></div>'+
      '</div>';
  };
  const servingRows=[
    ['Inference API live',health.live?.ok?'Online':'Offline'],
    ['Inference API ready',health.ready?.ok?'Ready':'Not ready'],
    ['BB-Rating loaded',health.bb_rating?.loaded?'Yes':'No'],
    ['BB-Rating uncertainty loaded',health.bb_rating?.uncertainty_loaded?'Yes':'No'],
    ['BB Prediction Model loaded',health.bb_rating_prediction?.loaded?'Yes':'No'],
    ['BB Prediction Model version',health.bb_rating_prediction?.version||'—'],
    ['Future Performance loaded',health.future_performance?.loaded?'Yes':'No'],
    ['Automatic API restart',data.automatic_api_restart_configured?'Configured (restart + readiness check)':'Not configured (API_KEY missing)'],
  ];
  $('#model-release-serving').innerHTML=summaryRows(servingRows);
  $('#release-bb-rating').innerHTML=fmtRelease(models['bb-rating']||{});
  $('#release-future-performance').innerHTML=fmtRelease(models['future-performance']||{});
  $('#bb-rating-prediction-release').innerHTML=fmtRelease(release);
  const support=job.output?.oof_support||{};
  const outputRows=job.output?[
    ['Righe OOF valutate',job.output.oof_samples??'—'],
    ['Giocatori distinti OOF',support.unique_players??'—'],
    ['Transizioni giocatore-stagione',support.unique_player_season_transitions??'—'],
    ['Contesti target unici',support.unique_target_contexts??'—'],
    ['MAE modello calibrato',fmt(job.output.mae_calibrated_model??job.output.mae)],
    ['MAE policy finale (dopo fallback)',fmt(job.output.mae_after_scenario_fallback??job.output.mae)],
    ['MAE baseline',fmt(job.output.baseline_mae)],
    ['RMSE',fmt(job.output.rmse)],
    ['Coppie target duplicate',job.output.duplicate_target_context_pairs??'—'],
    ['Righe con identità team irrisolta',job.output.unresolved_team_identity_rows??'—'],
    ['Stagioni OOF',(job.output.walk_forward_target_seasons||job.output.target_seasons||[]).join(', ')||'—'],
    ['Gate qualità',job.status==='complete'?'SUPERATO':job.output.promotion_gate_passed===false?'NON SUPERATO':'—'],
    ['Gruppi oltre soglia',job.output.segment_regression_count??'—'],
    ['Scenari valutati',Object.keys(job.output.scenario_metrics||{}).length]
  ]:'';
  const outputMarkup=job.output
    ?'<div class="summary-list">'+outputRows.map(([a,b])=>'<div class="summary-row"><span>'+esc(a)+'</span><strong>'+esc(b??'—')+'</strong></div>').join('')+'</div>'+
      ((job.output.gate_failures||[]).length?'<div class="job-message"><strong>Cause del rifiuto:</strong> '+esc(job.output.gate_failures.join(' · '))+'</div>':'')+
      ((job.output.segment_regressions||[]).length?'<div class="summary-list"><div class="summary-row"><span>Segmenti con regressione (dettaglio)</span><strong>'+esc(job.output.segment_regression_count||job.output.segment_regressions.length)+'</strong></div>'+job.output.segment_regressions.map(s=>'<div class="summary-row"><span>'+esc((s.dimension||'segmento')+' · '+(s.group||'—')+' (righe='+String(s.rows_evaluated??s.n??'—')+', giocatori='+String(s.unique_players??'—')+')')+'</span><strong>MAE '+esc(fmt(s.mae))+' / baseline '+esc(fmt(s.baseline_mae))+'</strong></div>').join('')+'</div>':'')
    :'';
  $('#bb-rating-prediction-job').innerHTML='<div class="job-head"><div><div class="job-status">'+esc(job.stage||'Ready')+'</div><div class="job-message">'+esc(job.error||job.message||'Nessun training eseguito.')+'</div></div><span class="registry-badge '+(['running','queued'].includes(job.status)?'candidate':'')+'">'+esc(job.status||'idle')+'</span></div>'+
    outputMarkup+
    progressMarkup(job);
  if(['running','queued'].includes(job.status)) {
    $('#bb-rating-prediction-job').insertAdjacentHTML('beforeend','<div class="button-row"><button id="bb-rating-prediction-cancel" class="btn btn-danger btn-small" '+(job.cancel_requested?'disabled':'')+'>'+(job.cancel_requested?'Interruzione richiesta…':'Interrompi training')+'</button></div>');
    const cancel=$('#bb-rating-prediction-cancel');
    cancel.onclick=async()=>{cancel.disabled=true;try{await api('/admin-api/jobs/bb-rating-prediction/cancel',{method:'POST',body:{}});toast('Interruzione training richiesta');await renderModelReleases();}catch(e){toast(e.message,'error');cancel.disabled=false;}};
  }
  const start=$('#bb-rating-prediction-start');
  start.disabled=['running','queued'].includes(job.status);
  start.textContent=['running','queued'].includes(job.status)?'Training in corso…':'Allena candidate';
  start.onclick=async()=>{
    start.disabled=true;
    try{await api('/admin-api/bb-rating-prediction/train',{method:'POST',body:{}});toast('Training BB Prediction Model avviato');await renderModelReleases();}
    catch(e){toast(e.message,'error');start.disabled=false;}
  };
  $('#model-releases-refresh').onclick=()=>renderModelReleases().catch(e=>toast(e.message,'error'));

  const causalButtons=[$('#causal-refresh-team-switch'),$('#causal-refresh-teammate'),$('#causal-teammate-diagnose')].filter(Boolean);
  const causalResult=$('#causal-refresh-result');
  const causalProgress=$('#causal-refresh-progress');
  const causalCancel=$('#causal-refresh-cancel');
  const showCausalJob=job=>{
    if(!causalResult||!job)return;
    if(causalProgress)causalProgress.innerHTML=progressMarkup(job);
    const cancelAvailable=['queued','running'].includes(job.status);
    if(causalCancel){
      causalCancel.classList.toggle('hidden',!cancelAvailable);
      causalCancel.disabled=!cancelAvailable||!!job.cancel_requested;
      causalCancel.textContent=job.cancel_requested?'Interruzione richiesta…':'Interrompi refresh';
    }
    const lines=[job.message||'Refresh causale in attesa.'];
    if(job.stage)lines.unshift(job.stage+(Number.isFinite(Number(job.progress))?' · '+Number(job.progress)+'%':''));
    if(job.summary){
      const s=job.summary;
      lines.push('Contesti: '+(s.contexts_total??0)+' · aggiornati: '+(s.contexts_refreshed??0)+' · senza coorte: '+(s.contexts_skipped??0)+' · errori: '+(s.contexts_failed??0)+' · righe scritte: '+(s.rows_written??0));
    }
    if(job.error)lines.push('Errore: '+job.error);
    const issues=(job.results||[]).filter(row=>row.status==='failed'||row.status==='not_written');
    if(issues.length)lines.push('Contesti con anomalie:\n'+issues.slice(0,12).map(row=>row.league+'/'+row.competition+' ('+row.through_season+'): '+(row.reason||row.status)).join('\n'));
    causalResult.textContent=lines.join('\n');
    const active=['queued','running'].includes(job.status);
    causalButtons.forEach(button=>button.disabled=active);
  };
  const pollCausalJob=async()=>{
    const job=await api('/admin-api/causal-panels/refresh-all');
    showCausalJob(job);
    return job;
  };
  const runCausalRefresh=async kind=>{
    if(causalButtons.some(button=>button.disabled))return;
    causalButtons.forEach(button=>button.disabled=true);
    if(causalResult)causalResult.textContent='Avvio del refresh globale…';
    try{
      const started=await api('/admin-api/causal-panels/refresh-all',{method:'POST',body:{kind}});
      let job=started.job||{};
      showCausalJob(job);
      while(['queued','running'].includes(job.status)){
        await new Promise(resolve=>setTimeout(resolve,1500));
        job=await pollCausalJob();
      }
      if(job.status==='complete')toast('Refresh causale completato su tutte le leghe e competizioni','success');
      else if(job.status==='partial')toast('Refresh causale concluso con alcuni errori; controlla il dettaglio','error');
      else toast('Refresh causale non completato: '+(job.error||job.message||'errore'),'error');
    }catch(e){
      if(causalResult)causalResult.textContent='Refresh causale non riuscito: '+e.message;
      toast('Refresh causale non riuscito: '+e.message,'error');
    }finally{
      try{await pollCausalJob();}catch(_){}
    }
  };
  const causalTeammateDiagnosticsResult=$('#causal-teammate-diagnostics-result');
  const causalTeammateDiagnosticsDownload=$('#causal-teammate-diagnostic-download');
  const runTeammateSupportDiagnostics=async()=>{
    if(causalButtons.some(button=>button.disabled))return;
    causalButtons.forEach(button=>button.disabled=true);
    if(causalTeammateDiagnosticsResult)causalTeammateDiagnosticsResult.textContent='Verifica supporto, overlap e bilanciamento per compagno/lega/competizione/stagione…';
    if(causalTeammateDiagnosticsDownload){
      causalTeammateDiagnosticsDownload.classList.add('hidden');
      causalTeammateDiagnosticsDownload.href='#';
    }
    try{
      const report=await api('/admin-api/causal-panels/teammate-support-diagnostics',{method:'POST',body:{}});
      const s=report.summary||{};
      const lines=[
        'Righe persistite: '+(s.panel_rows??0)+' · gruppi distinti: '+(s.groups_total??0),
        'Righe complete per il modello: '+(s.total_eligible_complete_case_rows??0)+' · righe escluse: '+(s.total_excluded_complete_case_rows??0),
        'Gruppi con supporto grezzo sufficiente (≥60 righe, ≥20 trattati, ≥20 controlli): '+(s.groups_raw_support_passed??0),
        'Gruppi con gate statistici superati: '+(s.groups_with_statistical_gates_passed??0)+' · diagnostiche eseguite: '+(s.groups_diagnostics_evaluated??0),
        'Gruppi non valutati per limite operativo: '+(s.groups_not_evaluated_due_limit??0),
        'Gruppi con problemi di integrità delle identità: '+(s.groups_with_identity_integrity_issues??0),
        'Righe non classificabili per contesto incompleto: '+(s.unclassified_panel_rows??0),
        'Ultimo refresh: '+(s.last_refresh||'non disponibile')
      ];
      const statusCounts=s.status_counts||{};
      const statuses=Object.entries(statusCounts).sort((a,b)=>b[1]-a[1]).map(([name,count])=>name+': '+count);
      if(statuses.length)lines.push('Esiti: '+statuses.join(' · '));
      const missingness=s.missingness_by_field||{};
      const missingnessLabels={
        missing_baseline_bb_rating_rows:'BB-Rating baseline',
        missing_minutes_pre_rows:'minuti pre-trattamento',
        missing_usage_pre_rows:'usage pre-trattamento',
        missing_age_pre_rows:'età',
        missing_team_strength_pre_rows:'forza squadra pre-trattamento',
        invalid_outcome_rows:'BB-Rating outcome non valido',
        invalid_treatment_rows:'trattamento non valido',
        missing_focal_identity_rows:'identità giocatore',
        missing_teammate_identity_rows:'identità compagno',
        missing_league_rows:'lega',
        missing_competition_rows:'competizione',
        missing_transition_season_rows:'stagione di transizione'
      };
      const missingnessTop=Object.entries(missingness)
        .filter(([key,value])=>missingnessLabels[key]&&Number(value?.rows)>0)
        .sort((a,b)=>Number(b[1]?.percent||0)-Number(a[1]?.percent||0))
        .slice(0,8);
      if(missingnessTop.length){
        lines.push('Valori mancanti/non validi per campo (le percentuali si sovrappongono):');
        missingnessTop.forEach(([key,value])=>{
          lines.push('• '+missingnessLabels[key]+': '+Number(value.rows||0).toLocaleString('it-IT')+' ('+Number(value.percent||0).toFixed(2)+'%)');
        });
      }
      const best=s.best_supported_groups||[];
      if(best.length){
        lines.push('Gruppi con più trattati:');
        best.slice(0,8).forEach(row=>{
          let detail=String(row.teammate_global_id||'ID sconosciuto')+' · '+String(row.league_key||'?')+'/'+String(row.competition||'?')+' '+String(row.transition_season||'?');
          detail+=' — trattati '+(row.treated??0)+' / controlli '+(row.control??0)+' — '+String(row.status||'n/d');
          if(row.propensity_overlap_share!==null&&row.propensity_overlap_share!==undefined)detail+=' — overlap '+(Number(row.propensity_overlap_share)*100).toFixed(1)+'%';
          if(row.max_abs_smd_after!==null&&row.max_abs_smd_after!==undefined)detail+=' — max |SMD| '+Number(row.max_abs_smd_after).toFixed(3);
          lines.push(detail);
        });
      }
      if(s.interpretation)lines.push(s.interpretation);
      if(causalTeammateDiagnosticsResult)causalTeammateDiagnosticsResult.textContent=lines.join('\n');
      if(causalTeammateDiagnosticsDownload&&report.download_url){
        causalTeammateDiagnosticsDownload.href=report.download_url;
        causalTeammateDiagnosticsDownload.classList.remove('hidden');
      }
      const passed=Number(s.groups_with_statistical_gates_passed||0);
      toast(
        passed>0
          ? 'Diagnostica completata: alcuni gruppi superano i gate statistici, ma restano osservazionali.'
          : 'Diagnostica completata: nessun gruppo ha superato tutti i gate statistici.',
        'success'
      );
    }catch(e){
      if(causalTeammateDiagnosticsResult)causalTeammateDiagnosticsResult.textContent='Diagnostica non riuscita: '+e.message;
      toast('Diagnostica teammate exposure non riuscita: '+e.message,'error');
    }finally{
      try{await pollCausalJob();}catch(_){causalButtons.forEach(button=>button.disabled=false);}
    }
  };
  $('#causal-teammate-diagnose')?.addEventListener('click',runTeammateSupportDiagnostics);
  $('#causal-refresh-team-switch')?.addEventListener('click',()=>runCausalRefresh('team_switch'));
  $('#causal-refresh-teammate')?.addEventListener('click',()=>runCausalRefresh('teammate_exposure'));
  causalCancel?.addEventListener('click',async()=>{
    causalCancel.disabled=true;
    try{
      const result=await api('/admin-api/causal-panels/refresh-all/cancel',{method:'POST',body:{}});
      showCausalJob(result.job||await pollCausalJob());
      toast(result.message||'Interruzione refresh richiesta');
    }catch(e){
      toast(e.message,'error');
      causalCancel.disabled=false;
    }
  });
  pollCausalJob().catch(()=>{});


  $$('.release-promote').forEach(button=>{
    const model=button.dataset.model;
    const item=models[model]||{};
    button.disabled=!(item.candidate&&item.candidate.valid);
    button.onclick=async()=>{
      if(!confirm('Promuovere il candidate '+model+'? Il sistema riavvierà automaticamente l’API e verificherà la readiness.'))return;
      button.disabled=true;
      try{const result=await api('/admin-api/model-releases/'+encodeURIComponent(model)+'/promote',{method:'POST',body:{confirm:true,candidate_id:item.candidate?.id}});const restart=result.api_restart||{};$('#model-release-result').textContent=(result.message||'Promozione completata')+' Versione: '+(result.version||'—')+(restart.ok?' · API riavviata e pronta.':' · ATTENZIONE: riavvio non confermato: '+(restart.error||'verifica necessaria'));toast(restart.ok?'Candidate promosso; API pronta':'Promozione eseguita ma API non confermata pronta',restart.ok?'success':'error');await renderModelReleases();}
      catch(e){toast(e.message,'error');button.disabled=false;}
    };
  });
  $$('.release-rollback').forEach(button=>{
    const model=button.dataset.model;
    const item=models[model]||{};
    button.disabled=!(item.previous&&item.previous.valid);
    button.onclick=async()=>{
      if(!confirm('Ripristinare la release precedente di '+model+'? Il sistema riavvierà automaticamente l’API e verificherà la readiness.'))return;
      button.disabled=true;
      try{const result=await api('/admin-api/model-releases/'+encodeURIComponent(model)+'/rollback',{method:'POST',body:{confirm:true}});const restart=result.api_restart||{};$('#model-release-result').textContent=(result.message||'Rollback completato')+' Versione: '+(result.version||'—')+(restart.ok?' · API riavviata e pronta.':' · ATTENZIONE: riavvio non confermato: '+(restart.error||'verifica necessaria'));toast(restart.ok?'Rollback completato; API pronta':'Rollback eseguito ma API non confermata pronta',restart.ok?'success':'error');await renderModelReleases();}
      catch(e){toast(e.message,'error');button.disabled=false;}
    };
  });
  if(['running','queued'].includes(job.status)&&App.page==='model-releases')App.bbRatingPredictionTimer=setTimeout(()=>renderModelReleases().catch(e=>toast(e.message,'error')),2500);
}

function chartSvg(points){if(!points||points.length<1)return empty('⌁','Dati non disponibili','I fold walk-forward appariranno dopo il training.');const vals=points.map(p=>Number(p.rmse)).filter(Number.isFinite);if(!vals.length)return '';const min=Math.min(...vals),max=Math.max(...vals),range=Math.max(max-min,.001),w=640,h=210,pad=34;const coords=points.map((p,i)=>{const x=pad+(i*(w-pad*2)/Math.max(1,points.length-1));const y=h-pad-((Number(p.rmse)-min)/range)*(h-pad*2);return{x,y,p};});return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><line class="chart-grid-line" x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}"/><line class="chart-grid-line" x1="${pad}" y1="${pad}" x2="${w-pad}" y2="${pad}"/><polyline class="chart-line" points="${coords.map(c=>`${c.x},${c.y}`).join(' ')}"/>${coords.map(c=>`<circle class="chart-dot" cx="${c.x}" cy="${c.y}" r="5"/><text class="chart-label" x="${c.x}" y="${h-9}" text-anchor="middle">${esc(c.p.target_season??'')}</text>`).join('')}</svg>`;}
async function renderBacktests(){const d=await api('/admin-api/backtests'),r=d.report||{},o=r.overall||{},q=d.quality||{};$('#backtest-kpis').innerHTML=[kpi('⌁','Ensemble RMSE',fmt(o.rmse),'<span>OOT</span>'),kpi('↗','Base RMSE',fmt(r.base_rmse),'<span>XGBoost base</span>','green'),kpi('◫','Persistence RMSE',fmt(r.persistence_rmse),'<span>previous season</span>','violet'),kpi('◎','Interval coverage',pct(o.interval_coverage),'<span>conformal</span>','orange')].join('');$('#backtest-chart').innerHTML=chartSvg(q.folds||[]);$('#backtest-summary').innerHTML=d.candidate?.run_id?summaryRows([['Run ID',shortId(d.candidate.run_id)],['Samples',o.n??'—'],['Competitions',q.competitions??'—'],['Valid',r.valid?'Yes':'No']]):empty('▤','Nessun candidate','Completa un training run per ispezionare il report.','Avvia training','training');const comps=Object.entries(r.by_competition||{});$('#competition-body').innerHTML=comps.length?comps.map(([name,m])=>`<tr><td><strong>${esc(name)}</strong></td><td>${esc(m.n??'—')}</td><td>${esc(fmt(m.rmse))}</td><td>${esc(fmt(m.mae))}</td><td>${esc(fmt(m.bias))}</td><td>${esc(pct(m.interval_coverage))}</td></tr>`).join(''):'<tr><td colspan="6" class="muted-inline">Nessuna metrica per competition disponibile.</td></tr>';$$$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));}

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
async function renderRegistry(){const d=await api('/admin-api/registry');$('#top-model').textContent=shortId(d.production?.run_id);$('#registry-cards').innerHTML=registryCard('Production',d.production)+registryCard('Candidate',d.candidate,'candidate');const hist=d.history||[];$('#registry-history').innerHTML=hist.length?[...hist].reverse().map(h=>`<tr><td>${esc(h.at||'—')}</td><td>${esc(h.event||'—')}</td><td>${esc(h.run_id||'—')}</td></tr>`).join(''):'<tr><td colspan="3" class="muted-inline">Nessun evento nel registry.</td></tr>';const check=$('#registry-confirm'),prom=$('#promote-btn'),roll=$('#rollback-btn');check.addEventListener('change',()=>{prom.disabled=!check.checked||!d.candidate;roll.disabled=!check.checked||!d.previous;});prom.addEventListener('click',async()=>{prom.disabled=true;try{const r=await api('/admin-api/registry/promote',{method:'POST',body:{confirm:true}});toast(r.api_restart?.ok?'Candidate promoted; API ready':(r.api_restart?.error||r.reason||'Promotion completed, but API restart was not verified'),r.api_restart?.ok?'success':'error');await renderRegistry();}catch(e){toast(e.message,'error');prom.disabled=false;}});roll.addEventListener('click',async()=>{roll.disabled=true;try{const r=await api('/admin-api/registry/rollback',{method:'POST',body:{confirm:true}});toast(r.api_restart?.ok?'Rollback completed; API ready':(r.api_restart?.error||r.reason||'Rollback completed, but API restart was not verified'),r.api_restart?.ok?'success':'error');await renderRegistry();}catch(e){toast(e.message,'error');roll.disabled=false;}});}


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

$('#login-form').addEventListener('submit',async ev=>{ev.preventDefault();const error=$('#login-error'),button=$('button',ev.currentTarget);error.textContent='';button.disabled=true;try{const d=await api('/admin-api/login',{method:'POST',body:{username:$('#login-username').value,password:$('#login-password').value}});await enterBackOffice(d.user);}catch(e){error.textContent=e.message;}finally{button.disabled=false;}});
$('#logout-btn').addEventListener('click',async()=>{try{await api('/admin-api/logout',{method:'POST',body:{}});}catch(_){}showLogin();});
$$('.nav-item').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));

(async function boot(){try{const s=await api('/admin-api/session');await enterBackOffice(s.user);}catch(_){showLogin();}})();


async function renderAiJobs(){
  const errorNode=$('#ai-jobs-error');
  const scheduleForm=$('#ai-jobs-schedule-form');
  const intervalField=$('#ai-schedule-interval-field');
  const intervalInput=scheduleForm?.interval_hours;
  const dailyTimeField=$('#ai-schedule-daily-time-field');
  const dailyTimeInput=scheduleForm?.daily_time;
  const leagueField=$('#ai-schedule-league-field');
  const leaguePicker=$('#ai-schedule-league-picker');
  const leagueToggle=$('#ai-schedule-league-toggle');
  const leagueSummary=$('#ai-schedule-league-summary');
  const leagueOptions=$('#ai-schedule-league-options');

  const setLeagueDropdownOpen=open=>{
    if(!leagueOptions||!leagueToggle)return;
    leagueOptions.classList.toggle('hidden',!open);
    leagueToggle.setAttribute('aria-expanded',String(open));
  };
  leagueToggle?.addEventListener('click',()=>{
    setLeagueDropdownOpen(leagueOptions.classList.contains('hidden'));
  });
  leaguePicker?.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      setLeagueDropdownOpen(false);
      leagueToggle?.focus();
    }
  });
  // Bind one delegated outside-click handler for all future mounts of this template.
  if(!window.__aiScheduleLeagueOutsideClickBound){
    document.addEventListener('click',event=>{
      const picker=document.querySelector('#ai-schedule-league-picker');
      const options=document.querySelector('#ai-schedule-league-options');
      const toggle=document.querySelector('#ai-schedule-league-toggle');
      if(picker&&!picker.contains(event.target)){
        options?.classList.add('hidden');
        toggle?.setAttribute('aria-expanded','false');
      }
    });
    window.__aiScheduleLeagueOutsideClickBound=true;
  }

  const selectedLeagues=()=>{
    const allBox=$('#ai-schedule-all-leagues');
    if(allBox?.checked)return ['ALL'];
    return $$('input[name="schedule_league"]:checked',leagueOptions).map(input=>input.value);
  };
  const syncLeagueSelection=()=>{
    const allBox=$('#ai-schedule-all-leagues');
    const boxes=$$('input[name="schedule_league"]',leagueOptions);
    if(!allBox)return;
    boxes.forEach(input=>{input.disabled=allBox.checked;});
    const selected=boxes.filter(input=>input.checked).map(input=>input.value);
    if(leagueSummary){
      leagueSummary.textContent=allBox.checked?'Tutte':selected.length===1?selected[0]:selected.length+' leghe selezionate';
      if(!allBox.checked&&!selected.length)leagueSummary.textContent='Seleziona leghe';
    }
  };
  const loadScheduleOptions=async(preserveSelection=false)=>{
    const previousSelection=preserveSelection?selectedLeagues():['ALL'];
    const data=await api('/admin-api/scheduler/options');
    const leagues=[...new Set((data.leagues||[]).map(value=>String(value).trim().toUpperCase()).filter(Boolean))].sort();
    if(!leagueOptions)return;
    const allSelected=previousSelection.includes('ALL')||previousSelection.length===0;
    leagueOptions.innerHTML=
      '<label class="schedule-league-option schedule-league-all"><input id="ai-schedule-all-leagues" type="checkbox" '+(allSelected?'checked':'')+'> <span><strong>Tutte</strong><small>Tutte le leghe attualmente presenti</small></span></label>'+
      leagues.map(league=>'<label class="schedule-league-option"><input type="checkbox" name="schedule_league" value="'+esc(league)+'" '+(!allSelected&&previousSelection.includes(league)?'checked':'')+'> <span><strong>'+esc(league)+'</strong></span></label>').join('');
    $('#ai-schedule-all-leagues')?.addEventListener('change',()=>{
      if($('#ai-schedule-all-leagues')?.checked){
        $$('input[name="schedule_league"]',leagueOptions).forEach(input=>{input.checked=false;});
      }
      syncLeagueSelection();
    });
    $$('input[name="schedule_league"]',leagueOptions).forEach(input=>input.addEventListener('change',()=>{
      if(input.checked&&$('#ai-schedule-all-leagues'))$('#ai-schedule-all-leagues').checked=false;
      syncLeagueSelection();
    }));
    syncLeagueSelection();
  };
  const syncScheduleFields=()=>{
    const task=scheduleForm?.task_key.value||'';
    const needsLeague=['refresh_simulation_sources','refresh_scenario_serving','refresh_training_serving'].includes(task);
    if(leagueField)leagueField.classList.toggle('hidden',!needsLeague);
    if(!needsLeague)setLeagueDropdownOpen(false);
    const daily=scheduleForm?.schedule_mode.value==='daily';
    if(intervalField)intervalField.classList.toggle('hidden',daily);
    if(intervalInput)intervalInput.required=!daily;
    if(dailyTimeField)dailyTimeField.classList.toggle('hidden',!daily);
    if(dailyTimeInput)dailyTimeInput.required=daily;
  };
  const taskFields=()=>{
    const task=scheduleForm.task_key.value;
    const params={};
    if(['refresh_context_seasonal','refresh_simulation_sources','refresh_scenario_serving','refresh_training_serving'].includes(task)){
      params.season=Number(scheduleForm.season.value);
    }
    if(['refresh_simulation_sources','refresh_scenario_serving','refresh_training_serving'].includes(task)){
      params.leagues=selectedLeagues();
    }
    return params;
  };
  const refreshSchedules=async()=>{
    const data=await api('/admin-api/scheduler/jobs');
    $('#ai-jobs-schedule-list').innerHTML=(data.jobs||[]).map(job=>{
      const parameters=job.parameters&&typeof job.parameters==='object'?job.parameters:{};
      const leagueSummary=Array.isArray(parameters.leagues)?(parameters.leagues.includes('ALL')?'Tutte le leghe':parameters.leagues.join(', ')):(parameters.league||'');
      const compSummary=parameters.competition?' · legacy '+esc(parameters.competition):' · tutte le competizioni presenti';
      const paramsSummary=leagueSummary?' · '+esc(leagueSummary)+compSummary:'';
      const status=job.last_status||'programmato';
      const error=job.last_error?' · '+esc(job.last_error):'';
      const cadence=job.schedule_mode==='daily'
        ?'ogni giorno alle '+esc(String(job.daily_time||'—').slice(0,5))+' (Europe/Rome)'
        :'ogni '+esc(Math.round(job.interval_minutes/60))+' h';
      const dateFrom=job.start_date?' · dal '+esc(new Date(String(job.start_date).slice(0,10)+'T00:00:00').toLocaleDateString('it-IT')):'';
      const dateTo=job.end_date?' · fino al '+esc(new Date(String(job.end_date).slice(0,10)+'T00:00:00').toLocaleDateString('it-IT')):'';
      const next=job.next_run_at?new Date(job.next_run_at).toLocaleString('it-IT',{timeZone:'Europe/Rome'}):'—';
      return '<div class="summary-row schedule-job-row"><span><strong>'+esc(job.name)+'</strong><br><small>'+esc(job.task_key)+' · '+cadence+dateFrom+dateTo+paramsSummary+' · prossimo: '+esc(next)+error+'</small></span><span class="schedule-job-actions"><strong>'+esc(status)+'</strong> <label class="schedule-switch" title="'+(job.enabled?'Disattiva pianificazione':'Attiva pianificazione')+'"><input type="checkbox" data-toggle-schedule="'+job.job_id+'" '+(job.enabled?'checked':'')+' aria-label="'+(job.enabled?'Disattiva':'Attiva')+' '+esc(job.name)+'"><span class="schedule-switch-track" aria-hidden="true"></span></label> <button class="btn btn-danger btn-small" data-delete-schedule="'+job.job_id+'">Elimina</button></span></div>';
    }).join('')||'<p class="microcopy">Nessuna pianificazione configurata.</p>';
    $('#ai-jobs-run-history').innerHTML=(data.runs||[]).map(run=>{
      const details=run.result&&typeof run.result==='object'?run.result:{};
      const count=details.contexts_total!==undefined?' · '+(details.contexts_completed||0)+'/'+details.contexts_total+' contesti':'';
      return '<div class="summary-row"><span>'+esc(run.task_key)+' · '+esc(run.started_at||'—')+count+'</span><strong>'+esc(run.status)+(run.error?' · '+esc(run.error):'')+'</strong></div>';
    }).join('')||'<p class="microcopy">Nessuna esecuzione registrata.</p>';
    $$('[data-toggle-schedule]').forEach(toggle=>toggle.addEventListener('change',async()=>{
      const requested=toggle.checked;
      toggle.disabled=true;
      try{
        await api('/admin-api/scheduler/jobs/'+toggle.dataset.toggleSchedule+'/toggle',{method:'POST',body:{}});
        await refreshSchedules();
      }catch(e){
        toggle.checked=!requested;
        toggle.disabled=false;
        toast(e.message,'error');
      }
    }));
    $$('[data-delete-schedule]').forEach(button=>button.addEventListener('click',async()=>{
      if(!confirm('Eliminare questa pianificazione?'))return;
      button.disabled=true;
      try{await api('/admin-api/scheduler/jobs/'+button.dataset.deleteSchedule,{method:'DELETE'});await refreshSchedules();}
      catch(e){toast(e.message,'error');button.disabled=false;}
    }));
  };
  $('#ai-jobs-schedule-refresh')?.addEventListener('click',()=>refreshSchedules().catch(e=>toast(e.message,'error')));
  $('#ai-jobs-refresh')?.addEventListener('click',()=>Promise.all([refreshSchedules(),loadScheduleOptions(true)]).catch(e=>toast(e.message,'error')));
  scheduleForm?.task_key.addEventListener('change',syncScheduleFields);
  scheduleForm?.schedule_mode.addEventListener('change',syncScheduleFields);
  scheduleForm?.addEventListener('submit',async event=>{
    event.preventDefault();
    const task=scheduleForm.task_key.value;
    const parameters=taskFields();
    const scheduleMode=scheduleForm.schedule_mode.value;
    const startDate=scheduleForm.start_date.value||null;
    const endDate=scheduleForm.end_date.value||null;
    if(scheduleMode==='daily'&&!scheduleForm.daily_time.value){toast('Seleziona l’orario giornaliero','error');return;}
    if(startDate&&endDate&&endDate<startDate){toast('La data fine deve essere uguale o successiva alla data inizio','error');return;}
    const intervalHours=Number(scheduleForm.interval_hours.value);
    if(['refresh_context_seasonal','refresh_simulation_sources','refresh_scenario_serving','refresh_training_serving'].includes(task)){
      if(!Number.isInteger(parameters.season)||parameters.season<2000||parameters.season>2100){toast('Seleziona una stagione valida','error');return;}
    }
    if(['refresh_simulation_sources','refresh_scenario_serving','refresh_training_serving'].includes(task)&&!parameters.leagues?.length){
      toast('Seleziona Tutte le leghe oppure almeno una lega','error');return;
    }
    const button=$('button[type="submit"]',scheduleForm);
    button.disabled=true;
    try{
      await api('/admin-api/scheduler/jobs',{method:'POST',body:{
        name:scheduleForm.elements.namedItem('name').value.trim(),task_key:task,
        interval_minutes:scheduleMode==='daily'?1440:intervalHours*60,
        schedule_mode:scheduleMode,
        daily_time:scheduleMode==='daily'?scheduleForm.daily_time.value:null,
        start_date:startDate,end_date:endDate,
        enabled:scheduleForm.enabled.checked,
        parameters
      }});
      toast('Pianificazione salvata');
      scheduleForm.reset();
      scheduleForm.interval_hours.value='24';
      await loadScheduleOptions();
      syncScheduleFields();
      await refreshSchedules();
    }catch(e){toast('Salvataggio pianificazione non riuscito: '+e.message,'error');}
    finally{button.disabled=false;}
  });
  try{
    await loadScheduleOptions();
    syncScheduleFields();
    await refreshSchedules();
  }catch(e){
    if(errorNode)errorNode.textContent='Scheduler non disponibile: verifica la connessione PostgreSQL e i permessi sulle tabelle canoniche. '+e.message;
  }
}

