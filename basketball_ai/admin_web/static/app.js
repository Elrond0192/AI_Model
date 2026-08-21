const App={user:null,page:'overview',overview:null,trainingTimer:null};
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
  const res=await fetch(path,{credentials:'same-origin',...options,headers});
  let data={};try{data=await res.json();}catch(_){data={};}
  if(res.status===401){showLogin();throw new Error('Sessione scaduta');}
  if(!res.ok)throw new Error(data.detail||data.message||`HTTP ${res.status}`);
  return data;
}

function showLogin(){App.user=null;$('#app-shell').classList.add('hidden');$('#login-screen').classList.remove('hidden');}
function showApp(user){App.user=user;$('#login-screen').classList.add('hidden');$('#app-shell').classList.remove('hidden');$('#user-name').textContent=user.username;$('#user-role').textContent=user.role==='admin'?'Administrator':user.role;$('#user-avatar').textContent=(user.username||'A')[0].toUpperCase();$$('.admin-only').forEach(el=>el.classList.toggle('hidden',user.role!=='admin'));}
function updateClock(){const d=new Date();$('#top-utc').textContent=d.toISOString().replace('T',' ').slice(0,19);}
setInterval(updateClock,1000);updateClock();

function activateNav(page){$$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.page===page));}
function mountTemplate(page){const t=$(`#page-${page}`);if(!t){root.innerHTML='<div class="loading">Pagina non disponibile</div>';return;}root.replaceChildren(t.content.cloneNode(true));$$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));}
async function navigate(page){clearTimeout(App.trainingTimer);App.page=page;activateNav(page);mountTemplate(page);try{if(page==='overview')await renderOverview();if(page==='data')await renderData();if(page==='training')await renderTraining();if(page==='backtests')await renderBacktests();if(page==='scenarios')await renderScenarios();if(page==='registry')await renderRegistry();if(page==='health')await renderHealth();if(page==='audit')await renderAudit();if(page==='settings')await renderSettings();}catch(e){toast(e.message,'error');}}

function kpi(icon,label,value,sub='',tone=''){return `<div class="kpi-card"><div class="kpi-icon ${tone}">${icon}</div><div class="kpi-label">${esc(label)}</div><div class="kpi-value ${tone==='green'?'online':''}">${esc(value)}</div><div class="kpi-sub">${sub}</div></div>`;}
function updateTop(o){$('#top-db').textContent=o.database||'not selected';$('#top-model').textContent=shortId(o.production?.run_id);$('#top-api').textContent=o.api?.online?'● V2':'● OFFLINE';$('#top-api').className=o.api?.online?'meta-ok':'';}

async function renderOverview(){
  const o=await api('/admin-api/overview');App.overview=o;updateTop(o);
  const s=o.summary||{};$('#overview-kpis').innerHTML=[
    kpi('▤','Database',o.database||'Not selected',`<span class="dot ${o.data_loaded?'green':'yellow'}"></span>${o.data_loaded?'Connesso e caricato':'Da caricare'}`),
    kpi('▥','Dati caricati',fmtInt(s.player_rows),'<span>righe player/competition</span>','green'),
    kpi('◉','Competitions',String((s.competitions||[]).length||'—'),'<span>caricate</span>','violet'),
    kpi('⬡','Modello attivo',shortId(o.production?.run_id),o.production?'<span class="dot green"></span>Production':'<span class="dot red"></span>Not promoted','orange'),
    kpi('⌘','Stato API',o.api?.online?'Online':'Offline',`<span class="dot ${o.api?.online?'green':'red'}"></span>V2`,'green')
  ].join('');
  $('#readiness-body').innerHTML=o.readiness.map(r=>`<tr><td><strong>${esc(r.component)}</strong></td><td><span class="status ${r.status}"><span class="status-dot">${r.status==='ok'?'✓':r.status==='danger'?'×':'△'}</span>${esc(r.label)}</span></td><td>${esc(r.detail)}</td><td class="right"><button class="btn btn-small btn-secondary readiness-action" data-page="${esc(r.action==='data'?'data':r.action)}">${r.action==='registry'?'Apri registry':r.action==='training'?'Apri training':'Apri dati'}</button></td></tr>`).join('');
  $$('.readiness-action').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));
  const prod=o.production||{},cand=o.candidate||{};
  $('#overview-registry').innerHTML=(prod.run_id||cand.run_id)?summaryRows([['Production',prod.run_id?shortId(prod.run_id):'—'],['Candidate',cand.run_id?shortId(cand.run_id):'—'],['Candidate RMSE',cand.overall_rmse!==undefined?fmt(cand.overall_rmse):'—']]):empty('⬡','Nessun modello registrato','Esegui un training per generare il primo candidate.','Avvia training','training');
  const q=o.candidate_quality||{};$('#overview-backtest').innerHTML=q.rmse!==null&&q.rmse!==undefined?summaryRows([['RMSE',fmt(q.rmse)],['MAE',fmt(q.mae)],['Coverage',pct(q.coverage)],['Folds',String(q.seasons||0)]]):empty('▤','Nessun backtest disponibile','Esegui un training per generare un candidate e il suo backtest.','Esegui training','training');
  const quality=[['RMSE',fmt(q.rmse)],['MAE',fmt(q.mae)],['R²',fmt(q.r2)],['Coverage',pct(q.coverage)],['Stagioni',q.seasons??'—'],['Competitions',q.competitions??'—']];
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
  $('#refresh-profiles').addEventListener('click',loadProfiles);$('#refresh-snapshots').addEventListener('click',loadSnapshots);$('#profile-form').addEventListener('submit',async ev=>{ev.preventDefault();const form=new FormData(ev.currentTarget),body=Object.fromEntries(form.entries());body.port=Number(body.port);const btn=$('button[type="submit"]',ev.currentTarget);btn.disabled=true;try{await api('/admin-api/profiles',{method:'POST',body});toast('Profilo salvato');ev.currentTarget.querySelector('[name=password]').value='';await loadProfiles();}catch(e){toast(e.message,'error');}finally{btn.disabled=false;}});await Promise.all([loadProfiles(),loadSnapshots()]);
}

async function renderTraining(){
  const d=await api('/admin-api/training'),s=d.summary||{},j=d.job||{};
  const datasetLoaded=s.player_rows!==undefined&&s.player_rows!==null;
  $('#training-kpis').innerHTML=[
    kpi('▤','Snapshot training',d.snapshot?.snapshot_id?shortId(d.snapshot.snapshot_id):'—',`<span class="dot ${datasetLoaded?'green':'yellow'}"></span>${datasetLoaded?'Storico pronto':'Prepara lo storico in Dati e snapshot'}`),
    kpi('◫','Season range',s.season_min?`${s.season_min}–${s.season_max}`:'—','<span>training history</span>','green'),
    kpi('◉','Competitions',String((s.competitions||[]).length||'—'),'<span>observed</span>','violet'),
    kpi('▥','Observations',fmtInt(s.player_rows),'<span>player rows</span>','orange')
  ].join('');
  $('#training-contract').innerHTML=(d.contract||[]).map(r=>`<tr><td><strong>${esc(r.stage)}</strong></td><td>${esc(r.seasons)}</td><td>${esc(r.purpose)}</td></tr>`).join('');
  $('#training-config').innerHTML=Object.entries(d.configuration||{}).map(([a,b])=>`<div class="config-item"><span>${esc(a.replaceAll('_',' '))}</span><strong>${esc(Array.isArray(b)?b.join(', '):b)}</strong></div>`).join('');
  const job=$('#training-job');
  const idleMessage=!datasetLoaded&&j.status==='idle'?'Prima prepara e attiva uno snapshot nella pagina Dati e snapshot. Il training non carica dati automaticamente.':(j.error||j.message||'');
  job.innerHTML=`<div class="job-head"><div><div class="job-status">${esc(j.stage||'Ready')}</div><div class="job-message">${esc(idleMessage)}</div></div><span class="registry-badge ${j.status==='running'?'candidate':''}">${esc(j.status||'idle')}</span></div><div class="progress-track"><div class="progress-bar" style="width:${Math.max(0,Math.min(100,Number(j.progress)||0))}%"></div></div>${j.run_id?`<div class="job-message">Run ID: ${esc(j.run_id)}</div>`:''}`;
  const start=$('#start-training');
  start.disabled=!datasetLoaded||!d.snapshot?.snapshot_id||j.status==='running';
  start.addEventListener('click',async()=>{
    const originalText=start.textContent;
    start.disabled=true;
    try{
      if(!datasetLoaded||!d.snapshot?.snapshot_id)throw new Error('Prepara e attiva prima uno snapshot in Dati e snapshot');
      start.textContent='Controllo snapshot…';
      const loaded={summary:s,snapshot:d.snapshot};
      const seasons=loaded.summary?.seasons||[];
      if(seasons.length<5)throw new Error(`Servono almeno cinque stagioni per il training. Disponibili: ${seasons.length}.`);
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

function chartSvg(points){if(!points||points.length<1)return empty('⌁','Dati non disponibili','I fold walk-forward appariranno dopo il training.');const vals=points.map(p=>Number(p.rmse)).filter(Number.isFinite);if(!vals.length)return '';const min=Math.min(...vals),max=Math.max(...vals),range=Math.max(max-min,.001),w=640,h=210,pad=34;const coords=points.map((p,i)=>{const x=pad+(i*(w-pad*2)/Math.max(1,points.length-1));const y=h-pad-((Number(p.rmse)-min)/range)*(h-pad*2);return{x,y,p};});return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><line class="chart-grid-line" x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}"/><line class="chart-grid-line" x1="${pad}" y1="${pad}" x2="${w-pad}" y2="${pad}"/><polyline class="chart-line" points="${coords.map(c=>`${c.x},${c.y}`).join(' ')}"/>${coords.map(c=>`<circle class="chart-dot" cx="${c.x}" cy="${c.y}" r="5"/><text class="chart-label" x="${c.x}" y="${h-9}" text-anchor="middle">${esc(c.p.target_season??'')}</text>`).join('')}</svg>`;}
async function renderBacktests(){const d=await api('/admin-api/backtests'),r=d.report||{},o=r.overall||{},q=d.quality||{};$('#backtest-kpis').innerHTML=[kpi('⌁','Ensemble RMSE',fmt(o.rmse),'<span>OOT</span>'),kpi('↗','Base RMSE',fmt(r.base_rmse),'<span>XGBoost base</span>','green'),kpi('◫','Persistence RMSE',fmt(r.persistence_rmse),'<span>previous season</span>','violet'),kpi('◎','Interval coverage',pct(o.interval_coverage),'<span>conformal</span>','orange')].join('');$('#backtest-chart').innerHTML=chartSvg(q.folds||[]);$('#backtest-summary').innerHTML=d.candidate?.run_id?summaryRows([['Run ID',shortId(d.candidate.run_id)],['Samples',o.n??'—'],['Competitions',q.competitions??'—'],['Valid',r.valid?'Yes':'No']]):empty('▤','Nessun candidate','Completa un training run per ispezionare il report.','Avvia training','training');const comps=Object.entries(r.by_competition||{});$('#competition-body').innerHTML=comps.length?comps.map(([name,m])=>`<tr><td><strong>${esc(name)}</strong></td><td>${esc(m.n??'—')}</td><td>${esc(fmt(m.rmse))}</td><td>${esc(fmt(m.mae))}</td><td>${esc(fmt(m.bias))}</td><td>${esc(pct(m.interval_coverage))}</td></tr>`).join(''):'<tr><td colspan="6" class="muted-inline">Nessuna metrica per competition disponibile.</td></tr>';$$('.jump',root).forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));}

function csvValues(value){return String(value||'').split(',').map(v=>v.trim()).filter(Boolean);}
async function renderScenarios(){const form=$('#scenario-form'),result=$('#scenario-result'),submit=$('#scenario-submit');const overview=App.overview||await api('/admin-api/overview');form.season.value=overview.summary?.season_max||new Date().getFullYear();form.addEventListener('submit',async ev=>{ev.preventDefault();let parameters={};try{const raw=form.parameters.value.trim();parameters=raw?JSON.parse(raw):{};}catch(_){toast('Parameters JSON non valido','error');return;}const payload={scenario:form.scenario.value,player_global_ids:csvValues(form.player_global_ids.value),team_global_ids:csvValues(form.team_global_ids.value),source_league:form.source_league.value.trim()||null,target_league:form.target_league.value.trim()||null,season:Number(form.season.value),competition:form.competition.value.trim()||'RS',top_n:Number(form.top_n.value||5),parameters:{...parameters,simulations:Number(form.simulations.value||5000)}};submit.disabled=true;submit.textContent='Valutazione…';result.textContent='Il motore scenari sta elaborando la richiesta…';try{const data=await api('/admin-api/scenarios/evaluate',{method:'POST',body:payload});result.textContent=JSON.stringify(data,null,2);}catch(e){result.textContent='';toast(e.message,'error');}finally{submit.disabled=false;submit.textContent='Valuta scenario';}});}

function registryCard(title,entry,tone=''){if(!entry||!entry.run_id)return `<div class="registry-card">${empty('⬡',`Nessun ${title.toLowerCase()}`,`Non è presente un run ${title.toLowerCase()} nel registry.`)}</div>`;return `<div class="registry-card"><span class="registry-badge ${tone}">${esc(title)}</span><div class="registry-run">${esc(entry.run_id)}</div><div class="registry-meta">Version: ${esc(entry.model_version||'—')}<br>Data cutoff: ${esc(entry.data_cutoff||'—')}<br>OOT RMSE: ${esc(fmt(entry.overall_rmse))}<br>Registered: ${esc(entry.registered_at||entry.promoted_at||'—')}</div></div>`;}
async function renderRegistry(){const d=await api('/admin-api/registry');$('#top-model').textContent=shortId(d.production?.run_id);$('#registry-cards').innerHTML=registryCard('Production',d.production)+registryCard('Candidate',d.candidate,'candidate');const hist=d.history||[];$('#registry-history').innerHTML=hist.length?[...hist].reverse().map(h=>`<tr><td>${esc(h.at||'—')}</td><td>${esc(h.event||'—')}</td><td>${esc(h.run_id||'—')}</td></tr>`).join(''):'<tr><td colspan="3" class="muted-inline">Nessun evento nel registry.</td></tr>';const check=$('#registry-confirm'),prom=$('#promote-btn'),roll=$('#rollback-btn');check.addEventListener('change',()=>{prom.disabled=!check.checked||!d.candidate;roll.disabled=!check.checked||!d.previous;});prom.addEventListener('click',async()=>{prom.disabled=true;try{const r=await api('/admin-api/registry/promote',{method:'POST',body:{confirm:true}});toast(r.reason||'Candidate promoted');await renderRegistry();}catch(e){toast(e.message,'error');prom.disabled=false;}});roll.addEventListener('click',async()=>{roll.disabled=true;try{const r=await api('/admin-api/registry/rollback',{method:'POST',body:{confirm:true}});toast(r.reason||'Rollback completed');await renderRegistry();}catch(e){toast(e.message,'error');roll.disabled=false;}});}

async function renderHealth(){async function refresh(){const d=await api('/admin-api/api-health');const live=!!d.live?.ok,ready=!!d.ready?.ok;$('#top-api').textContent=live?'● V2':'● OFFLINE';$('#top-api').className=live?'meta-ok':'';$('#health-kpis').innerHTML=[kpi('⌘','API process',live?'Online':'Offline',`<span class="dot ${live?'green':'red'}"></span>/health/live`,'green'),kpi('✓','Inference ready',ready?'Ready':'Not ready',`<span class="dot ${ready?'green':'yellow'}"></span>/health/ready`),kpi('↗','Internal base',d.base_url||'—','<span>Docker network</span>','violet'),kpi('⬡','API contract','V2','<span>server-to-server</span>','orange')].join('');$('#health-checks').innerHTML=[['Live endpoint',d.live],['Readiness endpoint',d.ready]].map(([name,x])=>`<div class="health-item"><div class="health-copy"><strong>${esc(name)}</strong><span>${esc(x?.error||JSON.stringify(x?.body||{}).slice(0,160))}</span></div><span class="status ${x?.ok?'ok':'danger'}">${x?.ok?'✓ OK':'× KO'}</span></div>`).join('');$('#health-endpoints').innerHTML=(d.endpoints||[]).map(e=>`<div class="endpoint-item"><code>${esc(e)}</code><span class="registry-badge">V2</span></div>`).join('');}$('#refresh-health').addEventListener('click',()=>refresh().catch(e=>toast(e.message,'error')));await refresh();}

async function renderAudit(){async function refresh(){const d=await api('/admin-api/audit?limit=200');const rows=d.entries||[];$('#audit-body').innerHTML=rows.length?rows.map(r=>`<tr><td>${esc(r.ts||'—')}</td><td><strong>${esc(r.action||'—')}</strong></td><td>${esc(r.actor||'—')}</td><td>${esc(r.target||'—')}</td><td>${esc(r.details||'')}</td></tr>`).join(''):'<tr><td colspan="5" class="muted-inline">Nessun evento audit disponibile.</td></tr>';}$('#refresh-audit').addEventListener('click',()=>refresh().catch(e=>toast(e.message,'error')));await refresh();}

async function renderSettings(){const d=await api('/admin-api/settings');$('#runtime-settings').innerHTML=Object.entries(d.runtime||{}).map(([k,v])=>`<div class="settings-row"><span>${esc(k.replaceAll('_',' '))}</span><strong>${esc(v??'—')}</strong></div>`).join('');$('#promotion-settings').innerHTML=Object.entries(d.promotion_gates||{}).map(([k,v])=>`<div class="settings-row"><span>${esc(k.replaceAll('_',' '))}</span><strong>${esc(v)}</strong></div>`).join('');$('#password-form').addEventListener('submit',async ev=>{ev.preventDefault();const body=Object.fromEntries(new FormData(ev.currentTarget).entries());const btn=$('button',ev.currentTarget);btn.disabled=true;try{await api('/admin-api/settings/password',{method:'POST',body});toast('Password aggiornata. Effettua nuovamente il login.');showLogin();}catch(e){toast(e.message,'error');btn.disabled=false;}});}

$('#login-form').addEventListener('submit',async ev=>{ev.preventDefault();const error=$('#login-error'),button=$('button',ev.currentTarget);error.textContent='';button.disabled=true;try{const d=await api('/admin-api/login',{method:'POST',body:{username:$('#login-username').value,password:$('#login-password').value}});showApp(d.user);await navigate('overview');}catch(e){error.textContent=e.message;}finally{button.disabled=false;}});
$('#logout-btn').addEventListener('click',async()=>{try{await api('/admin-api/logout',{method:'POST',body:{}});}catch(_){}showLogin();});
$$('.nav-item').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.page)));

(async function boot(){try{const s=await api('/admin-api/session');showApp(s.user);await navigate('overview');}catch(_){showLogin();}})();
