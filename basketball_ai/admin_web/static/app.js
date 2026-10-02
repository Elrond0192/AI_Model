const App={user:null,page:'overview',overview:null,trainingTimer:null,bbRatingTimer:null,futurePerformanceTimer:null};
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
async function navigate(page){clearTimeout(App.trainingTimer);clearTimeout(App.bbRatingTimer);clearTimeout(App.futurePerformanceTimer);App.page=page;activateNav(page);mountTemplate(page);try{if(page==='overview')await renderOverview();if(page==='data')await renderData();if(page==='training')await renderTraining();if(page==='backtests')await renderBacktests();if(page==='bb-rating')await renderBBRating();if(page==='future-performance')await renderFuturePerformancePage();if(page==='scenarios')await renderScenarios();if(page==='registry')await renderRegistry();if(page==='health')await renderHealth();if(page==='audit')await renderAudit();if(page==='settings')await renderSettings();}catch(e){toast(e.message,'error');}}

function kpi(icon,label,value,sub='',tone=''){return `<div class="kpi-card"><div class="kpi-icon ${tone}">${icon}</div><div class="kpi-label">${esc(label)}</div><div class="kpi-value ${tone==='green'?'online':''}">${esc(value)}</div><div class="kpi-sub">${sub}</div></div>`;}
function updateTop(o){$('#top-db').textContent=o.database||'not selected';$('#top-model').textContent=shortId(o.production?.run_id);const bb=o.bb_rating||{};$('#top-bb-rating').textContent=bb.bb_rating_version?'v'+esc(bb.bb_rating_version):'—';$('#top-api').textContent=o.api?.online?'● V2':'● OFFLINE';$('#top-api').className=o.api?.online?'meta-ok':'';}

async function renderOverview(){
  const o=await api('/admin-api/overview');App.overview=o;updateTop(o);
  const s=o.summary||{},bb=o.bb_rating||{};$('#overview-kpis').innerHTML=[
    kpi('▤','Database',o.database||'Not selected',`<span class="dot ${o.data_loaded?'green':'yellow'}"></span>${o.data_loaded?'Connesso e caricato':'Da caricare'}`),
    kpi('▥','Dati caricati',fmtInt(s.player_rows),'<span>righe player/competition</span>','green'),
    kpi('◉','Competitions',String((s.competitions||[]).length||'—'),'<span>caricate</span>','violet'),
    kpi('⬡','Prediction Model',shortId(o.production?.run_id),o.production?'<span class="dot green"></span>Production':'<span class="dot red"></span>Not promoted','orange'),
    kpi('◆','BB-Rating',bb.bb_rating_version?'v'+bb.bb_rating_version:'—',bb.ready?'<span class="dot green"></span>Ready':'<span class="dot yellow"></span>Da verificare','violet'),
    kpi('⌘','Stato API',o.api?.online?'Online':'Offline',`<span class="dot ${o.api?.online?'green':'red'}"></span>V2`,'green')
  ].join('');
  $('#readiness-body').innerHTML=o.readiness.map(r=>{const page=r.action==='data'?'data':r.action;const label=r.action==='registry'?'Apri registry':r.action==='training'?'Apri training':r.action==='bb-rating'?'Apri BB-Rating':r.action==='future-performance'?'Apri Future Performance':'Apri dati';return `<tr><td><strong>${esc(r.component)}</strong></td><td><span class="status ${r.status}"><span class="status-dot">${r.status==='ok'?'✓':r.status==='danger'?'×':'△'}</span>${esc(r.label)}</span></td><td>${esc(r.detail)}</td><td class="right"><button class="btn btn-small btn-secondary readiness-action" data-page="${esc(page)}">${label}</button></td></tr>`;}).join('');
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