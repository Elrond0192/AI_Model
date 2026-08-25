from pathlib import Path

path = Path("basketball_ai/admin_web/static/index.html")
text = path.read_text(encoding="utf-8")

nav_anchor = '        <button class="nav-item" data-page="settings"><span class="nav-icon">⚙</span><span>Settings</span></button>'
nav_insert = '        <button class="nav-item" data-page="contact"><span class="nav-icon">✉</span><span>Contatti</span></button>\n'

contact_template = '''  <template id="page-contact">
    <section class="page">
      <div class="page-heading"><div><h1>Contatti</h1><p>Hai una domanda, vuoi segnalare un problema o proporre un miglioramento? Invia un messaggio direttamente al team.</p></div></div>
      <div class="split-grid">
        <section class="panel">
          <div class="panel-heading"><div><h2>Invia un messaggio</h2><p class="microcopy">Compila il form e descrivi la tua richiesta nel modo più dettagliato possibile.</p></div></div>
          <form class="form-grid" onsubmit="event.preventDefault();const f=this,b=f.querySelector('button[type=submit]');b.disabled=true;b.textContent='Messaggio inviato ✓';f.querySelector('.contact-success').classList.remove('hidden');setTimeout(()=>{f.reset();b.disabled=false;b.textContent='Invia messaggio';},1200);">
            <label>Nome<input name="name" type="text" autocomplete="name" placeholder="Il tuo nome" required></label>
            <label>Email<input name="email" type="email" autocomplete="email" placeholder="nome@azienda.it" required></label>
            <label class="span-2">Oggetto<input name="subject" type="text" placeholder="Come possiamo aiutarti?" required></label>
            <label class="span-2">Messaggio<textarea name="message" rows="8" placeholder="Scrivi qui il tuo messaggio..." required></textarea></label>
            <div class="span-2 button-row" style="align-items:center;justify-content:space-between">
              <span class="microcopy"><strong>Privacy:</strong> i dati saranno usati esclusivamente per rispondere alla richiesta.</span>
              <button class="btn btn-primary" type="submit">Invia messaggio</button>
            </div>
            <div class="span-2 contact-success hidden" style="font-size:12px;color:var(--success,#48c78e)">Messaggio inviato correttamente. In questo mockup l'invio è simulato.</div>
          </form>
        </section>
        <aside class="panel">
          <div class="panel-heading"><h2>Prima di scriverci</h2></div>
          <div class="summary-list">
            <div class="summary-row"><span>Problemi tecnici</span><strong>Indica pagina ed errore</strong></div>
            <div class="summary-row"><span>Funzionalità</span><strong>Descrivi il risultato atteso</strong></div>
            <div class="summary-row"><span>Oggetto</span><strong>Usa un titolo chiaro</strong></div>
          </div>
        </aside>
      </div>
    </section>
  </template>

'''

if 'data-page="contact"' not in text:
    if nav_anchor not in text:
        raise SystemExit("menu anchor not found")
    text = text.replace(nav_anchor, nav_insert + nav_anchor, 1)

if 'id="page-contact"' not in text:
    template_anchor = '  <template id="page-settings">'
    if template_anchor not in text:
        raise SystemExit("template anchor not found")
    text = text.replace(template_anchor, contact_template + template_anchor, 1)

path.write_text(text, encoding="utf-8")
print("Contact mockup applied")
