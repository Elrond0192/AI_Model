(() => {
  const csrfToken = () => {
    const item = document.cookie
      .split('; ')
      .find((value) => value.startsWith('hm_ai_admin_csrf='));
    return item ? decodeURIComponent(item.split('=').slice(1).join('=')) : '';
  };

  const notify = (message, type = 'success') => {
    if (typeof window.toast === 'function') {
      window.toast(message, type);
      return;
    }
    window.alert(message);
  };

  const decorateProfiles = () => {
    document.querySelectorAll('#profiles-list .profile-card').forEach((card) => {
      if (card.querySelector('.profile-delete')) return;
      const sourceButton = card.querySelector('.profile-load, .profile-test');
      const row = card.querySelector('.button-row');
      const name = sourceButton?.dataset?.name;
      if (!row || !name) return;

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn btn-danger btn-small profile-delete';
      button.dataset.name = name;
      button.textContent = 'Delete profile';
      row.appendChild(button);
    });
  };

  const refreshProfiles = () => {
    const refresh = document.querySelector('#refresh-profiles');
    if (refresh) refresh.click();
  };

  document.addEventListener('click', async (event) => {
    const button = event.target.closest('.profile-delete');
    if (!button) return;

    const name = button.dataset.name;
    if (!name) return;
    if (!window.confirm(`Eliminare definitivamente il profilo “${name}”?`)) return;

    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = 'Deleting…';

    try {
      const response = await fetch(`/admin-api/profiles/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        credentials: 'same-origin',
        headers: {
          Accept: 'application/json',
          'X-CSRF-Token': csrfToken(),
        },
      });

      let payload = {};
      try {
        payload = await response.json();
      } catch (_) {
        payload = {};
      }

      if (!response.ok) {
        throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
      }

      notify(`Profilo ${name} eliminato`);
      refreshProfiles();
    } catch (error) {
      notify(error.message || 'Impossibile eliminare il profilo', 'error');
      button.disabled = false;
      button.textContent = originalText;
    }
  });

  const observer = new MutationObserver(decorateProfiles);
  const start = () => {
    const root = document.querySelector('#page-root') || document.body;
    observer.observe(root, { childList: true, subtree: true });
    decorateProfiles();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
})();
