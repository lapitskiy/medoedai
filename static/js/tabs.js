// Общая логика табов
// data-tabs="main" — идентификатор набора табов
// Кнопки: [data-tab-btn][data-target="#panelId"]
// Панели: [data-tab-panel][id="panelId"]

(function(){
  function debugLog(){
    // Debug logging disabled
  }
  const escapeId = (function(){
    try{ if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape; }catch(_){ }
    return function(s){ return String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&'); };
  })();
  function activateTab(container, targetId){
    debugLog('activateTab -> targetId=', targetId);
    try{
      const btns = container.querySelectorAll('[data-tab-btn]');
      const panels = container.querySelectorAll('[data-tab-panel]');
      btns.forEach(b=>b.classList.remove('active'));
      panels.forEach(p=>p.classList.remove('active'));
      const btn = container.querySelector(`[data-tab-btn][data-target="#${escapeId(targetId)}"]`);
      const panel = container.querySelector(`[data-tab-panel][id="${escapeId(targetId)}"]`);
      if(btn) btn.classList.add('active');
      if(panel) panel.classList.add('active');
      try{
        if(panel){
          // Прокрутка к началу активной панели
          try{ panel.scrollIntoView({behavior:'smooth', block:'start'}); }catch(_){ }
        }
      }catch(_){ }
      debugLog('activateTab result:', {
        btnFound: !!btn,
        panelFound: !!panel,
        btnTarget: btn ? btn.getAttribute('data-target') : null,
        panelId: panel ? panel.id : null,
        btnsCount: btns.length,
        panelsCount: panels.length
      });
    }catch(_){ }
  }

  function initTabs(root){
    const containers = (root || document).querySelectorAll('[data-tabs]');
    containers.forEach(container => {
      // Навесим клики
      container.addEventListener('click', (e)=>{
        const btn = e.target.closest('[data-tab-btn]');
        if(!btn) return;
        const target = btn.getAttribute('data-target');
        if(target && target.startsWith('#')){
          debugLog('click tab-btn ->', target);
          const id = target.slice(1);
          activateTab(container, id);
        }
      });
      // Активируем таб по умолчанию: если уже есть .active на кнопке — уважаем её,
      // иначе включаем первую кнопку
      try{
        const preset = container.querySelector('[data-tab-btn].active');
        if (preset) {
          const t = preset.getAttribute('data-target');
          if (t && t.startsWith('#')) activateTab(container, t.slice(1));
        } else {
          const firstBtn = container.querySelector('[data-tab-btn]');
          if(firstBtn){
            const t = firstBtn.getAttribute('data-target');
            if(t && t.startsWith('#')) activateTab(container, t.slice(1));
          }
        }
      }catch(_){ }
    });
  }

  // Экспортируем функции глобально для интеграции с сайдбаром
  window.__tabs = {
    initTabs,
    activateTabByAnchor: function(containerSelector, anchor){
      try{
        const id = String(anchor||'').replace(/^#/, '');
        const cont = document.querySelector(containerSelector);
        if(cont) activateTab(cont, id);
      }catch(_){ }
    },
    dump: function(){
      try{
        document.querySelectorAll('[data-tabs]').forEach((container, idx) => {
          const actBtn = container.querySelector('[data-tab-btn].active');
          const actPanel = container.querySelector('[data-tab-panel].active');
          debugLog('dump container#'+idx, {
            activeBtnTarget: actBtn ? actBtn.getAttribute('data-target') : null,
            activePanelId: actPanel ? actPanel.id : null,
            btns: Array.from(container.querySelectorAll('[data-tab-btn]')).map(b=>b.getAttribute('data-target')),
            panels: Array.from(container.querySelectorAll('[data-tab-panel]')).map(p=>p.id)
          });
        });
      }catch(_){ }
    }
  };

  document.addEventListener('DOMContentLoaded', function(){
    try{ document.body.classList.add('tabs-initialized'); }catch(_){ }
    debugLog('DOMContentLoaded: init');
    initTabs(document);
    // Активируем таб по hash, если есть
    try{
      if (location.hash) {
        debugLog('hash present ->', location.hash);
        window.__tabs && window.__tabs.activateTabByAnchor('[data-tabs="main"]', location.hash);
      }
    }catch(_){ }
  });

  // Синхронизация с изменениями хеша адресной строки
  window.addEventListener('hashchange', function(){
    debugLog('hashchange ->', location.hash);
    try{ window.__tabs && window.__tabs.activateTabByAnchor('[data-tabs="main"]', location.hash); }catch(_){ }
  });
})();


