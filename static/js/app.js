/* =============================================
   MediAI Suite — Shared JS  v2
   ============================================= */

// ── Mobile nav toggle ────────────────────────
function initNavToggle() {
  const btn   = document.getElementById('mobileMenuBtn');
  const links = document.getElementById('navLinks');
  if (!btn || !links) return;

  btn.addEventListener('click', () => {
    const open = links.classList.toggle('open');
    // Swap between hamburger and X icon via SVG use href
    const use = btn.querySelector('use');
    if (use) use.setAttribute('href', open ? '#icon-x' : '#icon-menu');
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!btn.contains(e.target) && !links.contains(e.target)) {
      links.classList.remove('open');
      const use = btn.querySelector('use');
      if (use) use.setAttribute('href', '#icon-menu');
    }
  });
}

// ── Animate number counters ──────────────────
function animateCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target   = parseFloat(el.dataset.count) || 0;
    const isFloat  = String(el.dataset.count).includes('.');
    const duration = 1400;
    const start    = performance.now();

    (function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased    = 1 - Math.pow(1 - progress, 3);
      const value    = target * eased;
      el.textContent = isFloat ? value.toFixed(1) : Math.round(value);
      if (progress < 1) requestAnimationFrame(update);
    })(start);
  });
}

// ── Ripple on buttons ────────────────────────
function initRipple() {
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function (e) {
      const r    = document.createElement('span');
      const rect = this.getBoundingClientRect();
      Object.assign(r.style, {
        position: 'absolute',
        width: '22px', height: '22px',
        background: 'rgba(255,255,255,0.3)',
        borderRadius: '50%',
        left: (e.clientX - rect.left - 11) + 'px',
        top:  (e.clientY - rect.top  - 11) + 'px',
        animation: 'ripple 0.55s ease-out forwards',
        pointerEvents: 'none',
      });
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      this.appendChild(r);
      setTimeout(() => r.remove(), 600);
    });
  });

  // Inject ripple keyframe once
  if (!document.getElementById('ripple-style')) {
    const s = document.createElement('style');
    s.id = 'ripple-style';
    s.textContent = '@keyframes ripple{0%{transform:scale(1);opacity:1;}100%{transform:scale(5);opacity:0;}}';
    document.head.appendChild(s);
  }
}

// ── Scroll reveal ────────────────────────────
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity    = '1';
        e.target.style.transform  = 'translateY(0)';
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.reveal').forEach(el => {
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(22px)';
    el.style.transition = 'opacity 0.55s ease, transform 0.55s ease';
    observer.observe(el);
  });
}

// ── Toast notifications ──────────────────────
function showToast(msg, type = 'info') {
  const palette = {
    info:    '#3b82f6',
    success: '#10b981',
    danger:  '#f43f5e',
    warn:    '#f59e0b',
  };
  const icons = {
    info:    'ℹ',
    success: '✓',
    danger:  '⚠',
    warn:    '⚡',
  };
  const color = palette[type] || palette.info;
  const t     = document.createElement('div');
  Object.assign(t.style, {
    position:   'fixed',
    bottom:     '28px',
    right:      '28px',
    zIndex:     '9999',
    background: 'rgba(5,12,26,0.96)',
    border:     `1px solid ${color}`,
    color:      '#eef4ff',
    padding:    '14px 20px',
    borderRadius: '14px',
    fontSize:   '0.88rem',
    fontFamily: "'DM Sans', sans-serif",
    boxShadow:  '0 12px 40px rgba(0,0,0,0.55)',
    animation:  'fade-up 0.35s ease both',
    maxWidth:   '340px',
    display:    'flex',
    alignItems: 'center',
    gap:        '10px',
  });
  t.innerHTML = `<span style="color:${color};font-size:1rem;">${icons[type]}</span>${msg}`;
  document.body.appendChild(t);
  setTimeout(() => {
    t.style.transition = 'opacity 0.3s';
    t.style.opacity    = '0';
    setTimeout(() => t.remove(), 320);
  }, 3600);
}

// ── Image preview ────────────────────────────
function setupImagePreview(inputId, previewId) {
  const input   = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  if (!input || !preview) return;

  input.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      preview.src          = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  });
}

// ── Drag-and-drop upload zone ─────────────────
function setupDragDrop(zoneId, inputId) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  if (!zone || !input) return;

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      input.dispatchEvent(new Event('change'));
    }
  });
}

// ── Navbar scroll shadow ──────────────────────
function initNavScroll() {
  const nav = document.querySelector('.navbar');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.style.background = window.scrollY > 10
      ? 'rgba(2,8,16,0.92)'
      : 'rgba(2,8,16,0.75)';
  }, { passive: true });
}

// ── Active nav link highlight ─────────────────
function highlightActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if (href && path === href) a.classList.add('active');
    else if (href && href !== '/' && path.startsWith(href) && href.length > 1) {
      a.classList.add('active');
    }
  });
}

// ── Init ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNavToggle();
  initNavScroll();
  highlightActiveNav();
  initRipple();
  initScrollReveal();
  animateCounters();
});

