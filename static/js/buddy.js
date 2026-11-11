// Floating pixel-art buddy (Clippy style) on post pages, loaded by
// layouts/partials/extra-head.html. Idles with random poses, points at
// {{< point >}} blocks, reacts to the quiz, dozes off when the reader is idle.
// Clicking it puts it to sleep (remembered per browser) or wakes it up.
(() => {
  // Order of the frames in static/buddy/sprite.webp (assets/buddy/make_sprite.py).
  const FRAMES = ['idle', 'blink', 'lookLeft', 'lookRight', 'thinking', 'pointing', 'explaining',
    'reading', 'excited', 'surprised', 'happy', 'walking', 'running', 'jump', 'sleep', 'wave', 'celebrate'];
  const IDLE = ['blink', 'blink', 'blink', 'lookLeft', 'lookRight', 'thinking', 'reading',
    'explaining', 'happy', 'wave', 'jump'];
  const DOZE_AFTER = 60000; // ms without scrolling or moving the pointer

  const script = document.currentScript;
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Buddy sits at the inline start (left in LTR, right in RTL); the sprite faces right.
  const flipToText = getComputedStyle(document.documentElement).direction === 'rtl';
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const remember = (v) => { try { localStorage.setItem('buddy', v); } catch {} };
  let asleep = false;
  try { asleep = localStorage.getItem('buddy') === 'asleep'; } catch {}

  const el = document.createElement('div');
  el.className = 'buddy';
  el.innerHTML = '<p class="buddy-say" aria-live="polite" hidden></p><button class="buddy-me" type="button"></button>';
  const me = el.lastChild, say = el.firstChild;
  me.style.backgroundImage = `url("${script.dataset.sprite}")`;
  me.setAttribute('aria-label', script.dataset.label);

  let timer, busy = false, dozing = false, lastActive = Date.now();
  const show = (f, flip = false) => {
    me.style.backgroundPositionX = `${FRAMES.indexOf(f) / (FRAMES.length - 1) * 100}%`;
    me.classList.toggle('buddy-flip', flip);
  };
  const rest = () => {
    busy = false;
    say.hidden = true;
    document.querySelectorAll('.buddy-lit').forEach((t) => t.classList.remove('buddy-lit'));
    show(asleep || dozing ? 'sleep' : 'idle');
  };
  const play = (f, ms = 1400, flip = false) => {
    if (asleep) return;
    clearTimeout(timer);
    dozing = false;
    show(f, flip);
    timer = setTimeout(rest, ms);
  };

  // Random idle poses, and dozing off.
  const idle = () => {
    setTimeout(idle, 3000 + Math.random() * 5000);
    if (busy || asleep || dozing) return;
    if (Date.now() - lastActive > DOZE_AFTER) { dozing = true; show('sleep'); return; }
    if (calm) return;
    const f = pick(IDLE);
    play(f, f === 'blink' ? 180 : 1400);
  };
  const wake = () => {
    lastActive = Date.now();
    if (dozing && !busy) play('surprised', 900);
  };
  addEventListener('scroll', wake, { passive: true });
  addEventListener('pointermove', wake, { passive: true });

  // Point at each marked block once, when it reaches the middle of the screen.
  const seen = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (!e.isIntersecting || asleep) return;
    seen.unobserve(e.target);
    play('pointing', 5000, flipToText);
    busy = true;
    say.textContent = e.target.dataset.say;
    say.hidden = !say.textContent;
    e.target.classList.add('buddy-lit');
  }), { rootMargin: '-35% 0px -35% 0px' });
  document.querySelectorAll('.buddy-target').forEach((t) => seen.observe(t));

  // Quiz reactions; quiz.js has already marked the answer when this bubbles up.
  document.addEventListener('click', (e) => {
    const b = e.target.closest('.quiz-q button');
    if (!b || busy) return;
    const quiz = b.closest('.quiz');
    if (!quiz.querySelector('.quiz-q:not(:disabled)')) play('celebrate', 2500);
    else play(b.classList.contains('quiz-wrong') ? 'surprised' : 'happy');
  });

  me.addEventListener('click', () => {
    asleep = !asleep;
    remember(asleep ? 'asleep' : 'awake');
    el.classList.toggle('buddy-asleep', asleep);
    if (asleep) { clearTimeout(timer); rest(); } else play('wave');
  });

  document.body.append(el);
  el.classList.toggle('buddy-asleep', asleep);
  if (asleep) show('sleep');
  else if (calm) show('idle');
  else { // walk in, then wave
    show('walking', flipToText);
    el.classList.add('buddy-enter');
    el.addEventListener('animationend', () => busy || play('wave', 1500), { once: true });
  }
  setTimeout(idle, 4000);
})();
