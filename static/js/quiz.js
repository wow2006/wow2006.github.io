// End-of-post quiz, see layouts/shortcodes/quiz.html.
// One answer per question; the first answer is final and is sent to Google
// Analytics (gtag exists only when the GA snippet loaded, e.g. no Do Not Track).
document.querySelectorAll('.quiz').forEach((quiz) => {
  const questions = quiz.querySelectorAll('.quiz-q');
  let answered = 0, right = 0;

  questions.forEach((q) => q.addEventListener('click', (e) => {
    const choice = e.target.closest('button');
    if (!choice || q.disabled) return;
    const ok = choice.value === q.dataset.answer;
    q.disabled = true;
    q.querySelector(`button[value="${q.dataset.answer}"]`).classList.add('quiz-right');
    if (!ok) choice.classList.add('quiz-wrong');
    q.querySelector('.quiz-verdict').textContent = ok ? quiz.dataset.right : quiz.dataset.wrong;
    q.querySelector('.quiz-explain').hidden = false;

    if (typeof window.gtag === 'function') {
      window.gtag('event', 'quiz_answer', {
        quiz_question: q.dataset.id,
        quiz_correct: ok ? 'yes' : 'no',
        quiz_choice: choice.value,
        // GA4 cuts parameter values at 100 characters.
        quiz_text: q.querySelector('legend').textContent.trim().slice(0, 100),
      });
    }

    right += ok;
    if (++answered === questions.length) {
      const score = quiz.querySelector('.quiz-score');
      score.textContent = quiz.dataset.score.replace('%r', right).replace('%n', questions.length);
      score.hidden = false;
    }
  }));
});
