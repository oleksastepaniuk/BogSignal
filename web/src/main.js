import './style.css';

// Highlight the nav link for the section currently in view.
const links = document.querySelectorAll('.nav a');
const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links) {
        link.classList.toggle('active', link.hash === `#${entry.target.id}`);
      }
    }
  },
  { rootMargin: '-50% 0px -50% 0px' },
);
document.querySelectorAll('main section').forEach((s) => observer.observe(s));
