const qs = (selector, scope = document) => scope.querySelector(selector);
const qsa = (selector, scope = document) => [...scope.querySelectorAll(selector)];

const progress = qs("#scroll-progress");
addEventListener("scroll", () => {
  const height = document.documentElement.scrollHeight - innerHeight;
  progress.style.width = height > 0 ? (scrollY / height * 100) + "%" : "0";
}, { passive: true });

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add("visible");
    if (entry.target.classList.contains("metric")) animateMetric(qs("b", entry.target));
    observer.unobserve(entry.target);
  });
}, { threshold: .14 });
qsa(".reveal").forEach(element => observer.observe(element));

function animateMetric(element) {
  if (!element || element.dataset.done) return;
  element.dataset.done = "true";
  const target = Number(element.dataset.count);
  const decimals = Number(element.dataset.decimals ?? (String(target).includes(".") ? 1 : 0));
  const prefix = element.dataset.prefix || "";
  const suffix = element.dataset.suffix || "";
  const start = performance.now();
  const duration = 1100;
  const tick = now => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    element.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const scenarios = {
  bau: {
    description: "Physical exclusions only; growth follows learned accessibility and agglomeration patterns.",
    share: "31.7", area: "3,126 ha", vh: "2,020 ha", hci: "0.89"
  },
  eco: {
    description: "Agricultural, forest, and riparian penalties redirect allocation while keeping total demand unchanged.",
    share: "30.0", area: "2,956 ha", vh: "1,774 ha", hci: "0.84"
  }
};
qsa("[data-scenario]").forEach(button => button.addEventListener("click", () => {
  qsa("[data-scenario]").forEach(item => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", active);
  });
  const value = scenarios[button.dataset.scenario];
  qs("#scenario-description").textContent = value.description;
  qs("#scenario-share").textContent = value.share;
  qs("#scenario-area").textContent = value.area;
  qs("#scenario-vh").textContent = value.vh;
  qs("#scenario-hci").textContent = value.hci;
  qs("#risk-fill").style.width = value.share + "%";
}));

const schemes = {
  ahp: [35, 35, 20, 10],
  equal: [25, 25, 25, 25],
  rank: [39, 39, 13, 10],
  pca: [7, 81, 10, 2]
};
const barIds = ["flood", "seismic", "bio", "fire"];
function renderScheme(name) {
  schemes[name].forEach((value, index) => {
    qs("#bar-" + barIds[index]).style.width = value + "%";
    qs("#out-" + barIds[index]).textContent = (value / 100).toFixed(2);
  });
}
qsa("[data-scheme]").forEach(button => button.addEventListener("click", () => {
  qsa("[data-scheme]").forEach(item => item.classList.toggle("active", item === button));
  renderScheme(button.dataset.scheme);
}));
renderScheme("ahp");

const dialog = qs("#lightbox");
const dialogImage = qs("#lightbox-image");
qsa("[data-full]").forEach(button => button.addEventListener("click", () => {
  const thumbnail = qs("img", button);
  dialogImage.src = button.dataset.full;
  dialogImage.alt = thumbnail.alt;
  dialog.showModal();
}));
qs("#lightbox-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => {
  if (event.target === dialog) dialog.close();
});
addEventListener("keydown", event => {
  if (event.key === "Escape" && dialog.open) dialog.close();
});
