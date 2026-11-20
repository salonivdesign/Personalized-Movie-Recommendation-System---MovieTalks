/* MovieTalks frontend — vanilla JS, no build step.
 *
 * Flow: user action -> state change -> fetch FastAPI -> render cards.
 * Ratings and genre preferences live in localStorage; the server is stateless
 * and receives the ratings with each /api/recommend request.
 * All server text is inserted with textContent (never innerHTML) to avoid XSS.
 */
(() => {
  "use strict";

  // ---- config ----------------------------------------------------------------
  const params = new URLSearchParams(location.search);
  const isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
  // Same-origin by default. If the page is opened from file:// or a different
  // local port (e.g. `python -m http.server 5500`), talk to the local API.
  const API_BASE = params.get("api") ||
    ((location.protocol === "file:" || (isLocal && location.port !== "8000")) ? "http://127.0.0.1:8000" : "");
  const MIN_FOR_CF = 5;
  const STORE_KEY = "movietalks.v1";

  const state = {
    ratings: {},      // movieId -> {rating, title, year}
    genres: [],       // selected genre filters
    tab: "foryou",    // foryou | top | similar
    demoUser: null,
    similar: null,    // {movie_id, title}
    dirty: false,     // ratings changed since the last For You load
  };

  const $ = (sel) => document.querySelector(sel);
  const grid = $("#grid");

  // ---- tiny DOM helper ---------------------------------------------------------
  function h(tag, props = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v === false || v == null) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid);
    return el;
  }

  // ---- persistence ---------------------------------------------------------------
  function loadStore() {
    try {
      const s = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
      for (const [id, r] of Object.entries(s.ratings || {})) {
        if (/^\d+$/.test(id) && r && r.rating >= 0.5 && r.rating <= 5 && typeof r.title === "string") {
          state.ratings[id] = { rating: Number(r.rating), title: r.title, year: r.year ?? null };
        }
      }
      if (Array.isArray(s.genres)) state.genres = s.genres.filter((g) => typeof g === "string").slice(0, 10);
    } catch { /* storage unavailable or corrupt: start fresh */ }
  }
  function saveStore() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify({ ratings: state.ratings, genres: state.genres })); }
    catch { /* ignore quota/privacy-mode errors */ }
  }

  // ---- API ---------------------------------------------------------------------------
  async function api(path, { method = "GET", body, signal } = {}) {
    let res;
    try {
      res = await fetch(API_BASE + path, {
        method, signal,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (err) {
      if (err.name === "AbortError") throw err;
      throw new Error("Cannot reach the API. Is the backend running?");
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error body */ }
    if (!res.ok) {
      let msg = data && data.detail;
      if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
      throw new Error(msg || `Request failed (${res.status})`);
    }
    return data;
  }

  function toast(message, kind = "info") {
    const el = h("div", { class: `toast ${kind === "error" ? "error" : ""}`, role: "status", text: message });
    $("#toasts").append(el);
    setTimeout(() => el.remove(), 4500);
  }

  // ---- movie card ------------------------------------------------------------------------
  function hueFor(text) {
    let x = 0;
    for (const c of text) x = (x * 31 + c.charCodeAt(0)) % 360;
    return x;
  }

  function poster(item) {
    // No external poster API (keeps the project key-free): a deterministic
    // gradient from the first genre + title initials.
    const hue = hueFor(item.genres[0] || "Movie");
    const initials = item.title.replace(/[^A-Za-z0-9 ]/g, "").split(/\s+/).filter(Boolean)
      .slice(0, 2).map((w) => w[0]).join("").toUpperCase() || "?";
    const el = h("div", { class: "poster", "aria-hidden": "true" }, initials,
      item.year ? h("span", { class: "poster-year", text: String(item.year) }) : null,
      item.predicted_rating ? h("span", {
        class: "poster-pred", title: "Predicted rating for you",
        text: `≈ ${item.predicted_rating.toFixed(1)} ★`,
      }) : null);
    el.style.background = `linear-gradient(135deg, hsl(${hue} 60% 38%), hsl(${(hue + 50) % 360} 65% 28%))`;
    return el;
  }

  function stars(movieId, onRate) {
    const wrap = h("div", { class: "stars", role: "group", "aria-label": "Your rating" });
    const current = () => (state.ratings[movieId] || {}).rating || 0;
    const els = [];
    const paint = (v) => els.forEach((s, i) =>
      s.style.setProperty("--fill", `${Math.max(0, Math.min(1, v - i)) * 100}%`));
    for (let i = 1; i <= 5; i++) {
      const star = h("span", { class: "star" });
      for (const half of [0.5, 1]) {
        const value = i - 1 + half;
        star.append(h("button", {
          type: "button", class: `half ${half === 0.5 ? "left" : "right"}`,
          "aria-label": `Rate ${value} stars`,
          onclick: () => onRate(value),
          onmouseenter: () => paint(value), onfocus: () => paint(value), onblur: () => paint(current()),
        }));
      }
      els.push(star);
      wrap.append(star);
    }
    wrap.addEventListener("mouseleave", () => paint(current()));
    wrap._refresh = () => paint(current());
    paint(current());
    return wrap;
  }

  function reasonText(item) {
    if (state.tab === "similar") {
      return { content: "Similar genres & tags", cf: "Liked by the same kind of viewers", pop: "Popular pick" }[item.reason_code] || "";
    }
    if (item.because) return `Because you liked ${item.because}`;
    return item.reason || "";
  }

  function card(item) {
    const meta = item.avg_rating != null
      ? `★ ${item.avg_rating.toFixed(2)} avg · ${item.n_ratings.toLocaleString()} ratings`
      : "No ratings yet";
    const reason = reasonText(item);
    return h("article", { class: "card" },
      poster(item),
      h("div", { class: "card-body" },
        h("h3", { class: "card-title", text: item.title }),
        h("div", { class: "genres" }, item.genres.slice(0, 3).map((g) => h("span", { class: "genre-tag", text: g }))),
        h("div", { class: "meta", text: meta }),
        reason ? h("div", { class: "reason", text: reason }) : null,
        h("div", { class: "card-actions" },
          stars(item.movie_id, (v) => setRating(item, v)),
          h("button", { class: "btn", type: "button", text: "Similar", onclick: () => showSimilar(item) })),
        item.imdb_url ? h("a", { href: item.imdb_url, target: "_blank", rel: "noopener noreferrer", text: "IMDb ↗" }) : null));
  }

  // ---- ratings ------------------------------------------------------------------------------
  function setRating(item, value) {
    const id = String(item.movie_id);
    if (state.ratings[id] && state.ratings[id].rating === value) delete state.ratings[id];   // click again = remove
    else state.ratings[id] = { rating: value, title: item.title, year: item.year ?? null };
    saveStore();
    afterRatingChange();
  }

  function afterRatingChange() {
    document.querySelectorAll(".stars").forEach((el) => el._refresh && el._refresh());
    renderDrawer();
    $("#rating-count").textContent = Object.keys(state.ratings).length;
    if (!state.demoUser) { state.dirty = true; updateRefresh(); }
  }

  function updateRefresh() {
    $("#refresh").hidden = !(state.dirty && state.tab === "foryou" && !state.demoUser);
  }

  // ---- loading + rendering ----------------------------------------------------------------------
  let loadSeq = 0;

  function skeleton(n = 12) {
    grid.replaceChildren(...Array.from({ length: n }, () => h("div", { class: "skeleton" })));
  }

  function setStatus(text, nRated = null) {
    $("#status-text").textContent = text || "";
    const bar = $("#progress");
    if (nRated == null) { bar.hidden = true; return; }
    bar.hidden = false;
    $("#progress-bar").style.width = `${Math.min(nRated / MIN_FOR_CF, 1) * 100}%`;
  }

  async function load() {
    const seq = ++loadSeq;
    skeleton();
    state.dirty = false;
    updateRefresh();
    const gq = state.genres.length ? `&genres=${encodeURIComponent(state.genres.join(","))}` : "";
    try {
      let data, title, anchor = null;
      if (state.tab === "foryou") {
        if (state.demoUser) {
          data = await api(`/api/users/${state.demoUser}/recommend?n=18${gq}`);
          title = `Recommended for sample dataset user #${state.demoUser}`;
        } else {
          data = await api("/api/recommend", {
            method: "POST",
            body: {
              ratings: Object.entries(state.ratings).map(([id, r]) => ({ movie_id: Number(id), rating: r.rating })),
              genres: state.genres, n: 18,
            },
          });
          title = data.mode === "popular" ? "Popular right now — rate a few to personalise" : "Recommended for you";
        }
      } else if (state.tab === "top") {
        data = await api(`/api/popular?n=24${gq}`);
        title = "Top picks (Bayesian weighted rating)";
      } else {
        data = await api(`/api/movies/${state.similar.movie_id}/similar?n=17${gq}`);
        title = `Movies similar to “${data.movie.title}”`;
        anchor = card(data.movie);
        anchor.classList.add("is-anchor");
      }
      if (seq !== loadSeq) return;                       // a newer request superseded this one
      $("#section-title").textContent = title;
      const showProgress = state.tab === "foryou" && !state.demoUser && data.ratings_needed_for_cf > 0;
      setStatus(data.mode_description + (showProgress ? `  (${data.n_ratings_used}/${MIN_FOR_CF} ratings)` : ""),
        showProgress ? data.n_ratings_used : null);
      const cards = data.items.map(card);
      if (anchor) cards.unshift(anchor);
      if (!cards.length) grid.replaceChildren(h("div", { class: "empty", text: "No movies match these filters. Try removing a genre." }));
      else grid.replaceChildren(...cards);
    } catch (err) {
      if (seq !== loadSeq) return;
      if (state.demoUser && state.tab === "foryou") { state.demoUser = null; syncDemo(); }
      setStatus("");
      grid.replaceChildren(h("div", { class: "empty" },
        h("p", { text: err.message }),
        h("button", { class: "btn", type: "button", text: "Retry", onclick: load })));
      toast(err.message, "error");
    }
  }

  // ---- tabs ---------------------------------------------------------------------------------------
  function switchTab(tab) {
    state.tab = tab;
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t.dataset.tab === tab;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", String(on));
    });
    load();
  }

  function showSimilar(item) {
    state.similar = { movie_id: item.movie_id, title: item.title };
    const tab = $("#tab-similar");
    tab.hidden = false;
    tab.textContent = `Similar to ${item.title.length > 22 ? item.title.slice(0, 21) + "…" : item.title}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
    switchTab("similar");
  }

  // ---- genre chips + demo user -----------------------------------------------------------------------
  function renderChips(all) {
    const box = $("#genre-chips");
    box.replaceChildren(...all.map((g) => h("button", {
      type: "button", class: "chip", text: g, "aria-pressed": String(state.genres.includes(g)),
      onclick: (e) => {
        state.genres = state.genres.includes(g) ? state.genres.filter((x) => x !== g) : [...state.genres, g].slice(0, 10);
        e.currentTarget.setAttribute("aria-pressed", String(state.genres.includes(g)));
        $("#clear-genres").hidden = !state.genres.length;
        saveStore();
        load();
      },
    })));
    $("#clear-genres").hidden = !state.genres.length;
  }

  function syncDemo() {
    $("#demo-exit").hidden = !state.demoUser;
    $("#demo-user").value = state.demoUser ?? "";
  }

  // ---- search -----------------------------------------------------------------------------------------
  const searchInput = $("#search");
  const resultsBox = $("#search-results");
  let searchTimer = null, searchCtl = null, activeIdx = -1;

  function hideResults() { resultsBox.hidden = true; resultsBox.replaceChildren(); activeIdx = -1; }

  function renderResults(list) {
    activeIdx = -1;
    if (!list.length) {
      resultsBox.replaceChildren(h("li", { class: "sr-empty", text: "No movies found" }));
    } else {
      resultsBox.replaceChildren(...list.map((m) => h("li", {
        role: "option", tabindex: "-1",
        onclick: () => { hideResults(); searchInput.value = ""; showSimilar(m); },
      }, h("span", { text: m.title }),
        h("span", { class: "sr-meta", text: [m.year, m.genres[0]].filter(Boolean).join(" · ") }))));
    }
    resultsBox.hidden = false;
  }

  async function runSearch(q) {
    if (searchCtl) searchCtl.abort();
    searchCtl = new AbortController();
    try {
      renderResults(await api(`/api/movies/search?q=${encodeURIComponent(q)}&limit=10`, { signal: searchCtl.signal }));
    } catch (err) {
      if (err.name !== "AbortError") { resultsBox.replaceChildren(h("li", { class: "sr-empty", text: err.message })); resultsBox.hidden = false; }
    }
  }

  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const q = searchInput.value.trim();
    if (!q) { hideResults(); return; }
    searchTimer = setTimeout(() => runSearch(q), 250);
  });
  searchInput.addEventListener("keydown", (e) => {
    const items = [...resultsBox.querySelectorAll("li[role=option]")];
    if (e.key === "Escape") { hideResults(); return; }
    if (!items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      activeIdx = (activeIdx + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.setAttribute("aria-selected", String(i === activeIdx)));
      items[activeIdx].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && activeIdx >= 0) {
      e.preventDefault();
      items[activeIdx].click();
    }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) hideResults(); });

  // ---- ratings drawer ---------------------------------------------------------------------------------
  const drawer = $("#drawer");
  let lastFocus = null;

  function openDrawer() {
    lastFocus = document.activeElement;
    drawer.classList.add("is-open");
    drawer.setAttribute("aria-hidden", "false");
    $("#scrim").hidden = false;
    $("#close-ratings").focus();
  }
  function closeDrawer() {
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    $("#scrim").hidden = true;
    if (lastFocus) lastFocus.focus();
  }

  function renderDrawer() {
    const entries = Object.entries(state.ratings).sort((a, b) => a[1].title.localeCompare(b[1].title));
    const list = $("#ratings-list");
    if (!entries.length) {
      list.replaceChildren(h("li", { class: "rl-empty", text: "No ratings yet. Rate movies on any card." }));
    } else {
      list.replaceChildren(...entries.map(([id, r]) => h("li", {},
        h("div", { class: "rl-top" },
          h("span", { text: r.year ? `${r.title} (${r.year})` : r.title }),
          h("button", {
            class: "rl-remove", type: "button", "aria-label": `Remove rating for ${r.title}`, text: "✕",
            onclick: () => { delete state.ratings[id]; saveStore(); afterRatingChange(); },
          })),
        stars(Number(id), (v) => setRating({ movie_id: Number(id), title: r.title, year: r.year }, v)))));
    }
    $("#clear-ratings").hidden = !entries.length;
  }

  // ---- evaluation dialog ----------------------------------------------------------------------------------
  function table(headers, rows) {
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, headers.map((x) => h("th", { text: x })))),
      h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", { text: String(c) })))))));
  }

  async function openMetrics() {
    const dlg = $("#metrics-dialog");
    const body = $("#metrics-body");
    body.textContent = "Loading…";
    if (typeof dlg.showModal === "function") dlg.showModal();
    try {
      const m = await api("/api/metrics");
      const f = (x, d = 4) => (typeof x === "number" ? x.toFixed(d) : "–");
      const parts = [];
      if (m.rating_prediction) {
        const rp = m.rating_prediction;
        parts.push(h("h3", { text: "Rating prediction on held-out ratings (lower is better)" }),
          table(["Model", "RMSE", "MAE"], ["global_mean", "item_bias", "user+item_bias", "matrix_factorization"]
            .filter((k) => rp[k]).map((k) => [k, f(rp[k].rmse), f(rp[k].mae)])));
      }
      if (m.ranking) {
        const rows = Object.entries(m.ranking).filter(([k]) => !k.startsWith("_")).map(([k, r]) =>
          [k, f(r["precision@10"]), f(r["recall@10"]), f(r["ndcg@10"]), f(r["coverage@10"]), f(r["diversity@10"])]);
        parts.push(h("h3", { text: `Top-10 ranking (${m.ranking._n_users_evaluated} users)` }),
          table(["Model", "Precision@10", "Recall@10", "NDCG@10", "Coverage", "Diversity"], rows));
      }
      parts.push(h("p", { class: "note", text:
        `Relevant = held-out rating ≥ ${m.split?.relevance_threshold ?? 4}. Chronological per-user split; ` +
        `trained ${m.meta?.trained_at_utc ?? "?"}. Produced by scripts/train.py.` }));
      body.replaceChildren(...parts);
    } catch (err) {
      body.textContent = err.message;
    }
  }

  // ---- banner / init -------------------------------------------------------------------------------------------
  function showBanner(msg) {
    const b = $("#banner");
    b.replaceChildren(h("strong", { text: "Backend not ready. " }), msg + " ",
      h("code", { text: "python scripts/train.py" }), " then ", h("code", { text: "uvicorn api.main:app --reload" }));
    b.hidden = false;
  }

  function wire() {
    $("#open-ratings").addEventListener("click", openDrawer);
    $("#close-ratings").addEventListener("click", closeDrawer);
    $("#scrim").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && drawer.classList.contains("is-open")) closeDrawer(); });
    $("#clear-ratings").addEventListener("click", () => {
      if (window.confirm("Remove all your ratings?")) { state.ratings = {}; saveStore(); afterRatingChange(); }
    });
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
    $("#refresh").addEventListener("click", load);
    $("#clear-genres").addEventListener("click", () => {
      state.genres = []; saveStore();
      document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      $("#clear-genres").hidden = true;
      load();
    });
    $("#demo-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const id = Number($("#demo-user").value);
      if (!Number.isInteger(id) || id < 1) { toast("Enter a valid sample dataset user id (1, 2, 3 …)", "error"); return; }
      state.demoUser = id;
      syncDemo();
      state.tab === "foryou" ? load() : switchTab("foryou");
    });
    $("#demo-exit").addEventListener("click", () => { state.demoUser = null; syncDemo(); load(); });
    $("#open-metrics").addEventListener("click", openMetrics);
    $("#close-metrics").addEventListener("click", () => $("#metrics-dialog").close());
  }

  // ---- theme ---------------------------------------------------------------------
  // Key must match the inline <script> in index.html's <head>.
  const THEME_KEY = "cinematch.theme";
  const mq = window.matchMedia("(prefers-color-scheme: light)");

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || (mq.matches ? "light" : "dark");
  }

  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    const btn = $("#theme-toggle");
    if (btn) {
      const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
      btn.setAttribute("aria-label", label);
      btn.title = label;
    }
    if (persist) {
      try { localStorage.setItem(THEME_KEY, theme); } catch { /* ignore */ }
    }
  }

  function wireTheme() {
    applyTheme(currentTheme(), false);
    $("#theme-toggle").addEventListener("click", () =>
      applyTheme(currentTheme() === "dark" ? "light" : "dark", true));
  }

  async function init() {
    loadStore();
    wire();
    wireTheme();
    renderDrawer();
    $("#rating-count").textContent = Object.keys(state.ratings).length;
    try {
      const health = await api("/api/health");
      if (health.status !== "ok") { showBanner("The model has not been trained yet. Run"); setStatus(""); grid.replaceChildren(h("div", { class: "empty", text: "No model loaded." })); return; }
    } catch (err) {
      showBanner(err.message + " Start it with");
      grid.replaceChildren(h("div", { class: "empty", text: "API unreachable." }));
      return;
    }
    try { renderChips(await api("/api/genres")); } catch (err) { toast(err.message, "error"); }
    load();
  }

  init();
})();