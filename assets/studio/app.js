/* Style Studio: reads the store through /api, writes picks and owner actions through POST /api.
   Artifact mode (window.STUDIO_ARTIFACT set by studio_artifact.py): reads the published snapshot files and
   saves every action as a row in the artifact's db ("actions"); Claude applies the rows with
   studio_artifact.py apply. All values from files are text: set with textContent or escaped before HTML. */
(function () {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const screen = $("#screen");
  const KIND = {trained: "אומן", described: "מתיאור", derived: "נגזר"};
  const STATUS = {ready: "מוכן", training: "באימון", archived: "בארכיון"};
  const STEP = {plan: "תוכנית", frames: "פריימים", motion: "תנועה"};
  const SOURCE = {law: "חוק של הסגנון", idea: "מבנק הרעיונות", invention: "רעיון חדש"};
  let data = null;
  const ART = window.STUDIO_ARTIFACT || null;
  let db, pending = [];
  let cur = location.hash;
  const go = h => { if (ART) { cur = h; route(); } else location.hash = h; };

  // ------------------------------------------------------------ io
  async function useDb() {
    if (db !== undefined) return db;
    try { db = (window.claude && window.claude.use) ? await window.claude.use("db") : null; } catch (e) { db = null; }
    return db;
  }
  const docsOf = s => ((s && (s.docs || s.documents)) || (Array.isArray(s) ? s : [])).map(x => typeof x.data === "function" ? x.data() : (x.data || x));
  async function loadPending() {
    const d = await useDb();
    if (!d) return;
    try {
      const rows = docsOf(await d.collection("actions").get());
      pending = rows.filter(r => r && r.route && r.at > (ART.built_at || "")).sort((a, b) => a.at < b.at ? -1 : 1);
    } catch (e) { pending = []; }
  }
  function overlayOverview(o) {
    for (const r of pending) {
      let m;
      if ((m = r.route.match(/^\/api\/lesson\/([^/]+)\/(approve|reject)$/))) {
        const l = o.lessons.find(x => x.id === m[1]); if (l) { l.status = m[2] === "approve" ? "rule" : "rejected"; l.queued = true; }
      } else if ((m = r.route.match(/^\/api\/idea\/([^/]+)\/remove$/))) {
        o.ideas = o.ideas.filter(i => i.id !== m[1]);
      } else if ((m = r.route.match(/^\/api\/style\/([^/]+)\/rename$/))) {
        const s = o.styles.find(x => x.slug === m[1]); if (s) { s.name = (r.body || {}).name || s.name; s.queued = true; }
      }
    }
    return o;
  }
  function overlayRoom(pid, room) {
    for (const r of pending) {
      const m = r.route.match(/^\/api\/project\/([^/]+)\/(pick|motion-mode)$/);
      if (!m || decodeURIComponent(m[1]) !== pid) continue;
      const b = r.body || {};
      room.picks = room.picks || {};
      if (m[2] === "motion-mode") { room.picks.motion = room.picks.motion || {beats: {}}; room.picks.motion.mode = b.mode; continue; }
      let sec;
      if (b.step === "motion") { room.picks.motion = room.picks.motion || {beats: {}}; sec = room.picks.motion.beats = room.picks.motion.beats || {}; }
      else sec = room.picks[b.step] = room.picks[b.step] || {};
      sec[String(b.beat)] = {picked: b.id || null, more: !!b.more, own: b.own || "", queued: true};
      if (b.note != null) (room.picks.notes = room.picks.notes || {})[String(b.beat)] = b.note;
    }
    return room;
  }
  async function get(url) {
    if (ART) {
      const m = url.match(/^\/api\/project\/(.+)$/);
      const pid = m && decodeURIComponent(m[1]);
      const entry = m && data && data.projects.find(p => p.id === pid);
      const file = url === "/api/overview" ? "data/overview.json" : entry && entry.room_file;
      if (!file) throw new Error("לא נמצא");
      const r = await fetch(file);
      if (!r.ok) throw new Error("הקובץ חסר בארטיפקט");
      const j = await r.json();
      if (!m) { await loadPending(); return overlayOverview(j); }
      return overlayRoom(pid, j);
    }
    const r = await fetch(url, {credentials: "same-origin"});
    const j = await r.json().catch(() => ({error: "bad response"}));
    if (!r.ok) throw new Error(j.error || r.status);
    return j;
  }
  async function post(url, body) {
    if (ART) {
      const d = await useDb();
      if (!d) throw new Error("השמירה לא זמינה כאן. פתחו את הארטיפקט ב־claude.ai כשאתם מחוברים");
      const row = {route: url, body: body || {}, at: new Date().toISOString(), applied: false};
      const id = row.at.replace(/[^0-9]/g, "") + "-" + Math.random().toString(36).slice(2, 6);
      await d.collection("actions").doc(id).set(row);
      pending.push(row);
      return {ok: true, queued: true, out: "", err: ""};
    }
    const r = await fetch(url, {method: "POST", credentials: "same-origin", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body || {})});
    const j = await r.json().catch(() => ({error: "bad response"}));
    if (!r.ok) throw new Error(j.error || r.status);
    return j;
  }
  function ask(text, input) {
    return new Promise(res => {
      const m = document.createElement("div");
      m.className = "modal";
      m.innerHTML = `<div class="mbox" role="dialog" aria-modal="true"><p></p>${input != null ? '<input class="note-in" id="ask-in">' : ""}<div class="acts"><button class="btn pri" data-a="ok" type="button">אישור</button><button class="btn" data-a="no" type="button">ביטול</button></div></div>`;
      m.querySelector("p").textContent = text;
      const inp = m.querySelector("#ask-in");
      if (inp) inp.value = input;
      document.body.append(m);
      (inp || m.querySelector("[data-a=ok]")).focus();
      const done = ok => { m.remove(); res(ok ? (inp ? inp.value : true) : (inp ? null : false)); };
      m.addEventListener("click", e => { const a = e.target.dataset && e.target.dataset.a; if (a) done(a === "ok"); else if (e.target === m) done(false); });
      m.addEventListener("keydown", e => { if (e.key === "Escape") done(false); if (e.key === "Enter" && inp) done(true); });
    });
  }
  const saved = r => r && r.queued ? "נשמר. Claude יחיל אחרי \"בחרתי\"" : null;
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.add("on");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("on"), 2400);
  }
  async function load() {
    data = await get("/api/overview");
    $("#ct-styles").textContent = data.styles.length;
    $("#ct-clients").textContent = data.clients.length;
    $("#ct-ideas").textContent = data.ideas.length;
    $("#ct-projects").textContent = data.projects.length;
    const cand = data.lessons.filter(l => l.status === "candidate" || l.status === "proposed").length;
    const b = $("#ct-learning"); b.hidden = !cand; b.textContent = cand;
    const miss = Object.entries(data.caps.missing || {});
    $("#conn").innerHTML = data.caps.checked
      ? `<b><span class="dot${miss.length ? " miss" : ""}"></span>חיבורים</b><p>${miss.length ? esc(miss.map(([k, v]) => k + ": " + v.join(", ")).join(" · ")) : "הכול מחובר"}</p>`
      : `<b><span class="dot miss"></span>חיבורים</b><p>עוד לא נבדק (doctor.py)</p>`;
  }

  // ------------------------------------------------------------ shell
  function crumb(parts) { $("#crumb").innerHTML = parts.map((p, i) => i === parts.length - 1 ? `<b>${esc(p)}</b>` : esc(p)).join(" / "); }
  function nav(on) { document.querySelectorAll("#nav .ni").forEach(b => b.classList.toggle("is-on", b.dataset.go === on)); }
  // two-tone title: the last word in the light violet (text only, built with DOM nodes)
  function twoTone(root) {
    root.querySelectorAll(".h1 h1, .selhead h2").forEach(h => {
      const t = h.textContent.trim(), i = t.lastIndexOf(" ");
      if (i < 1) return;
      const em = document.createElement("em");
      em.textContent = t.slice(i + 1);
      h.textContent = t.slice(0, i + 1);
      h.append(em);
      h.classList.add("tt");
    });
  }
  function pad(html) { screen.innerHTML = `<div class="pad">${html}</div>`; twoTone(screen); }
  const thumb = (url, ar) => url
    ? (/\.(mp4|webm|mov)$|\/clip$/.test(url) ? `<div class="thbox"><video src="${esc(url)}" muted loop playsinline preload="metadata"></video></div>`
      : `<div class="thbox"><img src="${esc(url)}" alt="" loading="lazy"></div>`)
    : `<div class="thbox ph">אין תמונה${ar ? " · " + esc(ar) : ""}</div>`;

  // ------------------------------------------------------------ screens
  function styles() {
    nav("styles"); crumb(["סגנונות"]);
    if (!data.styles.length) return pad(`<div class="h1"><h1>סגנונות</h1></div><div class="empty">עוד אין סגנונות. בצ'אט: "תאמן סגנון חדש".</div>`);
    pad(`<div class="h1"><h1>סגנונות</h1><span>${data.styles.length}</span></div>
      <p class="lead">כל סגנון שאומן. הפריים שמייצג אותו נבחר ב־cover.py ובאישורך.</p>
      <div class="grid">${data.styles.map(s => `<article class="card" data-style="${esc(s.slug)}" tabindex="0">
        ${thumb(s.cover, s.aspect)}
        <div class="cb"><div class="row"><h3>${esc(s.name)}</h3><span class="badge ${s.status === "ready" ? "ready" : "train"}"><i></i>${esc(STATUS[s.status] || s.status)}</span></div>
          <div class="meta">${esc(KIND[s.kind] || s.kind || "")} · ${esc(s.aspect || "")} · ${s.rules_count || 0} חוקים</div>
          <div class="kv"><span class="k">לקוחות</span><span class="v">${s.clients.length ? s.clients.map(c => `<span class="av"><i>${esc((c || "?")[0])}</i>${esc(c)}</span>`).join("") : '<span class="none">אין</span>'}</span></div>
          <div class="kv"><span class="k">חיבורים</span><span class="v">${(s.requires || []).map(r => `<span class="cx${(data.caps.missing && Object.values(data.caps.missing).flat().includes(r.name)) ? " missing" : ""}"><i></i>${esc(r.name)}</span>`).join("") || '<span class="none">-</span>'}</span></div>
        </div></article>`).join("")}</div>`);
    screen.querySelectorAll("[data-style]").forEach(c => c.addEventListener("click", () => go("#/style/" + c.dataset.style)));
  }

  function styleOne(slug) {
    const s = data.styles.find(x => x.slug === slug);
    if (!s) return styles();
    nav("styles"); crumb(["סגנונות", s.name]);
    const rules = data.lessons.filter(l => l.status === "rule" && l.layer === "style" && l.target === slug);
    pad(`<div class="selhead">${thumb(s.cover, s.aspect)}
        <div class="sh-b"><span class="sh-k">${esc(KIND[s.kind] || "")} · ${esc(s.aspect || "")}</span><h2>${esc(s.name)}</h2>
        <p class="meta">${esc(s.description || "")}</p>
        <div class="stats"><div><b>${s.rules_count || 0}</b><span>חוקים מאושרים</span></div><div><b>${s.clients.length}</b><span>לקוחות</span></div><div><b>${data.ideas.filter(i => (i.made_for || {}).style === slug).length}</b><span>רעיונות</span></div></div>
        <div class="acts"><button class="btn" id="rename" type="button">שינוי שם</button></div></div></div>
      <h2 class="meta" style="margin:8px 0">חוקים של הסגנון</h2>
      <div class="list">${rules.map(r => `<div class="row-item"><span class="mono">${esc(r.id)}</span><div class="grow"><b>${esc(r.measurable)}</b><p>${esc(r.text)}</p></div></div>`).join("") || '<div class="empty">עוד אין חוקים שאושרו לסגנון הזה</div>'}</div>`);
    $("#rename").addEventListener("click", async () => {
      const name = await ask("שם חדש לסגנון", s.name);
      if (!name || name === s.name) return;
      try { const r = await post(`/api/style/${encodeURIComponent(slug)}/rename`, {name}); if (!r.ok) throw new Error(r.err); toast(saved(r) || "השם נשמר"); await load(); styleOne(slug); }
      catch (e) { toast("לא נשמר: " + e.message); }
    });
  }

  function clients() {
    nav("clients"); crumb(["לקוחות"]);
    if (!data.clients.length) return pad(`<div class="h1"><h1>לקוחות</h1></div><div class="empty">עוד אין לקוחות. בצ'אט: "לקוח חדש" עם צבע מותג אחד.</div>`);
    pad(`<div class="h1"><h1>לקוחות</h1><span>${data.clients.length}</span></div>
      <div class="grid">${data.clients.map(c => `<article class="card" data-client="${esc(c.slug)}" tabindex="0">
        ${c.logo ? `<div class="thbox"><img src="${esc(c.logo)}" alt="" style="object-fit:contain;padding:18px"></div>` : `<div class="thbox ph">${esc(c.name)}</div>`}
        <div class="cb"><h3>${esc(c.name)}</h3>
          <div class="swatch">${Object.values(c.palette).map(h => `<i style="background:${/^#[0-9a-f]{3,8}$/i.test(h) ? h : "transparent"}" title="${esc(h)}"></i>`).join("")}</div>
          <div class="kv"><span class="k">סגנונות</span><span class="v">${c.styles.map(x => esc(x)).join(", ") || '<span class="none">אין</span>'}</span></div>
        </div></article>`).join("")}</div>`);
    screen.querySelectorAll("[data-client]").forEach(c => c.addEventListener("click", () => go("#/client/" + c.dataset.client)));
  }

  function clientOne(slug) {
    const c = data.clients.find(x => x.slug === slug);
    if (!c) return clients();
    nav("clients"); crumb(["לקוחות", c.name]);
    const rules = data.lessons.filter(l => l.status === "rule" && l.layer === "client" && l.target === slug);
    const projects = data.projects.filter(p => p.client === slug);
    pad(`<div class="h1"><h1>${esc(c.name)}</h1></div>
      <div class="ins" style="max-width:760px">
        <section><h6>פלטה</h6><div class="swatch">${Object.entries(c.palette).map(([k, h]) => `<i style="background:${/^#[0-9a-f]{3,8}$/i.test(h) ? h : "transparent"};width:44px;height:44px" title="${esc(k)} ${esc(h)}"></i>`).join("")}</div></section>
        <section><h6>פונטים</h6><dl>${Object.entries(c.fonts).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></section>
        <section><h6>כן / לא</h6><p>${c.dos.map(esc).join(" · ") || "-"}</p><p>${c.donts.map(esc).join(" · ") || "-"}</p></section>
        <section><h6>חוקים של הלקוח</h6>${rules.map(r => `<p><span class="mono">${esc(r.id)}</span> ${esc(r.measurable)}</p>`).join("") || "<p class='none'>אין</p>"}</section>
        <section><h6>פרויקטים</h6>${projects.map(p => `<p><a href="#/project/${encodeURIComponent(p.id)}/plan">${esc(p.title)}</a> · ${esc(p.stage)}</p>`).join("") || "<p class='none'>אין</p>"}</section>
      </div>`);
  }

  const ideaFilter = {kind: ""};
  function ideas() {
    nav("ideas"); crumb(["רעיונות"]);
    const kinds = [...new Set(data.ideas.map(i => i.kind))];
    const list = data.ideas.filter(i => !ideaFilter.kind || i.kind === ideaFilter.kind);
    pad(`<div class="h1"><h1>בנק רעיונות</h1><span>${data.ideas.length}</span></div>
      <div class="bar"><button class="chip${!ideaFilter.kind ? " is-on" : ""}" data-k="">הכול</button>${kinds.map(k => `<button class="chip${ideaFilter.kind === k ? " is-on" : ""}" data-k="${esc(k)}">${esc(k)}</button>`).join("")}</div>
      ${list.length ? `<div class="grid">${list.map(i => `<article class="card" tabindex="0">
        ${i.clip ? `<div class="thbox"><video src="${esc(i.clip)}" ${i.still ? `poster="${esc(i.still)}"` : ""} muted loop playsinline preload="none"></video></div>` : thumb(i.still, i.aspect)}
        <div class="cb"><div class="row"><h3>${esc(i.title)}</h3><span class="tag">${esc(i.kind)}</span></div>
          <div class="meta sm">${esc((i.made_for || {}).style || "כללי")} · נבחר ${i.picked || 0} מתוך ${i.offered || 0} · ${esc(i.source)}</div>
          <div class="row"><span class="mono">${esc(i.id)}</span><button class="btn ghost" data-rm="${esc(i.id)}" type="button" style="margin-inline-start:auto">הסרה</button></div>
        </div></article>`).join("")}</div>` : '<div class="empty">אין רעיונות בסינון הזה</div>'}`);
    screen.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => { ideaFilter.kind = b.dataset.k; ideas(); }));
    screen.querySelectorAll("video").forEach(v => { v.closest(".card").addEventListener("mouseenter", () => v.play().catch(() => {})); v.closest(".card").addEventListener("mouseleave", () => v.pause()); });
    screen.querySelectorAll("[data-rm]").forEach(b => b.addEventListener("click", async e => {
      e.stopPropagation();
      if (!await ask("להסיר את הרעיון? הוא עובר לארכיון, לא נמחק.")) return;
      try { const r = await post(`/api/idea/${b.dataset.rm}/remove`); if (!r.ok) throw new Error(r.err); toast(saved(r) || "הוסר לארכיון"); await load(); ideas(); }
      catch (err) { toast("לא הוסר: " + err.message); }
    }));
  }

  function learning() {
    nav("learning"); crumb(["למידה"]);
    const open = data.lessons.filter(l => l.status === "candidate" || l.status === "proposed");
    const rules = data.lessons.filter(l => l.status === "rule");
    const row = (l, acts) => `<div class="row-item"><span class="mono">${esc(l.id)}</span><div class="grow"><b>${esc(l.measurable)}</b>${l.queued ? ' <span class="queued">ממתין ל־Claude</span>' : ""}<p>${esc(l.text)}</p><p class="meta sm">${esc(l.layer)}${l.target ? " · " + esc(l.target) : ""}${l.scope ? " · " + esc(l.scope) : ""}</p></div>${acts ? `<div class="acts"><button class="btn pri" data-ok="${esc(l.id)}" data-st="${esc(l.status)}" type="button">לאשר כחוק</button><button class="btn" data-no="${esc(l.id)}" type="button">לדחות</button></div>` : ""}</div>`;
    pad(`<div class="h1"><h1>למידה</h1><span>${open.length} ממתינים</span></div>
      <p class="lead">הערות שחזרו על עצמן. רק האישור שלך הופך אותן לחוק.</p>
      <div class="list">${open.map(l => row(l, true)).join("") || '<div class="empty">אין הצעות שממתינות לך</div>'}</div>
      <h2 class="meta" style="margin:22px 0 8px">חוקים (${rules.length})</h2>
      <div class="list">${rules.map(l => row(l, false)).join("")}</div>`);
    screen.querySelectorAll("[data-ok]").forEach(b => b.addEventListener("click", async () => {
      try {
        let r = await post(`/api/lesson/${b.dataset.ok}/approve`, {one_note: b.dataset.st === "candidate"});
        if (!r.ok && /yes-no-conflict/.test(r.err) && await ask("יש חוקים שאולי סותרים:\n" + r.out + "\nלאשר בכל זאת?"))
          r = await post(`/api/lesson/${b.dataset.ok}/approve`, {one_note: b.dataset.st === "candidate", yes_no_conflict: true});
        if (!r.ok) throw new Error(r.err || r.out);
        toast(saved(r) || "אושר כחוק"); await load(); learning();
      } catch (e) { toast("לא אושר: " + e.message); }
    }));
    screen.querySelectorAll("[data-no]").forEach(b => b.addEventListener("click", async () => {
      const reason = await ask("למה לדחות? (לא חובה)", "");
      if (reason == null) return;
      try { const r = await post(`/api/lesson/${b.dataset.no}/reject`, {reason}); if (!r.ok) throw new Error(r.err); toast(saved(r) || "נדחה"); await load(); learning(); }
      catch (e) { toast("לא נדחה: " + e.message); }
    }));
  }

  function projects() {
    nav("projects"); crumb(["פרויקטים"]);
    pad(`<div class="h1"><h1>פרויקטים</h1><span>${data.projects.length}</span></div>
      <div class="list">${data.projects.map(p => `<a class="row-item" href="#/project/${encodeURIComponent(p.id)}/plan" style="text-decoration:none;color:inherit">
        <div class="grow"><b>${esc(p.title)}</b><p>${esc(p.style)}${p.client ? " · " + esc(p.client) : ""} · שלב ${esc(p.stage)} · ${esc(p.status)}</p></div>
        <span class="meta sm">${esc((p.updated || "").slice(0, 10))}</span></a>`).join("") || '<div class="empty">עוד אין פרויקטים</div>'}</div>`);
  }

  // ------------------------------------------------------------ project room
  function optionsFor(room, step, n) {
    if (step === "plan") {
      const b = (room.plan.beats || []).find(x => +x.n === +n) || {};
      const opts = b.options || [];
      const last = Math.max(0, ...opts.map(o => +o.set || 1));
      return {beat: b, opts: opts.filter(o => (+o.set || 1) === last), rec: b.recommended};
    }
    const doc = step === "frames" ? room.frames : room.motion;
    const bt = (doc.beats || {})[String(n)] || {};
    const sets = bt.sets || [];
    const lastSet = sets[sets.length - 1] || {variants: []};
    const b = (room.plan.beats || []).find(x => +x.n === +n) || {};
    return {beat: b, opts: lastSet.variants || [], rec: (lastSet.variants || [])[0] && lastSet.variants[0].id};
  }

  async function room(pid, step, beatN) {
    nav("projects");
    let r;
    try { r = await get("/api/project/" + encodeURIComponent(pid)); }
    catch (e) { return pad(`<div class="empty">לא נמצא: ${esc(e.message)}</div>`); }
    const beats = (r.plan.beats || []);
    const n = +beatN || (beats[0] && beats[0].n) || 1;
    crumb(["פרויקטים", r.entry.title, STEP[step]]);
    const sec = step === "motion" ? (r.picks.motion.beats || {}) : (r.picks[step] || {});
    const {beat, opts, rec} = optionsFor(r, step, n);
    const pick = sec[String(n)] || {};
    const media = o => step === "frames" ? (o.still ? r.media + o.still : null) : step === "motion" ? (o.clip ? r.media + o.clip : null) : null;
    screen.innerHTML = `<div class="three">
      <div class="col"><h5>ביטים</h5>${beats.map(b => { const p = sec[String(b.n)] || {};
        return `<button class="beat${+b.n === n ? " now" : p.picked || p.own ? " done" : ""}" data-n="${b.n}" type="button"><span class="n">${b.n}</span><span>${esc((b.script || "").slice(0, 34))}</span><small>${p.picked ? esc(p.picked) : p.more ? "עוד רעיונות" : ""}${p.queued ? ' <span class="queued">ממתין</span>' : ""}</small></button>`; }).join("")}</div>
      <div class="center">
        <div class="steps">${Object.keys(STEP).map(s => `<a class="chip${s === step ? " is-on" : ""}" href="#/project/${encodeURIComponent(pid)}/${s}/${n}">${STEP[s]}</a>`).join("")}</div>
        ${r.problems.length ? `<div class="probs">${r.problems.map(esc).join("<br>")}</div>` : ""}
        <div class="bhead"><span class="pill">ביט ${n}</span><span class="mono">${beat.start != null ? (+beat.start).toFixed(1) + "–" + (+beat.end).toFixed(1) + " s" : ""}</span></div>
        <div class="words">${esc(beat.script || "")}</div>
        ${step === "motion" ? `<div class="bar"><span class="meta">מצב תנועה:</span><button class="chip${(r.picks.motion.mode || "three") === "three" ? " is-on" : ""}" data-mode="three">3 וריאציות</button><button class="chip${r.picks.motion.mode === "one" ? " is-on" : ""}" data-mode="one">אחת, Claude מחליט</button></div>` : ""}
        ${opts.length ? `<div class="opts">${opts.map((o, i) => `<div class="opt${pick.picked === o.id ? " is-picked" : ""}">
            ${step !== "plan" ? (media(o) ? (step === "motion" ? `<video class="fr" src="${esc(media(o))}" autoplay muted loop playsinline></video>` : `<img class="fr" src="${esc(media(o))}" alt="" loading="lazy">`) : `<div class="fr thbox ph">עוד אין ${step === "frames" ? "פריים" : "קליפ"}</div>`) : ""}
            <div class="oh"><span class="key">${esc(o.key || "abc"[i] || i + 1)}</span><h4>${esc(o.title || o.entrance || o.id)}</h4>${o.wild ? '<span class="tag wild">קלף פראי</span>' : ""}${o.id === rec ? '<span class="tag rec">ההמלצה שלי</span>' : ""}</div>
            <p>${esc(o.desc || o.treatment || (o.ease ? o.ease.name + " · " + (o.timing || {}).dur_s + " s · " + (o.camera || {}).prop : ""))}</p>
            ${o.source ? `<p class="meta sm">${esc(SOURCE[o.source] || o.source)}${o.law ? " · " + esc(o.law) : ""}</p>` : ""}
            <button class="btn" data-pick="${esc(o.id)}" type="button">${pick.picked === o.id ? "✓ נבחר" : "לבחור בזה"}</button></div>`).join("")}</div>`
          : `<div class="empty">עוד אין אפשרויות לשלב הזה בביט הזה</div>`}
        <div class="morebar"><button class="btn${pick.more ? " pri" : ""}" id="more" type="button">עוד רעיונות</button>
          <input class="note-in" id="note" placeholder="הערה לביט (לא חובה)" value="${esc((r.picks.notes || {})[String(n)] || "")}" style="flex:1;min-width:200px">
          <span class="hint">${ART ? "כל בחירה נשמרת כאן. כשסיימתם, כתבו ל־Claude \"בחרתי\"" : "אחרי הבחירות, כתבו ל־Claude \"בחרתי\""}</span></div>
      </div>
      <div class="col"><div class="ins">
        <section><h6>פרויקט</h6><div class="big">${esc(r.entry.title)}</div><dl><dt>סגנון</dt><dd>${esc(r.entry.style)}</dd><dt>לקוח</dt><dd>${esc(r.entry.client || "-")}</dd><dt>שלב</dt><dd>${esc(r.entry.stage)}</dd></dl></section>
        ${r.spend.length ? `<section><h6>הוצאות מאושרות</h6>${r.spend.map(b => `<p>${esc(b.id)} · ${esc(b.status)} · ${(+b.spent || 0).toFixed(2)} / ${(+b.total || 0).toFixed(2)} ${esc(b.currency)}</p>`).join("")}</section>` : ""}
      </div></div></div>`;
    screen.querySelectorAll("[data-n]").forEach(b => b.addEventListener("click", () => go(`#/project/${encodeURIComponent(pid)}/${step}/${b.dataset.n}`)));
    const save = async body => {
      try { const res = await post(`/api/project/${encodeURIComponent(pid)}/pick`, Object.assign({step, beat: n}, body)); toast(saved(res) || "נשמר"); room(pid, step, n); }
      catch (e) { toast("לא נשמר: " + e.message); }
    };
    screen.querySelectorAll("[data-pick]").forEach(b => b.addEventListener("click", () => save({id: b.dataset.pick, note: $("#note").value})));
    $("#more").addEventListener("click", () => save({more: !pick.more, id: pick.picked || null, note: $("#note").value}));
    $("#note").addEventListener("change", e => save({note: e.target.value, id: pick.picked || null, more: !!pick.more}));
    screen.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", async () => {
      try { const res = await post(`/api/project/${encodeURIComponent(pid)}/motion-mode`, {mode: b.dataset.mode}); toast(saved(res) || "נשמר"); room(pid, step, n); }
      catch (e) { toast("לא נשמר: " + e.message); }
    }));
    document.onkeydown = e => {
      if (e.target.closest("input,textarea")) return;
      const i = "123".indexOf(e.key);
      if (i >= 0 && opts[i]) save({id: opts[i].id});
      const idx = beats.findIndex(b => +b.n === n);
      if (e.key === "ArrowDown" && beats[idx + 1]) go(`#/project/${encodeURIComponent(pid)}/${step}/${beats[idx + 1].n}`);
      if (e.key === "ArrowUp" && beats[idx - 1]) go(`#/project/${encodeURIComponent(pid)}/${step}/${beats[idx - 1].n}`);
    };
  }

  // ------------------------------------------------------------ router
  async function route() {
    document.onkeydown = null;
    const p = (ART ? cur : location.hash).replace(/^#\/?/, "").split("/").map(decodeURIComponent);
    if (!data) { try { await load(); } catch (e) { return pad(`<div class="empty">אין חיבור לסטודיו: ${esc(e.message)}</div>`); } }
    if (p[0] === "style") return styleOne(p[1]);
    if (p[0] === "clients") return clients();
    if (p[0] === "client") return clientOne(p[1]);
    if (p[0] === "ideas") return ideas();
    if (p[0] === "learning") return learning();
    if (p[0] === "projects") return projects();
    if (p[0] === "project") return room(p[1], STEP[p[2]] ? p[2] : "plan", p[3]);
    return styles();
  }
  document.querySelectorAll("#nav .ni").forEach(b => b.addEventListener("click", () => go("#/" + b.dataset.go)));
  document.addEventListener("click", e => {
    const a = ART && e.target.closest && e.target.closest('a[href^="#/"]');
    if (a) { e.preventDefault(); go(a.getAttribute("href")); }
  });
  $("#reload").addEventListener("click", async () => { data = null; await route(); toast("רוענן"); });
  const root = document.documentElement;
  try { const t = localStorage.getItem("style-lab:theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $("#theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("style-lab:theme", root.dataset.theme); } catch (e) {}
  });
  window.addEventListener("hashchange", route);
  route();
})();
