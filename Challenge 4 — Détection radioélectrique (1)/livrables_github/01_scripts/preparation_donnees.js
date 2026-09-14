/* Aval de la partie données — Challenge 4 (Niveau 1)
   Amont (équipe données) : uploads/Doc hackathon/Filtrage_hackathon.py
     SUP_SUPPORT.txt + SUP_NATURE.txt -> sites_France.csv (95 202) / sites_32.csv (437)
   Ce script (aval, artefact) : sites_*.csv -> JSON de data/
     1. jointure code département via SUP_SUPPORT.txt (INSEE, puis CP ; 2A/2B gérés)
     2. correction du signe des longitudes ouest (l'amont compare l'hémisphère à "O",
        l'ANFR code "W" : 17 723 sites sortent du CSV en positif — correctif amont : hemi_neg="W")
     3. splits nationaux par cluster spatial 500 m, 75/10/15, graine 42 (EF-20)
     4. split dédié 70/15/15 pour le pilote Gers
     5. agrégats (densité par département, stats globales)
   Exécution : node scripts/preparation_donnees.js — déterministe. */
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const read = f => fs.readFileSync(path.join(ROOT, f), 'utf8');

function parseCSV(txt) {
  const out = [], lines = txt.split('\n');
  for (let i = 1; i < lines.length; i++) {
    const l = lines[i]; if (!l.trim()) continue;
    const f = []; let cur = '', q = false;
    for (const ch of l) { if (ch === '"') q = !q; else if (ch === ',' && !q) { f.push(cur); cur = ''; } else cur += ch; }
    f.push(cur.trim()); out.push(f);
  }
  return out;
}
const numFR = s => { if (s === '' || s == null) return null; const v = parseFloat(String(s).replace(',', '.')); return isFinite(v) ? v : null; };
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const s = arr[i]; arr[i] = arr[j]; arr[j] = s; }
  return arr;
}

/* 1. id -> hémisphère lon / CP / INSEE, depuis le référentiel brut */
const idInfo = new Map();
for (const l of read('uploads/Doc hackathon/SUP_SUPPORT.txt').split('\n').slice(1)) {
  if (!l) continue; const c = l.split(';');
  if (!idInfo.has(c[0])) idInfo.set(c[0], { ew: c[10], cp: c[17], insee: c[18] });
}
const dept = id => {
  const r = idInfo.get(id); if (!r) return null;
  const ins = r.insee || '';
  if (/^2[AB]/.test(ins)) return ins.slice(0, 2);
  if (/^\d{5}/.test(ins)) return ins.slice(0, 2);
  return String(r.cp || '').padStart(5, '0').slice(0, 2);
};
/* 2. Correctif hémisphère : W (ANFR) vs "O" attendu par l'amont */
const fixLon = (id, lon) => { const r = idInfo.get(id); return (r && r.ew === 'W') ? -Math.abs(lon) : lon; };

/* 3. sites_France.csv -> métropole + splits nationaux */
let fixed = 0;
const sites = parseCSV(read('uploads/Doc hackathon/sites_France.csv')).map(f => {
  const lon0 = +f[2], lon = fixLon(f[0], lon0); if (lon !== lon0) fixed++;
  return { id: f[0], lat: +f[1], lon, type: f[3], haut: numFR(f[4]), dept: dept(f[0]) };
});
const metro = sites.filter(s => s.dept && (/^2[AB]/.test(s.dept) || (+s.dept >= 1 && +s.dept <= 95)));

const ckey = r => Math.round(r.lon * 111320 * Math.cos(r.lat * Math.PI / 180) / 500)
  + '|' + Math.round(r.lat * 111320 / 500);
const clusters = new Map();
for (const r of metro) { const k = ckey(r); (clusters.get(k) || clusters.set(k, []).get(k)).push(r); }
const keys = shuffle([...clusters.keys()], mulberry32(42));
const N = metro.length; let nTst = 0, nVl = 0;
for (const k of keys) {
  const c = clusters.get(k);
  if (nTst < 0.15 * N) { for (const r of c) r.split = 2; nTst += c.length; }
  else if (nVl < 0.10 * N) { for (const r of c) r.split = 1; nVl += c.length; }
  else for (const r of c) r.split = 0;
}
const sc = [0, 0, 0]; for (const r of metro) sc[r.split]++;

/* 4. sites_32.csv -> jeu pilote Gers, split dédié 70/15/15 */
const gers = parseCSV(read('uploads/Doc hackathon/sites_32.csv')).map(f => ({
  id: 'SUP-' + f[0], lat: +(+f[1]).toFixed(6), lon: +fixLon(f[0], +f[2]).toFixed(6),
  type: f[3], haut: numFR(f[4]), cp: (idInfo.get(f[0]) || {}).cp || null,
}));
const gshuf = shuffle([...gers], mulberry32(42));
const nT = Math.floor(gshuf.length * 0.15), nV = Math.floor(gshuf.length * 0.15);
gshuf.forEach((s, i) => { s.split = i < nT ? 'test' : (i < nT + nV ? 'validation' : 'train'); });
const gersDataset = [...gshuf].sort((a, b) => a.id < b.id ? -1 : 1);
const gsc = { train: 0, validation: 0, test: 0 }; for (const s of gersDataset) gsc[s.split]++;

/* 5. Sorties */
const franceAll = metro.map(r => [r.id, +r.lat.toFixed(5), +r.lon.toFixed(5), r.dept, r.split, r.haut, r.type]);
const densite = {}; for (const r of metro) densite[r.dept] = (densite[r.dept] || 0) + 1;
const stats = {
  source: 'Filtrage_hackathon.py (nettoyage ANFR) -> sites_France.csv / sites_32.csv ; aval : jointure dept + correction hémisphère W + splits',
  sitesFrance: sites.length, metropole: metro.length, outreMerOuSansDept: sites.length - metro.length,
  lonCorrigees: fixed,
  clusters: clusters.size, splits: { train: sc[0], validation: sc[1], test: sc[2] }, graine: 42,
  departements: Object.keys(densite).length,
  gers: { sites: gersDataset.length, splits: gsc }, byDept: densite,
};
const out = (f, d) => fs.writeFileSync(path.join(ROOT, 'data', f), d);
out('france_all.json', JSON.stringify(franceAll));
out('france_densite.json', JSON.stringify(densite));
out('france_stats.json', JSON.stringify(stats, null, 1));
out('gers_dataset.json', JSON.stringify(gersDataset));
console.log('OK —', metro.length, 'sites métropole,', fixed, 'longitudes corrigées,', clusters.size, 'clusters, splits', sc.join('/'), '· Gers', gersDataset.length, gsc.train + '/' + gsc.validation + '/' + gsc.test);
