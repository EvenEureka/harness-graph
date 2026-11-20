/* Compares docs/builder.js with the Python grammar: node site/check_builder.js
   - the nine variables give 3,600 distinct names
   - the number of nodes per graph has the distribution in results/summary/space.json
   - every hand-written graph and every graph of the search panel is one of those names */
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
const B = require(path.join(root, 'docs', 'builder.js'));
const space = JSON.parse(fs.readFileSync(path.join(root, 'results/summary/space.json')));
const panel = JSON.parse(fs.readFileSync(path.join(root, 'graphs_panel/panel.json')));
const names = new Set(), hist = {};
let raw = 0, maxEdges = 0;
for (const plan of [false, true]) for (const first of [false, true]) for (const tool of [false, true])
for (const verify of ['none', 'same', 'fresh']) for (const gate of [false, true])
for (const onfail of ['continue', 'reflect', 'reset']) for (const retries of [1, 2, 3])
for (const compact of ['none', '16k', '32k']) for (const stall of ['none', 'nudge', 'reflect', 'reset', 'compact']) {
  raw++;
  const c = {plan, first, tool, verify, gate, onfail, retries, compact, stall};
  const r = B.draw(c);
  if (names.has(r.name)) continue;                       // without the check, the two retry variables change nothing
  names.add(r.name);
  hist[r.nodes] = (hist[r.nodes] || 0) + 1;
  maxEdges = Math.max(maxEdges, r.edges);
  if (!/^<svg[^]*<\/svg>$/.test(r.svg) || /NaN|undefined/.test(r.svg)) throw new Error('bad drawing for ' + r.name);
}
const fail = (m) => { console.error('FAIL ' + m); process.exitCode = 1; };
if (raw !== space.raw_combinations) fail(`raw settings ${raw} != ${space.raw_combinations}`);
if (names.size !== space.distinct_graph_signatures) fail(`distinct names ${names.size} != ${space.distinct_graph_signatures}`);
if (JSON.stringify(hist) !== JSON.stringify(space.nodes_per_graph)) fail('node histogram differs: ' + JSON.stringify(hist));
for (const [h, n] of Object.entries(space.hand_written_points)) if (!names.has(n)) fail(`${h} = ${n} is not drawn`);
for (const n of Object.keys(panel.configs)) if (!names.has(n)) fail(`panel graph ${n} is not drawn`);
/* the "draw" buttons of the page give exactly the graphs the live searches returned */
const cand = JSON.parse(fs.readFileSync(path.join(root, 'graphs_select/candidates.json')));
const selPath = path.join(root, 'results/summary/selection.json');
const picked = fs.existsSync(selPath) && /Stage 4 · Selection/.test(fs.readFileSync(path.join(root, 'docs/index.html'), 'utf8'))
  ? [JSON.parse(fs.readFileSync(selPath)).searched_graph] : [];
const want = Object.values(cand.live).map((r) => r.recommendation).concat(picked).sort();
const page = fs.readFileSync(path.join(root, 'docs/index.html'), 'utf8');
const drawn = [...page.matchAll(/data-config='([^']*)'/g)].map((m) => B.graphName(Object.assign({}, B.DEF, JSON.parse(m[1])))).sort();
if (JSON.stringify(drawn) !== JSON.stringify(want)) fail(`draw buttons give ${drawn}, searches returned ${want}`);
else console.log(`${drawn.length} draw buttons match the graphs returned by the live searches` + (picked.length ? ' and the selected graph' : ''));
console.log(`${raw} settings, ${names.size} distinct graphs, nodes per graph ${JSON.stringify(hist)}, at most ${maxEdges} edges; ` +
            `${Object.keys(space.hand_written_points).length} hand-written and ${Object.keys(panel.configs).length} panel graphs found`);
