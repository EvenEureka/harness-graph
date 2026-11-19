/* Draws the harness graph that a setting of the nine design variables compiles to.
   A port of compile_config() in harness_graph/grammar.py; site/check_builder.js compares it with the Python grammar
   on all 3,600 settings. */
var W = 110, H = 52, VW = 1100;
var DEF = {plan: false, first: false, tool: false, verify: 'none', gate: false, onfail: 'continue', retries: 2, compact: 'none', stall: 'none'};
var PRESETS = {
  plain: {},
  gate: {gate: true},
  mid: {plan: true, gate: true, onfail: 'reflect', compact: '16k'},
  full: {plan: true, first: true, tool: true, verify: 'fresh', gate: true, onfail: 'reset', retries: 3, compact: '16k', stall: 'reset'}
};

/* the name the repository gives this graph (Config.name in grammar.py) */
function graphName(c) {
  return ['g', 'p' + (c.plan ? 1 : 0), 'f' + (c.first ? 'r' : 'n'), 't' + (c.tool ? 1 : 0), 'v' + c.verify[0],
          c.gate ? 'k' + c.onfail.slice(0, 3) + c.retries : 'k0',
          'c' + (c.compact === 'none' ? 0 : parseInt(c.compact, 10)), 's' + c.stall.slice(0, 3)].join('-');
}

function draw(c) {
  var out = [], nodes = 0, edges = 0;
  function box(n) {
    if (!n.end) nodes++;
    var s = '<rect class="box' + (n.prog ? ' prog' : '') + '" x="' + n.x + '" y="' + n.y + '" width="' + W + '" height="' + H + '" rx="' + (n.end ? 26 : 4) + '"/>';
    if (n.end) s += '<text class="t" x="' + (n.x + W / 2) + '" y="' + (n.y + 31) + '" text-anchor="middle">' + n.label + '</text>';
    else s += '<text class="t" x="' + (n.x + W / 2) + '" y="' + (n.y + 22) + '" text-anchor="middle">' + n.label + '</text>' +
              '<text class="code" x="' + (n.x + W / 2) + '" y="' + (n.y + 40) + '" text-anchor="middle">' + n.type + '</text>';
    out.push(s);
  }
  function arrow(pts, label, lx, ly, anchor) {
    edges++;
    out.push('<polyline class="ar" points="' + pts.map(function (p) { return p[0] + ',' + p[1]; }).join(' ') + '" marker-end="url(#ab)"/>');
    if (label) out.push('<text class="e" x="' + lx + '" y="' + ly + '" text-anchor="' + (anchor || 'middle') + '">' + label + '</text>');
  }

  /* the main line: [plan] -> [reproduce] -> work -> [summarize ->] [self-review] -> [run tests] -> submit */
  var main = [];
  if (c.plan) main.push({label: 'Plan', type: 'plan', lab: 'next'});
  if (c.first) main.push({label: 'Reproduce', type: 'reproduce', lab: 'done'});
  var work = {label: 'Edit code', type: (c.first ? 'fix' : 'solve') + (c.tool ? ' + tool' : ''), lab: 'done'};
  main.push(work);
  if (c.verify === 'fresh') main.push({label: 'Summarize', type: 'compact', lab: 'next'});
  if (c.verify !== 'none') main.push({label: 'Self-review', type: 'verify', lab: 'done'});
  var check = null;
  if (c.gate) { check = {label: 'Run tests', type: 'check', prog: true, lab: 'pass'}; main.push(check); }
  main.push({label: 'Submit', end: true});

  /* the side loop taken when the agent has made no edit for 20 steps */
  var stall = [];
  if (c.stall === 'reflect' || c.stall === 'reset') stall.push({label: 'Find cause', type: 'reflect'});
  if (c.stall === 'reset') stall.push({label: 'Start over', type: 'reset', prog: true});
  if (c.stall === 'compact') stall.push({label: 'Summarize', type: 'compact'});

  var before = main.indexOf(work);
  var pad = Math.max(0, stall.length - before);
  var S = pad + main.length;
  var gap = Math.min(80, (VW - 40 - S * W) / (S - 1));
  var x0 = (VW - (S * W + (S - 1) * gap)) / 2;
  function sx(i) { return Math.round(x0 + i * (W + gap)); }
  main.forEach(function (n, i) { n.slot = pad + i; n.x = sx(n.slot); n.y = 140; });
  main.forEach(box);
  for (var i = 0; i < main.length - 1; i++) {
    var a = main[i], b = main[i + 1];
    arrow([[a.x + W, 166], [b.x - 2, 166]], a.lab, (a.x + W + b.x) / 2, 158);
  }

  var wx = work.x, ret = wx + 75, so = wx + 30, sr = wx + 8;
  var minY = 112, maxY = 215;

  /* the side loop taken when the conversation grows past the threshold */
  if (c.compact !== 'none') {
    box({label: 'Summarize', type: 'compact', x: wx, y: 30});
    arrow([[wx + 35, 140], [wx + 35, 84]], 'over ' + c.compact, wx + 29, 116, 'end');
    arrow([[wx + 75, 82], [wx + 75, 138]], 'next', wx + 81, 116, 'start');
    minY = 14;
  }

  /* the way back from a failed check */
  if (check) {
    var lab = 'fail (up to ' + c.retries + ')', cx = check.x + 55;
    if (c.onfail === 'continue') {
      arrow([[cx, 192], [cx, 356], [ret, 356], [ret, 194]], lab, (cx + ret) / 2, 348);
      maxY = 376;
    } else {
      var rf = {label: 'Find cause', type: 'reflect', x: check.x, y: 350};
      box(rf);
      arrow([[cx, 192], [cx, 348]], lab, cx + 7, 280, 'start');
      if (c.onfail === 'reflect') {
        arrow([[rf.x, 376], [ret, 376], [ret, 194]], 'next', (rf.x + ret) / 2, 368);
      } else {
        var rslot = check.slot - 1;
        var rs = {label: 'Start over', type: 'reset', prog: true, x: sx(rslot), y: 350};
        box(rs);
        arrow([[rf.x, 376], [rs.x + W + 2, 376]], 'next', (rf.x + rs.x + W) / 2, 368);
        if (rslot === work.slot) arrow([[ret, 350], [ret, 194]], 'next', ret + 7, 280, 'start');
        else arrow([[rs.x, 376], [ret, 376], [ret, 194]], 'next', (rs.x + ret) / 2, 368);
      }
      maxY = 420;
    }
  }

  if (c.stall === 'nudge') {
    arrow([[so, 192], [so, 236], [sr, 236], [sr, 194]], 'stalled: reminder', wx - 4, 240, 'end');
    maxY = Math.max(maxY, 258);
  } else if (stall.length) {
    stall.forEach(function (n, k) { n.x = sx(work.slot - 1 - k); n.y = 250; box(n); });
    var s1 = stall[0], last = stall[stall.length - 1];
    arrow([[so, 192], [so, 276], [s1.x + W + 2, 276]], 'stalled', so + 6, 240, 'start');
    if (stall.length === 2) arrow([[s1.x, 276], [last.x + W + 2, 276]], 'next', (s1.x + last.x + W) / 2, 268);
    arrow([[last.x + 55, 250], [last.x + 55, 221], [sr, 221], [sr, 194]], 'next', (last.x + 55 + sr) / 2, 215);
    maxY = Math.max(maxY, 320);
  }

  var text = ['Main line: ' + main.map(function (n) { return n.label.toLowerCase(); }).join(' → ') + '.'];
  if (c.tool) text.push('While editing, the agent may call the test checker itself.');
  if (check) text.push('If the tests fail: ' + {continue: 'go back to editing with the failure output', reflect: 'write down the cause, then go back to editing', reset: 'write down the cause, restore the code, clear the conversation and start again with that note'}[c.onfail] + ', at most ' + c.retries + (c.retries === 1 ? ' time.' : ' times.'));
  if (c.compact !== 'none') text.push('If the conversation passes ' + c.compact + ' tokens: replace it with a summary and continue.');
  if (c.stall !== 'none') text.push('If 20 steps pass without an edit: ' + {nudge: 'remind the agent and continue', reflect: 'write down the cause and continue', reset: 'write down the cause and start over', compact: 'summarize the conversation and continue'}[c.stall] + '.');

  var svg = '<svg viewBox="0 ' + minY + ' ' + VW + ' ' + (maxY - minY) + '" role="img" aria-label="' + text.join(' ') + '">' +
    '<defs><marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path class="ah" d="M0 0L10 5L0 10z"/></marker></defs>' +
    out.join('') + '</svg>';
  return {svg: svg, nodes: nodes, edges: edges, text: text, name: graphName(c)};
}

function init() {
  var ids = ['plan', 'first', 'tool', 'verify', 'gate', 'onfail', 'retries', 'compact', 'stall'];
  var el = {};
  ids.forEach(function (k) { el[k] = document.getElementById('s-' + k); });
  function read() {
    return {plan: el.plan.value === '1', first: el.first.value === '1', tool: el.tool.value === '1', verify: el.verify.value,
            gate: el.gate.value === '1', onfail: el.onfail.value, retries: +el.retries.value, compact: el.compact.value, stall: el.stall.value};
  }
  function render() {
    var c = read();
    el.onfail.disabled = el.retries.disabled = !c.gate;
    var r = draw(c);
    document.getElementById('gsvg').innerHTML = r.svg;
    document.getElementById('gtext').innerHTML = '<b>' + r.nodes + (r.nodes === 1 ? ' node, ' : ' nodes, ') + r.edges + (r.edges === 1 ? ' edge' : ' edges') +
      '</b> · name in the repository <code>' + r.name + '</code><br>' + r.text.join(' ');
  }
  function apply(p) {
    var c = Object.assign({}, DEF, p);
    el.plan.value = c.plan ? '1' : '0'; el.first.value = c.first ? '1' : '0'; el.tool.value = c.tool ? '1' : '0';
    el.verify.value = c.verify; el.gate.value = c.gate ? '1' : '0'; el.onfail.value = c.onfail;
    el.retries.value = String(c.retries); el.compact.value = c.compact; el.stall.value = c.stall;
    render();
  }
  ids.forEach(function (k) { el[k].addEventListener('change', render); });
  Array.prototype.forEach.call(document.querySelectorAll('[data-preset]'), function (b) {
    b.addEventListener('click', function () { apply(PRESETS[b.getAttribute('data-preset')]); });
  });
  /* a button anywhere on the page that carries a setting: draw it and bring the builder into view */
  Array.prototype.forEach.call(document.querySelectorAll('[data-config]'), function (b) {
    b.addEventListener('click', function () {
      apply(JSON.parse(b.getAttribute('data-config')));
      document.getElementById('builder').scrollIntoView({behavior: 'smooth'});
    });
  });
  apply(PRESETS.mid);
}
if (typeof document !== 'undefined') init();
if (typeof module !== 'undefined') module.exports = {draw: draw, graphName: graphName, PRESETS: PRESETS, DEF: DEF};
