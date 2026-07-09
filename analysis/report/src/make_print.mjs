// Generate a fully-static, print-optimized professional PDF-source HTML from the canonical dataset.
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const DATA = require('./data.js');

/* ---- print palette (light, print-safe) ---- */
const C = { cann:'#C8102E', cannInk:'#A50D26', cuda:'#5F9400', cudaInk:'#4C7A00',
  ink:'#16191D', ink2:'#3A4048', muted:'#5B636E', faint:'#8A929C',
  hair:'#DCDFDA', hair2:'#EBEDE9', graphite:'#2E3440', signal:'#9A6410', surf:'#FFFFFF', surf2:'#F7F8F5', surf3:'#EEF0EC' };
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

/* ========== SVG chart bakers ========== */
function radarSVG(){
  const ax=DATA.axes, n=ax.length, size=340, cx=size/2, cy=size/2+4, R=size/2-58;
  const pt=(i,r)=>{const a=-Math.PI/2+i/n*2*Math.PI;return[+(cx+Math.cos(a)*r).toFixed(1),+(cy+Math.sin(a)*r).toFixed(1)];};
  let s=`<svg viewBox="0 0 ${size} ${size+8}" width="360" xmlns="http://www.w3.org/2000/svg">`;
  [0.25,0.5,0.75,1].forEach(f=>{ s+=`<polygon points="${ax.map((_,i)=>pt(i,R*f).join(',')).join(' ')}" fill="none" stroke="${C.hair}" stroke-width="1"/>`; });
  ax.forEach((a,i)=>{ const[x,y]=pt(i,R); s+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${C.hair}" stroke-width="1"/>`;
    const[lx,ly]=pt(i,R+24); const anc=Math.abs(lx-cx)<14?'middle':(lx>cx?'start':'end');
    s+=`<text x="${lx}" y="${ly+3}" text-anchor="${anc}" font-family="sans-serif" font-size="11" fill="${C.muted}">${esc(a.label.split(' ')[0])}</text>`; });
  const poly=(key,col)=>{ const pts=ax.map((a,i)=>pt(i,R*a[key]/100));
    s+=`<polygon points="${pts.map(p=>p.join(',')).join(' ')}" fill="${col}" fill-opacity="0.13" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>`;
    pts.forEach(p=>{ s+=`<circle cx="${p[0]}" cy="${p[1]}" r="3" fill="${col}"/>`; }); };
  poly('cann',C.cann); poly('cuda',C.cuda);
  s+='</svg>'; return s;
}
function hbarSVG(rows,col,inkCol,{w=430,labW=150}={}){
  const max=Math.max(...rows.map(r=>r.n)), rh=22;
  let s=`<svg viewBox="0 0 ${w} ${rows.length*rh+6}" width="100%" xmlns="http://www.w3.org/2000/svg">`;
  rows.forEach((r,i)=>{ const y=i*rh+4, bw=Math.max(3,r.n/max*(w-labW-40));
    s+=`<text x="0" y="${y+13}" font-family="sans-serif" font-size="10.5" fill="${C.muted}">${esc(r.cat)}</text>`;
    s+=`<rect x="${labW}" y="${y+2}" width="${bw.toFixed(1)}" height="15" rx="3.5" fill="${col}"/>`;
    s+=`<text x="${(labW+bw+6).toFixed(1)}" y="${y+13.5}" font-family="sans-serif" font-size="10.5" font-weight="700" fill="${inkCol}">${r.n}</text>`; });
  s+='</svg>'; return s;
}
function stackedSVG(){
  const rows=[{k:'cann-samples',data:DATA.fileTypes.cann,col:C.cann},{k:'cuda-samples',data:DATA.fileTypes.cuda,col:C.cuda}];
  const W=430, barH=26, gap=40; let s=`<svg viewBox="0 0 ${W} ${rows.length*(barH+gap)}" width="100%" xmlns="http://www.w3.org/2000/svg">`;
  rows.forEach((r,ri)=>{ const total=r.data.reduce((a,d)=>a+d.n,0), y=ri*(barH+gap)+16;
    s+=`<text x="0" y="${y-4}" font-family="sans-serif" font-size="10.5" font-weight="700" fill="${r.col}">${r.k} · ${total} files</text>`;
    let x=0; const sc=W/total;
    r.data.forEach((d,di)=>{ const bw=d.n*sc, shade=(0.35+0.6*(1-di/r.data.length)).toFixed(2);
      s+=`<rect x="${x.toFixed(1)}" y="${y}" width="${Math.max(0,bw-1.2).toFixed(1)}" height="${barH}" rx="2" fill="${r.col}" fill-opacity="${shade}"/>`;
      if(bw>42)s+=`<text x="${(x+bw/2).toFixed(1)}" y="${y+barH/2+3.5}" text-anchor="middle" font-family="monospace" font-size="8.5" font-weight="700" fill="#fff">${esc(d.ext.split(' ')[0])}</text>`;
      x+=bw; }); });
  s+='</svg>'; return s;
}
function gaugeSVG(){
  let cann=0,cuda=0,tie=0; DATA.openness.forEach(r=>{r.edge==='cann'?cann++:r.edge==='cuda'?cuda++:tie++;});
  const tot=DATA.openness.length, W=430, bh=28;
  let s=`<svg viewBox="0 0 ${W} 46" width="100%" xmlns="http://www.w3.org/2000/svg">`; let x=0;
  [['cann',cann,C.cann],['tie',tie,C.surf3],['cuda',cuda,C.cuda]].forEach(([k,v,col])=>{ const bw=v/tot*W; if(bw>0){
    s+=`<rect x="${(x+(x>0?1.5:0)).toFixed(1)}" y="4" width="${Math.max(0,bw-1.5).toFixed(1)}" height="${bh}" rx="4" fill="${col}"/>`;
    if(bw>28)s+=`<text x="${(x+bw/2).toFixed(1)}" y="${4+bh/2+4}" text-anchor="middle" font-family="monospace" font-size="11" font-weight="700" fill="${k==='tie'?C.muted:'#fff'}">${v}</text>`;
    x+=bw; }});
  s+=`<text x="0" y="44" font-family="monospace" font-size="9.5" fill="${C.muted}">开放度占优：cuda ${cuda} · 持平 ${tie} · cann ${cann}（共 ${tot} 项）</text></svg>`;
  return s;
}
function metricBarsSVG(){
  const W=430, rh=48, pad=150;
  let s=`<svg viewBox="0 0 ${W} ${DATA.headline.length*rh+4}" width="100%" xmlns="http://www.w3.org/2000/svg">`;
  DATA.headline.forEach((h,i)=>{ const y=i*rh+6; const max=Math.max(h.cann,h.cuda);
    s+=`<text x="0" y="${y+10}" font-family="sans-serif" font-size="10" fill="${C.ink2}">${esc(h.label)}</text>`;
    [['cann',h.cann,C.cann,C.cannInk],['cuda',h.cuda,C.cuda,C.cudaInk]].forEach(([k,v,col,ink],j)=>{
      const by=y+16+j*13, bw=Math.max(2,v/max*(W-pad-46));
      s+=`<rect x="${pad}" y="${by}" width="${bw.toFixed(1)}" height="10" rx="2.5" fill="${col}"/>`;
      s+=`<text x="${(pad+bw+5).toFixed(1)}" y="${by+9}" font-family="monospace" font-size="9.5" font-weight="700" fill="${ink}">${v.toLocaleString()}</text>`; }); });
  s+='</svg>'; return s;
}
function covMatrixHTML(){
  const lvl=['—','基础','覆盖','深耕'];
  const cell=(v,col)=>{ const op=v===0?1:(0.18+v*0.26).toFixed(2); const bg=v===0?C.surf3:col;
    const fg=v>=2?'#fff':(v===0?C.faint:C.ink);
    return `<td class="cov-cell"><span class="heat" style="background:${bg};opacity:${op};color:${fg}">${lvl[v]}</span></td>`; };
  let rows=DATA.coverage.map(d=>`<tr><td class="cov-dom"><span class="tagdot">${esc(d.tag)}</span>${esc(d.domain)}</td>${cell(d.cann,C.cann)}${cell(d.cuda,C.cuda)}</tr>`).join('');
  return `<table class="covtbl"><thead><tr><th>领域</th><th class="c">cann</th><th class="c">cuda</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function openTableHTML(){
  const eb=e=>e==='cann'?'<span class="eb eb-cann">cann</span>':e==='cuda'?'<span class="eb eb-cuda">cuda</span>':'<span class="eb eb-tie">持平</span>';
  const rows=DATA.openness.map(r=>`<tr><td class="crit">${esc(r.crit)}</td><td class="${r.edge==='cann'?'ec':''}">${esc(r.cann)}</td><td class="${r.edge==='cuda'?'eu':''}">${esc(r.cuda)}</td><td class="c">${eb(r.edge)}</td></tr>`).join('');
  return `<table class="dtl"><thead><tr><th>评估维度</th><th>cann-samples</th><th>cuda-samples</th><th class="c">占优</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function catTableHTML(rows,total,name){
  const r=rows.map(x=>`<tr><td>${esc(x.cat)}</td><td class="num">${x.n}</td></tr>`).join('');
  return `<table class="dtl mini"><thead><tr><th>${name} 类目</th><th class="num">样例</th></tr></thead><tbody>${r}<tr class="tot"><td>合计</td><td class="num">${total}</td></tr></tbody></table>`;
}
function listHTML(arr,cls){ return `<ul class="feat ${cls}">${arr.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`; }

/* verification log rows (from the adversarial workflow) */
const VERIFY=[
  ['样例与类目计数','CONFIRMED','cuda cpp=205（10 类目）、python=33（4 类目）、合计 238；cann 顶层类目 4、顶层样例组 19（0_Intro 5 / 1_Features 4 / 2_Performance 9 / 3_Utilities 1）；matmul_recipes 7、grouped_matmul_recipes 4、matmul_tutorials 8 步。均逐目录枚举核对，精确匹配。'],
  ['内核源与扩展名','CONFIRMED','cann：.asc=116、.h=235、无 .cu/.cuh；cuda：.cu=189、.cuh=50、.h=219、无 .asc。.asc 仅见于 cann（Ascend C 内核），.cu/.cuh 仅见于 cuda。四项计数精确。'],
  ['许可证性质','CONFIRMED','cann LICENSE 第1行 = CANN OSL v2.0；§2.1 授权限「systems with Huawei AI Processors」、明写 revocable；§3.1 禁止服务其他处理器；§5.1(b)(ii) 专利/IP 诉讼即终止、§5.2 授权 void ab initio；非 OSI。cuda LICENSE = 逐字 BSD-3-Clause（NVIDIA 2022）+ CUDA EULA 引用；OSI 认证宽松。'],
  ['托管与 CI','PARTIALLY-CORRECT','均确认：cuda GitHub 原生、CHANGELOG 按 CUDA 版本、pre-commit.ci、run_tests.py；cann GitCode 主场、GitHub 镜像（.ci/mirror_update_time.txt）、run_ci_functional.py 功能门禁。修正：#435 为 v13.3 release PR（下界），线上实际更高；cuda 本地树无 .github/ Actions，CI 即 pre-commit.ci 外部服务；cann 的 CI 清单 tests/ci_functional_test.yaml 已被移除。'],
  ['领域独占性','PARTIALLY-CORRECT','cuda 独占（cann 全无）：图形互操作 D3D/Vulkan/GL、经典 HPC/金融、数值库 cuBLAS/cuFFT/cuSOLVER/NPP/CUB；确认无误。cann 独占（cuda 全无）：MoE 派发/合并、融合注意力 FIA、KV-RMSNorm-RoPE、MX 量化矩阵乘；确认。唯一重叠修正：cuda 在 cpp/9_CUDA_Tile/tileRope 提供孤立 RoPE 前向（另有 tileLayerNorm/tileBmm），故「cuda 完全不涉及 RoPE」为过度陈述——但 cann 的融合调优深度仍为 cuda 所无。'],
];
function verifyHTML(){
  const badge=v=>v==='CONFIRMED'?'<span class="vb vb-ok">已证实</span>':'<span class="vb vb-part">部分修正</span>';
  return VERIFY.map(([t,v,d])=>`<tr><td class="crit">${esc(t)}</td><td class="c">${badge(v)}</td><td>${esc(d)}</td></tr>`).join('');
}

/* ========== PAGE ASSEMBLY ========== */
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>cann-samples vs cuda-samples · 开源生态深度对比报告</title>
<style>
@page{ size:A4; margin:16mm 0 0 0; }
:root{ --cann:${C.cann}; --cannInk:${C.cannInk}; --cuda:${C.cuda}; --cudaInk:${C.cudaInk};
  --ink:${C.ink}; --ink2:${C.ink2}; --muted:${C.muted}; --faint:${C.faint}; --hair:${C.hair}; --hair2:${C.hair2};
  --graphite:${C.graphite}; --signal:${C.signal}; --surf:${C.surf}; --surf2:${C.surf2}; --surf3:${C.surf3}; }
*{ box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
html,body{ margin:0; padding:0; }
body{ font-family:"Liberation Sans","DejaVu Sans","WenQuanYi Zen Hei",sans-serif; color:var(--ink);
  font-size:9.6pt; line-height:1.62; }
.mono{ font-family:"DejaVu Sans Mono","Liberation Mono",monospace; }
.page{ padding:0 16mm 18mm; }
.page.break{ page-break-before:always; }
h1,h2,h3,h4{ margin:0; line-height:1.2; }
p{ margin:0 0 7pt; }
a{ color:var(--cudaInk); text-decoration:none; }
.small{ font-size:8.4pt; color:var(--muted); }
.faint{ color:var(--faint); }
code{ font-family:"DejaVu Sans Mono",monospace; font-size:8.4pt; background:var(--surf3); padding:0.5pt 3pt; border-radius:3px; }

/* cover */
.cover{ height:calc(297mm - 16mm); display:flex; flex-direction:column; padding:0 18mm; page-break-after:always; position:relative; }
.cover-top{ padding-top:30mm; }
.cover .ey{ font-family:"DejaVu Sans Mono",monospace; letter-spacing:.22em; text-transform:uppercase; font-size:8.5pt; color:var(--faint); }
.cover h1{ font-size:33pt; font-weight:800; letter-spacing:-.5pt; margin-top:9mm; line-height:1.05; }
.cover h1 .a{ color:var(--cannInk); } .cover h1 .b{ color:var(--cudaInk); } .cover h1 .vs{ color:var(--faint); font-weight:400; font-size:.5em; }
.cover .subt{ font-size:12pt; color:var(--ink2); margin-top:7mm; max-width:150mm; line-height:1.55; }
.cover-cards{ display:flex; gap:6mm; margin-top:14mm; }
.cc{ flex:1; border:0.6pt solid var(--hair); border-radius:9px; padding:6mm; }
.cc.a{ border-top:2.4pt solid var(--cann); } .cc.b{ border-top:2.4pt solid var(--cuda); }
.cc h3{ font-size:13pt; } .cc .rl{ font-family:"DejaVu Sans Mono",monospace; font-size:7.6pt; color:var(--faint); margin:1mm 0 3mm; }
.cc dl{ display:grid; grid-template-columns:auto 1fr; gap:1.6mm 4mm; font-size:8.6pt; margin:0; }
.cc dt{ color:var(--faint); font-family:"DejaVu Sans Mono",monospace; font-size:7.6pt; }
.cc dd{ margin:0; color:var(--ink2); }
.pill{ display:inline-block; font-family:"DejaVu Sans Mono",monospace; font-size:7pt; padding:1pt 5pt; border-radius:8px; }
.pill.o{ background:rgba(95,148,0,.14); color:var(--cudaInk); } .pill.r{ background:rgba(200,16,46,.11); color:var(--cannInk); }
.cover-foot{ margin-top:auto; padding-bottom:12mm; border-top:0.6pt solid var(--hair); padding-top:5mm; display:flex; justify-content:space-between; font-size:8pt; color:var(--muted); }
.cover-badges{ display:flex; gap:3mm; margin-top:6mm; flex-wrap:wrap; }
.cbadge{ font-family:"DejaVu Sans Mono",monospace; font-size:8pt; border:0.6pt solid var(--hair); border-radius:20px; padding:2pt 8pt; color:var(--ink2); }

/* section chrome */
.sec-ey{ font-family:"DejaVu Sans Mono",monospace; letter-spacing:.16em; text-transform:uppercase; font-size:8pt; color:var(--faint); }
.sec-ey .n{ color:var(--cannInk); font-weight:700; }
h2.sec{ font-size:17pt; font-weight:750; margin:2mm 0 3mm; letter-spacing:-.3pt; }
.lead{ color:var(--muted); font-size:9.8pt; margin-bottom:5mm; max-width:160mm; }
.rule{ height:2pt; background:linear-gradient(90deg,var(--cann),var(--cuda)); border-radius:2px; margin:0 0 6mm; width:34mm; }

/* toc */
.toc{ margin-top:8mm; } .toc h2{ font-size:15pt; margin-bottom:5mm; }
.toc ol{ list-style:none; padding:0; margin:0; counter-reset:t; }
.toc li{ display:flex; align-items:baseline; gap:3mm; padding:2.4mm 0; border-bottom:0.5pt dotted var(--hair); font-size:10pt; }
.toc li .tn{ font-family:"DejaVu Sans Mono",monospace; color:var(--cannInk); font-weight:700; width:8mm; }
.toc li .tt{ font-weight:600; } .toc li .td{ color:var(--muted); font-size:8.4pt; margin-left:auto; }

/* generic blocks */
.grid2{ display:grid; grid-template-columns:1fr 1fr; gap:6mm; }
.card{ border:0.6pt solid var(--hair); border-radius:8px; padding:5mm; break-inside:avoid; }
.card h3{ font-size:11pt; margin-bottom:1mm; } .card .cs{ color:var(--muted); font-size:8.2pt; margin-bottom:3mm; }
.chart{ break-inside:avoid; } .chart svg{ display:block; }
.legend{ display:flex; gap:6mm; font-family:"DejaVu Sans Mono",monospace; font-size:8pt; color:var(--muted); margin-top:2mm; }
.legend i{ width:9px; height:9px; border-radius:2px; display:inline-block; margin-right:3px; vertical-align:middle; }

/* tables */
table{ border-collapse:collapse; width:100%; font-size:8.6pt; }
.dtl th,.dtl td{ text-align:left; padding:2.4mm 3mm; border-bottom:0.5pt solid var(--hair2); vertical-align:top; }
.dtl thead th{ background:var(--surf2); font-family:"DejaVu Sans Mono",monospace; font-size:7.4pt; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); }
.dtl .crit{ font-weight:700; } .dtl .num{ text-align:right; font-family:"DejaVu Sans Mono",monospace; }
.dtl .c{ text-align:center; } .dtl tr.tot td{ font-weight:700; border-top:1pt solid var(--hair); background:var(--surf2); }
.dtl td.ec{ box-shadow:inset 2.4pt 0 0 var(--cann); } .dtl td.eu{ box-shadow:inset 2.4pt 0 0 var(--cuda); }
.dtl.mini th,.dtl.mini td{ padding:1.7mm 3mm; }
.eb{ font-family:"DejaVu Sans Mono",monospace; font-size:7pt; padding:1pt 5pt; border-radius:8px; }
.eb-cann{ background:rgba(200,16,46,.11); color:var(--cannInk); } .eb-cuda{ background:rgba(95,148,0,.14); color:var(--cudaInk); } .eb-tie{ background:var(--surf3); color:var(--muted); }
.covtbl{ border-collapse:collapse; width:100%; font-size:8.4pt; }
.covtbl th{ text-align:left; padding:2mm 3mm; border-bottom:0.6pt solid var(--hair); font-family:"DejaVu Sans Mono",monospace; font-size:7.4pt; text-transform:uppercase; color:var(--faint); }
.covtbl th.c{ text-align:center; width:22mm; }
.covtbl td{ padding:1.5mm 3mm; border-bottom:0.5pt solid var(--hair2); }
.covtbl .cov-dom{ font-size:8.6pt; } .covtbl .cov-cell{ text-align:center; }
.covtbl .heat{ display:inline-block; min-width:16mm; padding:1.4mm 0; border-radius:4px; font-family:"DejaVu Sans Mono",monospace; font-size:7.6pt; font-weight:700; text-align:center; }
.tagdot{ font-family:"DejaVu Sans Mono",monospace; font-size:6.8pt; color:var(--faint); border:0.5pt solid var(--hair); border-radius:3px; padding:0 3pt; margin-right:4pt; }

/* trees */
pre.tree{ font-family:"DejaVu Sans Mono",monospace; font-size:7.6pt; line-height:1.55; color:var(--ink2); margin:0; white-space:pre-wrap; }
pre.tree b{ color:var(--ink); }

/* callouts / feature lists */
.callout{ border-left:2.4pt solid var(--signal); background:rgba(154,100,16,.07); border-radius:0 6px 6px 0; padding:3.5mm 5mm; font-size:9pt; color:var(--ink2); break-inside:avoid; margin:4mm 0; }
.callout b{ color:var(--ink); }
ul.feat{ list-style:none; padding:0; margin:2mm 0 0; display:grid; gap:1.8mm; }
ul.feat li{ position:relative; padding-left:5mm; font-size:8.8pt; color:var(--ink2); line-height:1.5; }
ul.feat li::before{ content:""; position:absolute; left:0; top:3.2pt; width:6px; height:6px; border-radius:2px; }
ul.feat.a li::before{ background:var(--cann); } ul.feat.b li::before{ background:var(--cuda); }
.kv{ display:grid; grid-template-columns:auto 1fr; gap:1.6mm 5mm; font-size:8.8pt; }
.kv dt{ font-family:"DejaVu Sans Mono",monospace; font-size:7.8pt; color:var(--faint); }
.kv dd{ margin:0; color:var(--ink2); }
.note{ font-size:8pt; color:var(--faint); font-style:italic; margin-top:3mm; }

/* rec cards */
.rec{ display:grid; grid-template-columns:1fr 1fr; gap:6mm; }
.rc{ border:0.6pt solid var(--hair); border-radius:8px; padding:5mm; break-inside:avoid; }
.rc.a{ border-top:2.4pt solid var(--cann); } .rc.b{ border-top:2.4pt solid var(--cuda); }
.rc .when{ font-family:"DejaVu Sans Mono",monospace; font-size:7.4pt; text-transform:uppercase; letter-spacing:.06em; color:var(--faint); }
.rc h3{ font-size:12pt; margin:1mm 0 2mm; }
.rc ul{ margin:0; padding-left:4.5mm; font-size:8.8pt; color:var(--ink2); display:grid; gap:1.4mm; }
.vb{ font-family:"DejaVu Sans Mono",monospace; font-size:7pt; padding:1pt 5pt; border-radius:8px; }
.vb-ok{ background:rgba(95,148,0,.14); color:var(--cudaInk); } .vb-part{ background:rgba(154,100,16,.13); color:var(--signal); }
h3.blk{ font-size:11.5pt; margin:6mm 0 2mm; padding-bottom:1.5mm; border-bottom:0.6pt solid var(--hair); }
h3.blk .dot{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5pt; vertical-align:middle; }
.dot.cann{ background:var(--cann); } .dot.cuda{ background:var(--cuda); }
</style></head><body>

<!-- ============ COVER ============ -->
<section class="cover">
  <div class="cover-top">
    <div class="ey">开源开放生态 · 深度对比报告</div>
    <h1><span class="a">cann-samples</span><br><span class="vs">versus</span> <span class="b">cuda-samples</span></h1>
    <p class="subt">华为 <b>cann-samples</b>（分支 <span class="mono">sync/upstream/master</span>）与英伟达 <b>NVIDIA/cuda-samples</b>（<span class="mono">master · Release v13.3</span>）的目录级、定位级、覆盖级与开源生态级全方位对比分析。</p>
    <div class="cover-badges">
      <span class="cbadge">目录对比</span><span class="cbadge">定位对比</span><span class="cbadge">覆盖范围</span>
      <span class="cbadge">开源开放定位</span><span class="cbadge">开发者样例</span><span class="cbadge">生态治理</span>
    </div>
  </div>
  <div class="cover-cards">
    <div class="cc a"><h3>cann-samples</h3><div class="rl">gitcode.com/cann/cann-samples</div>
      <dl><dt>厂商</dt><dd>华为 Huawei</dd><dt>平台</dt><dd>Ascend NPU（昇腾 910B/C · 950）</dd>
      <dt>主托管</dt><dd>GitCode（GitHub 只读镜像）</dd><dt>内核</dt><dd>Ascend C <span class="mono">.asc</span></dd>
      <dt>许可</dt><dd>CANN OSL v2.0 <span class="pill r">受限 · 非 OSI</span></dd></dl></div>
    <div class="cc b"><h3>cuda-samples</h3><div class="rl">github.com/NVIDIA/cuda-samples</div>
      <dl><dt>厂商</dt><dd>英伟达 NVIDIA</dd><dt>平台</dt><dd>CUDA GPU（含 Tegra / DriveOS）</dd>
      <dt>主托管</dt><dd>GitHub（PR 原生协作）</dd><dt>内核</dt><dd>CUDA C++ <span class="mono">.cu / .cuh</span></dd>
      <dt>许可</dt><dd>BSD-3-Clause <span class="pill o">宽松 · OSI</span></dd></dl></div>
  </div>
  <div class="cover-foot"><span>数据快照 2026-07-09 · 计数来自两仓工作树直接枚举 · 关键论断经对抗式核验</span><span>独立技术分析 · 非厂商官方文档</span></div>
</section>

<!-- ============ TOC + EXEC SUMMARY ============ -->
<section class="page">
  <div class="toc">
    <h2>目录</h2>
    <ol>
      <li><span class="tn">00</span><span class="tt">执行摘要</span><span class="td">一页看懂两仓分野</span></li>
      <li><span class="tn">01</span><span class="tt">总览与加权能力画像</span><span class="td">七轴雷达 · 头部指标</span></li>
      <li><span class="tn">02</span><span class="tt">目录结构与定位映射</span><span class="td">平铺 vs 深嵌套 · 类目对应</span></li>
      <li><span class="tn">03</span><span class="tt">产品定位与开源开放定位</span><span class="td">许可证逐条对照 · 记分卡</span></li>
      <li><span class="tn">04</span><span class="tt">覆盖范围：广度 vs 深度</span><span class="td">领域热力矩阵 · 类目分布</span></li>
      <li><span class="tn">05</span><span class="tt">给开发者提供的样例</span><span class="td">教学范式 · 构建门槛 · 独占能力</span></li>
      <li><span class="tn">06</span><span class="tt">开源开放生态与治理</span><span class="td">托管 · 贡献 · CI · 成熟度</span></li>
      <li><span class="tn">07</span><span class="tt">结论与选型建议</span><span class="td">场景化推荐矩阵</span></li>
      <li><span class="tn">08</span><span class="tt">数据附录与核验日志</span><span class="td">全量计数 · 对抗式验证</span></li>
    </ol>
  </div>

  <div style="margin-top:12mm">
    <div class="sec-ey"><span class="n">00</span> &nbsp;执行摘要</div>
    <h2 class="sec">一页看懂两仓分野</h2><div class="rule"></div>
    <p>两个仓库名字都叫「samples」，本质却是两种不同的产物。<b>cuda-samples</b> 是一座「CUDA 工具链全景陈列馆」：以约 <b>238</b> 个自包含、结构一致的可运行样例，横扫从入门原语、并行算法、数值库（cuBLAS/cuFFT/cuSOLVER/NPP/CUB）、图形互操作（D3D/Vulkan/OpenGL）、经典 HPC/金融，到跨平台（Windows/Tegra/DriveOS）与 CUDA-Python 的极宽领域，并以 <b>BSD-3-Clause</b> 这一 OSI 认证的宽松许可自由分发。它的价值是<b>广度</b>与<b>低门槛</b>。</p>
    <p><b>cann-samples</b> 则是一部「昇腾算子调优知识库」：顶层仅 <b>4</b> 个类目、<b>19</b> 个样例组，却在 <span class="mono">2_Performance</span> 下用 9 个递进式 <span class="mono">*_story</span> 把矩阵乘、分组矩阵乘、MoE 派发/合并、融合注意力、KV-RMSNorm-RoPE、低比特（MX FP4/FP8、HiF8、INT8）量化等少数高价值 LLM 推理算子，从 <span class="mono">0_naive</span> 一路打穿到 <span class="mono">7_fullload</span>，配长文中文教程与 profiling 图。它的价值是<b>深度</b>与<b>调优方法论</b>——代价是受限于华为 AI 处理器场景的 <b>CANN OSL v2.0</b> 许可（非 OSI）与真机硬件门槛。</p>
    <div class="callout"><b>核心判断：</b>二者非替代关系。若你要<b>系统学习 GPU 生态广度、使用官方库、关心开源合规与跨平台</b>，选 cuda-samples；若你要在昇腾上<b>深挖单个 LLM 算子的极致性能与调优范式</b>，选 cann-samples。在「开放度」这一单项上，cuda-samples 因宽松许可、全球托管与用途自由而明显领先；在「单点深度与中文调优文档」上，cann-samples 反超。</p>
  </div>
</section>

<!-- ============ 01 OVERVIEW ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">01</span> &nbsp;总览 · 加权能力画像</div>
  <h2 class="sec">七轴能力画像与头部指标</h2><div class="rule"></div>
  <p class="lead">下列七轴（0–100 分）是本报告的<b>显式分析模型</b>，由目录事实、覆盖广度/深度、许可条款与治理信号推导得出，非任一厂商官方指标。交互版报告允许读者对这七轴自定义权重、实时重算契合度。</p>
  <div class="grid2">
    <div class="card"><h3>能力画像雷达</h3><div class="cs">面积越大代表该维度越强 · 原始分 0–100</div>
      <div class="chart">${radarSVG()}</div>
      <div class="legend"><span><i style="background:${C.cann}"></i>cann-samples</span><span><i style="background:${C.cuda}"></i>cuda-samples</span></div>
    </div>
    <div class="card"><h3>头部量化指标对比</h3><div class="cs">每项上条为 cann、下条为 cuda</div>
      <div class="chart">${metricBarsSVG()}</div>
    </div>
  </div>
  <div class="callout"><b>如何读这张画像：</b>cuda-samples 在<b>覆盖广度、开放度、跨平台、成熟度</b>四轴显著领先；cann-samples 在<b>单点深度、教学/文档、LLM 算子聚焦</b>三轴反超。两条多边形几乎互补——这正是「广度型工具链陈列」与「深度型算子知识库」的形状差异。</div>
  <table class="dtl" style="margin-top:5mm"><thead><tr><th>能力轴</th><th class="num">cann</th><th class="num">cuda</th><th>说明</th></tr></thead><tbody>
  ${DATA.axes.map(a=>`<tr><td class="crit">${esc(a.label)}</td><td class="num" style="color:${C.cannInk};font-weight:700">${a.cann}</td><td class="num" style="color:${C.cudaInk};font-weight:700">${a.cuda}</td><td class="small">${esc({breadth:'领域/类目覆盖面',depth:'单个算子/主题的打穿深度',openness:'许可宽松度与用途自由',portable:'跨平台/跨硬件可移植性',pedagogy:'教程完整度与文档密度',maturity:'项目年龄/社区/连续性',llmops:'大模型推理算子聚焦度'}[a.key])}</td></tr>`).join('')}
  </tbody></table>
</section>

<!-- ============ 02 DIRECTORY ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">02</span> &nbsp;目录结构与定位映射</div>
  <h2 class="sec">相同的编号骨架，不同的血肉</h2><div class="rule"></div>
  <p class="lead">两仓都采用「数字前缀分层」的样例骨架，但 cuda 是<b>扁平陈列</b>（category/sample 两级到底），cann 是<b>深层叙事式</b>（category/story/{tutorials|recipes}/step/include，最深达 6–7 层）。cuda 靠<b>样例数量</b>铺开广度，cann 靠 story 内部的 <b>step/example</b> 展开深度。</p>
  <div class="grid2">
    <div class="card"><h3><span class="dot cann"></span>cann-samples 目录树</h3><div class="cs">4 顶层类目 · 深藏 story / recipes / tutorials</div>
<pre class="tree">Samples/
├─ <b>0_Introduction</b>/        入门 · 5 组
│   vector_add · matmul · npu_execution …
├─ <b>1_Features</b>/            特性 · 15 叶
│   ├─ hardware_features/   hif8·simt·vector_function
│   ├─ instruction_optimization/ mte2·unit_flag…
│   ├─ memory_optimization/ full_load·bank_conflict
│   └─ system_optimization/ streamk·tail_rebalance
├─ <b>2_Performance</b>/         性能调优 · 9 story
│   ├─ matmul_story/  recipes(7)+tutorials(0→7)
│   ├─ grouped_matmul_story/ recipes(4)
│   ├─ moe_dispatch_and_combine_story/
│   ├─ moe_init_routing_story/
│   ├─ full_quant_fused_infer_attention_score_story/
│   ├─ kv_rms_norm_rope_cache_story/ membase·regbase
│   ├─ rms_norm_quant_story/ · simd_vf_story/
│   └─ scalar_story/
└─ <b>3_Utilities</b>/           工具 · 1
    simulation-based-vf-profiling</pre></div>
    <div class="card"><h3><span class="dot cuda"></span>cuda-samples 目录树</h3><div class="cs">cpp/ 10 类目（205）+ python/ 4 类目（33）</div>
<pre class="tree">cpp/
├─ <b>0_Introduction</b>/          46  vectorAdd·matrixMul
├─ <b>1_Utilities</b>/             3   deviceQuery
├─ <b>2_Concepts_and_Techniques</b>/ 32  reduction·scan
├─ <b>3_CUDA_Features</b>/         24  graphs·tensorCore
├─ <b>4_CUDA_Libraries</b>/        39  cuBLAS·cuFFT·NPP
├─ <b>5_Domain_Specific</b>/       36  BlackScholes·nbody
├─ <b>6_Performance</b>/           5   transpose
├─ <b>7_libNVVM</b>/               9   NVVM IR / ptxgen
├─ <b>8_Platform_Specific</b>/     1   Tegra
└─ <b>9_CUDA_Tile</b>/             10  tileMatmul·tileRope
python/
├─ <b>1_GettingStarted</b>/        8   vectorAdd·deviceQuery
├─ <b>2_CoreConcepts</b>/          20  cudaGraphs·prefixSum
├─ <b>3_FrameworkInterop</b>/      2   PyTorch·TensorFlow
└─ <b>4_DistributedComputing</b>/  3   simpleP2P·multiGPU</pre></div>
  </div>

  <h3 class="blk">类目语义对应关系</h3>
  <table class="dtl"><thead><tr><th>语义领域</th><th>cann 对应</th><th>cuda 对应</th><th class="c">关系</th></tr></thead><tbody>
    <tr><td class="crit">入门原语</td><td>0_Introduction</td><td>0_Introduction / py 1_GettingStarted</td><td class="c"><span class="eb eb-tie">对齐</span></td></tr>
    <tr><td class="crit">并行算法/概念</td><td>—（仅 simd_vf 含 reduce）</td><td>2_Concepts_and_Techniques</td><td class="c"><span class="eb eb-cuda">cuda 独有</span></td></tr>
    <tr><td class="crit">硬件/指令/访存特性</td><td>1_Features（4 子类）</td><td>3_CUDA_Features（部分）</td><td class="c"><span class="eb eb-cann">cann 更细</span></td></tr>
    <tr><td class="crit">数值库</td><td>—（仅 ops-tensor 子模块）</td><td>4_CUDA_Libraries</td><td class="c"><span class="eb eb-cuda">cuda 独有</span></td></tr>
    <tr><td class="crit">领域应用/图形/金融</td><td>—</td><td>5_Domain_Specific</td><td class="c"><span class="eb eb-cuda">cuda 独有</span></td></tr>
    <tr><td class="crit">算子性能调优</td><td>2_Performance（9 story）</td><td>6_Performance（5 样例）</td><td class="c"><span class="eb eb-cann">cann 独深</span></td></tr>
    <tr><td class="crit">LLM 推理算子</td><td>2_Performance（MoE/FIA/RoPE）</td><td>—（仅 9_CUDA_Tile 孤立原语）</td><td class="c"><span class="eb eb-cann">cann 独有</span></td></tr>
    <tr><td class="crit">底层 IR / 编译</td><td>—</td><td>7_libNVVM</td><td class="c"><span class="eb eb-cuda">cuda 独有</span></td></tr>
    <tr><td class="crit">Python / 框架互操作</td><td>—（Python 仅校验脚本）</td><td>python/ 3_FrameworkInterop</td><td class="c"><span class="eb eb-cuda">cuda 独有</span></td></tr>
    <tr><td class="crit">跨平台/嵌入式</td><td>—</td><td>8_Platform_Specific（Tegra）</td><td class="c"><span class="eb eb-cuda">cuda 独有</span></td></tr>
  </tbody></table>
  <div class="callout"><b>结构哲学差异：</b>cuda 是「字典型」仓库——每个样例自包含（<code>.cu</code>+README+CMakeLists）、结构高度一致，配全局 <code>Common/</code>（33 项 helper 头文件）与集中式 <code>run_tests.py</code>，适合「查 X 功能的最小示例」。cann 是「教程型」仓库——无全局 common，共享代码就近放在各 story 内，结构因样例而异（有的 tutorials/recipes 双分支、有的按 <span class="mono">membase/regbase</span> 分硬件路径），适合「学单算子从朴素到极致的优化过程」。</div>
</section>

<!-- ============ 03 OPENNESS ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">03</span> &nbsp;产品定位 · 开源开放定位</div>
  <h2 class="sec">「真开源」与「源码可得」之间</h2><div class="rule"></div>
  <p class="lead">这是全对比中最具决策分量的一节。二者「开放」的含义并不相同：cuda-samples 是 OSI 认证的宽松开源；cann-samples 是<b>限定用途、可撤销</b>的源码可得（source-available）许可。</p>

  <div class="grid2">
    <div class="card"><h3>许可证逐条对照</h3><div class="cs">直接研读两仓 LICENSE 全文</div>
      <table class="dtl mini"><tbody>
        <tr><td class="crit">许可名称</td></tr>
        <tr><td>cann：<b>CANN Open Software License Agreement v2.0</b></td></tr>
        <tr><td>cuda：<b>BSD 3-Clause</b>（+ CUDA EULA 引用）</td></tr>
        <tr><td class="crit" style="padding-top:3mm">用途限制</td></tr>
        <tr><td>cann §2.1/§3.1：仅可为<b>搭载华为 AI 处理器</b>的系统开发/分发，禁止服务其他厂商处理器</td></tr>
        <tr><td>cuda：<b>无任何用途或硬件限制</b>，可商用可分发</td></tr>
        <tr><td class="crit" style="padding-top:3mm">可撤销性</td></tr>
        <tr><td>cann §2.1 明写 revocable；§5 华为可随时终止、§5.2 授权 void ab initio（自始无效、须删除）</td></tr>
        <tr><td>cuda：永久授权，无终止条款</td></tr>
        <tr><td class="crit" style="padding-top:3mm">专利/IP 反制</td></tr>
        <tr><td>cann §5.1(b)(ii)：你若指控 Software 侵犯你 IP，华为可终止授权</td></tr>
        <tr><td>cuda：无任何报复机制</td></tr>
        <tr><td class="crit" style="padding-top:3mm">OSI 认证</td></tr>
        <tr><td>cann：<span class="eb eb-cann">否</span> &nbsp; cuda：<span class="eb eb-cuda">是</span></td></tr>
      </tbody></table>
    </div>
    <div class="card"><h3>开放度记分卡</h3><div class="cs">逐项占优计分</div>
      <div class="chart">${gaugeSVG()}</div>
      <h4 style="font-size:9.4pt;margin:4mm 0 1mm">cuda 定位</h4>
      <p class="small">面向全体 CUDA 开发者的工具链特性陈列——「展示 CUDA Toolkit 能做什么」。广、浅、可运行、可自由取用。</p>
      <h4 style="font-size:9.4pt;margin:3mm 0 1mm">cann 定位</h4>
      <p class="small">面向昇腾算子工程师的实战调优知识库——「教你把某算子在昇腾上打到极致」。窄、深、重文档、限昇腾用途。</p>
    </div>
  </div>

  <h3 class="blk">开源开放度记分卡（逐条）</h3>
  ${openTableHTML()}
  <div class="callout"><b>合规要点：</b>BSD-3 仅要求「保留版权声明、不得以 NVIDIA 名义背书」，此外无任何限制、授权永久。CANN OSL v2.0 则把授权范围<b>硬绑定到华为 AI 处理器</b>，并保留在违约或专利诉讼时随时收回授权、要求删除衍生作品的权利。因此在「可自由分发 / 可用于非昇腾平台 / 长期法律确定性」上，cuda-samples 明显更开放；这也是选型时不可忽视的法律维度。</div>
</section>

<!-- ============ 04 COVERAGE ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">04</span> &nbsp;覆盖范围 · 广度 vs 深度</div>
  <h2 class="sec">谁覆盖得更广，谁挖得更深</h2><div class="rule"></div>
  <p class="lead">领域热力矩阵量化两仓的覆盖投入（无 / 基础 / 覆盖 / 深耕）。规模上 cuda 约为 cann 的 8 倍（238 vs 62 叶子样例、189 <code>.cu</code> vs 116 <code>.asc</code>），但 cann 在少数 LLM 算子上的纵深为 cuda 所无。</p>
  ${covMatrixHTML()}
  <div class="callout"><b>公平性说明：</b>cann 在「注意力 / LLM 推理算子」上的深耕（融合 RMSNorm+RoPE+KV-Cache、MoE 派发/合并、全量化 FIA）确为 cuda 所无；但非零重叠——cuda 在 <code>cpp/9_CUDA_Tile</code> 提供了<b>孤立原语</b> <code>tileRope</code>（RoPE 前向）、<code>tileLayerNorm</code>、<code>tileBmm</code>。故矩阵中 cuda 该项记「基础」而非「无」：它有零件，但没有 cann 那种把整条算子链打穿的融合调优 story。</div>
  <div class="grid2" style="margin-top:5mm">
    <div class="card"><h3><span class="dot cuda"></span>cuda 类目分布</h3><div class="cs">cpp 205 + python 33 = 238</div>${catTableHTML([...DATA.cudaCpp,...DATA.cudaPython.map(p=>({cat:'py '+p.cat,n:p.n}))],238,'cuda')}</div>
    <div class="card"><h3><span class="dot cann"></span>cann 类目分布</h3><div class="cs">展开 recipes/tutorials/variants ≈ 62</div>${catTableHTML(DATA.cannCat,62,'cann')}
      <div class="chart" style="margin-top:3mm"><h4 style="font-size:8.8pt;margin-bottom:2mm">文件构成对比</h4>${stackedSVG()}</div>
    </div>
  </div>
</section>

<!-- ============ 05 DEVELOPER SAMPLES ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">05</span> &nbsp;给开发者提供的样例</div>
  <h2 class="sec">一个是「样例库」，一个是「教科书」</h2><div class="rule"></div>
  <p class="lead">同样叫 samples，开发体验却不同。cuda 每个样例是一段<b>聚焦的可运行程序</b>；cann 每个 story 是一部<b>逐步教程</b>，从 <span class="mono">0_naive</span> 优化到 <span class="mono">7_fullload</span>，配性能建模文档与 profiling 图。</p>
  <div class="grid2">
    <div class="card"><h3><span class="dot cann"></span>cann：递进式 story / recipe</h3>
      ${listHTML([
        '逐步教程：matmul_tutorials 0_naive→1_pingpong→2_block_swat→…→7_fullload，每步一个优化点',
        '多配方 recipe：同一算子的 MXFP4/MXFP8/HiF8/INT8 多量化实现并列',
        '重文档：67 篇 README/.md（均值 ~10.6KB 长文）+ 210 张 tiling/流水/profiling 图',
        '构建门槛高：cmake -DNPU_ARCH=dav-3510/2201 + ops-tensor 子模块 + 指定 Toolkit 版本',
        'Python 仅作数据生成/精度校验脚本，非可运行样例',
        '硬件门槛：需 Ascend 910B/C 或 950 真机，特性样例挑架构',
      ],'a')}
    </div>
    <div class="card"><h3><span class="dot cuda"></span>cuda：聚焦式一样例一特性</h3>
      ${listHTML([
        '单点可运行：一个样例演示一个 API/特性，读完即跑',
        'NVRTC 变体：9 个 *_nvrtc 目录专教运行期编译（JIT）',
        '跨平台 CMake：Linux / Windows / Tegra 交叉 / DriveOS 一套构建',
        '自动化回归：run_tests.py + test_args.json 逐样例参数化跑测',
        '真 Python 样例：cuda-python + PyTorch/TF 自定义算子互操作',
        '硬件门槛低：任意 NVIDIA GPU 即可，README 精简（均值 ~2.3KB 卡片）',
      ],'b')}
    </div>
  </div>
  <div class="callout"><b>文档密度是最直观的分野：</b>两仓 README 总量相近（各约 585KB），但 cuda 摊在 258 篇里、单篇均值仅 ~2.3KB（模板式卡片）；cann 集中在 67 篇里、单篇均值 ~10.6KB（长文教程），单篇最大 64KB。cann README 含约 7.6 万汉字与 184 张图；<code>simd_vf_story</code> 单篇 README 达 1644 行、<code>matmul_story</code> 1180 行并配 8 级递进教程。</div>

  <div class="grid2" style="margin-top:5mm">
    <div class="card"><h3><span class="dot cann"></span>仅 cann 独有</h3>${listHTML(DATA.onlyCann,'a')}</div>
    <div class="card"><h3><span class="dot cuda"></span>仅 cuda 独有</h3>${listHTML(DATA.onlyCuda,'b')}</div>
  </div>
</section>

<!-- ============ 06 ECOSYSTEM ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">06</span> &nbsp;开源开放生态 · 治理与维护</div>
  <h2 class="sec">全球 PR 社区 vs 结构化 SIG 门禁</h2><div class="rule"></div>
  <p class="lead">两仓代表两种开源运营范式：cuda 是 GitHub 原生、随 CUDA 工具链多年滚动演进的成熟全球生态；cann 是 GitCode 主场、GitHub 镜像、组织严密但年轻的中文优先社区。</p>
  <div class="grid2">
    <div class="card"><h3><span class="dot cann"></span>cann-samples 生态</h3>
      <dl class="kv" style="margin-top:2mm">
        <dt>主平台</dt><dd>GitCode（中国）· Issues/Discussions/SIG 均在此</dd>
        <dt>GitHub</dt><dd>只读镜像 <span class="mono">sync/upstream/master</span>（cann-robot 自动同步）</dd>
        <dt>治理</dt><dd>SIG ops-basic · Committer <span class="mono">/lgtm</span> → Maintainer <span class="mono">/approve</span></dd>
        <dt>贡献规范</dt><dd>620 行 CONTRIBUTING（A/B/C 三种 Sample 模板）+ OAT.xml 许可声明强制</dd>
        <dt>CI 门禁</dt><dd><span class="mono">run_ci_functional.py</span> 真机功能测试</dd>
        <dt>依赖</dt><dd>git 子模块 ops-tensor / shmem（钉死于 gitcode）+ 45KB 第三方声明</dd>
        <dt>成熟度</dt><dd>年轻 · 镜像分支约 50 提交（2026-05→07）· 约 21 位作者（多为华为）</dd>
        <dt>语言</dt><dd>中文优先</dd>
      </dl>
    </div>
    <div class="card"><h3><span class="dot cuda"></span>cuda-samples 生态</h3>
      <dl class="kv" style="margin-top:2mm">
        <dt>主平台</dt><dd>GitHub（全球）· 原生 fork + PR 协作</dd>
        <dt>版本</dt><dd>随 CUDA Toolkit 发布（本次 v13.3）· CHANGELOG 上溯 CUDA 9.2/10.0</dd>
        <dt>治理</dt><dd>NVIDIA 维护 · 外部 Fork + PR · 94 行 CONTRIBUTING</dd>
        <dt>CI 门禁</dt><dd>pre-commit.ci 格式化（clang-format + ruff）+ <span class="mono">run_tests.py</span> 回归</dd>
        <dt>依赖</dt><dd>CPM 拉取 CCCL（pin v3.3.3）· 更去中心化</dd>
        <dt>成熟度</dt><dd>多年沉淀 · CHANGELOG 覆盖 CUDA 10.x→13.3 · 数十版本连续</dd>
        <dt>语言</dt><dd>英文 / 全球</dd>
      </dl>
    </div>
  </div>
  <div class="grid2" style="margin-top:5mm">
    <div class="callout" style="margin:0"><b>cann 生态强项：</b>结构化治理与真机 CI 功能门禁、极深的中文调优文档、面向 AI 大模型算子的前沿覆盖（MX 量化、MoE、注意力）。强制兼容性声明（硬件/CANN 版本/shape 约束）保证可复现。</div>
    <div class="callout" style="margin:0"><b>cuda 生态强项：</b>宽松许可带来的低法律摩擦、全球社区可达性、跨平台与数值库的极致广度、随工具链长期演进的成熟度、集中化自动测试。</div>
  </div>
</section>

<!-- ============ 07 RECOMMENDATION ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">07</span> &nbsp;结论与选型建议</div>
  <h2 class="sec">什么时候选哪一个</h2><div class="rule"></div>
  <p class="lead">二者不是替代关系，而是服务于不同目标、不同硬件栈的两种资源。</p>
  <div class="rec">
    <div class="rc a"><div class="when">选 cann-samples 当你…</div><h3><span class="dot cann"></span>攻昇腾算子深度</h3>
      <ul>
        <li>在 Ascend 910B/C 或 950 上做<b>高性能算子开发/调优</b></li>
        <li>需要 <b>LLM 推理算子</b>（MoE、融合注意力、RoPE+RMSNorm KV Cache）的工程实现范式</li>
        <li>需要 <b>MX/HiF8/INT8 低比特量化</b>矩阵乘的完整 recipe</li>
        <li>想系统学习「访存/指令/系统」三层<b>逐步优化方法论</b></li>
        <li>接受<b>限昇腾用途</b>的许可与真机门槛，看重中文深度文档</li>
      </ul>
    </div>
    <div class="rc b"><div class="when">选 cuda-samples 当你…</div><h3><span class="dot cuda"></span>学 CUDA 生态广度</h3>
      <ul>
        <li>要<b>系统入门/概览</b> CUDA 全栈能力与 API 特性</li>
        <li>需要 <b>cuBLAS/cuFFT/cuSOLVER/NPP/CUB/Thrust</b> 等库的可运行示例</li>
        <li>做<b>图形互操作</b>（D3D/Vulkan/GL）或<b>经典 HPC/金融</b>计算</li>
        <li>关心<b>开源合规、可商用、可分发</b>与跨平台（Windows/Tegra/DriveOS）</li>
        <li>想要<b>低硬件门槛</b>（任意 NVIDIA GPU）与全球社区支持</li>
      </ul>
    </div>
  </div>
  <h3 class="blk">一句话总览</h3>
  <table class="dtl"><thead><tr><th>维度</th><th>cann-samples</th><th>cuda-samples</th></tr></thead><tbody>
    <tr><td class="crit">一句定位</td><td>昇腾算子调优知识库</td><td>CUDA 工具链特性陈列馆</td></tr>
    <tr><td class="crit">取胜之道</td><td>深度 · 调优方法论 · 中文文档</td><td>广度 · 低门槛 · 开源合规</td></tr>
    <tr><td class="crit">规模</td><td>4 类目 / 19 组 / ≈62 叶 / 782 文件</td><td>14 类目 / 238 叶 / 2,022 文件</td></tr>
    <tr><td class="crit">许可</td><td>CANN OSL v2.0（受限 · 非 OSI）</td><td>BSD-3-Clause（宽松 · OSI）</td></tr>
    <tr><td class="crit">硬件门槛</td><td>特定 Ascend 真机 + 指定 Toolkit</td><td>任意 NVIDIA GPU</td></tr>
    <tr><td class="crit">最适合</td><td>昇腾算子工程师 / LLM 推理优化</td><td>CUDA 学习者 / 生态应用开发者</td></tr>
  </tbody></table>
</section>

<!-- ============ 08 APPENDIX ============ -->
<section class="page break">
  <div class="sec-ey"><span class="n">08</span> &nbsp;数据附录与核验日志</div>
  <h2 class="sec">全量计数与对抗式验证</h2><div class="rule"></div>

  <h3 class="blk">A · 头部量化指标</h3>
  <table class="dtl"><thead><tr><th>指标</th><th class="num">cann</th><th class="num">cuda</th><th>口径备注</th></tr></thead><tbody>
  ${DATA.headline.map(h=>`<tr><td class="crit">${esc(h.label)}</td><td class="num" style="color:${C.cannInk}">${h.cann.toLocaleString()}</td><td class="num" style="color:${C.cudaInk}">${h.cuda.toLocaleString()}</td><td class="small">${esc(h.note||'')}</td></tr>`).join('')}
  </tbody></table>

  <h3 class="blk">B · 文件类型构成</h3>
  <div class="grid2">
    <div>${(function(){const r=DATA.fileTypes.cann.map(d=>`<tr><td>${esc(d.ext)}</td><td class="num">${d.n}</td></tr>`).join('');return `<table class="dtl mini"><thead><tr><th>cann · 扩展名</th><th class="num">数量</th></tr></thead><tbody>${r}</tbody></table>`;})()}</div>
    <div>${(function(){const r=DATA.fileTypes.cuda.map(d=>`<tr><td>${esc(d.ext)}</td><td class="num">${d.n}</td></tr>`).join('');return `<table class="dtl mini"><thead><tr><th>cuda · 扩展名</th><th class="num">数量</th></tr></thead><tbody>${r}</tbody></table>`;})()}</div>
  </div>

  <h3 class="blk">C · 对抗式核验日志</h3>
  <p class="small" style="margin-bottom:3mm">下列关键论断由独立子代理对两仓工作树执行反证式核验（尝试推翻而非确认），结论如下。</p>
  <table class="dtl"><thead><tr><th>论断</th><th class="c">判定</th><th>核验结论与修正</th></tr></thead><tbody>
  ${verifyHTML()}
  </tbody></table>

  <h3 class="blk">D · 方法与免责声明</h3>
  <dl class="kv">
    <dt>对比对象</dt><dd>cann-samples <span class="mono">sync/upstream/master</span>；NVIDIA/cuda-samples <span class="mono">master</span> Release v13.3。快照 2026-07-09。</dd>
    <dt>计数口径</dt><dd>文件数/类目/样例均由 <span class="mono">find/ls</span> 直接枚举（排除 .git）。cuda 叶子样例=cpp 205 + python 33=238；cann 叶子样例展开 recipes/tutorials/variants ≈62，顶层样例组 19。</dd>
    <dt>能力评分</dt><dd>七轴 0–100 为显式分析模型，供交互版重加权，非厂商官方指标。</dd>
    <dt>局限</dt><dd>cuda-samples 为浅克隆，历史以 PR 编号与 CHANGELOG 佐证；领域深度分级含主观成分。</dd>
  </dl>
  <p class="note">本报告为独立技术对比分析，非任何厂商官方文档；商标与许可条款归各自权利人所有。数值以快照日两仓实际内容为准。</p>
</section>

</body></html>`;

fs.writeFileSync('../print.html', html);
console.log('print.html written:', html.length, 'bytes');
