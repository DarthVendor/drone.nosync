#!/usr/bin/env python3
"""Render a decisions trace (from trace_decisions.py) as a scrubbable page.

    python scripts/decision_page.py decisions.json out.html
"""
import json, sys
SRC, OUT = sys.argv[1:3]
D = json.load(open(SRC))
cs, hs = D["boxes"]["c"], D["boxes"]["h"]
X0 = min(c[0]-h[0] for c,h in zip(cs,hs))-2; X1 = max(c[0]+h[0] for c,h in zip(cs,hs))+2
Y0 = min(c[1]-h[1] for c,h in zip(cs,hs))-2; Y1 = max(c[1]+h[1] for c,h in zip(cs,hs))+2
sx = lambda x: round(x-X0,2); sy = lambda y: round(Y1-y,2)
rects = [dict(x=sx(c[0]-h[0]), y=sy(c[1]+h[1]), w=round(2*h[0],2), h=round(2*h[1],2)) for c,h in zip(cs,hs)]
flights = []
for f in D["flights"]:
    dec = [dict(t=d["t"], step=int(round(d["t"])), beams=d["beams"], az=d["beam_az"], patches=d["patches"],
                sub=[sx(d["sub_world"][0]), sy(d["sub_world"][1])], sub_ego=d["sub_ego"], goal_ego=d["goal_ego"],
                alpha=d["alpha"][0], gate=d["gate"][0], hd=d["heading_delta"], hg=d["heading_gate"], value=d["value"],
                a_sub=d["attn_sub"], sal=d["sal"], cf=d["cf"], chain=d["chain_len"]) for d in f["decisions"]]
    flights.append(dict(label=f"episode {f['episode']} — {f['outcome']} at {f['end_step']*D['dt']:.1f} s", outcome=f["outcome"],
                        end=f["end_step"], path=[[sx(x), sy(y)] for x,y in f["path"]], yaw=f["yaw"],
                        goals=[[sx(g[0]), sy(g[1])] for g in f["goals"]], dec=dec, nb=f["n_beams"]))
J = json.dumps(dict(rects=rects, W=round(X1-X0,2), H=round(Y1-Y0,2), flights=flights, grid=D["patch_grid"], reach=D["reach"], dt=D["dt"]), separators=(",",":"))
n_cr = sum(1 for f in flights if f["outcome"] == "crash")
html = """<title>Decision Head</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{--ground:#F2EFE8;--panel:#FBF9F5;--edge:#DCD5C8;--ink:#1B2430;--body:#3C4550;--muted:#7A736A;--block:#CFC8BA;--teal:#0B8FA6;--red:#C2412F;--amber:#B8801F;--grid:#E4DED2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0F151D;--panel:#161F2B;--edge:#26313F;--ink:#EDE8DF;--body:#C2C9D2;--muted:#8B95A2;--block:#26313F;--teal:#24A3B8;--red:#D9563F;--amber:#E8A33D;--grid:#1E2733}}
:root[data-theme="dark"]{--ground:#0F151D;--panel:#161F2B;--edge:#26313F;--ink:#EDE8DF;--body:#C2C9D2;--muted:#8B95A2;--block:#26313F;--teal:#24A3B8;--red:#D9563F;--amber:#E8A33D;--grid:#1E2733}
*{box-sizing:border-box} body{background:var(--ground);color:var(--body);font-family:Newsreader,Georgia,serif;font-size:16px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 72px}
.eyebrow{font-family:"JetBrains Mono",monospace;font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--amber);margin:0}
h1{font-family:Archivo,system-ui,sans-serif;font-weight:700;font-size:clamp(30px,4.5vw,46px);line-height:1.05;letter-spacing:-.02em;color:var(--ink);margin:.15em 0 .3em}
h2{font-family:Archivo,sans-serif;font-weight:600;font-size:15px;letter-spacing:.02em;color:var(--ink);margin:0 0 8px}
.lede{font-size:18px;max-width:70ch;margin:0 0 18px} .lede b{color:var(--ink);font-weight:500}
.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--edge);padding:12px 16px;margin:14px 0 18px}
.bar input[type=range]{flex:1;min-width:220px} .bar button,.bar select{font-family:Archivo,sans-serif;font-size:12.5px;color:var(--body);background:transparent;border:1px solid var(--edge);padding:6px 12px;cursor:pointer}
.bar button:focus-visible,.bar select:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
.mono{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums;color:var(--ink)}
.grid{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:16px} @media(max-width:960px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--edge);padding:14px 16px} .card svg{display:block;width:100%;height:auto}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px} .kv .k{color:var(--muted);font-family:Archivo,sans-serif;font-size:12px;letter-spacing:.04em;text-transform:uppercase;align-self:center}
.note{font-size:13.5px;color:var(--muted);margin:8px 0 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--muted);margin-top:8px} .legend i{display:inline-block;width:12px;height:12px;margin-right:5px;vertical-align:-2px}
.wide{grid-column:1 / -1} .crash{color:var(--red)}
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--edge);font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--muted);line-height:1.8}
</style>
<div class="wrap">
<p class="eyebrow">Singapore CBD &middot; composer v2 &middot; __NF__ flights, __NC__ of them crashes</p>
<h1>Decision Head</h1>
<p class="lede">Every 0.2 s the composer reads its front fan, its camera, its own stream and the goal, and writes a sub-goal, term weights and a heading. For each decision of each flight: what it <b>saw</b>, what it <b>chose</b>, where it <b>looked</b>, what it is <b>sensitive</b> to, and what it <b>depends</b> on. On the crashes, scrub to the last few decisions: the counterfactual bars say whether anything the drone perceived was moving its choice at all.</p>
<div class="bar"><select id="flight" aria-label="flight"></select><button id="play" aria-pressed="false">Play</button><input id="scrub" type="range" min="0" max="1" value="0" aria-label="decision"><span class="mono" id="tlabel"></span></div>
<div class="grid">
 <div class="card"><h2>Where it is, and where it was told to go</h2><div id="map"></div>
  <div class="legend"><span><i style="background:var(--teal)"></i>path so far</span><span><i style="background:var(--amber)"></i>sub-goal</span><span><i style="background:var(--red);border-radius:50%"></i>leg goal</span><span><i style="background:var(--red)"></i>impact</span></div></div>
 <div class="card"><h2>What it saw: the front fan</h2><div id="fan"></div><p class="note">Bar length = range (short = close). Fill = saliency of that beam. Outer dots = the sub-goal query's attention.</p></div>
 <div class="card"><h2>What it chose</h2><div class="kv" id="kv"></div><div id="cf"></div><p class="note">Counterfactual shift: metres the sub-goal moves when that input is removed.</p></div>
 <div class="card"><h2>What it saw: the camera</h2><div id="cam"></div><p class="note">10 &times; 5 patches, nearest depth (dark = close); outline = saliency.</p></div>
 <div class="card wide"><h2>Over the flight</h2><div id="attn"></div>
  <div class="legend"><span><i style="background:var(--teal)"></i>beams</span><span><i style="background:var(--amber)"></i>camera</span><span><i style="background:var(--red)"></i>goal</span><span><i style="background:var(--muted)"></i>self + stream</span></div>
  <p class="note">Top: the sub-goal query's attention, grouped, per decision. Bottom: what removing each input would have done to the sub-goal, per decision. A red mark is the impact.</p><div id="cft"></div></div>
</div>
<footer>composer v2 at its current checkpoint (co-training in progress) &middot; low level: per-beam city genome extended for the front fan, downward fan and tilt &middot; sensors: 24-beam 120&deg; front fan, 4-beam downward fan, 20&times;10 depth camera, tilt &middot; attention: read layer, head-averaged &middot; saliency: &Vert;&part;&Vert;sub-goal&Vert;/&part;token&Vert; &middot; counterfactuals: sub-goal shift with that input masked &middot; regenerate: scripts/trace_decisions.py then scripts/decision_page.py</footer>
</div>
<script>
const D=__DATA__;const NS="http://www.w3.org/2000/svg";const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e};
let F=D.flights[0], dec=F.dec, n=dec.length, i=0, playing=false;
const scrub=document.getElementById("scrub"), tl=document.getElementById("tlabel"), sel=document.getElementById("flight");
D.flights.forEach((f,j)=>{const o=document.createElement("option"); o.value=j; o.textContent=f.label; sel.appendChild(o);});
const map=el("svg",{viewBox:`0 0 ${D.W} ${D.H}`,role:"img","aria-label":"plan view"});
for(const r of D.rects) map.appendChild(el("rect",{x:r.x,y:r.y,width:r.w,height:r.h,fill:"var(--block)",opacity:0.8}));
const goalsG=el("g",{}); map.appendChild(goalsG);
const trail=el("path",{fill:"none",stroke:"var(--teal)","stroke-width":0.18,"stroke-linejoin":"round"}); map.appendChild(trail);
const subLine=el("line",{stroke:"var(--amber)","stroke-width":0.12,"stroke-dasharray":"0.4 0.3"}); map.appendChild(subLine);
const sub=el("circle",{r:0.38,fill:"var(--amber)",opacity:0.9}); map.appendChild(sub);
const impact=el("path",{stroke:"var(--red)","stroke-width":0.2,fill:"none"}); map.appendChild(impact);
const body=el("circle",{r:0.32,fill:"var(--ink)"}), nose=el("line",{stroke:"var(--ink)","stroke-width":0.16}); map.appendChild(body); map.appendChild(nose);
document.getElementById("map").appendChild(map);
const FW=360,FH=220,cx=FW/2,cy=FH-24,RMAX=150; const fan=el("svg",{viewBox:`0 0 ${FW} ${FH}`,role:"img","aria-label":"front fan"}); const beamG=el("g",{}); fan.appendChild(beamG); document.getElementById("fan").appendChild(fan);
const [GW,GH]=D.grid; const cam=el("svg",{viewBox:`0 0 ${GW*30} ${GH*30}`,role:"img","aria-label":"camera patches"}); const cells=[];
for(let r=0;r<GH;r++)for(let c=0;c<GW;c++){const q=el("rect",{x:c*30+1,y:r*30+1,width:28,height:28,fill:"var(--teal)",stroke:"var(--amber)","stroke-width":0}); cells.push(q); cam.appendChild(q);}
document.getElementById("cam").appendChild(cam);
function groups(d){const nb=F.nb, a=d.a_sub||[]; const g={self:a[0]||0,goal:a[1]||0,beams:0,cam:0,stream:0};
 for(let k=2;k<a.length;k++){ if(k<2+nb) g.beams+=a[k]; else if(k<a.length-1||d.chain===0) g.cam+=a[k]; else g.stream+=a[k]; } return g;}
let attnCursor, cfCursor, attnPx;
function buildCharts(){ const A=document.getElementById("attn"), C=document.getElementById("cft"); A.innerHTML=""; C.innerHTML="";
 const W=980,H=120,L=36,B=22,T=8,R=8; const px=k=>L+(n>1?k/(n-1):0)*(W-L-R), py=v=>H-B-v*(H-B-T); attnPx=px;
 const s=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"attention over time"});
 let acc=new Array(n).fill(0);
 for(const [k,col] of [["beams","var(--teal)"],["cam","var(--amber)"],["goal","var(--red)"],["other","var(--muted)"]]){
  const top=[]; for(let j=0;j<n;j++){const g=groups(dec[j]); top.push(acc[j]+(k==="other"?g.self+g.stream:g[k]));}
  let d=`M${px(0)},${py(acc[0])}`; for(let j=0;j<n;j++) d+=` L${px(j)},${py(top[j])}`; for(let j=n-1;j>=0;j--) d+=` L${px(j)},${py(acc[j])}`; d+="Z";
  s.appendChild(el("path",{d,fill:col,opacity:0.55})); acc=top;}
 for(const v of [0,0.5,1]){s.appendChild(el("line",{x1:L,x2:W-R,y1:py(v),y2:py(v),stroke:"var(--grid)","stroke-width":1})); const t=el("text",{x:L-6,y:py(v)+4,"text-anchor":"end",fill:"var(--muted)","font-size":10,"font-family":"JetBrains Mono, monospace"}); t.textContent=v; s.appendChild(t);}
 attnCursor=el("line",{y1:T,y2:H-B,stroke:"var(--ink)","stroke-width":1.2}); s.appendChild(attnCursor); A.appendChild(s);
 const s2=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"counterfactual shifts over time"});
 const mx=Math.max(0.5,...dec.map(d=>Math.max(d.cf.no_goal,d.cf.no_beams,d.cf.no_camera,d.cf.no_stream))); const py2=v=>H-B-Math.min(v,mx)/mx*(H-B-T);
 for(const [k,col] of [["no_goal","var(--red)"],["no_beams","var(--teal)"],["no_camera","var(--amber)"],["no_stream","var(--muted)"]]){
  let d=""; dec.forEach((q,j)=>{d+=(j?" L":"M")+px(j)+","+py2(q.cf[k])}); s2.appendChild(el("path",{d,fill:"none",stroke:col,"stroke-width":2}));}
 for(const v of [0,mx/2,mx]){s2.appendChild(el("line",{x1:L,x2:W-R,y1:py2(v),y2:py2(v),stroke:"var(--grid)","stroke-width":1})); const t=el("text",{x:L-6,y:py2(v)+4,"text-anchor":"end",fill:"var(--muted)","font-size":10,"font-family":"JetBrains Mono, monospace"}); t.textContent=v.toFixed(1)+"m"; s2.appendChild(t);}
 if(F.outcome==="crash"){ for(const svg of [s,s2]) svg.appendChild(el("line",{x1:px(n-1),x2:px(n-1),y1:T,y2:H-B,stroke:"var(--red)","stroke-width":2})); }
 cfCursor=el("line",{y1:T,y2:H-B,stroke:"var(--ink)","stroke-width":1.2}); s2.appendChild(cfCursor); C.appendChild(s2); }
function loadFlight(j){ F=D.flights[j]; dec=F.dec; n=dec.length; scrub.max=n-1; while(goalsG.firstChild) goalsG.removeChild(goalsG.firstChild);
 for(const g of F.goals) goalsG.appendChild(el("circle",{cx:g[0],cy:g[1],r:0.5,fill:"none",stroke:"var(--red)","stroke-width":0.18}));
 const e=F.path[F.path.length-1]; impact.setAttribute("d", F.outcome==="crash" ? `M${e[0]-0.5},${e[1]-0.5} L${e[0]+0.5},${e[1]+0.5} M${e[0]-0.5},${e[1]+0.5} L${e[0]+0.5},${e[1]-0.5}` : "");
 buildCharts(); show(F.outcome==="crash" ? Math.max(0,n-6) : 0); }
function show(k){ i=k; const d=dec[k]; scrub.value=k; tl.textContent=`decision ${k+1}/${n}  t=${(d.t*D.dt).toFixed(1)}s  stream ${d.chain} tokens`+(F.outcome==="crash"&&k>=n-3?"  — about to hit":"");
 const step=Math.min(d.step, F.path.length-1); const p=F.path[step], yaw=F.yaw[step];
 trail.setAttribute("d", F.path.slice(0,step+1).map((q,j)=>(j?"L":"M")+q[0]+","+q[1]).join(" "));
 body.setAttribute("cx",p[0]); body.setAttribute("cy",p[1]); nose.setAttribute("x1",p[0]); nose.setAttribute("y1",p[1]); nose.setAttribute("x2",p[0]+Math.cos(yaw)*0.9); nose.setAttribute("y2",p[1]-Math.sin(yaw)*0.9);
 sub.setAttribute("cx",d.sub[0]); sub.setAttribute("cy",d.sub[1]); subLine.setAttribute("x1",p[0]); subLine.setAttribute("y1",p[1]); subLine.setAttribute("x2",d.sub[0]); subLine.setAttribute("y2",d.sub[1]);
 while(beamG.firstChild) beamG.removeChild(beamG.firstChild);
 const nb=F.nb, salE=d.sal.entities, smax=Math.max(1e-9,...salE), a=d.a_sub||[], amax=Math.max(1e-9,...a.slice(2,2+nb));
 for(let b=0;b<nb;b++){const az=d.az[b], r=d.beams[b]*RMAX; beamG.appendChild(el("line",{x1:cx,y1:cy,x2:cx+Math.sin(az)*r,y2:cy-Math.cos(az)*r,stroke:"var(--amber)","stroke-width":5,"stroke-linecap":"round",opacity:(0.15+0.85*salE[b]/smax).toFixed(2)}));
  beamG.appendChild(el("circle",{cx:cx+Math.sin(az)*(RMAX+8),cy:cy-Math.cos(az)*(RMAX+8),r:4,fill:"var(--teal)",opacity:(0.1+0.9*(a[2+b]||0)/amax).toFixed(2)}));}
 beamG.appendChild(el("line",{x1:cx,y1:cy,x2:cx,y2:cy-RMAX-16,stroke:"var(--muted)","stroke-width":1,"stroke-dasharray":"3 3"}));
 const ga=Math.atan2(d.goal_ego[1],d.goal_ego[0]); beamG.appendChild(el("circle",{cx:cx+Math.sin(ga)*RMAX*0.98,cy:cy-Math.cos(ga)*RMAX*0.98,r:5,fill:"none",stroke:"var(--red)","stroke-width":2}));
 const sa=Math.atan2(d.sub_ego[1],d.sub_ego[0]), sr=Math.min(1,Math.hypot(d.sub_ego[0],d.sub_ego[1])/D.reach)*RMAX; beamG.appendChild(el("circle",{cx:cx+Math.sin(sa)*sr,cy:cy-Math.cos(sa)*sr,r:5,fill:"var(--amber)"}));
 const pm=d.patches, sp=salE.slice(nb), spm=Math.max(1e-9,...sp); cells.forEach((q,j)=>{q.setAttribute("opacity",(0.15+0.85*(1-pm[j])).toFixed(2)); q.setAttribute("stroke-width",(3*sp[j]/spm).toFixed(2));});
 const g=groups(d);
 document.getElementById("kv").innerHTML=[["sub-goal (body)",`${d.sub_ego[0].toFixed(1)}, ${d.sub_ego[1].toFixed(1)}, ${d.sub_ego[2].toFixed(1)} m`],["goal (body)",`${d.goal_ego[0].toFixed(1)}, ${d.goal_ego[1].toFixed(1)}, ${d.goal_ego[2].toFixed(1)} m`],["priority &alpha;",d.alpha.toFixed(3)],["gate",d.gate.toFixed(3)],["heading",`${(d.hd*57.3).toFixed(0)}&deg; &middot; gate ${d.hg.toFixed(2)}`],["value",d.value.toFixed(2)],["looked at",`beams ${(100*g.beams).toFixed(0)}% &middot; camera ${(100*g.cam).toFixed(0)}% &middot; goal ${(100*g.goal).toFixed(0)}%`]].map(([k,v])=>`<div class="k">${k}</div><div class="mono">${v}</div>`).join("");
 const cf=d.cf, cmx=Math.max(0.2,cf.no_goal,cf.no_beams,cf.no_camera,cf.no_stream);
 document.getElementById("cf").innerHTML=[["goal removed",cf.no_goal,"var(--red)"],["beams blanked",cf.no_beams,"var(--teal)"],["camera blanked",cf.no_camera,"var(--amber)"],["stream forgotten",cf.no_stream,"var(--muted)"]].map(([k,v,c])=>`<div style="display:grid;grid-template-columns:120px 1fr 52px;gap:8px;align-items:center;font-size:13px;margin-top:6px"><span style="color:var(--muted)">${k}</span><span style="height:8px;background:${c};width:${(100*v/cmx).toFixed(0)}%;border-radius:4px"></span><span class="mono">${v.toFixed(2)} m</span></div>`).join("");
 attnCursor.setAttribute("x1",attnPx(k)); attnCursor.setAttribute("x2",attnPx(k)); cfCursor.setAttribute("x1",attnPx(k)); cfCursor.setAttribute("x2",attnPx(k)); }
scrub.addEventListener("input",e=>show(+e.target.value)); sel.addEventListener("change",e=>loadFlight(+e.target.value));
document.getElementById("play").addEventListener("click",function(){playing=!playing; this.setAttribute("aria-pressed",playing); this.textContent=playing?"Pause":"Play"; if(playing) tick();});
function tick(){ if(!playing) return; show((i+1)%n); setTimeout(tick,120); }
const mq=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)"); if(mq&&mq.matches) document.getElementById("play").disabled=true;
loadFlight(0);
</script>
"""
html = html.replace("__DATA__", J).replace("__NF__", str(len(flights))).replace("__NC__", str(n_cr))
open(OUT, "w").write(html); print(f"  {OUT}: {len(html)/1e6:.2f} MB, {len(flights)} flights ({n_cr} crashes)")
