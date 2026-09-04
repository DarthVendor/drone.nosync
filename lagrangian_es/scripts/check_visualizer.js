/*
 * Headless render check for scripts/visualizer.html.
 *
 *   node scripts/check_visualizer.js scripts/visualizer.html data.json
 *
 * Stubs just enough DOM and canvas to run the page's real render path, then
 * drives every robot through every mode, episode and several frames.  Two
 * classes of bug motivated it, both of which killed the whole page rather than
 * one overlay, because an exception inside requestAnimationFrame stops the loop:
 *
 *   - reading a camera's fields (landmarks, mount, image) off a range sensor,
 *     because the overlay was gated on `R.sensor` being truthy rather than on
 *     its type;
 *   - a sprite kind handled in paint()'s draw loop but not its collection loop.
 *
 * It also asserts that every declared overlay actually produces sprites --
 * rendering without throwing is not the same as drawing anything.
 */
const fs = require('fs');

const CSS = {'--grid':'#C7D0D9','--line':'#D3D9E0','--ink':'#15202B','--muted':'#7A8896',
  '--target':'#5B4BC4','--accent':'#0B7A85','--warn':'#C0490F','--ok':'#1F7A45',
  '--surface-2':'#F4F6F9'};
const ctxStub = new Proxy({}, {get:(t,k)=> k==='measureText' ? ()=>({width:10})
  : (typeof k==='string' ? ()=>{} : undefined), set:()=>true});
function mkEl(id){
  return {id, style:{}, hidden:false, textContent:'', innerHTML:'', className:'',
    children:[], clientWidth:240, value:0, max:0,
    getContext:()=>ctxStub, getBoundingClientRect:()=>({width:640,height:400}),
    setAttribute(){}, addEventListener(){}, appendChild(c){this.children.push(c);},
    onclick:null, oninput:null, onchange:null, classList:{add(){},remove(){}},
    parentElement:{clientWidth:240}, closest:()=>null};
}
const els = {};
globalThis.document = {getElementById:(id)=>(els[id] ||= mkEl(id)),
  createElement:()=>mkEl('new'), documentElement:{}, body:{}, addEventListener(){}};
globalThis.getComputedStyle = ()=>({getPropertyValue:(n)=>CSS[n]||'#888',
                                    fontFamily:'sans-serif'});
globalThis.matchMedia = ()=>({matches:false, addEventListener(){}});
globalThis.ResizeObserver = class { observe(){} };
globalThis.MutationObserver = class { observe(){} };
globalThis.requestAnimationFrame = ()=>0;
globalThis.performance = {now:()=>0};
globalThis.devicePixelRatio = 1;
globalThis.addEventListener = ()=>{};

const [tplPath, dataPath] = process.argv.slice(2);
const tpl = fs.readFileSync(tplPath, 'utf8');
const js = tpl.split('<script>')[1].split('</script>')[0]
  .replace('JSON.parse(document.getElementById("data").textContent)', '__DATA__');
globalThis.__DATA__ = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
(0, eval)(js + '\nglobalThis.__T = {selectRobot, render, updateHUD, drawImagePlane,'
  + ' drawRangePanel, obstacleSprites, landmarkSprites, beamSprites, cableSprites,'
  + ' ROBOTS, setFrame:(f)=>{frame=f;}, setMode:(m)=>{mode=m;}, setEp:(e)=>{ep=e;}};');

const T = globalThis.__T;
let fails = 0;

for (let i = 0; i < T.ROBOTS.length; i++){
  const R = T.ROBOTS[i];
  try {
    T.selectRobot(i);
    for (const mode of ['both','trained','prior']){
      T.setMode(mode);
      for (const ep of [0, R.runs.trained.length - 1]){
        T.setEp(ep);
        for (const f of [0, 1, Math.floor(R.n_frames/2), R.n_frames - 1]){
          T.setFrame(f);
          T.updateHUD(); T.render();
          const who = mode === 'prior' ? 'prior' : 'trained';
          T.drawImagePlane(who, f); T.drawRangePanel(who, f);
        }
      }
    }
    console.log(`  ok   ${R.key.padEnd(18)} renders`);
  } catch (e) {
    fails++;
    console.log(`  FAIL ${R.key.padEnd(18)} ${e.constructor.name}: ${e.message}`);
    console.log('       ' + (e.stack || '').split('\n')[1]);
  }
}

for (let i = 0; i < T.ROBOTS.length; i++){
  const R = T.ROBOTS[i];
  T.selectRobot(i); T.setMode('trained'); T.setEp(0);
  const f = Math.floor(R.n_frames/2); T.setFrame(f);
  const ep0 = R.runs.trained[0];
  const got = {obstacles: T.obstacleSprites('trained').length,
               landmarks: T.landmarkSprites('trained', f).length,
               beams:     T.beamSprites('trained', f).length,
               cable:     T.cableSprites('trained', f, [1,1,1]).length};
  const want = [];
  if (ep0.obstacles) want.push('obstacles');
  if (R.sensor && R.sensor.type === 'landmark_camera') want.push('landmarks');
  if (R.sensor && R.sensor.type === 'range') want.push('beams');
  if (ep0.cable) want.push('cable');
  const missing = want.filter(k => got[k] === 0);
  if (missing.length){
    fails++;
    console.log(`  EMPTY ${R.key.padEnd(18)} declares ${missing} but draws none`);
  }
}

console.log(fails ? `\n${fails} failure(s)` : '\nvisualizer ok');
process.exit(fails ? 1 : 0);
