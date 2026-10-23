// Renders side-by-side replays of s1 eval game 1713 with Playwright (frame-by-frame screenshots).
// Usage: node record_replay.js OUT_FRAMES_DIR   (needs `npm i playwright`; run from this dir)
const fs=require('fs'),path=require('path');const {chromium}=require(process.env.PW||'playwright');
const D=path.join(__dirname,'..');const FPS=24;
function load(d){const t=fs.readFileSync(path.join(D,`traj_s1_e1713_d${d}.txt`),'utf8').trim().split(/\s+/).slice(1);
  return {b:t.map(x=>x.split(':')[0]),s:t.map(x=>+x.split(':')[1])}}
const G={1:load(1),4:load(4),5:load(5)};
const K16=G[5].b.findIndex(h=>h.includes('e'));
// segments: [title, left depth, right depth, seconds, step mapping fn(u)->d5 step, hold seconds]
const N5=G[5].b.length-1;
const SEG=[
 {title:'Same game (s1 eval #1713): depth 1 vs depth 5',L:1,R:5,sec:15,hold:2,map:u=>u*N5},
 {title:'Highlight: depth 4 vs depth 5 — d5 reaches 16384',L:4,R:5,sec:20,hold:4,
  map:u=>u<0.6?u/0.6*(K16-400):(K16-400)+(u-0.6)/0.4*(N5-(K16-400))}];
const html=`<html><head><meta charset="utf-8"></head><body style="margin:0;background:#f7f6f3;font-family:Helvetica,Arial,sans-serif">
<canvas id=c width=1280 height=720></canvas><script>
const G=${JSON.stringify(G)};const COL=['#eee4da','#ede0c8','#f2b179','#f59563','#f67c5f','#f65e3b','#edcf72','#edcc61','#edc850','#edc53f','#edc22e','#3c3a32'];
const DC={1:'#6b7280',4:'#7c3aed',5:'#c2410c'};const x=document.getElementById('c').getContext('2d');
function rr(X,Y,W,H,r){x.beginPath();x.roundRect(X,Y,W,H,r);x.fill()}
function board(d,i,X){const g=G[d],n=g.b.length;i=Math.min(Math.max(0,Math.round(i)),n-1);const h=g.b[i];let m=0;
 x.fillStyle=DC[d];x.font='bold 30px Helvetica';x.textAlign='center';x.fillText('Expectimax depth '+d,X+250,150);
 x.fillStyle='#1c1f24';x.font='24px Helvetica';
 for(let k=0;k<16;k++)m=Math.max(m,parseInt(h[15-k],16));
 x.fillText('score '+g.s[i].toLocaleString('en-US')+'   max tile '+(1<<m).toLocaleString('en-US'),X+250,190);
 x.fillStyle='#bbada0';rr(X,210,500,440,16);
 for(let k=0;k<16;k++){const v=parseInt(h[15-k],16),cx=X+12+(k%4)*121,cy=222+Math.floor(k/4)*104;
  x.fillStyle=v?COL[Math.min(v-1,11)]:'#cdc1b4';rr(cx,cy,113,96,8);
  if(v){const s=String(1<<v);x.fillStyle=v<=2?'#776e65':'#f9f6f2';x.font='bold '+(s.length>=5?30:s.length==4?36:44)+'px Helvetica';x.fillText(s,cx+56,cy+63)}}
 x.font='20px Helvetica';x.fillStyle=i>=n-1?'#b91c1c':'#6b7280';
 x.fillText(i>=n-1?'GAME OVER · '+(n-1).toLocaleString('en-US')+' moves':'move '+i.toLocaleString('en-US'),X+250,690)}
window.frame=(t,L,R,step)=>{x.fillStyle='#f7f6f3';x.fillRect(0,0,1280,720);x.fillStyle='#1c1f24';x.textAlign='center';
 x.font='bold 34px Helvetica';x.fillText(t,640,70);x.font='20px Helvetica';x.fillStyle='#6b7280';
 x.fillText('n-tuple TD(0) value net + expectimax · replay of logged trajectories',640,105);board(L,step,80);board(R,step,700)}
</script></body></html>`;
(async()=>{const out=process.argv[2];fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'player.html'),html);
 const b=await chromium.launch();const p=await b.newPage({viewport:{width:1280,height:720}});
 await p.goto('file://'+path.resolve(out,'player.html'));let f=0;
 for(const s of SEG){const nf=s.sec*FPS,nh=s.hold*FPS;
  for(let j=0;j<nf+nh;j++){const u=Math.min(1,j/(nf-1));await p.evaluate(a=>frame(...a),[s.title,s.L,s.R,s.map(u)]);
   await p.screenshot({path:path.join(out,`f${String(f++).padStart(5,'0')}.png`)})}}
 await b.close();console.log('frames',f,'K16',K16,'N5',N5)})();
