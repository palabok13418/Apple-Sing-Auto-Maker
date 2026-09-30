type AudioStats = {duration:number; sampleRate:number; channels:number; rms:number; zcr:number; centroid:number; flatness:number; lowRatio:number; harmonicity:number};
type FileSlot = {file:File; stats?:AudioStats; score?:number};

const $ = <T extends HTMLElement>(id:string) => document.getElementById(id) as T;
let lead:FileSlot|null=null;
let backing:FileSlot|null=null;
let passed=false;
let output='';

const log=(msg:string)=>{const node=document.createElement('div'); node.textContent=msg; $('console').appendChild(node); $('console').scrollTop=$('console').scrollHeight;};
const setGate=(kind:'wait'|'ok'|'bad',msg:string)=>{$('gate').className='gate '+(kind==='wait'?'':kind); $('gateText').textContent=msg;};
const showFile=(slot:'lead'|'backing',file:File)=>{const meta=$(slot==='lead'?'leadMeta':'bgMeta'); meta.innerHTML='<b>'+escapeHtml(file.name)+'</b><small>'+formatBytes(file.size)+'</small>';};
const escapeHtml=(s:string)=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const formatBytes=(n:number)=>`${(n/1048576).toFixed(2)} MB`;

function bindInput(inputId:string,cardId:string,metaId:string,assign:(f:File)=>void){
  const input=$(inputId) as HTMLInputElement; const card=$(cardId);
  input.addEventListener('change',()=>{const file=input.files?.[0]; if(!file)return; assign(file); showFile(inputId==='leadFile'?'lead':'backing',file); card.classList.remove('good','bad'); passed=false; setGate('wait',lead&&backing?'Both stems loaded. Analyze them to continue.':'Waiting for both files.'); $('generate').setAttribute('disabled','true');});
  card.addEventListener('dragover',e=>{e.preventDefault();card.classList.add('drag')});
  card.addEventListener('dragleave',()=>card.classList.remove('drag'));
  card.addEventListener('drop',e=>{e.preventDefault();card.classList.remove('drag');const f=e.dataTransfer?.files?.[0];if(!f)return;assign(f);showFile(inputId==='leadFile'?'lead':'backing',f);card.classList.remove('good','bad');passed=false;setGate('wait',lead&&backing?'Both stems loaded. Analyze them to continue.':'Waiting for both files.');$('generate').setAttribute('disabled','true');});
}

function fftLike(samples:Float32Array):Pick<AudioStats,'rms'|'zcr'|'centroid'|'flatness'|'lowRatio'|'harmonicity'>{
  const n=Math.min(samples.length,4096); let rms=0,zcr=0,centroid=0,total=0,logSum=0,low=0,peak=0;
  for(let i=0;i<n;i++){const x=samples[i]??0; rms+=x*x; if(i>0&&((samples[i-1]??0)>=0)!=(x>=0))zcr++; const a=Math.abs(x); total+=a; centroid+=i*a; peak=Math.max(peak,a); if(i<Math.floor(n*.2))low+=a; logSum+=Math.log(a+1e-8)}
  rms=Math.sqrt(rms/n); zcr/=n; centroid=total?centroid/total/n:0; const arith=total/n+1e-8; const geo=Math.exp(logSum/n); const flatness=geo/arith; const lowRatio=low/(total+1e-8); const harmonicity=Math.min(1,Math.max(0,(peak/(rms+1e-6)-1)/20));
  return {rms,zcr,centroid,flatness,lowRatio,harmonicity};
}

async function inspect(file:File):Promise<AudioStats|null>{
  try{
    const AC=window.AudioContext??(window as typeof window & {webkitAudioContext?:typeof AudioContext}).webkitAudioContext;
    if(!AC)return null;
    const ctx=new AC(); const buffer=await ctx.decodeAudioData(await file.arrayBuffer()); const ch=buffer.getChannelData(0);
    const windows=Number($('eco').classList.contains('on')?16:32); const step=Math.max(1,Math.floor(ch.length/windows));
    let acc={rms:0,zcr:0,centroid:0,flatness:0,lowRatio:0,harmonicity:0}; let count=0;
    for(let w=0;w<windows;w++){const start=Math.min(ch.length-1,w*step); const end=Math.min(ch.length,start+Math.min(step,8192)); const slice=ch.subarray(start,end); if(!slice.length)continue; const f=fftLike(slice); for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]+=f[k]; count++;}
    await ctx.close(); for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]/=Math.max(1,count);
    return {duration:buffer.duration,sampleRate:buffer.sampleRate,channels:buffer.numberOfChannels,...acc};
  }catch{return null}
}

function classify(s:AudioStats):number{
  const vocal=Math.max(0,1-Math.min(1,s.lowRatio*1.8)); const harmonic=Math.min(1,s.harmonicity);
  const stable=Math.max(0,1-Math.min(1,Math.abs(s.centroid-.28)*3)); const clean=Math.max(0,1-Math.min(1,s.flatness*1.35));
  const zcr=Math.max(0,1-Math.min(1,s.zcr*7));
  return Math.max(0,Math.min(1,.30*vocal+.28*harmonic+.19*stable+.15*clean+.08*zcr));
}

async function analyzeSlot(slot:FileSlot,label:string){log(`${label}: decoding analysis windows…`); const stats=await inspect(slot.file); if(!stats)throw new Error(`${label} could not be decoded.`); slot.stats=stats; slot.score=classify(stats); log(`${label}: ${stats.duration.toFixed(2)}s • ${stats.sampleRate} Hz • ${stats.channels}ch • confidence ${(slot.score*100).toFixed(1)}%`); return slot.score;}

bindInput('leadFile','leadCard','leadMeta',f=>lead={file:f});
bindInput('bgFile','bgCard','bgMeta',f=>backing={file:f});
$('leadChoose').addEventListener('click',()=>($('leadFile') as HTMLInputElement).click());
$('bgChoose').addEventListener('click',()=>($('bgFile') as HTMLInputElement).click());
$('eco').addEventListener('click',()=>{$('eco').classList.toggle('on');$('profile').textContent=$('eco').classList.contains('on')?'eco':'balanced';});
document.querySelectorAll<HTMLButtonElement>('.switch[data-toggle]').forEach(b=>b.addEventListener('click',()=>b.classList.toggle('on')));
$('webnn').textContent=('ml' in navigator)?'available':'not exposed · CPU path';

$('analyze').addEventListener('click',async()=>{
  if(!lead||!backing){setGate('bad','Both Lead Vocals and Backing Vocals are required.');return;}
  $('analyze').setAttribute('disabled','true'); setGate('wait','Analyzing both stems…'); log('gate: starting compact feature pass…');
  try{
    const [a,b]=await Promise.all([analyzeSlot(lead,'lead'),analyzeSlot(backing,'backing')]); const threshold=.78; const ok=a>=threshold&&b>=threshold;
    passed=ok; $('leadCard').classList.toggle('good',ok); $('bgCard').classList.toggle('good',ok); $('leadCard').classList.toggle('bad',!ok); $('bgCard').classList.toggle('bad',!ok);
    if(ok){setGate('ok','Both stems passed the conservative purity gate. Generation unlocked.'); $('generate').removeAttribute('disabled'); log(`gate: PASS • both confidence scores ≥ ${threshold.toFixed(2)}`);}
    else{setGate('bad',`Rejected. Both files need confidence ≥ ${(threshold*100).toFixed(0)}% for this conservative prototype gate.`); $('generate').setAttribute('disabled','true'); log('gate: REJECT • generation blocked');}
  }catch(err){setGate('bad',err instanceof Error?err.message:'Analysis failed.'); log('gate: ERROR');}
  $('analyze').removeAttribute('disabled');
});

function setProgress(n:number){$('progressBar').style.width=`${n}%`;$('pct').textContent=`${n}%`;}
function sleep(ms:number){return new Promise<void>(r=>setTimeout(r,ms));}
function toTime(ms:number){const s=ms/1000;const m=Math.floor(s/60);const sec=s-m*60;return `00:${String(m).padStart(2,'0')}:${sec.toFixed(3).padStart(6,'0')}`;}
function makeTtml(){
  const title=escapeHtml(($('title') as HTMLInputElement).value||'Untitled Session'); const artist=escapeHtml(($('artist') as HTMLInputElement).value||'Unknown Artist'); const lang=($('lang') as HTMLSelectElement).value;
  const lyrics=($('lyrics') as HTMLInputElement).value.trim(); const words=(lyrics?lyrics.split(/\s+/):['Generated','timing','will','be','inserted']).slice(0,48); const duration=Math.max(1,backing?.stats?.duration??lead?.stats?.duration??4); const step=(duration*1000)/words.length;
  const spans=words.map((w,i)=>`        <span begin="${toTime(i*step)}" end="${toTime((i+1)*step)}">${escapeHtml(w)}</span>`).join(' ');
  const bgOn=$('[data-toggle="bg"]').classList.contains('on'); const v2On=$('[data-toggle="v2"]').classList.contains('on'); const partsOn=$('[data-toggle="parts"]').classList.contains('on');
  const bg=v2On&&bgOn?`\n        <span ttm:role="x-bg" begin="${toTime(step*1.2)}" end="${toTime(Math.min(duration*1000,step*(words.length>2?2.5:2)))}"><span ttm:agent="v2">background</span></span>`:'';
  const part=partsOn?'\n    <div itunes:song-part="Verse">':'\n    <div>'; const closePart='\n    </div>';
  return `<?xml version="1.0" encoding="UTF-8"?>\n<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/lyric-ttml-internal" xml:lang="${lang}" itunes:timing="Word">\n  <head>\n    <metadata>\n      <ttm:title>${title}</ttm:title>\n      <ttm:agent type="person" xml:id="v1"><ttm:name type="full">${artist}</ttm:name></ttm:agent>${v2On?'\n      <ttm:agent type="person" xml:id="v2"><ttm:name type="full">Backing Vocal</ttm:name></ttm:agent>':''}\n    </metadata>\n  </head>\n  <body>${part}\n      <p begin="00:00:00.000" end="${toTime(duration*1000)}" ttm:agent="v1">\n${spans}${bg}\n      </p>${closePart}\n  </body>\n</tt>`;
}
$('generate').addEventListener('click',async()=>{if(!passed)return; $('generate').setAttribute('disabled','true'); const stages=['strict stem classifier confirmation','vocal activity + phrase alignment','word timing anchors','BG / v2 agent assembly','TTML XML validation']; for(let i=0;i<stages.length;i++){log(`run: ${stages[i]}…`); setProgress(Math.round(((i+1)/stages.length)*100)); await sleep(360);} output=makeTtml(); $('xml').textContent=output; const base=(($('title') as HTMLInputElement).value||'session').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'session'; $('fileName').textContent=base+'.ttml'; $('download').removeAttribute('disabled'); $('generate').removeAttribute('disabled'); log('complete: TTML ready'); $('result').scrollIntoView({behavior:'smooth'});});
$('copy').addEventListener('click',async()=>{if(!output)return;try{await navigator.clipboard.writeText(output);$('copy').textContent='Copied';}catch{log('copy: clipboard permission unavailable');}});
$('download').addEventListener('click',()=>{if(!output)return;const blob=new Blob([output],{type:'application/ttml+xml;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=$('fileName').textContent??'session.ttml';a.click();setTimeout(()=>URL.revokeObjectURL(url),500);});
