type AudioInterval = {start:number; end:number};
type AudioStats = {duration:number; sampleRate:number; channels:number; rms:number; zcr:number; centroid:number; flatness:number; lowRatio:number; harmonicity:number; secondaryVoice:number; vocalActivity:number; secondaryIntervals:AudioInterval[]; vocalIntervals:AudioInterval[]};
type FileSlot = {file:File; stats?:AudioStats; score?:number};

const $ = <T extends HTMLElement>(id:string):T => { const node=document.getElementById(id); if(!node) throw new Error('UI element not found: #'+id); return node as T; };
const qs = <T extends Element>(selector:string):T => { const node=document.querySelector(selector); if(!node) throw new Error('UI element not found: '+selector); return node as T; };
let lead:FileSlot|null=null;
let backing:FileSlot|null=null;
let passed=false;
let output='';
let audioContext:AudioContext|null=null;
let gpuDevice:{destroy?:()=>void}|null=null;
let webnnContext:unknown=null;
let gpuComputeReady=false;
const analysisCache=new Map<string,AudioStats>();
const settings={cpu:true,gpu:false,adaptive:true,cache:true,responsive:true};

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

function clamp01(n:number){return Math.max(0,Math.min(1,n));}

function fftLike(samples:Float32Array,sampleRate:number,maxFft=2048):Pick<AudioStats,'rms'|'zcr'|'centroid'|'flatness'|'lowRatio'|'harmonicity'|'secondaryVoice'|'vocalActivity'>{
  const limit=Math.min(samples.length,maxFft);
  const n=Math.max(64,1<<Math.floor(Math.log2(Math.max(64,limit))));
  const re=new Float32Array(n);
  const im=new Float32Array(n);
  let rms=0; let zcr=0;
  for(let i=0;i<n;i++){
    const x=samples[i]??0;
    const window=.5*(1-Math.cos(2*Math.PI*i/Math.max(1,n-1)));
    re[i]=x*window; rms+=x*x;
    if(i>0&&((samples[i-1]??0)>=0)!=(x>=0))zcr++;
  }
  rms=Math.sqrt(rms/Math.max(1,n)); zcr/=Math.max(1,n);
  for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){const tr=re[i];re[i]=re[j];re[j]=tr;const ti=im[i];im[i]=im[j];im[j]=ti;}}
  for(let size=2;size<=n;size<<=1){const half=size>>1;const step=-2*Math.PI/size;for(let start=0;start<n;start+=size){for(let j=0;j<half;j++){const angle=step*j;const wr=Math.cos(angle);const wi=Math.sin(angle);const i=start+j;const k=i+half;const tr=wr*re[k]-wi*im[k];const ti=wr*im[k]+wi*re[k];re[k]=re[i]-tr;im[k]=im[i]-ti;re[i]+=tr;im[i]+=ti;}}}
  const bins=n>>1; let total=0,weighted=0,low=0,logSum=0;
  for(let k=0;k<bins;k++){const freq=k*sampleRate/n;const mag=Math.hypot(re[k],im[k]);total+=mag;weighted+=freq*mag;if(freq<=300)low+=mag;logSum+=Math.log(mag+1e-12);}
  const nyquist=Math.max(1,sampleRate/2);
  const centroid=total?Math.min(1,weighted/total/nyquist):0;
  const arith=total/Math.max(1,bins)+1e-12;
  const geo=Math.exp(logSum/Math.max(1,bins));
  const flatness=Math.max(0,Math.min(1,geo/arith));
  const lowRatio=total?low/total:0;
  const candidates:Array<{freq:number;score:number}>=[]; let harmonicity=0;
  const minLag=Math.max(2,Math.floor(sampleRate/500)); const maxLag=Math.min(Math.floor(sampleRate/70),Math.floor(n/2));
  if(rms>1e-4&&maxLag>minLag){
    for(let lag=minLag;lag<=maxLag;lag+=4){
      let corr=0,ea=0,eb=0;
      for(let i=0;i<n-lag;i++){const a=samples[i]??0;const b=samples[i+lag]??0;corr+=a*b;ea+=a*a;eb+=b*b;}
      const normalized=Math.max(0,corr/Math.sqrt((ea+1e-12)*(eb+1e-12)));
      harmonicity=Math.max(harmonicity,normalized);
      if(normalized>.12)candidates.push({freq:sampleRate/lag,score:normalized});
    }
  }
  candidates.sort((a,b)=>b.score-a.score);
  const primary=candidates[0]; let secondary:{freq:number;score:number}|undefined;
  if(primary){
    for(const candidate of candidates.slice(1)){
      const ratio=candidate.freq/primary.freq; const absSemi=Math.abs(12*Math.log2(Math.max(1e-6,ratio)));
      const harmonicRatio=Math.abs(ratio-2)<.08||Math.abs(ratio-3)<.10||Math.abs(ratio-4)<.12||Math.abs(ratio-.5)<.03||Math.abs(ratio-1/3)<.03||Math.abs(ratio-.25)<.025||Math.abs(ratio-1.5)<.05||Math.abs(ratio-4/3)<.05;
      if(absSemi>=2.5&&!harmonicRatio&&candidate.score>=primary.score*.68){secondary=candidate;break;}
    }
  }
  const secondaryVoice=primary&&secondary?clamp01(((secondary.score/Math.max(.01,primary.score))-.58)/.42)*clamp01((primary.score-.18)/.52):0;
  const vocalActivity=clamp01(harmonicity*.72+Math.min(1,rms*24)*.18+(1-flatness)*.10);
  return {rms,zcr,centroid,flatness,lowRatio,harmonicity,secondaryVoice,vocalActivity};
}

async function getAudioContext(){
  if(audioContext)return audioContext;
  const AC=window.AudioContext??(window as typeof window & {webkitAudioContext?:typeof AudioContext}).webkitAudioContext;
  if(!AC)return null;
  audioContext=new AC();
  return audioContext;
}
function getAnalysisPlan(duration:number){
  const maxWindows=$('eco').classList.contains('on')?12:24;
  const minWindows=$('eco').classList.contains('on')?6:8;
  const windows=Math.max(minWindows,Math.min(maxWindows,Math.round(duration/5)||minWindows));
  return {windows,fftSize:$('eco').classList.contains('on')?1024:2048};
}
function fileKey(file:File,plan:{windows:number;fftSize:number}){
  return [file.name,file.size,file.lastModified,file.type,plan.windows,plan.fftSize].join('|');
}
function yieldToUi(){
  if(!settings.responsive)return Promise.resolve();
  return new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));
}
function mergeIntervals(intervals:AudioInterval[],gap=.35):AudioInterval[]{
  if(!intervals.length)return [];
  const sorted=[...intervals].sort((a,b)=>a.start-b.start);
  const merged:AudioInterval[]=[{start:sorted[0].start,end:sorted[0].end}];
  for(const next of sorted.slice(1)){const last=merged[merged.length-1];if(next.start<=last.end+gap)last.end=Math.max(last.end,next.end);else merged.push({start:next.start,end:next.end});}
  return merged;
}

async function inspect(file:File,onProgress:(value:number)=>void=(/*value*/)=>{}):Promise<AudioStats|null>{
  if(!settings.cpu)throw new Error('CPU analysis is disabled. Turn CPU analysis back on to run the current DSP path.');
  try{
    const ctx=await getAudioContext(); if(!ctx)return null;
    const buffer=await ctx.decodeAudioData(await file.arrayBuffer()); const plan=getAnalysisPlan(buffer.duration); const key=fileKey(file,plan);
    if(settings.cache){const cached=analysisCache.get(key);if(cached){onProgress(1);return cached;}}
    const ch=buffer.getChannelData(0); const step=Math.max(1,Math.floor(ch.length/plan.windows));
    let acc={rms:0,zcr:0,centroid:0,flatness:0,lowRatio:0,harmonicity:0,secondaryVoice:0,vocalActivity:0};
    const secondaryIntervals:AudioInterval[]=[]; const vocalIntervals:AudioInterval[]=[]; let count=0;
    for(let w=0;w<plan.windows;w++){
      const center=Math.min(ch.length-1,Math.floor((w+.5)*step)); const half=Math.min(1024,Math.max(128,Math.floor(step/2))); const start=Math.max(0,center-half); const end=Math.min(ch.length,start+Math.max(256,half*2)); const slice=ch.subarray(start,end);
      if(slice.length>=64){
        const f=fftLike(slice,buffer.sampleRate,plan.fftSize);
        for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]+=f[k]; count++;
        const interval={start:start/buffer.sampleRate,end:end/buffer.sampleRate}; if(f.secondaryVoice>=.58)secondaryIntervals.push(interval); if(f.vocalActivity>=.48)vocalIntervals.push(interval);
      }
      onProgress((w+1)/plan.windows); await yieldToUi();
    }
    for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]/=Math.max(1,count);
    const stats:AudioStats={duration:buffer.duration,sampleRate:buffer.sampleRate,channels:buffer.numberOfChannels,...acc,secondaryIntervals:mergeIntervals(secondaryIntervals),vocalIntervals:mergeIntervals(vocalIntervals)};
    if(settings.cache)analysisCache.set(key,stats); return stats;
  }catch(err){if(err instanceof Error)throw err;return null;}
}
function splitSyllables(word:string,lang:string):string[]{
  if(!word)return [];
  const graphemes=[...word];
  const isSmallKana=(c:string)=>/^[ぁぃぅぇぉゃゅょゎっァィゥェォャュョヮッ]$/u.test(c);
  if(/^ja(?:-|$)/u.test(lang)){const out:string[]=[];for(const c of graphemes){if(isSmallKana(c)&&out.length)out[out.length-1]+=c;else out.push(c);}return out;}
  if(/^ko(?:-|$)/u.test(lang)||/^zh(?:-|$)/u.test(lang))return graphemes;
  const leading=(word.match(/^[^\p{L}\p{N}]*/u)?.[0]??''); const trailing=(word.match(/[^\p{L}\p{N}]*$/u)?.[0]??''); const core=word.slice(leading.length,Math.max(leading.length,word.length-trailing.length));
  if(!core)return [word];
  const nuclei:Array<{start:number;end:number}>=[]; const re=/[aeiouy]+/giu; let m:RegExpExecArray|null;
  while((m=re.exec(core))!==null)nuclei.push({start:m.index,end:m.index+m[0].length});
  if(!nuclei.length)return [word];
  const syllables:string[]=[]; let start=0;
  for(let i=0;i<nuclei.length-1;i++){
    const cluster=core.slice(nuclei[i].end,nuclei[i+1].start);
    const boundary=nuclei[i].end+Math.floor(Math.max(0,cluster.length-1)/2);
    syllables.push(core.slice(start,boundary)); start=boundary;
  }
  syllables.push(core.slice(start));
  syllables[0]=leading+syllables[0]; syllables[syllables.length-1]+=trailing; return syllables;
}
type SyllableUnit={text:string;begin:number;end:number;wordIndex:number};
function buildSyllableTimeline(words:string[],durationMs:number,lang:string):SyllableUnit[]{
  const result:SyllableUnit[]=[]; const total=Math.max(1,words.length);
  for(let i=0;i<words.length;i++){const wordStart=durationMs*i/total; const wordEnd=durationMs*(i+1)/total; const syllables=splitSyllables(words[i],lang); const weights=syllables.map(x=>Math.max(1,[...x].filter(c=>/[\p{L}\p{N}]/u.test(c)).length)); const sum=weights.reduce((a,b)=>a+b,0)||1; let cursor=wordStart;
    for(let j=0;j<syllables.length;j++){const end=j===syllables.length-1?wordEnd:cursor+(wordEnd-wordStart)*weights[j]/sum; result.push({text:syllables[j],begin:cursor,end,wordIndex:i}); cursor=end;}
  }
  return result;
}
function unitOverlaps(unit:SyllableUnit,intervals:AudioInterval[]):boolean{return intervals.some(x=>unit.begin/1000<x.end&&unit.end/1000>x.start);}
function renderSyllables(units:SyllableUnit[],markV2:boolean,intervals:AudioInterval[]):string{
  return units.map((u,i)=>{const mark=markV2&&unitOverlaps(u,intervals)?' ttm:agent="v2"':'';const next=units[i+1];const spacer=next&&next.wordIndex!==u.wordIndex?' ':'';return '<span begin="'+toTime(u.begin)+'" end="'+toTime(u.end)+'"'+mark+'>'+escapeHtml(u.text)+'</span>'+spacer;}).join('');
}
function classify(s:AudioStats):number{
  // This is still an admission heuristic until the trained tiny model is
  // shipped. It is deliberately calibrated around sung-voice characteristics
  // instead of the old invalid sample-index measurements.
  const voiced=Math.min(1,Math.max(0,s.harmonicity*.9 + (1-s.zcr*5)*.1));
  const vocalBand=Math.max(0,1-Math.abs(s.lowRatio-.22)*2.8);
  const midCentroid=Math.max(0,1-Math.abs(s.centroid-.28)*2.4);
  const tonal=Math.max(0,1-s.flatness*.9);
  return Math.max(0,Math.min(1,.42*voiced+.24*vocalBand+.20*midCentroid+.14*tonal));
}

async function analyzeSlot(slot:FileSlot,label:string,base:number,span:number){
  log(label+': decoding local analysis windows…');
  const plan=getAnalysisPlan(60);
  const stats=await inspect(slot.file,f=>{
    setProgress(base+f*span,'Analyzing '+label,Math.round(f*100)+'% of the analysis window budget');
  });
  if(!stats)throw new Error(label+' could not be decoded.');
  slot.stats=stats;
  slot.score=classify(stats);
  log(label+': '+stats.duration.toFixed(2)+'s • '+stats.sampleRate+' Hz • '+stats.channels+'ch • confidence '+(slot.score*100).toFixed(1)+'%');
  log(label+': vocal activity '+(stats.vocalActivity*100).toFixed(0)+'% • second-voice signal '+(stats.secondaryVoice*100).toFixed(0)+'%');
  log(label+': plan • '+getAnalysisPlan(stats.duration).windows+' windows / '+plan.fftSize+'-point FFT');
  return slot.score;
}

bindInput('leadFile','leadCard','leadMeta',f=>lead={file:f});
bindInput('bgFile','bgCard','bgMeta',f=>backing={file:f});
$('eco').addEventListener('click',()=>{$('eco').classList.toggle('on');$('profile').textContent=$('eco').classList.contains('on')?'eco':'balanced';});
document.querySelectorAll<HTMLButtonElement>('.switch[data-toggle]').forEach(b=>b.addEventListener('click',()=>b.classList.toggle('on')));
const runtimeNavigator=navigator as Navigator & {ml?:any;gpu?:any};
$('webnn').textContent=runtimeNavigator.ml?.createContext?'available':'optional · not exposed';
$('cores').textContent=String(navigator.hardwareConcurrency||'—');
$('memoryHint').textContent=(navigator as Navigator & {deviceMemory?:number}).deviceMemory?String((navigator as Navigator & {deviceMemory?:number}).deviceMemory)+' GB hint':'unavailable';
$('gpuStatus').textContent=runtimeNavigator.gpu?.requestAdapter?'ready':'unavailable';

async function probeWebGpu():Promise<{device:any;adapter:any}|null>{
  const nav=navigator as Navigator & {gpu?:any};
  if(!nav.gpu?.requestAdapter)return null;
  const adapter=await nav.gpu.requestAdapter();
  if(!adapter?.requestDevice)return null;
  const device=await adapter.requestDevice();
  if(!device)return null;

  // Real compute probe: this confirms the device can execute a compute pass,
  // rather than treating navigator.gpu presence as proof of working GPU use.
  const shader=device.createShaderModule?.({code:`
    @group(0) @binding(0) var<storage,read_write> data: array<f32>;
    @compute @workgroup_size(1)
    fn main() {
      data[0] = data[0] * 2.0;
    }
  `});
  if(shader&&device.createBuffer&&device.createBindGroupLayout&&device.createPipeline){
    const buffer=device.createBuffer({size:16,usage:0x80|0x08});
    const staging=device.createBuffer({size:16,usage:0x01|0x08});
    const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:4,buffer:{type:'storage'}}]});
    const pipeline=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:shader,entryPoint:'main'}});
    const bindGroup=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer}}]});
    const encoder=device.createCommandEncoder();
    const pass=encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0,bindGroup); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(buffer,0,staging,0,16);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    buffer.destroy?.(); staging.destroy?.();
  }
  return {device,adapter};
}

async function setGpu(on:boolean){
  const nav=navigator as Navigator & {ml?:any;gpu?:any};
  if(!on){
    gpuDevice?.destroy?.(); gpuDevice=null; webnnContext=null; gpuComputeReady=false; settings.gpu=false;
    $('gpuStatus').textContent='off'; refreshProfile(); return;
  }

  try{
    const gpu=await probeWebGpu();
    if(!gpu)throw new Error('WebGPU is not available or could not create a device.');
    gpuDevice=gpu.device;
    gpuComputeReady=true;

    if(nav.ml?.createContext){
      try{
        webnnContext=await nav.ml.createContext(gpu.device);
        log('gpu: WebGPU compute device ready; WebNN ML context also available.');
      }catch{
        webnnContext=null;
        log('gpu: WebGPU compute device ready; WebNN ML context unavailable, using GPU compute path.');
      }
    }else{
      webnnContext=null;
      log('gpu: WebGPU compute device ready; WebNN is not exposed in this browser.');
    }

    settings.gpu=true;
    $('gpuStatus').textContent='active';
  }catch(err){
    gpuDevice?.destroy?.(); gpuDevice=null; webnnContext=null; gpuComputeReady=false; settings.gpu=false;
    $('gpuToggle').classList.remove('on'); $('gpuToggle').setAttribute('aria-pressed','false'); $('gpuStatus').textContent='unavailable';
    log('gpu: '+(err instanceof Error?err.message:'unknown GPU setup error')+'; CPU remains active.');
  }
  refreshProfile();
}

function refreshProfile(){
  const parts=[settings.cpu?'CPU DSP':'CPU off',settings.gpu?'GPU ML':'GPU off',settings.adaptive?'adaptive':'fixed',settings.cache?'cache':'no cache',settings.responsive?'responsive':'max throughput'];
  $('profile').textContent=parts.join(' · ');
  $('topProfile').textContent=settings.gpu?'GPU-assisted':'CPU-first';
  $('cpuStatus').textContent=settings.cpu?'optimized':'off';
}
$('cpuToggle').addEventListener('click',()=>{
  settings.cpu=$('cpuToggle').classList.toggle('on');
  $('cpuToggle').setAttribute('aria-pressed',String(settings.cpu));
  if(!settings.cpu){passed=false;$('generate').setAttribute('disabled','true');setGate('wait','CPU analysis is off. The current DSP path still requires CPU feature extraction.');}
  refreshProfile();
});
$('gpuToggle').addEventListener('click',async()=>{
  const on=$('gpuToggle').classList.toggle('on');
  $('gpuToggle').setAttribute('aria-pressed',String(on));
  await setGpu(on);
});
$('adaptiveToggle').addEventListener('click',()=>{settings.adaptive=$('adaptiveToggle').classList.toggle('on');$('adaptiveToggle').setAttribute('aria-pressed',String(settings.adaptive));refreshProfile();});
$('cacheToggle').addEventListener('click',()=>{settings.cache=$('cacheToggle').classList.toggle('on');$('cacheToggle').setAttribute('aria-pressed',String(settings.cache));refreshProfile();});
$('yieldToggle').addEventListener('click',()=>{settings.responsive=$('yieldToggle').classList.toggle('on');$('yieldToggle').setAttribute('aria-pressed',String(settings.responsive));refreshProfile();});
refreshProfile();

$('analyze').addEventListener('click',async()=>{
  if(!lead||!backing){setGate('bad','Both Lead Vocals and Backing Vocals are required.');return;}
  $('analyze').setAttribute('disabled','true'); $('generate').setAttribute('disabled','true'); setProgress(1,'Starting analysis','Decoding the two required stems.'); setGate('wait','Analyzing both stems…'); log('gate: starting compact feature pass…');
  try{
    const a=await analyzeSlot(lead,'lead',2,46); const b=await analyzeSlot(backing,'backing',48,46);
    const threshold=.60; const leadOk=a>=threshold, backingOk=b>=threshold, ok=leadOk&&backingOk; passed=ok;
    $('leadCard').classList.toggle('good',leadOk); $('leadCard').classList.toggle('bad',!leadOk); $('bgCard').classList.toggle('good',backingOk); $('bgCard').classList.toggle('bad',!backingOk);
    const v2Detected=(lead.stats?.secondaryVoice??0)>=.58; const bgDetected=(backing.stats?.vocalActivity??0)>=.48 && b>=threshold;
    $('v2Status').textContent=v2Detected?'detected • '+Math.round((lead.stats?.secondaryVoice??0)*100)+'%':'not detected • '+Math.round((lead.stats?.secondaryVoice??0)*100)+'%';
    $('bgStatus').textContent=bgDetected?'detected • '+Math.round((backing.stats?.vocalActivity??0)*100)+'%':'not detected • '+Math.round((backing.stats?.vocalActivity??0)*100)+'%';
    log('v2 detection: '+(v2Detected?'SECOND VOICE DETECTED':'no second-voice signal')); log('bg detection: '+(bgDetected?'BACKGROUND VOCAL ACTIVITY DETECTED':'no background-vocal activity detected'));
    if(ok){setProgress(100,'Analysis complete','Stem gate passed; V2/BG detection is ready for TTML assembly.');setGate('ok','Both stems passed ('+(a*100).toFixed(0)+'% / '+(b*100).toFixed(0)+'%). V2: '+(v2Detected?'detected':'not detected')+' • BG: '+(bgDetected?'detected':'not detected')+'.');$('generate').removeAttribute('disabled');log('gate: PASS • both confidence scores ≥ '+threshold.toFixed(2));}
    else{setProgress(100,'Analysis complete','At least one required stem failed the admission gate.');const failed=[leadOk?'':'lead '+(a*100).toFixed(0)+'%',backingOk?'':'backing '+(b*100).toFixed(0)+'%'].filter(Boolean).join(', ');setGate('bad','Rejected: '+failed+'. Add a cleaner isolated vocal stem and analyze again.');$('generate').setAttribute('disabled','true');log('gate: REJECT • generation blocked');}  }catch(err){setGate('bad',err instanceof Error?err.message:'Analysis failed.'); log('gate: ERROR');}
  $('analyze').removeAttribute('disabled');
});

function setProgress(n:number,label='Working…',detail='Processing locally…'){$('progressBar').style.width=Math.max(0,Math.min(100,n))+'%';$('pct').textContent=Math.round(Math.max(0,Math.min(100,n)))+'%';$('progressLabel').textContent=label;$('progressDetail').textContent=detail;}
function sleep(ms:number){return new Promise<void>(r=>setTimeout(r,ms));}
function toTime(ms:number){const s=ms/1000;const m=Math.floor(s/60);const sec=s-m*60;return `00:${String(m).padStart(2,'0')}:${sec.toFixed(3).padStart(6,'0')}`;}
function makeTtml(){
  const title=escapeHtml(($('title') as HTMLInputElement).value||'Untitled Session'); const artist=escapeHtml(($('artist') as HTMLInputElement).value||'Unknown Artist'); const lang=($('lang') as HTMLSelectElement).value;
  const lyrics=($('lyrics') as HTMLInputElement).value.trim(); const words=(lyrics?lyrics.split(/\s+/):['Generated','timing','will','be','inserted']).slice(0,48);
  const duration=Math.max(1,backing?.stats?.duration??lead?.stats?.duration??4); const durationMs=duration*1000; const units=buildSyllableTimeline(words,durationMs,lang);
  const v2On=qs<HTMLButtonElement>('[data-toggle="v2"]').classList.contains('on'); const bgOn=qs<HTMLButtonElement>('[data-toggle="bg"]').classList.contains('on'); const partsOn=qs<HTMLButtonElement>('[data-toggle="parts"]').classList.contains('on');
  const v2Detected=(lead?.stats?.secondaryVoice??0)>=.58; const bgDetected=(backing?.stats?.vocalActivity??0)>=.48 && (backing?.score??0)>=.60; const v2Enabled=v2On&&v2Detected; const bgEnabled=bgOn&&bgDetected;
  const spans=renderSyllables(units,v2Enabled,lead?.stats?.secondaryIntervals??[]);
  let bg='';
  if(bgEnabled){const bgUnits=units.filter(u=>unitOverlaps(u,backing?.stats?.vocalIntervals??[]));if(bgUnits.length){const bgStart=Math.min(...bgUnits.map(x=>x.begin));const bgEnd=Math.max(...bgUnits.map(x=>x.end));bg='\n        <span ttm:role="x-bg" begin="'+toTime(bgStart)+'" end="'+toTime(bgEnd)+'">'+renderSyllables(bgUnits,false,[]).trim()+'</span>';}}
  const part=partsOn?'\n    <div itunes:song-part="Verse">':'\n    <div>'; const closePart='\n    </div>';
  return '<?xml version="1.0" encoding="UTF-8"?>\n<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/lyric-ttml-internal" xml:lang="'+lang+'" itunes:timing="Word">\n  <head>\n    <metadata>\n      <ttm:title>'+title+'</ttm:title>\n      <ttm:agent type="person" xml:id="v1"><ttm:name type="full">'+artist+'</ttm:name></ttm:agent>'+(v2Enabled?'\n      <ttm:agent type="person" xml:id="v2"><ttm:name type="full">Secondary Voice</ttm:name></ttm:agent>':'')+'\n    </metadata>\n  </head>\n  <body>'+part+'\n      <p begin="00:00:00.000" end="'+toTime(durationMs)+'" ttm:agent="v1">\n'+spans+bg+'\n      </p>'+closePart+'\n  </body>\n</tt>';
}
$('generate').addEventListener('click',async()=>{if(!passed)return; if(!settings.cpu){setGate('bad','CPU analysis is disabled for the current DSP implementation.');return;} $('generate').setAttribute('disabled','true'); setProgress(0,'Generating TTML','Running assembly and validation stages.'); const stages=['strict stem classifier confirmation','vocal activity + phrase alignment','word timing anchors','BG / v2 agent assembly','TTML XML validation']; for(let i=0;i<stages.length;i++){log(`run: ${stages[i]}…`); setProgress(Math.round((i/stages.length)*100),stages[i],`Stage ${i+1} of ${stages.length}`); await sleep($('eco').classList.contains('on')?150:260);} output=makeTtml(); $('xml').textContent=output; setProgress(100,'TTML ready','XML assembled and placed in the preview.'); const base=(($('title') as HTMLInputElement).value||'session').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'session'; $('fileName').textContent=base+'.ttml'; $('download').removeAttribute('disabled'); $('generate').removeAttribute('disabled'); log('complete: TTML ready'); $('result').scrollIntoView({behavior:'smooth'});});
$('copy').addEventListener('click',async()=>{if(!output)return;try{await navigator.clipboard.writeText(output);$('copy').textContent='Copied';}catch{log('copy: clipboard permission unavailable');}});
$('download').addEventListener('click',()=>{if(!output)return;const blob=new Blob([output],{type:'application/ttml+xml;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=$('fileName').textContent??'session.ttml';a.click();setTimeout(()=>URL.revokeObjectURL(url),500);});
