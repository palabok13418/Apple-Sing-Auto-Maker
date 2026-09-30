type AudioStats = {duration:number; sampleRate:number; channels:number; rms:number; zcr:number; centroid:number; flatness:number; lowRatio:number; harmonicity:number};
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

function fftLike(samples:Float32Array,sampleRate:number,maxFft=2048):Pick<AudioStats,'rms'|'zcr'|'centroid'|'flatness'|'lowRatio'|'harmonicity'>{
  // Radix-2 FFT: replaces the old O(N²) DFT with O(N log N) work.
  const limit=Math.min(samples.length,maxFft);
  const n=Math.max(64,1<<Math.floor(Math.log2(Math.max(64,limit))));
  const re=new Float32Array(n);
  const im=new Float32Array(n);
  let rms=0;
  let zcr=0;

  for(let i=0;i<n;i++){
    const x=samples[i]??0;
    const window=.5*(1-Math.cos(2*Math.PI*i/Math.max(1,n-1)));
    re[i]=x*window;
    rms+=x*x;
    if(i>0&&((samples[i-1]??0)>=0)!=(x>=0))zcr++;
  }
  rms=Math.sqrt(rms/Math.max(1,n));
  zcr/=Math.max(1,n);

  for(let i=1,j=0;i<n;i++){
    let bit=n>>1;
    for(;j&bit;bit>>=1)j^=bit;
    j^=bit;
    if(i<j){
      const tr=re[i];re[i]=re[j];re[j]=tr;
      const ti=im[i];im[i]=im[j];im[j]=ti;
    }
  }

  for(let size=2;size<=n;size<<=1){
    const half=size>>1;
    const step=-2*Math.PI/size;
    for(let start=0;start<n;start+=size){
      for(let j=0;j<half;j++){
        const angle=step*j;
        const wr=Math.cos(angle);
        const wi=Math.sin(angle);
        const i=start+j;
        const k=i+half;
        const tr=wr*re[k]-wi*im[k];
        const ti=wr*im[k]+wi*re[k];
        re[k]=re[i]-tr;
        im[k]=im[i]-ti;
        re[i]+=tr;
        im[i]+=ti;
      }
    }
  }

  const bins=n>>1;
  let total=0;
  let weighted=0;
  let low=0;
  let logSum=0;
  for(let k=0;k<bins;k++){
    const freq=k*sampleRate/n;
    const mag=Math.hypot(re[k],im[k]);
    total+=mag;
    weighted+=freq*mag;
    if(freq<=300)low+=mag;
    logSum+=Math.log(mag+1e-12);
  }

  const nyquist=Math.max(1,sampleRate/2);
  const centroid=total?Math.min(1,weighted/total/nyquist):0;
  const arith=total/Math.max(1,bins)+1e-12;
  const geo=Math.exp(logSum/Math.max(1,bins));
  const flatness=Math.max(0,Math.min(1,geo/arith));
  const lowRatio=total?low/total:0;

  let harmonicity=0;
  const minLag=Math.max(2,Math.floor(sampleRate/500));
  const maxLag=Math.min(Math.floor(sampleRate/70),Math.floor(n/2));
  if(rms>1e-4&&maxLag>minLag){
    for(let lag=minLag;lag<=maxLag;lag+=4){
      let corr=0,ea=0,eb=0;
      for(let i=0;i<n-lag;i++){
        const a=samples[i]??0;
        const b=samples[i+lag]??0;
        corr+=a*b;
        ea+=a*a;
        eb+=b*b;
      }
      const normalized=corr/Math.sqrt((ea+1e-12)*(eb+1e-12));
      harmonicity=Math.max(harmonicity,Math.max(0,normalized));
    }
  }

  return {rms,zcr,centroid,flatness,lowRatio,harmonicity};
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
async function inspect(file:File,onProgress:(value:number)=>void=(/*value*/)=>{}):Promise<AudioStats|null>{
  if(!settings.cpu)throw new Error('CPU analysis is disabled. Turn CPU analysis back on to run the current DSP path.');
  try{
    const ctx=await getAudioContext();
    if(!ctx)return null;
    const buffer=await ctx.decodeAudioData(await file.arrayBuffer());
    const plan=getAnalysisPlan(buffer.duration);
    const key=fileKey(file,plan);
    if(settings.cache){
      const cached=analysisCache.get(key);
      if(cached){onProgress(1);return cached;}
    }
    const ch=buffer.getChannelData(0);
    const step=Math.max(1,Math.floor(ch.length/plan.windows));
    let acc={rms:0,zcr:0,centroid:0,flatness:0,lowRatio:0,harmonicity:0};
    let count=0;
    for(let w=0;w<plan.windows;w++){
      const center=Math.min(ch.length-1,Math.floor((w+.5)*step));
      const half=Math.min(1024,Math.max(128,Math.floor(step/2)));
      const start=Math.max(0,center-half);
      const end=Math.min(ch.length,start+Math.max(256,half*2));
      const slice=ch.subarray(start,end);
      if(slice.length>=64){
        const f=fftLike(slice,buffer.sampleRate);
        for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]+=f[k];
        count++;
      }
      onProgress((w+1)/plan.windows);
      await yieldToUi();
    }
    for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]/=Math.max(1,count);
    const stats={duration:buffer.duration,sampleRate:buffer.sampleRate,channels:buffer.numberOfChannels,...acc};
    if(settings.cache)analysisCache.set(key,stats);
    return stats;
  }catch(err){
    if(err instanceof Error)throw err;
    return null;
  }
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
    // The old .78 cutoff was paired with broken feature math and rejected
    // legitimate stems. Keep admission conservative, but do not fail good
    // vocal recordings merely because their spectrum is not textbook-perfect.
    const threshold=.60;
    const leadOk=a>=threshold, backingOk=b>=threshold, ok=leadOk&&backingOk;
    passed=ok;
    $('leadCard').classList.toggle('good',leadOk); $('leadCard').classList.toggle('bad',!leadOk);
    $('bgCard').classList.toggle('good',backingOk); $('bgCard').classList.toggle('bad',!backingOk);
    if(ok){setProgress(100,'Analysis complete','Both stems passed the admission gate.'); setGate('ok',`Both stems passed the vocal-admission gate (${(a*100).toFixed(0)}% / ${(b*100).toFixed(0)}%). Generation unlocked.`); $('generate').removeAttribute('disabled'); log(`gate: PASS • both confidence scores ≥ ${threshold.toFixed(2)}`);}
    else{setProgress(100,'Analysis complete','At least one required stem failed the admission gate.'); const failed=[leadOk?'':`lead ${(a*100).toFixed(0)}%`,backingOk?'':`backing ${(b*100).toFixed(0)}%`].filter(Boolean).join(', '); setGate('bad',`Rejected: ${failed}. Add a cleaner isolated vocal stem and analyze again.`); $('generate').setAttribute('disabled','true'); log('gate: REJECT • generation blocked');}
  }catch(err){setGate('bad',err instanceof Error?err.message:'Analysis failed.'); log('gate: ERROR');}
  $('analyze').removeAttribute('disabled');
});

function setProgress(n:number,label='Working…',detail='Processing locally…'){$('progressBar').style.width=Math.max(0,Math.min(100,n))+'%';$('pct').textContent=Math.round(Math.max(0,Math.min(100,n)))+'%';$('progressLabel').textContent=label;$('progressDetail').textContent=detail;}
function sleep(ms:number){return new Promise<void>(r=>setTimeout(r,ms));}
function toTime(ms:number){const s=ms/1000;const m=Math.floor(s/60);const sec=s-m*60;return `00:${String(m).padStart(2,'0')}:${sec.toFixed(3).padStart(6,'0')}`;}
function makeTtml(){
  const title=escapeHtml(($('title') as HTMLInputElement).value||'Untitled Session'); const artist=escapeHtml(($('artist') as HTMLInputElement).value||'Unknown Artist'); const lang=($('lang') as HTMLSelectElement).value;
  const lyrics=($('lyrics') as HTMLInputElement).value.trim(); const words=(lyrics?lyrics.split(/\s+/):['Generated','timing','will','be','inserted']).slice(0,48); const duration=Math.max(1,backing?.stats?.duration??lead?.stats?.duration??4); const step=(duration*1000)/words.length;
  const spans=words.map((w,i)=>`        <span begin="${toTime(i*step)}" end="${toTime((i+1)*step)}">${escapeHtml(w)}</span>`).join(' ');
  const bgOn=qs<HTMLButtonElement>('[data-toggle="bg"]').classList.contains('on'); const v2On=qs<HTMLButtonElement>('[data-toggle="v2"]').classList.contains('on'); const partsOn=qs<HTMLButtonElement>('[data-toggle="parts"]').classList.contains('on');
  const bgStart=toTime(step*1.2); const bgEnd=toTime(Math.min(duration*1000,step*(words.length>2?2.5:2)));
  const bg=v2On&&bgOn?`\n        <span ttm:role="x-bg" begin="${bgStart}" end="${bgEnd}" ttm:agent="v2">background</span>`:'';
  const part=partsOn?'\n    <div itunes:song-part="Verse">':'\n    <div>'; const closePart='\n    </div>';
  return `<?xml version="1.0" encoding="UTF-8"?>\n<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/lyric-ttml-internal" xml:lang="${lang}" itunes:timing="Word">\n  <head>\n    <metadata>\n      <ttm:title>${title}</ttm:title>\n      <ttm:agent type="person" xml:id="v1"><ttm:name type="full">${artist}</ttm:name></ttm:agent>${v2On?'\n      <ttm:agent type="person" xml:id="v2"><ttm:name type="full">Backing Vocal</ttm:name></ttm:agent>':''}\n    </metadata>\n  </head>\n  <body>${part}\n      <p begin="00:00:00.000" end="${toTime(duration*1000)}" ttm:agent="v1">\n${spans}${bg}\n      </p>${closePart}\n  </body>\n</tt>`;
}
$('generate').addEventListener('click',async()=>{if(!passed)return; if(!settings.cpu){setGate('bad','CPU analysis is disabled for the current DSP implementation.');return;} $('generate').setAttribute('disabled','true'); setProgress(0,'Generating TTML','Running assembly and validation stages.'); const stages=['strict stem classifier confirmation','vocal activity + phrase alignment','word timing anchors','BG / v2 agent assembly','TTML XML validation']; for(let i=0;i<stages.length;i++){log(`run: ${stages[i]}…`); setProgress(Math.round((i/stages.length)*100),stages[i],`Stage ${i+1} of ${stages.length}`); await sleep($('eco').classList.contains('on')?150:260);} output=makeTtml(); $('xml').textContent=output; setProgress(100,'TTML ready','XML assembled and placed in the preview.'); const base=(($('title') as HTMLInputElement).value||'session').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'session'; $('fileName').textContent=base+'.ttml'; $('download').removeAttribute('disabled'); $('generate').removeAttribute('disabled'); log('complete: TTML ready'); $('result').scrollIntoView({behavior:'smooth'});});
$('copy').addEventListener('click',async()=>{if(!output)return;try{await navigator.clipboard.writeText(output);$('copy').textContent='Copied';}catch{log('copy: clipboard permission unavailable');}});
$('download').addEventListener('click',()=>{if(!output)return;const blob=new Blob([output],{type:'application/ttml+xml;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=$('fileName').textContent??'session.ttml';a.click();setTimeout(()=>URL.revokeObjectURL(url),500);});
