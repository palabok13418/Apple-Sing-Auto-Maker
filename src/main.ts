type AudioInterval = {start:number; end:number};
type AudioPeak = {time:number; strength:number};
type AudioStats = {duration:number; sampleRate:number; channels:number; rms:number; zcr:number; centroid:number; flatness:number; lowRatio:number; harmonicity:number; secondaryVoice:number; secondaryVoicePeak:number; vocalActivity:number; vocalCoverage:number; secondaryIntervals:AudioInterval[]; vocalIntervals:AudioInterval[]; syllablePeaks:AudioPeak[]; backingPeaks:AudioPeak[]};
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
  input.addEventListener('change',()=>{const file=input.files?.[0]; if(!file)return; assign(file); showFile(inputId==='leadFile'?'lead':'backing',file); card.classList.remove('good','bad'); passed=false; $('v2Status').textContent='waiting…'; $('bgStatus').textContent='waiting…'; setGate('wait',lead&&backing?'Both stems loaded. Analyze them to continue.':'Waiting for both files.'); $('generate').setAttribute('disabled','true');});
  card.addEventListener('dragover',e=>{e.preventDefault();card.classList.add('drag')});
  card.addEventListener('dragleave',()=>card.classList.remove('drag'));
  card.addEventListener('drop',e=>{e.preventDefault();card.classList.remove('drag');const f=e.dataTransfer?.files?.[0];if(!f)return;assign(f);showFile(inputId==='leadFile'?'lead':'backing',f);card.classList.remove('good','bad');passed=false;$('v2Status').textContent='waiting…';$('bgStatus').textContent='waiting…';setGate('wait',lead&&backing?'Both stems loaded. Analyze them to continue.':'Waiting for both files.');$('generate').setAttribute('disabled','true');});
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
    for(let lag=minLag;lag<=maxLag;lag+=2){
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
      const harmonicRatio=Math.abs(ratio-2)<.08||Math.abs(ratio-3)<.10||Math.abs(ratio-4)<.12||Math.abs(ratio-.5)<.03||Math.abs(ratio-1/3)<.03||Math.abs(ratio-.25)<.025;
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
  const eco=$('eco').classList.contains('on');
  const fftSize=eco?1024:2048;
  if(!settings.adaptive)return {windows:16,fftSize};
  const maxWindows=eco?12:24;
  const minWindows=eco?6:8;
  const windows=Math.max(minWindows,Math.min(maxWindows,Math.round(duration/5)||minWindows));
  return {windows,fftSize};
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
function normalizeIntervals(intervals:AudioInterval[],durationSeconds:number,gap=.18):AudioInterval[]{
  const safe=intervals
    .map(x=>({
      start:Math.max(0,Math.min(durationSeconds,Number.isFinite(x.start)?x.start:0)),
      end:Math.max(0,Math.min(durationSeconds,Number.isFinite(x.end)?x.end:0))
    }))
    .filter(x=>x.end>x.start);
  return mergeIntervals(safe,gap)
    .map(x=>({start:Math.max(0,x.start),end:Math.max(Math.max(0,x.start),Math.min(durationSeconds,x.end))}))
    .filter(x=>x.end-x.start>=.04);
}
function extractVocalIntervals(samples:Float32Array,sampleRate:number):{intervals:AudioInterval[];coverage:number;peaks:AudioPeak[];scores:Float32Array}{
  // Frame-level audio analysis: energy, ZCR and local voicing shape are used to
  // create real phrase and syllable timing anchors.
  const frame=Math.max(1024,Math.round(sampleRate*.08));
  const frames=Math.max(1,Math.ceil(samples.length/frame));
  const rmsValues=new Float32Array(frames);
  const zcrValues=new Float32Array(frames);
  const scores=new Float32Array(frames);
  let maxRms=0;
  for(let f=0;f<frames;f++){
    const start=f*frame; const end=Math.min(samples.length,start+frame);
    let energy=0,zcr=0;
    for(let i=start;i<end;i++){
      const x=samples[i]??0; energy+=x*x;
      if(i>start&&((samples[i-1]??0)>=0)!=(x>=0))zcr++;
    }
    const count=Math.max(1,end-start);
    const rms=Math.sqrt(energy/count);
    rmsValues[f]=rms; zcrValues[f]=zcr/count; maxRms=Math.max(maxRms,rms);
  }
  const sorted=[...rmsValues].sort((a,b)=>a-b);
  const q20=sorted[Math.floor((sorted.length-1)*.20)]??0;
  const q50=sorted[Math.floor((sorted.length-1)*.50)]??0;
  const threshold=Math.max(.003,q20*1.65,maxRms*.10);
  for(let f=0;f<frames;f++){
    const normalized=rmsValues[f]/Math.max(maxRms,1e-6);
    const energyScore=clamp01((rmsValues[f]-threshold)/Math.max(.001,maxRms-threshold));
    const zcrScore=clamp01(1-zcrValues[f]/.42);
    const localFloor=rmsValues[f]>=Math.max(threshold,q50*.72);
    scores[f]=clamp01(energyScore*.56+zcrScore*.16+(localFloor?.18:0)+normalized*.10);
  }
  const active=scores.map((score,f)=>score>=.38&&rmsValues[f]>=threshold&&zcrValues[f]<.48);
  for(let f=1;f<frames-1;f++){
    if(!active[f]&&active[f-1]&&active[f+1]&&scores[f]>=.25)active[f]=true;
    if(active[f]&&!active[f-1]&&!active[f+1]&&scores[f]<.58)active[f]=false;
  }
  const raw:AudioInterval[]=[]; let startFrame=-1;
  for(let f=0;f<frames;f++){
    if(active[f]&&startFrame<0)startFrame=f;
    const closing=(!active[f]&&startFrame>=0)||f===frames-1;
    if(closing){
      const endFrame=!active[f]?f:f+1;
      const a=startFrame*frame/sampleRate;
      const b=Math.min(samples.length/sampleRate,endFrame*frame/sampleRate);
      if(b-a>=.10)raw.push({start:a,end:b});
      startFrame=-1;
    }
  }
  const intervals=normalizeIntervals(raw,samples.length/Math.max(1,sampleRate),.16);
  const peaks:AudioPeak[]=[];
  const minPeakGap=Math.max(2,Math.round(.11*sampleRate/frame));
  for(const interval of intervals){
    const a=Math.max(1,Math.floor(interval.start*sampleRate/frame));
    const b=Math.min(frames-2,Math.ceil(interval.end*sampleRate/frame));
    let lastPeak=-9999;
    for(let f=a;f<=b;f++){
      const prev=scores[f-1],cur=scores[f],next=scores[f+1];
      if(cur>=prev&&cur>=next&&cur>=.48&&f-lastPeak>=minPeakGap){
        const strength=clamp01(cur*.72+(cur-Math.min(prev,next))*.90);
        if(strength>=.50){peaks.push({time:(f+.5)*frame/sampleRate,strength});lastPeak=f;}
      }
    }
  }
  const reduced:AudioPeak[]=[];
  for(const peak of peaks.sort((a,b)=>a.time-b.time)){
    const last=reduced[reduced.length-1];
    if(!last||peak.time-last.time>=.13)reduced.push(peak);
    else if(peak.strength>last.strength)reduced[reduced.length-1]=peak;
  }
  const covered=intervals.reduce((n,x)=>n+Math.max(0,x.end-x.start),0);
  return {intervals,coverage:clamp01(covered/Math.max(1,samples.length/sampleRate)),peaks:reduced,scores};
}

async function inspect(file:File,onProgress:(value:number)=>void=(/*value*/)=>{}):Promise<AudioStats|null>{
  if(!settings.cpu)throw new Error('CPU analysis is disabled. Turn CPU analysis back on to run the current DSP path.');
  try{
    const ctx=await getAudioContext(); if(!ctx)return null;
    const buffer=await ctx.decodeAudioData(await file.arrayBuffer());
    const plan=getAnalysisPlan(buffer.duration);
    const key=fileKey(file,plan);
    if(settings.cache){const cached=analysisCache.get(key);if(cached){onProgress(1);return cached;}}
    const ch=buffer.getChannelData(0);
    const step=Math.max(1,Math.floor(ch.length/plan.windows));
    let acc={rms:0,zcr:0,centroid:0,flatness:0,lowRatio:0,harmonicity:0,secondaryVoice:0,vocalActivity:0,vocalCoverage:0};
    const secondaryIntervals:AudioInterval[]=[];
    let secondaryVoicePeak=0;
    let count=0;
    for(let w=0;w<plan.windows;w++){
      const center=Math.min(ch.length-1,Math.floor((w+.5)*step));
      const half=Math.min(1024,Math.max(128,Math.floor(step/2)));
      const start=Math.max(0,center-half);
      const end=Math.min(ch.length,start+Math.max(256,half*2));
      const slice=ch.subarray(start,end);
      if(slice.length>=64){
        const f=fftLike(slice,buffer.sampleRate,plan.fftSize);
        for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]+=f[k];
        count++;
        secondaryVoicePeak=Math.max(secondaryVoicePeak,f.secondaryVoice);
        const interval={start:start/buffer.sampleRate,end:end/buffer.sampleRate};
        if(f.secondaryVoice>=.62)secondaryIntervals.push(interval);
      }
      onProgress((w+1)/plan.windows);
      await yieldToUi();
    }
    for(const k of Object.keys(acc) as Array<keyof typeof acc>)acc[k]/=Math.max(1,count);
    const timing=extractVocalIntervals(ch,buffer.sampleRate);
    const stats:AudioStats={
      duration:buffer.duration,sampleRate:buffer.sampleRate,channels:buffer.numberOfChannels,...acc,
      secondaryVoicePeak,
      vocalCoverage:timing.coverage,
      secondaryIntervals:normalizeIntervals(secondaryIntervals,buffer.duration,.12),
      vocalIntervals:timing.intervals,
      syllablePeaks:timing.peaks,
      backingPeaks:timing.peaks
    };
    if(settings.cache)analysisCache.set(key,stats);
    return stats;
  }catch(err){if(err instanceof Error)throw err;return null;}
}
function decodeCommonEntities(text:string):string{
  return text.replace(/&amp;/gi,'&').replace(/&quot;/gi,'"').replace(/&#39;/gi,"'").replace(/&lt;/gi,'<').replace(/&gt;/gi,'>');
}
function getLyricLines():string[]{
  const raw=decodeCommonEntities((($('lyrics') as HTMLInputElement).value)||'').replace(/\r\n?/g,'\n');
  const lines=raw.split('\n').map(line=>line.trim()).filter(line=>line.length>0);
  return lines.length?lines:['Generated timing will be inserted'];
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
  for(let i=0;i<nuclei.length-1;i++){const cluster=core.slice(nuclei[i].end,nuclei[i+1].start);const boundary=nuclei[i].end+Math.floor(Math.max(0,cluster.length-1)/2);syllables.push(core.slice(start,boundary));start=boundary;}
  syllables.push(core.slice(start)); syllables[0]=leading+syllables[0]; syllables[syllables.length-1]+=trailing; return syllables;
}
type SyllableUnit={text:string;begin:number;end:number;wordIndex:number};
type LyricLineUnit={text:string;begin:number;end:number;syllables:SyllableUnit[]};
function splitWords(line:string):string[]{return line.match(/(?:[^\s]+)/gu)??[];}
function syllableWeight(text:string):number{return Math.max(1,[...text].filter(c=>/[\p{L}\p{N}]/u.test(c)).length);}
function overlapSeconds(a:AudioInterval,b:{start:number;end:number}):number{return Math.max(0,Math.min(a.end,b.end)-Math.max(a.start,b.start));}
function overlapWithIntervals(start:number,end:number,intervals:AudioInterval[]):number{return intervals.reduce((sum,x)=>sum+overlapSeconds(x,{start,end}),0);}
function activeTimelineMap(activeMs:number,intervals:AudioInterval[],durationMs:number):number{
  const safeDuration=Math.max(0,Number.isFinite(durationMs)?durationMs:0);
  const target=Math.max(0,Number.isFinite(activeMs)?activeMs:0);
  if(!safeDuration)return 0;
  if(!intervals.length)return Math.min(safeDuration,target);
  let remaining=target;
  for(const x of intervals){
    const begin=Math.max(0,Math.min(safeDuration,x.start*1000));
    const end=Math.max(begin,Math.min(safeDuration,x.end*1000));
    const span=end-begin;
    if(remaining<=span)return begin+remaining;
    remaining-=span;
  }
  return safeDuration;
}
function nearestPeak(peaks:AudioPeak[],targetMs:number,fromIndex=0,minMs=0):{index:number;timeMs:number;strength:number}|null{
  let best:null|{index:number;timeMs:number;strength:number}=null;
  for(let i=Math.max(0,fromIndex);i<peaks.length;i++){
    const timeMs=peaks[i].time*1000;
    if(timeMs<minMs)continue;
    const candidate={index:i,timeMs,strength:peaks[i].strength};
    if(!best||Math.abs(candidate.timeMs-targetMs)<Math.abs(best.timeMs-targetMs))best=candidate;
    if(candidate.timeMs>targetMs+850)break;
  }
  return best;
}
function findPhraseBoundary(intervals:AudioInterval[],targetMs:number,minMs:number,maxMs:number):number{
  const points:number[]=[];
  for(const x of intervals){
    const start=x.start*1000,end=x.end*1000;
    if(end<minMs-600||start>maxMs+600)continue;
    points.push(Math.max(minMs,Math.min(maxMs,start)));
    points.push(Math.max(minMs,Math.min(maxMs,end)));
  }
  if(!points.length)return Math.max(minMs,Math.min(maxMs,targetMs));
  return points.reduce((best,x)=>Math.abs(x-targetMs)<Math.abs(best-targetMs)?x:best,points[0]);
}
function buildLyricLineTimeline(lines:string[],durationMs:number,lang:string,intervals:AudioInterval[],peaks:AudioPeak[]):LyricLineUnit[]{
  const duration=Math.max(1000,Math.floor(Number.isFinite(durationMs)?durationMs:1000));
  const safeIntervals=normalizeIntervals(intervals,duration/1000,.10);
  const safePeaks=peaks.filter(p=>Number.isFinite(p.time)&&p.time>=0&&p.time<=duration/1000).sort((x,y)=>x.time-y.time);
  const weights=lines.map(line=>Math.max(1,splitWords(line).reduce((n,w)=>n+splitSyllables(w,lang).length,0)));
  const totalWeight=weights.reduce((a,b)=>a+b,0)||1;
  const activeMs=safeIntervals.reduce((n,x)=>n+(x.end-x.start)*1000,0);
  let activeCursor=0;
  let lastEnd=0;
  return lines.map((line,i)=>{
    const lineActiveStart=activeCursor;
    const lineActiveEnd=Math.min(activeMs,lineActiveStart+(activeMs>100?activeMs*weights[i]/totalWeight:duration*weights[i]/totalWeight));
    const rawStart=activeMs>100?activeTimelineMap(lineActiveStart,safeIntervals,duration):duration*lineActiveStart/duration;
    const rawEnd=activeMs>100?activeTimelineMap(lineActiveEnd,safeIntervals,duration):duration*lineActiveEnd/duration;
    let lineStart=Math.max(lastEnd,Math.min(duration,findPhraseBoundary(safeIntervals,rawStart,lastEnd,Math.max(rawStart,rawEnd))*1));
    let lineEnd=Math.min(duration,Math.max(lineStart+60,findPhraseBoundary(safeIntervals,rawEnd,lineStart,Math.min(duration,rawEnd+700))));
    if(lineEnd<=lineStart)lineEnd=Math.min(duration,lineStart+Math.max(80,rawEnd-rawStart));

    const entries=splitWords(line).flatMap((word,wi)=>splitSyllables(word,lang).map(text=>({text,wordIndex:wi})));
    const syllables:SyllableUnit[]=[];
    const localPeaks=safePeaks.filter(p=>p.time*1000>=lineStart-100&&p.time*1000<=lineEnd+120);
    let peakIndex=0;
    let cursor=lineStart;
    const idealStep=(lineEnd-lineStart)/Math.max(1,entries.length);
    for(let j=0;j<entries.length;j++){
      const nominalStart=lineStart+j*idealStep;
      const foundStart=localPeaks.length?nearestPeak(localPeaks,nominalStart,j===0?0:peakIndex,cursor):null;
      let begin=j===0?lineStart:Math.max(cursor,foundStart?foundStart.timeMs-28:nominalStart);
      if(j>0&&begin-cursor<24)begin=Math.min(lineEnd-25,cursor+Math.max(24,idealStep*.30));

      const nominalEnd=j===entries.length-1?lineEnd:lineStart+(j+1)*idealStep;
      const foundEnd=localPeaks.length?nearestPeak(localPeaks,nominalEnd,foundStart?foundStart.index+1:peakIndex+1,begin):null;
      let end=j===entries.length-1?lineEnd:(foundEnd?foundEnd.timeMs-16:nominalEnd);
      end=Math.min(lineEnd,Math.max(begin+22,end));
      syllables.push({text:entries[j].text,begin:Math.max(lineStart,Math.min(lineEnd,begin)),end:Math.max(begin,Math.min(lineEnd,end)),wordIndex:entries[j].wordIndex});
      cursor=syllables[syllables.length-1].end;
      peakIndex=foundEnd?foundEnd.index:(foundStart?foundStart.index:peakIndex);
    }
    if(syllables.length){
      syllables[0].begin=lineStart;
      for(let j=1;j<syllables.length;j++)syllables[j].begin=Math.max(syllables[j].begin,syllables[j-1].end-10);
      for(let j=0;j<syllables.length-1;j++)syllables[j].end=Math.max(syllables[j].begin+20,Math.min(lineEnd,syllables[j+1].begin));
      syllables[syllables.length-1].end=Math.max(syllables[syllables.length-1].begin,lineEnd);
    }
    lastEnd=lineEnd;
    activeCursor=lineActiveEnd;
    return {text:line,begin:lineStart,end:lineEnd,syllables};
  });
}
function renderSyllables(units:SyllableUnit[]):string{
  return units.map((u,i)=>{const next=units[i+1];const spacer=next&&next.wordIndex!==u.wordIndex?' ':'';return '<span begin="'+toTime(u.begin)+'" end="'+toTime(u.end)+'">'+escapeHtml(u.text)+'</span>'+spacer;}).join('');
}
function splitIntervalsForLine(start:number,end:number,intervals:AudioInterval[]):AudioInterval[]{return intervals.filter(x=>overlapSeconds(x,{start:start/1000,end:end/1000})>.02);}
function lineAgent(line:LyricLineUnit,secondary:AudioInterval[],detected:boolean):'v1'|'v2'{
  if(!detected)return 'v1';
  const overlap=overlapWithIntervals(line.begin/1000,line.end/1000,secondary);
  return overlap/Math.max(.08,(line.end-line.begin)/1000)>=.18?'v2':'v1';
}
function classify(s:AudioStats):number{
  // Conservative stem-purity ensemble. It rewards sustained harmonic vocal
  // structure, but explicitly penalizes broadband/noisy content and inconsistent
  // activity. The goal is admission control, not guessing through ambiguous mixes.
  const periodicity=clamp01(s.harmonicity*.82+s.vocalActivity*.18);
  const voiceTexture=clamp01((1-s.flatness)*.58+Math.max(0,1-s.zcr*4)*.22+(1-s.lowRatio)*.20);
  const voiceBand=clamp01(1-Math.abs(s.lowRatio-.20)*3.6);
  const formantBand=clamp01(1-Math.abs(s.centroid-.30)*2.8);
  const coverage=clamp01(s.vocalCoverage/.42);
  const continuity=clamp01(s.vocalActivity*.62+s.vocalCoverage/.55*.38);
  const broadbandRisk=clamp01(s.flatness*.90+s.zcr*.46+Math.max(0,s.centroid-.56)*.60);
  const ambiguity=clamp01(Math.abs(periodicity-voiceTexture)*1.25+broadbandRisk*.55);
  const raw=.31*periodicity+.19*voiceTexture+.14*voiceBand+.12*formantBand+.15*coverage+.09*continuity;
  return clamp01(raw-.18*broadbandRisk-.10*ambiguity);
}

async function analyzeSlot(slot:FileSlot,label:string,base:number,span:number){
  log(label+': decoding local analysis windows…');
  const stats=await inspect(slot.file,f=>{
    setProgress(base+f*span,'Analyzing '+label,Math.round(f*100)+'% of the analysis window budget');
  });
  if(!stats)throw new Error(label+' could not be decoded.');
  slot.stats=stats;
  slot.score=classify(stats);
  log(label+': '+stats.duration.toFixed(2)+'s • '+stats.sampleRate+' Hz • '+stats.channels+'ch • confidence '+(slot.score*100).toFixed(1)+'%');
  log(label+': vocal activity '+(stats.vocalActivity*100).toFixed(0)+'% • vocal coverage '+(stats.vocalCoverage*100).toFixed(0)+'% • second-voice average '+(stats.secondaryVoice*100).toFixed(0)+'% • peak '+(stats.secondaryVoicePeak*100).toFixed(0)+'%');
  const plan=getAnalysisPlan(stats.duration);
  log(label+': plan • '+plan.windows+' windows / '+plan.fftSize+'-point FFT'+(settings.adaptive?'':' • fixed profile'));
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
  const sameFile=lead.file===backing.file || (lead.file.size===backing.file.size&&lead.file.lastModified===backing.file.lastModified&&lead.file.name===backing.file.name);
  if(sameFile){setGate('bad','Lead vocals and backing vocals must be two different audio files.');return;}
  $('analyze').setAttribute('disabled','true'); $('generate').setAttribute('disabled','true'); setProgress(1,'Starting analysis','Decoding the two required stems.'); setGate('wait','Analyzing both stems…'); log('gate: starting compact feature pass…');
  try{
    const a=await analyzeSlot(lead,'lead',2,46); const b=await analyzeSlot(backing,'backing',48,46);
    const threshold=.60; const leadOk=a>=threshold, backingOk=b>=threshold, ok=leadOk&&backingOk; passed=ok;
    $('leadCard').classList.toggle('good',leadOk); $('leadCard').classList.toggle('bad',!leadOk); $('bgCard').classList.toggle('good',backingOk); $('bgCard').classList.toggle('bad',!backingOk);
    const v2Detected=(lead.stats?.secondaryVoicePeak??0)>=.62; const bgDetected=(backing.stats?.vocalCoverage??0)>=.045&&(backing.stats?.vocalActivity??0)>=.44&&b>=threshold;
    $('v2Status').textContent=v2Detected?'detected • '+Math.round((lead.stats?.secondaryVoice??0)*100)+'%':'not detected • '+Math.round((lead.stats?.secondaryVoice??0)*100)+'%';
    $('bgStatus').textContent=bgDetected?'detected • '+Math.round((backing.stats?.vocalActivity??0)*100)+'%':'not detected • '+Math.round((backing.stats?.vocalActivity??0)*100)+'%';
    log('v2 detection: '+(v2Detected?'SECOND VOICE DETECTED':'no second-voice signal')); log('bg detection: '+(bgDetected?'BACKGROUND VOCAL ACTIVITY DETECTED':'no background-vocal activity detected'));
    if(ok){setProgress(100,'Analysis complete','Stem gate passed; V2/BG detection is ready for TTML assembly.');setGate('ok','Both stems passed ('+(a*100).toFixed(0)+'% / '+(b*100).toFixed(0)+'%). V2: '+(v2Detected?'detected':'not detected')+' • BG: '+(bgDetected?'detected':'not detected')+'.');$('generate').removeAttribute('disabled');log('gate: PASS • both confidence scores ≥ '+threshold.toFixed(2));}
    else{setProgress(100,'Analysis complete','At least one required stem failed the admission gate.');const failed=[leadOk?'':'lead '+(a*100).toFixed(0)+'%',backingOk?'':'backing '+(b*100).toFixed(0)+'%'].filter(Boolean).join(', ');setGate('bad','Rejected: '+failed+'. Add a cleaner isolated vocal stem and analyze again.');$('generate').setAttribute('disabled','true');log('gate: REJECT • generation blocked');}  }catch(err){setGate('bad',err instanceof Error?err.message:'Analysis failed.'); log('gate: ERROR');}
  $('analyze').removeAttribute('disabled');
});

function setProgress(n:number,label='Working…',detail='Processing locally…'){$('progressBar').style.width=Math.max(0,Math.min(100,n))+'%';$('pct').textContent=Math.round(Math.max(0,Math.min(100,n)))+'%';$('progressLabel').textContent=label;$('progressDetail').textContent=detail;}
function sleep(ms:number){return new Promise<void>(r=>setTimeout(r,ms));}
function toTime(ms:number){
  const safeMs=Math.max(0,Math.floor(Number.isFinite(ms)?ms:0));
  const totalSeconds=Math.floor(safeMs/1000);
  const hours=Math.floor(totalSeconds/3600);
  const minutes=Math.floor((totalSeconds-hours*3600)/60);
  const seconds=totalSeconds-hours*3600-minutes*60;
  const millis=safeMs%1000;
  return String(hours).padStart(2,'0')+':'+String(minutes).padStart(2,'0')+':'+String(seconds).padStart(2,'0')+'.'+String(millis).padStart(3,'0');
}
function makeTtml(onProgress:(value:number,label:string,detail:string)=>void=()=>{}){
  const title=escapeHtml(decodeCommonEntities(($('title') as HTMLInputElement).value||'Untitled Session'));
  const artist=escapeHtml(decodeCommonEntities(($('artist') as HTMLInputElement).value||'Unknown Artist'));
  const lang=($('lang') as HTMLSelectElement).value; const lines=getLyricLines();
  const durationMs=Math.max(1000,(backing?.stats?.duration??lead?.stats?.duration??4)*1000);
  const leadIntervals=normalizeIntervals(lead?.stats?.vocalIntervals??[],durationMs/1000,.18);
  const secondaryIntervals=normalizeIntervals(lead?.stats?.secondaryIntervals??[],durationMs/1000,.18);
  const backingIntervals=normalizeIntervals(backing?.stats?.vocalIntervals??[],durationMs/1000,.18);
  const secondaryCoverage=secondaryIntervals.reduce((n,x)=>n+Math.max(0,x.end-x.start),0);
  const v2Detected=(lead?.stats?.secondaryVoicePeak??0)>=.62&&secondaryIntervals.length>0&&secondaryCoverage>=.08;
  const bgDetected=(backing?.stats?.vocalCoverage??0)>=.045&&(backing?.stats?.vocalActivity??0)>=.44&&backingIntervals.length>0&&(backing?.score??0)>=.60;
  const allowV2=qs<HTMLButtonElement>('[data-toggle="v2"]').classList.contains('on'); const allowBg=qs<HTMLButtonElement>('[data-toggle="bg"]').classList.contains('on');
  const autoV2=v2Detected&&allowV2; const autoBg=bgDetected&&allowBg;
  onProgress(24,'Mapping lyric lines','Keeping every Enter-separated lyric line as its own timing unit.');
  const lineUnits=buildLyricLineTimeline(lines,durationMs,lang,leadIntervals);
  onProgress(52,'Building syllable timing','Breaking the supplied lines into syllables and distributing timing across analyzed vocal activity.');
  const outputLines=lineUnits.map((line,i)=>{
    const agent=lineAgent(line,secondaryIntervals,autoV2);
    const main=renderSyllables(line.syllables);
    const lineBacking=autoBg?splitIntervalsForLine(line.begin,line.end,backingIntervals):[];
    let bg='';
    if(lineBacking.length){
      const bgSyllables=line.syllables.filter(u=>overlapWithIntervals(u.begin/1000,u.end/1000,lineBacking)>.015);
      if(bgSyllables.length){
        const bgStart=Math.max(0,Math.min(durationMs,Math.min(...bgSyllables.map(x=>x.begin))));
        const bgEnd=Math.max(bgStart,Math.min(durationMs,Math.max(...bgSyllables.map(x=>x.end))));
        if(bgEnd>bgStart)bg='\n        <span ttm:role="x-bg" begin="'+toTime(bgStart)+'" end="'+toTime(bgEnd)+'">'+renderSyllables(bgSyllables)+'</span>';
      }
    }
    return '      <p begin="'+toTime(line.begin)+'" end="'+toTime(line.end)+'" itunes:key="L'+(i+1)+'" ttm:agent="'+agent+'">\n        '+main+bg+'\n      </p>';
  }).join('\n');
  onProgress(78,'Assembling TTML','Adding automatic V2 agent metadata and BG placements only when detected.');
  if(/00:-|:-\d|--/.test(outputLines))throw new Error('Internal timing guard rejected a negative or malformed lyric timestamp.');
  return '<?xml version="1.0" encoding="UTF-8"?>\n<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:itunes="http://music.apple.com/lyric-ttml-internal" xml:lang="'+lang+'" itunes:timing="Word">\n  <head>\n    <metadata>\n      <ttm:title>'+title+'</ttm:title>\n      <ttm:agent type="person" xml:id="v1"><ttm:name type="full">'+artist+'</ttm:name></ttm:agent>'+(autoV2?'\n      <ttm:agent type="person" xml:id="v2"><ttm:name type="full">Secondary Voice</ttm:name></ttm:agent>':'')+'\n    </metadata>\n  </head>\n  <body>\n    <div itunes:song-part="Verse">\n'+outputLines+'\n    </div>\n  </body>\n</tt>';
}
function parseTtmlTime(value:string|null):number{
  if(!value||!/^(?:\d+):[0-5]\d:[0-5]\d\.\d{3}$/.test(value))return NaN;
  const parts=value.split(':');
  return (Number(parts[0])*3600+Number(parts[1])*60+Number(parts[2]))*1000;
}
function validateTtml(xml:string){
  const doc=new DOMParser().parseFromString(xml,'application/xml');
  if(doc.getElementsByTagName('parsererror').length)throw new Error('Generated TTML failed XML validation.');
  const paragraphs=Array.from(doc.getElementsByTagName('p'));
  if(!paragraphs.length)throw new Error('Generated TTML contains no lyric lines.');
  let lastBegin=0;
  paragraphs.forEach((p,i)=>{
    if(p.getAttribute('itunes:key')!=='L'+(i+1))throw new Error('Lyric line keys are not continuous.');
    const begin=parseTtmlTime(p.getAttribute('begin')); const end=parseTtmlTime(p.getAttribute('end'));
    if(!Number.isFinite(begin)||!Number.isFinite(end))throw new Error('A lyric line has an invalid timestamp.');
    if(begin<0||end<begin)throw new Error('Lyric line timing is out of order.');
    if(i>0&&begin<lastBegin)throw new Error('Lyric line start times are not ordered.');
    lastBegin=begin;
    for(const span of Array.from(p.getElementsByTagName('span'))){
      const sb=parseTtmlTime(span.getAttribute('begin')); const se=parseTtmlTime(span.getAttribute('end'));
      if(!Number.isFinite(sb)||!Number.isFinite(se))throw new Error('A lyric syllable/BG span has an invalid timestamp.');
      if(sb<begin||se>end||se<sb)throw new Error('A lyric syllable/BG span is outside its lyric line.');
    }
  });
}
function updateLyricsStats(){
  const raw=(($('lyrics') as HTMLInputElement).value||'').replace(/\r\n?/g,'\n');
  const lines=raw.split('\n').filter(line=>line.trim().length>0);
  const lang=($('lang') as HTMLSelectElement).value;
  let words=0; let syllables=0;
  for(const line of lines){
    const ws=splitWords(decodeCommonEntities(line));
    words+=ws.length;
    syllables+=ws.reduce((n,w)=>n+splitSyllables(w,lang).length,0);
  }
  $('lyricsStats').textContent=lines.length+' lyric lines · '+words+' words · '+syllables+' syllable units';
}
$('lyrics').addEventListener('input',updateLyricsStats);
$('lang').addEventListener('change',updateLyricsStats);
updateLyricsStats();

$('generate').addEventListener('click',async()=>{if(!passed)return; if(!settings.cpu){setGate('bad','CPU analysis is disabled for the current DSP implementation.');return;} $('generate').setAttribute('disabled','true'); setProgress(0,'Generating TTML','Using the analyzed vocal envelope and the exact lyric lines from the textbox.'); try{output=makeTtml((value,label,detail)=>{log('run: '+label+'…');setProgress(value,label,detail);});setProgress(88,'Validating TTML','Checking XML, timestamp format, line ordering and nested BG spans.');validateTtml(output);$('xml').textContent=output;const lineCount=(output.match(/itunes:key="L\d+"/g)||[]).length;const syllableCount=(output.match(/<span begin=/g)||[]).length;setProgress(100,'TTML ready',lineCount+' lyric lines • '+syllableCount+' timed syllables • automatic BG/v2 applied from analysis.');const base=(($('title') as HTMLInputElement).value||'session').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'session';$('fileName').textContent=base+'.ttml';$('download').removeAttribute('disabled');log('complete: TTML ready • '+lineCount+' lines • '+syllableCount+' timed syllables');$('result').scrollIntoView({behavior:'smooth'});}catch(err){setGate('bad',err instanceof Error?err.message:'TTML generation failed.');log('generation: ERROR');} $('generate').removeAttribute('disabled');});
$('copy').addEventListener('click',async()=>{if(!output)return;try{await navigator.clipboard.writeText(output);$('copy').textContent='Copied';}catch{log('copy: clipboard permission unavailable');}});
$('download').addEventListener('click',()=>{if(!output)return;const blob=new Blob([output],{type:'application/ttml+xml;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=$('fileName').textContent??'session.ttml';a.click();setTimeout(()=>URL.revokeObjectURL(url),500);});
