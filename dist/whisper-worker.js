const TRANSFORMERS_URL='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';
const MODEL_ID='onnx-community/whisper-tiny_timestamped';
const SAMPLE_RATE=16000;
let pipelineCache=new Map();

function post(type,payload={}){
  self.postMessage({type,...payload});
}

function configureRuntime(mod){
  try{
    const onnx=mod?.env?.backends?.onnx;
    if(onnx?.env){
      try{onnx.env.logLevel='error';}catch{}
    }
  }catch{}
}

async function getPipeline(requestedGpu){
  const device=requestedGpu&&self.navigator?.gpu?'webgpu':'wasm';
  const key=device;
  if(pipelineCache.has(key))return pipelineCache.get(key);
  const promise=(async()=>{
    post('status',{message:device==='webgpu'?'Loading Whisper timestamp model on WebGPU…':'Loading Whisper timestamp model on CPU/WASM…'});
    const mod=await import(TRANSFORMERS_URL);
    configureRuntime(mod);
    return mod.pipeline('automatic-speech-recognition',MODEL_ID,{
      device,
      dtype:device==='webgpu'
        ?{encoder_model:'fp32',decoder_model_merged:'q4'}
        :'q8',
      session_options:{logSeverityLevel:3}
    });
  })();
  pipelineCache.set(key,promise);
  try{return await promise;}catch(err){
    pipelineCache.delete(key);
    throw err;
  }
}

function normalizeToken(text){
  return String(text??'').normalize('NFKD').toLowerCase().replace(/\p{M}/gu,'').replace(/[^\p{L}\p{N}]+/gu,'').trim();
}

function mergeTimedChunks(target,source,offset){
  for(const chunk of Array.isArray(source)?source:[]){
    const text=String(chunk?.text??'').trim();
    const ts=chunk?.timestamp??chunk?.timestamps;
    if(!text||!Array.isArray(ts)||!Number.isFinite(ts[0])||!Number.isFinite(ts[1]))continue;
    const start=Math.max(0,Number(ts[0]))+offset;
    const end=Math.max(start,Number(ts[1]))+offset;
    if(end<=start)continue;
    const previous=target[target.length-1];
    const normalized=normalizeToken(text);
    if(previous&&normalizeToken(previous.text)===normalized&&Math.abs(previous.start-start)<1.25){
      previous.end=Math.max(previous.end,end);
      continue;
    }
    if(previous&&start<previous.end){
      const gap=start-previous.start;
      if(normalized===normalizeToken(previous.text)||gap<0.05)continue;
    }
    target.push({text,start,end});
  }
}

async function runAlignment(audio,language,requestedGpu){
  const totalSeconds=audio.length/SAMPLE_RATE;
  const windowSeconds=24;
  const overlapSeconds=4;
  const stepSeconds=windowSeconds-overlapSeconds;
  const chunks=[];
  let usedGpu=requestedGpu&&!!self.navigator?.gpu;

  let pipe;
  try{
    pipe=await getPipeline(usedGpu);
  }catch(error){
    if(!usedGpu)throw error;
    usedGpu=false;
    post('fallback',{message:'WebGPU model initialization failed. Retrying this alignment in CPU/WASM.'});
    pipe=await getPipeline(false);
  }

  const count=Math.max(1,Math.ceil(Math.max(0,totalSeconds-overlapSeconds)/stepSeconds));
  for(let index=0;index<count;index++){
    const offset=Math.min(index*stepSeconds,Math.max(0,totalSeconds-windowSeconds));
    const end=Math.min(totalSeconds,offset+windowSeconds);
    const startSample=Math.floor(offset*SAMPLE_RATE);
    const endSample=Math.min(audio.length,Math.ceil(end*SAMPLE_RATE));
    const segment=audio.subarray(startSample,endSample);

    post('progress',{
      value:12+(index/count)*82,
      message:'Recognizing vocal audio',
      detail:'Whisper window '+(index+1)+' / '+count+' • '+offset.toFixed(1)+'–'+end.toFixed(1)+'s'+(usedGpu?' • WebGPU':' • CPU/WASM')
    });

    let result;
    try{
      result=await pipe(segment,{
        return_timestamps:'word',
        chunk_length_s:29,
        stride_length_s:0,
        ...(language?{language,task:'transcribe'}:{task:'transcribe'})
      });
    }catch(error){
      if(!usedGpu)throw error;
      usedGpu=false;
      post('fallback',{message:'WebGPU inference failed. Retrying the active window in CPU/WASM.'});
      pipe=await getPipeline(false);
      result=await pipe(segment,{
        return_timestamps:'word',
        chunk_length_s:29,
        stride_length_s:0,
        ...(language?{language,task:'transcribe'}:{task:'transcribe'})
      });
    }

    mergeTimedChunks(chunks,result?.chunks,offset);
  }

  return {chunks,device:usedGpu?'webgpu':'wasm',duration:totalSeconds};
}

self.onmessage=async(event)=>{
  const data=event.data||{};
  if(data.type!=='align')return;
  try{
    const incoming=data.audio;
    if(!(incoming instanceof ArrayBuffer))throw new Error('Whisper worker received invalid audio data.');
    const audio=new Float32Array(incoming);
    if(audio.length<1)throw new Error('The vocal stem contains no decodable samples.');
    post('progress',{value:4,message:'Whisper worker ready',detail:'The main page stays responsive while the actual audio is recognized.'});
    const result=await runAlignment(audio,data.language||'',!!data.useGpu);
    post('done',result);
  }catch(error){
    post('error',{message:error instanceof Error?error.message:String(error),stack:error instanceof Error?error.stack:''});
  }
};
