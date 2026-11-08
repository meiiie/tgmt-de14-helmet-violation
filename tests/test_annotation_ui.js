// Business-logic/event regressions against the actual page script. This is not a real-browser visual test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../src/de14/annotation_ui.html'),'utf8');
const code=[...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].at(-1)[1];
const base={schema_version:1,status:'draft',scope:'sampled_source_frames',sample_stride:10,
 source:{name:'1102.mp4',sha256:'a'.repeat(64),width:200,height:200,fps:30,frame_count:60},
 frames:[{source_index:0,timestamp_s:0,image:'frames/000000.jpg',reviewed:false,vehicles:[]},
         {source_index:10,timestamp_s:1/3,image:'frames/000010.jpg',reviewed:false,vehicles:[]}]};
function app(){
 const elements=new Map(),storage=new Map(),urls=new Map(),downloads=[],keys={};
 class Element{
  constructor(tag='DIV'){this.tagName=tag.toUpperCase();this.value='';this.children=[];this.handlers={};this.dataset={};this.textContent='';this.style={};this.hidden=false;const classes=new Set();this.classList={toggle:(k,on)=>{on=on??!classes.has(k);on?classes.add(k):classes.delete(k)},contains:k=>classes.has(k)}}
  addEventListener(name,fn){this.handlers[name]=fn} replaceChildren(...items){this.children=items} append(...items){this.children.push(...items)} add(item){this.children.push(item)} scrollTo(){} setPointerCapture(){}
  getBoundingClientRect(){return{left:10,top:20,width:100,height:100}}
  getContext(){return{clearRect(){},drawImage(){},strokeRect(){},fillRect(){},fillText(){},setLineDash(){},measureText:t=>({width:String(t).length*15})}}
  click(){if(this.tagName==='A')downloads.push({name:this.download,blob:urls.get(this.href)});else this.onclick?.()}
 }
 const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
 get('annotation-data').textContent=JSON.stringify(base);
 const modes=['vehicle','head','select'].map(mode=>{let e=new Element('button');e.dataset.mode=mode;return e});
 class Image{set src(value){this.onload?.()}}
 const context=vm.createContext({console,structuredClone,Blob,Date,Math,JSON,Number,Object,Set,Map,Error,Image,
  Option:function(text,value){this.textContent=text;this.value=value},confirm:()=>true,setTimeout:()=>0,
  URL:{createObjectURL:blob=>{const key='blob:'+urls.size;urls.set(key,blob);return key},revokeObjectURL(){}},
  localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
  document:{getElementById:get,querySelectorAll:()=>modes,createElement:tag=>new Element(tag),addEventListener:(event,fn)=>keys[event]=fn}});
 vm.runInContext(code,context);
 const fire=(event,x=40,y=80)=>get('canvas').handlers[event]?.({button:0,pointerId:1,clientX:x,clientY:y,target:get('canvas')});
 const mode=name=>modes.find(m=>m.dataset.mode===name).onclick();
 const draw=(x1,y1,x2,y2)=>{fire('pointerdown',x1,y1);fire('pointermove',x2,y2);fire('pointerup',x2,y2)};
 const choose=(id,value)=>{get(id).value=value;get(id).onchange?.()};
 const key=key=>keys.keydown({key,target:{tagName:'BODY'},preventDefault(){},ctrlKey:false});
 const exportJSON=async()=>{get('export-button').onclick();assert(downloads.length,'JSON download missing');return JSON.parse(await downloads.at(-1).blob.text())};
 const importJSON=async data=>{await get('import-file').onchange({target:{files:[{text:async()=>JSON.stringify(data)}],value:'selected'}})};
 const vehicle=()=>{mode('vehicle');draw(20,30,100,110);choose('occupancy','occupied')};
 return{get,storage,fire,mode,draw,choose,key,exportJSON,importJSON,vehicle};
}
let passed=0,failed=0;
async function test(name,fn){try{await fn();passed++;console.log('PASS '+name)}catch(e){failed++;console.error('FAIL '+name+': '+e.message)}}
(async()=>{
 await test('source pixel coordinates and independent head labels',async()=>{const a=app();a.vehicle();a.mode('head');a.draw(30,40,40,50);a.choose('helmet-label','helmet');a.draw(50,40,60,50);const d=await a.exportJSON();assert.deepEqual(d.frames[0].vehicles[0].box,[20,20,180,180]);assert.deepEqual(d.frames[0].vehicles[0].heads.map(h=>h.helmet),['helmet','unknown']);assert.equal(d.frames[0].vehicles[0].rider_count,2);assert.equal(d.frames[0].reviewed,false)});
 await test('selection click preserves an already reviewed frame',async()=>{const a=app();a.vehicle();a.get('review-button').onclick();a.mode('select');a.fire('pointerdown');a.fire('pointerup');assert.equal((await a.exportJSON()).frames[0].reviewed,true)});
 for(const cancel of ['Escape','pointercancel','navigation'])await test('cancel '+cancel+' restores reviewed geometry',async()=>{const a=app();a.vehicle();a.get('review-button').onclick();a.mode('select');a.fire('pointerdown');a.fire('pointermove',50,90);if(cancel==='Escape')a.key('Escape');else if(cancel==='pointercancel')a.fire('pointercancel');else a.get('next-button').onclick();a.fire('pointerup',50,90);const d=await a.exportJSON();assert.deepEqual(d.frames[0].vehicles[0].box,[20,20,180,180]);assert.equal(d.frames[0].reviewed,true)});
 await test('committed move clears review and updates autosave',async()=>{const a=app();a.vehicle();a.get('review-button').onclick();a.mode('select');a.fire('pointerdown');a.fire('pointermove',50,90);a.fire('pointerup',50,90);const d=await a.exportJSON();assert.equal(d.frames[0].reviewed,false);assert.notDeepEqual(d.frames[0].vehicles[0].box,[20,20,180,180]);assert.equal(JSON.parse([...a.storage.values()][0]).data.frames[0].reviewed,false)});
 const invalid=[['missing timestamp',d=>delete d.frames[0].timestamp_s],['null timestamp',d=>d.frames[0].timestamp_s=null],['string timestamp',d=>d.frames[0].timestamp_s='0'],['numeric vehicle ID',d=>d.frames[0].vehicles[0].id=12],['missing vehicle ID',d=>delete d.frames[0].vehicles[0].id],['missing head ID',d=>delete d.frames[0].vehicles[0].heads[0].id]];
 for(const [name,mutate]of invalid)await test('reject '+name+' without corrupting saved labels',async()=>{const a=app();a.vehicle();a.mode('head');a.draw(30,40,40,50);a.get('review-button').onclick();const old=await a.exportJSON(),saved=[...a.storage.values()][0];const bad=structuredClone(old);mutate(bad);await a.importJSON(bad);assert.equal([...a.storage.values()][0],saved);assert.deepEqual((await a.exportJSON()).frames,old.frames)});
 await test('source mismatch rejected and JSON round trip retains IDs',async()=>{const a=app();a.vehicle();a.get('review-button').onclick();const old=await a.exportJSON(),saved=[...a.storage.values()][0];const bad=structuredClone(old);bad.source.sha256='b'.repeat(64);await a.importJSON(bad);assert.equal([...a.storage.values()][0],saved);await a.importJSON(old);assert.deepEqual((await a.exportJSON()).frames,old.frames)});
 await test('rename vehicle ID preserves its boxes and heads and clears review',async()=>{const a=app();a.vehicle();a.mode('head');a.draw(30,40,40,50);a.choose('helmet-label','helmet');a.get('review-button').onclick();const old=await a.exportJSON();a.get('vehicle-id').value='V020';a.get('rename-id-button').onclick();const d=await a.exportJSON();assert.equal(d.frames[0].vehicles[0].id,'V020');assert.deepEqual(d.frames[0].vehicles[0].box,old.frames[0].vehicles[0].box);assert.deepEqual(d.frames[0].vehicles[0].heads,old.frames[0].vehicles[0].heads);assert.equal(d.frames[0].reviewed,false)});
 await test('rename cannot give two different boxes the same ID in a frame',async()=>{const a=app();a.vehicle();a.vehicle();a.get('review-button').onclick();const old=await a.exportJSON();a.get('vehicle-id').value='V001';a.get('rename-id-button').onclick();assert.deepEqual((await a.exportJSON()).frames,old.frames)});
 await test('jump to source frame changes cursor without altering labels',async()=>{const a=app();a.vehicle();a.get('review-button').onclick();const old=await a.exportJSON();a.get('source-frame-input').value='10';a.get('jump-frame-button').onclick();assert.equal(a.get('frame-index').textContent,'Frame nguồn 10');assert.deepEqual((await a.exportJSON()).frames,old.frames);a.get('source-frame-input').value='999';a.get('jump-frame-button').onclick();assert.equal(a.get('frame-index').textContent,'Frame nguồn 10')});
 console.log(JSON.stringify({passed,failed,scope:'Actual page JavaScript in DOM/canvas boundary harness; not real browser rendering or pointer-device QA'}));if(failed)process.exitCode=1;
})();
