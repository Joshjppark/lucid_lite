


```
const s = window.__lucid.state.session;
  console.log({
    totalIdentities: s.identities.length,
    globalEntries: s.trackIdentityMap.size,
    frameOverrides: s.frameIdentityMap.size,
    identityNames: s.identities.map(i => i.name),
  }); 

  (() => {
    const s = window.__lucid.state.session;
    const fg = s.getFrameGroup(673); 
    const out = {};
    for (const cam of s.cameras) {
      const insts = fg.getInstances(cam.name) || [];     
      out[cam.name] = insts.map(i => i.trackIdx);
    }
    return out;                            
  })()     
```


outputs:
A: 
top:0: {perFrame: 2, global: 3}
top:1: {perFrame: 0, global: 0}
top:2: {perFrame: 3, global: 1}
top:3: {perFrame: 1, global: 2}
topL:0: {perFrame: 2, global: 3}
topL:1: {perFrame: 0, global: 0}
topL:2: {perFrame: 3, global: 1}
topL:3: {perFrame: 1, global: 2}[[Prototype]]:  Object

B: (matchFrameInstances not on __lucid — try option C instead)

C:
length: 0
[[Prototype]]: Array(0)


(() => {                                                                                     
    const s = window.__lucid.state.session;              
    const fg = s.getFrameGroup(149);
    const dups = [];                                                                           
    for (const cam of s.cameras.map(c => c.name)) {
      const insts = fg.getInstances(cam) || [];                                                
      const byId = {};                                                                         
      for (const inst of insts) {
        if (inst.trackIdx == null) continue;                                                   
        const gid = s.getIdentityIdForTrack(cam, inst.trackIdx, 149);
        if (gid == null) continue;                                                             
        (byId[gid] = byId[gid] || []).push(inst.trackIdx);
      }                                                                                        
      for (const [gid, tracks] of Object.entries(byId)) {
        if (tracks.length > 1) {                                                               
          dups.push({ cam, identity: +gid, sharedBy: tracks });                                
        }
      }                                                                                        
    }                                                    
    return dups.length ? dups : '(no duplicates at frame)';                                
  })()

  (() => {                                                                                     
    const s = window.__lucid.state.session;              
    // Find the first frame where any visible (cam, track) in ANY camera lacks a per-frame     
  override.                                                                                    
    const out = [];                                                                            
    for (let fi = 0; fi <= 1799 && out.length < 25; fi++) {                                    
      const fg = s.getFrameGroup(fi);                    
      if (!fg) continue;                                                                       
      const missing = [];
      for (const cam of s.cameras.map(c => c.name)) {                                          
        const insts = fg.getInstances(cam) || [];        
        for (const inst of insts) {                                                            
          if (inst.trackIdx == null) continue;
          const k = `${fi}:${cam}:${inst.trackIdx}`;                                           
          if (!s.frameIdentityMap.has(k)) missing.push(`${cam}:${inst.trackIdx}`);             
        }
      }                                                                                        
      if (missing.length) out.push({ frame: fi, missingOverrides: missing });
    }                                                                                          
    return out;
  })()     

(()=>{const s=window.__lucid.state.session;
  const fi=149;
  const groups=s.instanceGroups.get(fi)||[];
  return groups.map((g,i)=>({
    groupIdx:i,identityId:g.identityId,cams:[...g.instances.keys()],
    midL_track:g.instances.has('midL')?g.instances.get('midL').trackIdx:null}));
})()

