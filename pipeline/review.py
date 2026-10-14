"""Local review UI for picking the best silhouette per species and adjusting the
bill-tip / tail-tip endpoints used for scaling.

    uv run python pipeline/review.py            # http://127.0.0.1:8777

Choices are saved to data/review.json.
"""
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from geometry import geodesic

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ROOT / "data/species.json"
SILS = ROOT / "data/silhouettes"
PHOTOS = ROOT / "data/photos"
REVIEW = ROOT / "data/review.json"
TOP_K = 6


def load_review():
    return json.loads(REVIEW.read_text()) if REVIEW.exists() else {}


def species_payload():
    species = json.loads(SPECIES.read_text())
    out = []
    for sp in species:
        cpath = SILS / sp["slug"] / "candidates.json"
        if not cpath.exists():
            continue
        cands = [c for c in json.loads(cpath.read_text()) if c.get("sil_file")][:TOP_K]
        out.append({**sp, "candidates": cands})
    out.sort(key=lambda s: -(s.get("vic_observations") or 0))
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/species":
            return self.send(json.dumps({"species": species_payload(), "review": load_review()}).encode())
        for prefix, base in (("/sil/", SILS), ("/photo/", PHOTOS)):
            if path.startswith(prefix):
                f = (base / path[len(prefix):]).resolve()
                if base.resolve() in f.parents and f.is_file():
                    return self.send(f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            return self.send(b"not found", "text/plain", 404)
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        review = load_review()
        if data.get("clear"):
            review.pop(data["slug"], None)
        else:
            entry = {k: data[k] for k in ("status", "photo_id", "endpoints") if k in data}
            if entry.get("endpoints"):
                sil = Image.open(SILS / data["slug"] / f"{entry['photo_id']}.png")
                mask = np.asarray(sil)[..., 3] > 127
                length, path = geodesic(mask, *entry["endpoints"])
                entry.update(geo_length_px=round(length, 1), path=path)
            review[data["slug"]] = entry
        REVIEW.write_text(json.dumps(review, indent=1))
        self.send(json.dumps(review.get(data["slug"], {})).encode())


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Silhouette review</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f1ea;color:#222}
 header{display:flex;gap:16px;align-items:baseline;padding:12px 20px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0}
 h1{font-size:20px;margin:0} .sci{font-style:italic;color:#666} .status{margin-left:auto;font-size:14px}
 .help{padding:6px 20px;font-size:13px;color:#555}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:14px 20px}
 .card{background:#fff;border:3px solid transparent;border-radius:8px;padding:8px;position:relative;cursor:pointer}
 .card.sel{border-color:#d33}
 .card .n{position:absolute;top:6px;left:10px;font-weight:700;font-size:18px}
 .card .meta{font-size:11px;color:#777;margin-top:4px}
 .row{display:flex;gap:8px;align-items:center}
 .silbox{flex:1;height:260px;display:flex;align-items:center;justify-content:center}
 svg{max-width:100%;max-height:260px}
 .photo{width:120px;height:120px;object-fit:contain;background:#eee}
 .pt{fill:#2a7;stroke:#fff;stroke-width:2;cursor:grab}
 .pt.tail{fill:#2a7}
 .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;color:#fff}
 .ok{background:#2a7}.rej{background:#c33}.none{background:#999}
</style></head><body>
<header><h1 id="name"></h1><span class="sci" id="sci"></span><span id="len"></span>
<span class="status"><span id="badge"></span> <span id="pos"></span></span></header>
<div class="help">Click / <b>1–6</b>: choose candidate &nbsp; On the chosen card, drag the dots (or click twice) to set bill tip and tail tip; yellow = measured body path &nbsp;
<b>x</b>: no usable silhouette &nbsp; <b>u</b>: clear &nbsp; <b>←/→</b> or <b>j/k</b>: prev/next &nbsp; <b>n</b>: next unreviewed</div>
<div class="grid" id="grid"></div>
<script>
let species=[], review={}, idx=0, pendingPt=null;
const $=id=>document.getElementById(id);
async function load(){const r=await (await fetch('/api/species')).json();species=r.species;review=r.review;
  const first=species.findIndex(s=>!review[s.slug]);idx=first<0?0:first;render();}
async function save(slug,data){review[slug]=data;
  const r=await fetch('/api/review',{method:'POST',body:JSON.stringify({slug,...data})});
  review[slug]=await r.json();if(species[idx].slug===slug)render();}
function clearReview(slug){delete review[slug];fetch('/api/review',{method:'POST',body:JSON.stringify({slug,clear:true})});}
function render(){
  const sp=species[idx], rv=review[sp.slug];
  $('name').textContent=sp.common_name;$('sci').textContent=sp.scientific_name;
  $('len').textContent=`${sp.length_cm} cm · ${sp.vic_observations} VIC obs`;
  $('pos').textContent=`${idx+1}/${species.length} · ${Object.keys(review).length} reviewed`;
  $('badge').innerHTML=!rv?'<span class="badge none">unreviewed</span>':rv.status==='ok'?'<span class="badge ok">chosen</span>':'<span class="badge rej">rejected</span>';
  const g=$('grid');g.innerHTML='';
  sp.candidates.forEach((c,i)=>{
    const chosen=rv&&rv.status==='ok'&&rv.photo_id===c.photo_id;
    const ep=chosen&&rv.endpoints?rv.endpoints:c.endpoints;
    const path=(chosen&&rv.path?rv.path:c.path||ep).map(p=>p.join(',')).join(' ');
    const [w,h]=c.sil_size, r=Math.max(w,h)/90;
    const card=document.createElement('div');card.className='card'+(chosen?' sel':'');
    card.innerHTML=`<div class="n">${i+1}</div><div class="row"><div class="silbox">
      <svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
        <image href="/sil/${sp.slug}/${c.sil_file}" width="${w}" height="${h}"/>
        <polyline points="${path}" fill="none" stroke="#fc3" stroke-width="${r/2.5}" stroke-linejoin="round"/>
        <line x1="${ep[0][0]}" y1="${ep[0][1]}" x2="${ep[1][0]}" y2="${ep[1][1]}" stroke="#d33" stroke-width="${r/4}" stroke-dasharray="${r} ${r/2}"/>
        <circle class="pt" r="${r}" cx="${ep[0][0]}" cy="${ep[0][1]}" data-i="0"/>
        <circle class="pt tail" r="${r}" cx="${ep[1][0]}" cy="${ep[1][1]}" data-i="1"/>
      </svg></div><img class="photo" src="/photo/${sp.slug}/${c.file}"></div>
      <div class="meta">score ${c.score} · pose ${c.profile_prob} · sym ${c.symmetry} · soft ${c.softness} · edges ${c.edges_touched} · ${c.license}</div>`;
    const svg=card.querySelector('svg');
    card.addEventListener('click',e=>{
      if(e.target.classList.contains('pt'))return;
      if(!chosen||!svg.contains(e.target)){pendingPt=null;choose(i);return;}
      // On the chosen card, two clicks on the silhouette set the two ends
      const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
      const q=[Math.round(p.x*10)/10,Math.round(p.y*10)/10];
      if(!pendingPt){pendingPt=q;const d=document.createElementNS('http://www.w3.org/2000/svg','circle');
        d.setAttribute('cx',q[0]);d.setAttribute('cy',q[1]);d.setAttribute('r',r);d.setAttribute('fill','#f90');svg.appendChild(d);}
      else{const a=pendingPt;pendingPt=null;save(sp.slug,{status:'ok',photo_id:c.photo_id,endpoints:[a,q]});}
    });
    card.querySelectorAll('.pt').forEach(pt=>pt.addEventListener('pointerdown',e=>{
      e.preventDefault();e.stopPropagation();pt.setPointerCapture(e.pointerId);
      const pts=[...ep.map(p=>[...p])];const line=svg.querySelector('line');
      const move=ev=>{const m=svg.getScreenCTM().inverse();const p=new DOMPoint(ev.clientX,ev.clientY).matrixTransform(m);
        const k=+pt.dataset.i;pts[k]=[Math.round(p.x*10)/10,Math.round(p.y*10)/10];pt.setAttribute('cx',p.x);pt.setAttribute('cy',p.y);
        line.setAttribute('x'+(k+1),p.x);line.setAttribute('y'+(k+1),p.y);};
      const up=()=>{pt.removeEventListener('pointermove',move);pt.removeEventListener('pointerup',up);
        save(sp.slug,{status:'ok',photo_id:c.photo_id,endpoints:pts});render();};
      pt.addEventListener('pointermove',move);pt.addEventListener('pointerup',up);}));
    g.appendChild(card);});
}
function choose(i){const sp=species[idx],c=sp.candidates[i];if(!c)return;
  const rv=review[sp.slug];const keep=rv&&rv.photo_id===c.photo_id&&rv.endpoints;
  save(sp.slug,{status:'ok',photo_id:c.photo_id,endpoints:keep||c.endpoints});render();}
function go(d){pendingPt=null;idx=(idx+d+species.length)%species.length;render();window.scrollTo(0,0);}
document.addEventListener('keydown',e=>{
  if(e.key>='1'&&e.key<='9')choose(+e.key-1);
  else if(e.key==='ArrowRight'||e.key==='k')go(1);
  else if(e.key==='ArrowLeft'||e.key==='j')go(-1);
  else if(e.key==='x'){save(species[idx].slug,{status:'rejected'});render();}
  else if(e.key==='u'){clearReview(species[idx].slug);render();}
  else if(e.key==='n'){const n=species.findIndex((s,i)=>i>idx&&!review[s.slug]);if(n>=0){idx=n;render();}}
});
load();
</script></body></html>"""


if __name__ == "__main__":
    print("Review UI: http://127.0.0.1:8777")
    ThreadingHTTPServer(("127.0.0.1", 8777), Handler).serve_forever()
