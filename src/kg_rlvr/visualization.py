"""Bounded graph projections rendered as standalone, offline interactive SVG HTML.

The renderer intentionally has no PyVis/CDN dependency: the generated file works
offline, preserves directed relation IDs, and exposes answer/path selectors.
The serialized subgraph is a view derived from the saved canonical graph.
"""

from collections import deque
import html
import json
from pathlib import Path

from .graph_index import execute

COLORS = {"researcher": "#58c6e8", "university": "#8a9dff", "project": "#ed96d0", "publication": "#7ed6b0",
          "field": "#c1a3f8", "funder": "#efc477", "city": "#f39782", "country": "#a9c87b"}


def witness_paths(graph, question, per_answer=3):
    """Enumerate up to `per_answer` actual witnesses per answer, deterministically."""
    answers = set(question["answer_ids"])
    result = {a: [] for a in sorted(answers)}
    relations = question["relation_path"]
    for path in question["gold_entity_paths"]:
        if (len(path) != len(relations) + 1 or path[0] != question["start_entity"]
                or path[-1] not in answers
                or any(t not in graph.get_neighbors(h, r) for h, r, t in zip(path, relations, path[1:]))):
            raise ValueError("Invalid saved reference path")
        if path not in result[path[-1]] and len(result[path[-1]]) < per_answer:
            result[path[-1]].append(path)
    if any(not paths for paths in result.values()):
        raise ValueError("Saved reference paths do not cover every answer")
    def visit(path, depth):
        if depth == len(relations):
            if path[-1] in answers and len(result[path[-1]]) < per_answer and path not in result[path[-1]]:
                result[path[-1]].append(path)
            return
        for neighbor in graph.get_neighbors(path[-1], relations[depth]):
            visit(path + [neighbor], depth + 1)
    visit([question["start_entity"]], 0)
    return result


def neighborhood(graph, seeds, radius, max_nodes, entity_types=None, relation_types=None):
    selected, pending = set(seeds), deque((s, 0) for s in sorted(seeds))
    if len(selected) > max_nodes:
        raise ValueError(f"Reference paths require {len(selected)} nodes, exceeding max_nodes={max_nodes}; increase the limit")
    while pending:
        entity, depth = pending.popleft()
        if depth == radius:
            continue
        for relation in graph.get_relations(entity):
            if relation_types and relation.removeprefix("inv:") not in relation_types:
                continue
            for neighbor in graph.get_neighbors(entity, relation):
                if entity_types and graph.entities[neighbor]["type"] not in entity_types:
                    continue
                if neighbor not in selected and len(selected) < max_nodes:
                    selected.add(neighbor)
                    pending.append((neighbor, depth + 1))
    return selected


def visualization_payload(graph, *, question=None, entity=None, radius=2, max_nodes=120, max_edges=220,
                          entity_types=None, relation_types=None, paths_per_answer=3):
    if max_nodes < 1 or max_edges < 1 or radius < 0 or paths_per_answer < 1:
        raise ValueError("Positive node/edge/path limits and nonnegative radius are required")
    for kind in entity_types or []:
        if kind not in COLORS:
            raise ValueError(f"Unknown entity type filter: {kind}")
    for relation in relation_types or []:
        if relation not in graph.relations:
            raise ValueError(f"Unknown relation filter: {relation}")
    paths, gold_edges, core, layers = [], set(), set(), {}
    if question:
        if execute(graph, question["start_entity"], question["relation_path"]).answers != question["answer_ids"]:
            raise ValueError("Cannot visualize a question with an incorrect or incomplete answer set")
        witnesses = witness_paths(graph, question, paths_per_answer)
        for answer, alternatives in witnesses.items():
            for index, path in enumerate(alternatives):
                paths.append({"answer": answer, "index": index + 1, "entities": path})
                core.update(path)
                for depth, node in enumerate(path):
                    layers[node] = min(layers.get(node, depth), depth)
                gold_edges.update(zip(path, question["relation_path"], path[1:]))
        entity = question["start_entity"]
        mode = "Question inspection"
    elif entity:
        mode = "Entity neighborhood"
        graph.get_entity(entity)
    else:
        mode = "Graph overview"
        entity = next(e for e in sorted(graph.entities) if graph.entities[e]["type"] == "researcher")
    seeds = core or {entity}
    selected = neighborhood(graph, seeds, radius, max_nodes, entity_types, relation_types)
    # Preserve every displayed reference traversal, even when user filters hide context.
    edges = [{"from": h, "relation": r, "to": t, "gold": True} for h, r, t in sorted(gold_edges)]
    canonical_gold = {(t, r[4:], h) if r.startswith("inv:") else (h, r, t) for h, r, t in gold_edges}
    for head, relation, tail in graph.triples:
        if head not in selected or tail not in selected or (head, relation, tail) in canonical_gold:
            continue
        if relation_types and relation not in relation_types:
            continue
        if len(edges) >= max_edges:
            break
        edges.append({"from": head, "relation": relation, "to": tail, "gold": False})
    if len(gold_edges) > max_edges:
        raise ValueError(f"Reference paths require {len(gold_edges)} edges; increase max_edges={max_edges}")
    # Remove context nodes with no rendered edge after the edge limit is applied.
    incident = {e[k] for e in edges for k in ("from", "to")} | seeds
    selected &= incident
    answers = set(question["answer_ids"]) if question else set()
    nodes = []
    for node in sorted(selected):
        role = "start" if node == entity else "answer" if node in answers else "intermediate" if node in core else "context"
        nodes.append({"id": node, **graph.entities[node], "role": role, "color": COLORS[graph.entities[node]["type"]], "layer": layers.get(node)})
    return {"mode": mode, "start": entity, "nodes": nodes, "edges": edges, "paths": paths, "question": question,
            "colors": COLORS, "graph_counts": {"nodes": len(graph.entities), "edges": len(graph.triples)},
            "limits": {"nodes": max_nodes, "edges": max_edges, "radius": radius, "paths_per_answer": paths_per_answer}}


def render_visualization(graph, output, **options):
    payload = visualization_payload(graph, **options)
    title = f"{payload['mode']} · {payload['question']['id'] if payload['question'] else payload['start']}"
    serialized = json.dumps(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = HTML.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", serialized)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return payload


def inspection_bundle(graph, questions, directory):
    """Reproducible overview, neighborhood, multi-answer hop examples and index."""
    directory = Path(directory)
    rows = [("graph_overview.html", "Graph overview")]
    render_visualization(graph, directory / "graph_overview.html")
    researcher = "researcher_000042" if "researcher_000042" in graph.entities else next(
        e for e in sorted(graph.entities) if graph.entities[e]["type"] == "researcher")
    filename = f"entity_{researcher}.html"
    render_visualization(graph, directory / filename, entity=researcher)
    rows.append((filename, "Researcher neighborhood"))
    selected = []
    for hop in (2, 3, 4):
        choices = [q for q in questions if q["hop_count"] == hop]
        if not choices:
            raise ValueError(f"Cannot build inspection bundle without {hop}-hop questions")
        question = min(choices, key=lambda q: (-len(q["answer_ids"]), -sum(q["answer_path_counts"].values()), q["id"]))
        filename = f"question_{question['id']}.html"
        render_visualization(graph, directory / filename, question=question)
        rows.append((filename, f"{hop}-hop question · {question['id']} · {len(question['answer_ids'])} answers"))
        selected.append({"question_id": question["id"], "hop_count": hop, "answer_count": len(question["answer_ids"]), "file": filename})
    links = "\n".join(f'<a href="{html.escape(filename)}">{html.escape(label)} <span>Open inspector →</span></a>' for filename, label in rows)
    summary = f"{len(graph.entities):,} entities · {len(graph.triples):,} stored triples · {len(graph.relations)} relations · {len(questions):,} verified questions."
    index = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Research graph inspectors</title><style>body{background:#08131e;color:#e6eef6;font:16px/1.6 system-ui;max-width:850px;margin:70px auto;padding:0 24px}h1{font-size:36px}p{color:#9bb5c9}a{display:flex;justify-content:space-between;gap:20px;border:1px solid #385068;border-radius:9px;padding:20px;margin:12px 0;color:#85def2;text-decoration:none;background:#112331}a:hover{background:#1c3749}span{font-size:13px;color:#a9c5d9}.eyebrow{color:#73dbef;letter-spacing:3px;font-size:12px}</style><div class="eyebrow">FICTIONAL RESEARCH ATLAS</div><h1>Inspect the graph.<br>Trace every answer.</h1>'''
    index += f"<p>{summary}<br>Choose a view. Each inspector is a standalone offline HTML file.</p>"
    index += links + "<p>Scroll to zoom. Drag nodes or pan the background. Use search and filters to inspect neighbors. Question views include every answer and verified reference paths.</p></html>"
    (directory / "index.html").write_text(index)
    return selected


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:dark;font:14px/1.5 system-ui,sans-serif;color:#e6eef6;background:#08131e}*{box-sizing:border-box}
body{margin:0;height:100vh;display:grid;grid-template-columns:360px 1fr;overflow:hidden}aside{padding:24px;border-right:1px solid #26394b;overflow:auto;background:#0e1c29}
.eyebrow{color:#69cce4;font-size:11px;text-transform:uppercase;letter-spacing:2px;font-weight:700}h1{font-size:23px;line-height:1.25;margin:9px 0 16px}h2{font-size:13px;color:#acc0d1;margin-top:22px}p{margin:8px 0}.muted{color:#95adbf;font-size:12px}code{font-size:11px;overflow-wrap:anywhere}
input,select,button{font:inherit;color:#e6eef6;background:#172a3a;border:1px solid #3a5268;border-radius:6px;padding:8px;max-width:100%}select,input[type=search]{width:100%}button{cursor:pointer}button:hover{background:#294456}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.chip{background:#1b3548;border:1px solid #375469;border-radius:20px;padding:3px 10px;font-size:11px}.legend{display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:11px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}label{cursor:pointer}details{margin:14px 0}summary{cursor:pointer;color:#bcd3e5}
#stage{position:relative;min-width:0;background:radial-gradient(ellipse at center,#112838,#08131e 75%)}#graph{height:100%;width:100%;touch-action:none;cursor:grab}#graph:active{cursor:grabbing}.toolbar{position:absolute;top:16px;left:18px;right:18px;display:flex;gap:6px;align-items:center;pointer-events:none}.toolbar>*{pointer-events:auto}#status{margin-left:auto;color:#9bb5c9;background:#0c1e2ce8;padding:7px 12px;border-radius:8px;font-size:12px}
.edge{stroke:#48637b;stroke-width:1.1;opacity:.55}.edge.gold{stroke:#75e7ff;stroke-width:3;opacity:1}.edge-label{fill:#c5dbea;font-size:10px;paint-order:stroke;stroke:#0d1f2e;stroke-width:4px;pointer-events:none}.edge-label.context{opacity:0}.node text{fill:#d3e3f0;font-size:10px;paint-order:stroke;stroke:#0b1d2b;stroke-width:3px;pointer-events:none}.node{cursor:pointer}.node:hover text{fill:white;font-weight:700}.footer{position:absolute;bottom:16px;left:18px;background:#0e2233dc;border:1px solid #2e475b;border-radius:8px;padding:9px 13px;color:#abc2d3;font-size:12px;pointer-events:none}
#inspect{border-left:3px solid #73d3ef;padding-left:12px;margin:10px 0;font-size:12px}#inspect:empty{display:none}.steps{font-size:12px;padding-left:18px}.steps li{margin:5px 0}#question{font-size:16px;color:#f2f7fc}a{color:#7fdef5}.check{display:flex;align-items:center;gap:6px}.path-note{border:1px solid #385068;border-radius:6px;padding:10px;margin:12px 0;font-size:12px;color:#b1c5d6}@media(max-width:800px){body{grid-template-columns:290px 1fr}aside{padding:16px}.toolbar{flex-wrap:wrap}#status{margin-left:0}}
</style></head><body><aside>
<div class="eyebrow">Fictional research atlas</div><h1 id="heading"></h1><div id="question"></div><div id="chips" class="chips"></div>
<div id="question-controls"><h2>REFERENCE PATHS</h2><select id="path"><option value="all">All displayed reference paths</option></select><div class="path-note" id="path-note"></div><ol id="steps" class="steps"></ol></div>
<h2>FIND & INSPECT</h2><input type="search" id="search" aria-label="Find entity" placeholder="Find a name or entity ID…"><div id="inspect"></div>
<h2>ENTITY TYPES</h2><div id="legend" class="legend"></div><h2>RELATION FILTER</h2><select id="relation" aria-label="Relation filter"><option value="all">All relation operations</option></select>
<label class="check muted" style="margin-top:12px"><input type="checkbox" id="labels"> Show all relation labels</label>
<details id="metadata"><summary>Question metadata & answers</summary><div id="metadata-body"></div></details>
<p class="muted">Start: cyan ring · intermediate: violet ring · answers: amber ring. Other nodes provide context and distractors.</p>
<p class="muted">Edges show directed navigation operations. An <code>inv:</code> edge traverses a stored relation backwards. Context edges show the stored direction.</p>
</aside><main id="stage"><div class="toolbar"><button id="fit">Fit</button><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-out" aria-label="Zoom out">−</button><button id="relax">Re-layout</button><span id="status"></span></div><svg id="graph" role="img" aria-label="Interactive knowledge graph"><defs><marker id="arrow" viewBox="0 -5 10 10" refX="24" refY="0" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,-5L10,0L0,5" fill="#6f91aa"/></marker><marker id="gold-arrow" viewBox="0 -5 10 10" refX="22" refY="0" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,-5L10,0L0,5" fill="#75e7ff"/></marker></defs><g id="viewport"><g id="edges"></g><g id="nodes"></g></g></svg><div class="footer">Scroll to zoom · drag background to pan · drag nodes to arrange · click a node to inspect</div></main>
<script type="application/json" id="data">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const make=(tag,attrs={})=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e;};
const addText=(parent,tag,value)=>{const e=document.createElement(tag);e.textContent=value;parent.appendChild(e);return e;};
$('heading').textContent=data.mode;$('question').textContent=data.question?data.question.question:data.nodes.find(n=>n.id===data.start).name;
for(const value of [`${data.nodes.length} nodes`,`${data.edges.length} edges`,...(data.question?[`${data.question.hop_count} hops`,data.question.split]:[])]){const e=addText($('chips'),'span',value);e.className='chip';}
$('status').textContent=`Showing ${data.nodes.length} of ${data.graph_counts.nodes} entities`;
const selectedTypes=new Set(Object.keys(data.colors));
for(const[type,color]of Object.entries(data.colors)){const label=document.createElement('label');const check=document.createElement('input');check.type='checkbox';check.checked=true;check.style.marginRight='4px';const dot=document.createElement('span');dot.className='dot';dot.style.background=color;label.append(check,dot,document.createTextNode(type));$('legend').appendChild(label);check.onchange=()=>{check.checked?selectedTypes.add(type):selectedTypes.delete(type);applyFilters();};}
for(const relation of [...new Set(data.edges.map(e=>e.relation))].sort()){const option=addText($('relation'),'option',relation);option.value=relation;}
if(data.question){const q=data.question;for(let i=0;i<data.paths.length;i++){const path=data.paths[i],answer=data.nodes.find(n=>n.id===path.answer);const option=addText($('path'),'option',`${answer.name} · path ${path.index}`);option.value=String(i);}
$('path-note').textContent=`Showing up to ${data.limits.paths_per_answer} verified witnesses per answer. ${Object.values(q.answer_path_counts).reduce((a,b)=>a+b,0)} program-conforming walks exist. These reference paths need not be the only solutions.`;
const body=$('metadata-body');for(const[text,value]of Object.entries({'Question ID':q.id,'Starting entity':q.start_entity,'Answer IDs':q.answer_ids.join(', '),'Relation sequence':q.relation_path.join(' → '),'Navigation branching':q.difficulty.branching_factor.toFixed(2),'Distractor edge occurrences':q.difficulty.distractor_count,'Reference tool calls':q.difficulty.reference_tool_calls,'Shortest answer distances':JSON.stringify(q.shortest_path_hops)})){addText(body,'p',`${text}: ${value}`);}
}else{$('question-controls').hidden=true;$('metadata').hidden=true;}
const W=1200,H=850, graph=$('graph'), byId=new Map();graph.setAttribute('viewBox',`0 0 ${W} ${H}`);
data.nodes.forEach((node,i)=>{const theta=i*2.3999632297;node.x=node.layer!==null?150+node.layer*(900/Math.max(1,data.question?.hop_count||4)):600+Math.cos(theta)*(110+Math.sqrt(i)*22);node.y=425+Math.sin(theta)*(100+Math.sqrt(i)*16);node.vx=0;node.vy=0;byId.set(node.id,node);});
const edgeEls=data.edges.map(edge=>{const group=make('g'),line=make('line',{class:`edge${edge.gold?' gold':''}`,'marker-end':`url(#${edge.gold?'gold-arrow':'arrow'})`}),text=make('text',{class:`edge-label${edge.gold?'':' context'}`,'text-anchor':'middle'}),title=make('title');title.textContent=`${edge.from} — ${edge.relation} → ${edge.to}`;text.textContent=edge.relation;group.append(line,text,title);$('edges').appendChild(group);return {edge,group,line,text};});
const nodeEls=data.nodes.map(node=>{const group=make('g',{class:'node','data-entity':node.id,tabindex:0}),circle=make('circle',{r:node.role==='context'?8:14,fill:node.color,stroke:node.role==='start'?'#75e7ff':node.role==='answer'?'#ffd17a':node.role==='intermediate'?'#c2a4ff':'#112a3b','stroke-width':node.role==='context'?2:5}),text=make('text',{x:node.role==='context'?13:20,y:4}),title=make('title');text.textContent=node.name.length>31?node.name.slice(0,29)+'…':node.name;title.textContent=`${node.name}\n${node.id} · ${node.type} · ${node.role}`;group.append(circle,text,title);$('nodes').appendChild(group);group.addEventListener('pointerdown',event=>startNodeDrag(event,node));group.addEventListener('click',()=>inspect(node));group.addEventListener('keydown',event=>{if(event.key==='Enter')inspect(node);});return{node,group,circle};});
let transform={x:0,y:0,k:1}, iterations=0,animating=false,drag=null,pan=null;
function draw(){for(const{node,group}of nodeEls)group.setAttribute('transform',`translate(${node.x},${node.y})`);for(const{edge,line,text}of edgeEls){const a=byId.get(edge.from),b=byId.get(edge.to);line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);text.setAttribute('x',(a.x+b.x)/2);text.setAttribute('y',(a.y+b.y)/2-5);}}
function simulate(){if(iterations<=0){animating=false;return;}iterations--;const nodes=data.nodes;for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=nodes[i],b=nodes[j],dx=a.x-b.x,dy=a.y-b.y,d2=Math.max(dx*dx+dy*dy,150),f=750/d2;a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f;}
for(const e of data.edges){const a=byId.get(e.from),b=byId.get(e.to),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-(e.gold?190:110))*.0018;a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f;}
for(const n of nodes){const anchor=n.layer!==null?150+n.layer*(900/Math.max(1,data.question?.hop_count||4)):600;n.vx+=(anchor-n.x)*(n.layer!==null?.015:.0009);n.vy+=(425-n.y)*.001;n.vx*=.72;n.vy*=.72;if(n!==drag){n.x=Math.max(35,Math.min(1165,n.x+n.vx));n.y=Math.max(35,Math.min(815,n.y+n.vy));}}draw();requestAnimationFrame(simulate);}
function run(n=230){iterations=n;if(!animating){animating=true;requestAnimationFrame(simulate);}}
function updateTransform(){$('viewport').setAttribute('transform',`translate(${transform.x},${transform.y}) scale(${transform.k})`);}
function fit(){transform={x:0,y:0,k:1};updateTransform();}
function point(event){const p=graph.createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform(graph.getScreenCTM().inverse());}
function startNodeDrag(event,node){event.stopPropagation();drag=node;graph.setPointerCapture(event.pointerId);inspect(node);}
graph.addEventListener('pointerdown',event=>{const p=point(event);pan={x:p.x,y:p.y,tx:transform.x,ty:transform.y};graph.setPointerCapture(event.pointerId);});
graph.addEventListener('pointermove',event=>{const p=point(event);if(drag){drag.x=(p.x-transform.x)/transform.k;drag.y=(p.y-transform.y)/transform.k;drag.vx=drag.vy=0;draw();}else if(pan){transform.x=pan.tx+p.x-pan.x;transform.y=pan.ty+p.y-pan.y;updateTransform();}});
graph.addEventListener('pointerup',()=>{drag=null;pan=null;});graph.addEventListener('pointercancel',()=>{drag=null;pan=null;});
function zoom(f,p={x:600,y:425}){const k=Math.max(.2,Math.min(6,transform.k*f)),ratio=k/transform.k;transform.x=p.x-(p.x-transform.x)*ratio;transform.y=p.y-(p.y-transform.y)*ratio;transform.k=k;updateTransform();}
graph.addEventListener('wheel',event=>{event.preventDefault();zoom(event.deltaY<0?1.12:1/1.12,point(event));},{passive:false});$('fit').onclick=fit;$('zoom-in').onclick=()=>zoom(1.2);$('zoom-out').onclick=()=>zoom(1/1.2);$('relax').onclick=()=>run();
function inspect(node){$('inspect').replaceChildren();addText($('inspect'),'strong',node.name);addText($('inspect'),'p',`${node.id} · ${node.type} · ${node.role}`);const incident=data.edges.filter(e=>e.from===node.id||e.to===node.id);for(const e of incident.slice(0,12))addText($('inspect'),'p',`${e.from===node.id?'→':'←'} ${e.relation}: ${byId.get(e.from===node.id?e.to:e.from).name}`);if(incident.length>12)addText($('inspect'),'p',`+ ${incident.length-12} more displayed edges`);}
function applyFilters(){const search=$('search').value.toLowerCase(),relation=$('relation').value,chosen=$('path').value,selectedPath=chosen==='all'?null:data.paths[Number(chosen)],pathNodes=selectedPath?new Set(selectedPath.entities):null;const visible=new Set();
for(const{node,group,circle}of nodeEls){const show=selectedTypes.has(node.type);group.style.display=show?'':'none';if(show)visible.add(node.id);const match=!search||`${node.id} ${node.name}`.toLowerCase().includes(search);group.style.opacity=match&&(!pathNodes||pathNodes.has(node.id))?1:.18;circle.setAttribute('filter',match&&search?'drop-shadow(0px 0px 6px white)':'');}
for(const{edge,group,text}of edgeEls){group.style.display=visible.has(edge.from)&&visible.has(edge.to)&&(relation==='all'||relation===edge.relation)?'':'none';const onPath=!selectedPath||selectedPath.entities.some((id,i)=>id===edge.from&&selectedPath.entities[i+1]===edge.to&&data.question.relation_path[i]===edge.relation);group.style.opacity=onPath?1:.12;text.style.opacity=edge.gold||$('labels').checked?1:0;}
$('steps').replaceChildren();if(data.question){const path=selectedPath||data.paths[0];for(let i=0;i<data.question.relation_path.length;i++)addText($('steps'),'li',`${byId.get(path.entities[i]).name} → ${data.question.relation_path[i]} → ${byId.get(path.entities[i+1]).name}`);}}
$('search').oninput=applyFilters;$('relation').onchange=applyFilters;$('labels').onchange=applyFilters;$('path').onchange=applyFilters;draw();applyFilters();run();
window.kgInspector={data,byId,transform:()=>({...transform}),inspect,applyFilters};
</script></body></html>'''
