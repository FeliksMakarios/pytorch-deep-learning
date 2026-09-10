"""Execute each notebook in a fresh Python process without a socket-based kernel."""
from pathlib import Path
import ast, base64, contextlib, io, json, os, subprocess, sys, time, traceback
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def execute(path):
 os.chdir(ROOT)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 book=nbformat.read(path,as_version=4)
 namespace={'__name__':'__main__'}
 count=0
 for cell in book.cells:
  if cell.cell_type!='code': continue
  count+=1
  cell.execution_count=count
  cell.outputs=[]
  stream=io.StringIO()
  def show(*args,**kwargs):
   for fig_id in plt.get_fignums():
    buffer=io.BytesIO()
    plt.figure(fig_id).savefig(buffer,format='png',bbox_inches='tight')
    cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()}))
   plt.close('all')
  plt.show=show
  try:
   with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    tree=ast.parse(cell.source)
    # Preserve the visible value of a final expression, as a notebook would.
    last=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
    exec(compile(tree,str(path),'exec'),namespace)
    if last:
     value=eval(compile(ast.Expression(last.value),str(path),'eval'),namespace)
     if value is not None: print(repr(value))
   if stream.getvalue(): cell.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
  except Exception:
   print(f'FAILED CELL {count}:\n{cell.source}\n{stream.getvalue()}',flush=True)
   raise
 nbformat.validate(book)
 nbformat.write(book,path)
 print(json.dumps({'file':path.name,'status':'passed','code_cells':count}))

if len(sys.argv)>1 and sys.argv[1] != '--with-hf':
 execute(Path(sys.argv[1]))
else:
 results=[]
 for path in sorted((ROOT/'notebooks').glob('*.ipynb' if '--with-hf' in sys.argv else '0*.ipynb')):
  start=time.perf_counter()
  r=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(path)],cwd=ROOT,capture_output=True,text=True,timeout=900)
  print(r.stdout,flush=True)
  if r.returncode: print(r.stderr,flush=True)
  record={'file':path.name,'status':'passed' if r.returncode==0 else 'failed','seconds':round(time.perf_counter()-start,2),'method':'fresh Python process, sequential code-cell execution'}
  if r.returncode: record['error']=r.stderr[-4000:]
  results.append(record)
 (ROOT/'execution_report.json').write_text(json.dumps(results,indent=2))

 if any(r['status'] != 'passed' for r in results):
  sys.exit(1)
