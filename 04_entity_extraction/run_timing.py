"""Wall-clock timing for the complete CLI run and its nested processing stages."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def format_duration(seconds):
    centiseconds=round(max(0,seconds)*100)
    hours,remainder=divmod(centiseconds,360000)
    minutes,remainder=divmod(remainder,6000)
    secs=remainder/100
    return (f'{hours}시간 ' if hours else '')+f'{minutes}분 {secs:05.2f}초'


class RunTiming:
    def __init__(self,output,*,started,started_at,mode,reuse_qwen=None,reuse_yolo=None):
        self.path=Path(output)/'timing.json'
        self.started=started
        self.stack=[]
        self.data=dict(status='running',started_at_utc=started_at,mode=mode,
            startup_elapsed_sec=time.perf_counter()-started,
            measurement='wall clock using time.perf_counter',
            scope='Python script startup before heavy imports through analysis and final comparison export; excludes shell setup and interpreter startup',
            cache_note='Existing per-frame/model-response caches may be reused. Stage times include cache lookup, not necessarily fresh inference.',
            requested_cache_sources=dict(qwen=str(reuse_qwen) if reuse_qwen else None,
                yolo=str(reuse_yolo) if reuse_yolo else None),stages=[])
        self.save()

    def save(self):
        self.data['elapsed_sec']=time.perf_counter()-self.started
        self.data['elapsed_human']=format_duration(self.data['elapsed_sec'])
        self.data['updated_at_utc']=utc_now()
        temporary=None
        try:
            with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=self.path.parent,delete=False) as f:
                temporary=f.name
                json.dump(self.data,f,ensure_ascii=False,indent=2)
                f.write('\n')
            os.replace(temporary,self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):os.unlink(temporary)

    @contextmanager
    def stage(self,name,label,*,kind='stage'):
        started=time.perf_counter()
        row=dict(name=name,label=label,kind=kind,parent=self.stack[-1] if self.stack else None,
            status='running',started_at_utc=utc_now())
        self.data['stages'].append(row);self.stack.append(name);self.save()
        try:
            yield
        except BaseException as exc:
            row.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
            raise
        else:
            row['status']='complete'
        finally:
            row.update(elapsed_sec=time.perf_counter()-started,finished_at_utc=utc_now())
            row['elapsed_human']=format_duration(row['elapsed_sec'])
            self.stack.pop();self.save()
            print(f'[작업시간] {label}: {row["elapsed_human"]} ({row["elapsed_sec"]:.2f}초, {row["status"]})',flush=True)

    def finish(self,status,*,error=None):
        self.data.update(status=status,finished_at_utc=utc_now())
        if error is not None:self.data['error']=error
        self.save()
        print(f'\n[전체 작업시간] {self.data["elapsed_human"]} ({self.data["elapsed_sec"]:.2f}초, {status})',flush=True)
        print(f'[시간 기록] {self.path.resolve()}',flush=True)
