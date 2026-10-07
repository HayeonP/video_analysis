#!/usr/bin/env python3
"""Render the same vector figures as a portable PDF and visual previews.

Uses the host's librsvg/Cairo/fontconfig; no ML, browser, or network calls.
"""
import ctypes as C
from html import escape
from pathlib import Path
from PIL import Image, ImageDraw
from build_method_slides import opening_slides, ROOT
from method_review_figures import review_slides
from method_diagrams import *

class Viewport(C.Structure):
    _fields_=[('x',C.c_double),('y',C.c_double),('width',C.c_double),('height',C.c_double)]

def bind(lib,name,ret,args):
    fn=getattr(lib,name);fn.restype=ret;fn.argtypes=args;return fn

def main():
    cairo=C.CDLL('libcairo.so.2');rsvg=C.CDLL('librsvg-2.so.2');gobject=C.CDLL('libgobject-2.0.so.0');fc=C.CDLL('libfontconfig.so.1')
    ptr=C.c_void_p;double=C.c_double
    font_add=bind(fc,'FcConfigAppFontAddFile',C.c_int,[ptr,C.c_char_p])
    font_add(None,str(ROOT/'assets/NanumGothic-Regular.ttf').encode())
    new=bind(rsvg,'rsvg_handle_new_from_data',ptr,[C.c_char_p,C.c_size_t,ptr])
    render=bind(rsvg,'rsvg_handle_render_document',C.c_int,[ptr,ptr,C.POINTER(Viewport),ptr])
    unref=bind(gobject,'g_object_unref',None,[ptr])
    surface=bind(cairo,'cairo_image_surface_create',ptr,[C.c_int,C.c_int,C.c_int])
    pdf=bind(cairo,'cairo_pdf_surface_create',ptr,[C.c_char_p,double,double])
    create=bind(cairo,'cairo_create',ptr,[ptr]);destroy=bind(cairo,'cairo_destroy',None,[ptr])
    finish=bind(cairo,'cairo_surface_finish',None,[ptr]);destroy_surface=bind(cairo,'cairo_surface_destroy',None,[ptr])
    png=bind(cairo,'cairo_surface_write_to_png',C.c_int,[ptr,C.c_char_p])
    nextpage=bind(cairo,'cairo_show_page',None,[ptr])
    pdf_dir=ROOT/'presentation/output/pdf';pdf_dir.mkdir(parents=True,exist_ok=True)
    preview_dir=ROOT/'presentation/output/previews';preview_dir.mkdir(parents=True,exist_ok=True)
    target=pdf_dir/'METHOD_SLIDES.pdf';pdf_surface=pdf(str(target).encode(),960,540);pdf_ctx=create(pdf_surface)
    slides=opening_slides()+review_slides()
    preview_paths=[]
    for i,s in enumerate(slides,1):
        phase={'observe':TEAL,'link':BLUE,'review':PURPLE,'export':INK,'all':INK}[s['phase']]
        content=rect(0,0,1600,900,PAPER,PAPER,0)
        content+=rect(80,51,25,3,phase,phase,0)+text(120,60,s['kicker'],12,MUTED,700)
        content+=text(80,118,s['title'],38,INK,700)+text(80,156,s['subtitle'],19,MUTED)
        figure=svg(s['body'],i,s['title']).replace('viewBox="0 0 1440 570"','x="80" y="188" width="1440" height="570" viewBox="0 0 1440 570"')
        content+=figure
        content+=rect(80,779,3,31,phase,phase,0)+text(98,803,s['takeaway'],20,INK)
        content+=line(80,850,1520,850,LINE,1)
        for x,label,p in [(80,'관측 · 중복 정리','observe'),(249,'외형 연결','link'),(376,'Qwen 검토','review'),(522,'결과 기록','export')]:
            col={'observe':TEAL,'link':BLUE,'review':PURPLE,'export':INK}[p] if s['phase'] in {p,'all'} else MUTED
            content+=circle(x+3,870,2.5,col,col,0)+text(x+14,874,label,12,col,600)
        content+=text(1320,875,'04 / ENTITY EXTRACTION',11,MUTED,400,'end')+text(1520,875,f'{i:02d} / {len(slides):02d}',12,INK,400,'end')
        doc=f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" font-family="NanumGothic, sans-serif">{content}</svg>'
        data=doc.replace('MethodSans,system-ui,sans-serif','NanumGothic,sans-serif').encode()
        handle=new(data,len(data),None)
        if not handle:raise RuntimeError(f'Cannot render SVG page {i}')
        if not render(handle,pdf_ctx,C.byref(Viewport(0,0,960,540)),None):raise RuntimeError(f'PDF render failed: {i}')
        nextpage(pdf_ctx)
        img=surface(0,1600,900);ctx=create(img)
        if not render(handle,ctx,C.byref(Viewport(0,0,1600,900)),None):raise RuntimeError(f'Preview render failed: {i}')
        preview=preview_dir/f'{i:02d}.png'
        if png(img,str(preview).encode()):raise RuntimeError('Cannot save preview')
        preview_paths.append(preview)
        destroy(ctx);destroy_surface(img);unref(handle)
    destroy(pdf_ctx);finish(pdf_surface);destroy_surface(pdf_surface)
    sheet=Image.new('RGB',(1640,4*250+20),'#e5e8e5')
    draw=ImageDraw.Draw(sheet)
    for i,path in enumerate(preview_paths):
        with Image.open(path) as image:thumbnail=image.convert('RGB').resize((390,220))
        x=20+(i%4)*405;y=20+(i//4)*250;sheet.paste(thumbnail,(x,y));draw.text((x+3,y+224),f'{i+1:02d}',fill='#132638')
    sheet.save(preview_dir/'contact_sheet.png')
    print(f'Created {target} ({len(slides)} pages) and slide preview PNGs.')

if __name__=='__main__':main()
