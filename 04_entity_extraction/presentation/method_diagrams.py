"""Small vector primitives for the method presentation; no analysis imports."""
from html import escape

INK='#132638'; MUTED='#657581'; LINE='#dce2e5'; PAPER='#f7f6f2'
TEAL='#127f77'; BLUE='#2c65c8'; PURPLE='#7952b3'; AMBER='#ad7625'
PALE_TEAL='#e5f2ee'; PALE_BLUE='#eaf0fb'; PALE_PURPLE='#f0eafa'; PALE_AMBER='#fbf1dd'

def text(x,y,value,size=23,color=INK,weight=400,anchor='start'):
    rows=value if isinstance(value,(list,tuple)) else [value]
    spans=''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.5}">{escape(str(row))}</tspan>' for i,row in enumerate(rows))
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{spans}</text>'

def rect(x,y,w,h,fill='white',stroke=LINE,r=14,sw=1.8,dash=False):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' stroke-dasharray="8 7"' if dash else '')+'/>'

def circle(x,y,r,fill='white',stroke=LINE,sw=2):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def line(x1,y1,x2,y2,color=LINE,sw=2,dash=False):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"'+(' stroke-dasharray="8 7"' if dash else '')+'/>'

def arrow(x1,y1,x2,y2,color=BLUE,sw=2.5,dash=False,bend=None):
    key={TEAL:'teal',BLUE:'blue',PURPLE:'purple',AMBER:'amber',MUTED:'gray',INK:'ink'}.get(color,'gray')
    d=f'M{x1} {y1} L{x2} {y2}' if bend is None else f'M{x1} {y1} C{bend} {y1} {bend} {y2} {x2} {y2}'
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arr-{key})"'+(' stroke-dasharray="8 7"' if dash else '')+'/>'

def chip(x,y,label,color=TEAL,fill=PALE_TEAL,w=None):
    w=w or max(80,len(label)*15+28)
    return rect(x,y,w,36,fill,fill,r=18)+text(x+w/2,y+25,label,17,color,600,'middle')

def person(x,y,scale=1,color=TEAL,pose='front'):
    """Stylized same-subject icon, centered at x with top y, ~80x150."""
    face=circle(0,22,19,'#e6c7a8',color,2)
    hair=f'<path d="M-19 21 Q-21 -4 0 1 Q23 0 20 23 L11 14 Q-2 20 -14 11 Z" fill="{color}"/>'
    face+=hair
    if pose=='scarf':face+=f'<path d="M-17 27 Q0 23 17 27 L15 40 L-15 40Z" fill="{color}"/>'
    elif pose!='back':face+=circle(-6,23,1.4,INK,INK,0)+circle(6,23,1.4,INK,INK,0)
    body=f'<path d="M-20 48 Q0 40 20 48 L34 96 L23 99 L15 72 L18 115 L-18 115 L-15 72 L-23 99 L-34 96Z" fill="{color}"/>'
    legs=f'<path d="M-13 114 L-15 148 M13 114 L16 148" stroke="{INK}" stroke-width="11" stroke-linecap="round"/>'
    return f'<g transform="translate({x} {y}) scale({scale})">{face}{body}{legs}</g>'

def creature(x,y,scale=1,color=AMBER):
    return f'<g transform="translate({x} {y}) scale({scale})"><path d="M-12 0 Q-42 -43 -70 -21 L-38 13 L-15 12 Q-4 28 13 13 L32 4 Q58 1 71 -18 Q32 -13 15 -3 Q10 -19 -2 -16Z" fill="{color}"/><circle cx="4" cy="-9" r="2" fill="white"/></g>'

def frame(x,y,w,h,color=TEAL,subject='person',pose='front',pos=.5,boxed=True,label=None):
    body=rect(x,y,w,h,'#edf0ef',LINE,r=12)
    body+=f'<path d="M{x+1} {y+h*.75} L{x+w*.26} {y+h*.42} L{x+w*.48} {y+h*.7} L{x+w*.75} {y+h*.3} L{x+w-1} {y+h*.62} V{y+h-1} H{x+1}Z" fill="#dfe5e2"/>'
    scale=h/215;px=x+w*pos;py=y+h*.20
    if subject=='person':body+=person(px,py,scale,color,pose)
    else:body+=creature(px,y+h*.5,w/190,color)
    if boxed:
        if subject=='person':body+=rect(px-39*scale,py-6*scale,78*scale,167*scale,'none',color,3,2.3)
        else:body+=rect(x+w*.10,y+h*.21,w*.8,h*.52,'none',color,3,2.3)
    if label:body+=chip(x+9,y+9,label,color,'white',max(62,len(label)*11+18))
    return body

def card(x,y,w,h,title,color=TEAL,fill='white',subtitle=None):
    out=rect(x,y,w,h,fill,LINE)+text(x+24,y+39,title,24,color,700)
    if subtitle:out+=text(x+24,y+70,subtitle,18,MUTED)
    return out

def vector(x,y,color=BLUE,n=12,w=190,h=44):
    out=''
    for i in range(n):
        value=((i*7+3)%11)/11
        out+=f'<rect x="{x+i*w/n}" y="{y+h*(1-value)}" width="{w/n-4}" height="{max(5,h*value)}" rx="2" fill="{color}" opacity="{.4+.6*value}"/>'
    return out

def check(x,y,color=TEAL):
    return circle(x,y,15,color,color)+f'<path d="M{x-7} {y} L{x-2} {y+5} L{x+8} {y-6}" fill="none" stroke="white" stroke-width="2.5"/>'

def svg(body,index,title):
    markers=''.join(f'<marker id="arr-{key}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9 Z" fill="{color}"/></marker>' for key,color in [('teal',TEAL),('blue',BLUE),('purple',PURPLE),('amber',AMBER),('gray',MUTED),('ink',INK)])
    result=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 570" role="img" aria-labelledby="fig-title-{index}" font-family="MethodSans,system-ui,sans-serif"><title id="fig-title-{index}">{escape(title)}</title><defs>{markers}</defs>{body}</svg>'
    return result.replace('arr-',f'arr-{index}-')
