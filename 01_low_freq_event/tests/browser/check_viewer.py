"""Optional browser smoke check: pip install playwright; uses installed Google Chrome."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[2]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto('http://127.0.0.1:5001',wait_until='networkidle')
    assert response.status==200
    page.wait_for_selector('#window-select option', state='attached')
    data=page.locator('#data').evaluate('(n)=>JSON.parse(n.textContent)')
    windows={w['window_id']:w for w in data['windows']}
    window_count=len(windows)
    events=page.locator('#event-list .card').count()
    expected=sorted(data['events'],key=lambda e:(e['start_sec'],e['end_sec'],e['event_id']))
    assert events==len(expected)>0
    assert page.locator('#window-select').input_value()=='all'
    assert page.locator('#event-list .card').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.id)')==[e['event_id'] for e in expected]
    assert page.locator('#timeline .bar').count()==events
    assert page.locator('#entity-list .entity').count()==len(data['entities'])
    # A cross-window selection must keep every event visible, while switching
    # evidence and bbox lookup to the selected event's source window.
    for event in expected:
        page.locator(f'#event-list .card[data-id="{event["event_id"]}"]').click()
        assert page.locator('#window-select').input_value()=='all'
        assert page.locator('#event-list .card').count()==events
        assert page.locator('#frame-panel').get_attribute('data-window-id')==event['window_id']
        assert page.locator('#detail-description').inner_text()==event['description']
        window=windows[event['window_id']]
        assert page.locator('#all-frames .thumb').count()==len(window['frames'])
        if event['evidence']:
            frame=event['evidence'][0]
            assert page.locator('#frame-image').get_attribute('src')==frame['image']
            observations=[o for o in window['observations'] if o['frame_index']==frame['frame_index']]
            assert page.locator('#frame-boxes .box').count()==len(observations)
            assert page.locator('#frame-boxes .box').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.entityId)')==[o['entity_id'] for o in observations]
            for obs in observations:
                expected_label=page.evaluate('(id)=>entityLabel(id)',obs['entity_id'])
                assert page.locator(f'#frame-boxes .box[data-entity-id="{obs["entity_id"]}"]').inner_text().startswith(expected_label)
            assert page.locator('#detail .participant[data-entity-id]').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.entityId)')==list(dict.fromkeys(event['entity_ids']))
    later=next((e for e in expected if e['window_id']!=expected[0]['window_id']),expected[-1])
    page.locator(f'#timeline .bar[data-id="{later["event_id"]}"]').click()
    assert page.locator('#frame-panel').get_attribute('data-window-id')==later['window_id']
    assert page.locator('#event-list .card').count()==events
    # Events outside the initial window are searchable from the global list.
    page.locator('#search').fill(later['description'])
    assert page.locator(f'#event-list .card[data-id="{later["event_id"]}"]').count()==1
    assert page.locator('#event-list .card').count()==page.locator('#timeline .bar').count()
    page.locator('#search').fill('__no_matching_event__')
    assert page.locator('#event-list .card').count()==0
    assert '검색 조건에 맞는 이벤트가 없습니다.' in page.locator('#event-list').inner_text()
    page.locator('#search').fill('')
    assert page.locator('#event-list .card').count()==events
    page.locator('#event-list .card').first.click()
    assert '프레임 chain' in page.locator('#detail').inner_text()
    page.locator('#tab-frames').click()
    page.locator('#all-frames .thumb').last.click()
    page.wait_for_function("document.getElementById('frame-image').complete && document.getElementById('frame-image').naturalWidth>0")
    assert page.locator('#frame-boxes .box').count()>0
    # Wait for the requested decoded frame, not a late callback from an earlier
    # event selection, before testing a repeat click on a settled paused frame.
    page.wait_for_function("!video.seeking && displayedFrameTime!==null && Math.abs(displayedFrameTime-currentFrame.time_sec)<.5/data.video.source_fps && document.querySelectorAll('#video-boxes .box').length>0")
    page.locator('#all-frames .thumb').last.click()
    assert page.locator('#video-boxes .box').count()>0
    page.locator('#boxes').uncheck()
    assert page.locator('#frame-boxes .box').count()==0
    page.locator('#boxes').check()
    page.locator('#play-event').click()
    page.wait_for_timeout(500)
    assert page.locator('#video').evaluate('(v)=>v.readyState>=2&&v.currentTime>0')
    page.locator('#video').evaluate('(v)=>v.pause()')
    # A sampled bbox must disappear on the next decoded, unobserved frame.
    # timeupdate-based drawing left the 0s boxes visible until roughly 0.2s.
    page.locator('#tab-events').click()
    page.locator('#event-list .card').first.click()
    stale=page.evaluate('''async()=>{
        const v=document.getElementById('video');
        v.currentTime=0;
        await new Promise(resolve=>v.addEventListener('seeked',resolve,{once:true}));
        return await new Promise(async resolve=>{
            let count=0;const stale=[];
            function tick(now,metadata){
                const boxes=document.querySelectorAll('#video-boxes .box').length;
                const observed=current.observations.some(o=>Math.abs(o.time_sec-metadata.mediaTime)<.5/data.video.source_fps);
                if(boxes&&!observed)stale.push({time:metadata.mediaTime,boxes});
                if(++count<65)v.requestVideoFrameCallback(tick);
                else{v.pause();resolve(stale)}
            }
            v.requestVideoFrameCallback(tick);await v.play();
        });
    }''')
    assert not stale,stale
    # A real browser range request verifies seeking support.
    ranged=page.request.get('http://127.0.0.1:5001/video.mp4',headers={'Range':'bytes=0-31'})
    assert ranged.status==206 and len(ranged.body())==32
    for window_id,window in windows.items():
        page.locator('#window-select').select_option(window_id)
        assert page.locator('#all-frames .thumb').count()==len(window['frames'])
        assert page.locator('#event-list .card').count()==len(window['events'])
        assert page.locator('#timeline .bar').count()==len(window['events'])
        if page.locator('#event-list .card').count():
            page.locator('#event-list .card').last.click()
            assert '프레임 chain' in page.locator('#detail').inner_text()
        for frame in window['frames']:
            if frame.get('warnings'):
                page.locator('#tab-frames').click()
                page.locator(f'#all-frames .thumb[data-index="{frame["frame_index"]}"]').click()
                assert 'bbox 검출 실패' in page.locator('#frame-caption').inner_text()
    page.locator('#window-select').select_option('all')
    assert page.locator('#event-list .card').count()==events
    page.locator(f'#event-list .card[data-id="{later["event_id"]}"]').click()
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(root/'outputs/viewer-desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(root/'outputs/viewer-mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+2')
    assert not errors,errors
    print(json.dumps({'status':'passed','window_count':window_count,'events_shown_by_default':events,'range_status':ranged.status,'browser_errors':errors},ensure_ascii=False))
    browser.close()
