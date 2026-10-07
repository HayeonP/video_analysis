"""Real click/wheel regression for the fixed video and two event scroll panes."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def rect(page, selector):
    return page.locator(selector).bounding_box()


def assert_fixed(page, original):
    actual = rect(page, '#video')
    assert all(abs(actual[key] - original[key]) < 1 for key in original), (original, actual)
    assert page.evaluate('scrollY === 0 && scrollX === 0')
    assert page.evaluate('document.scrollingElement.scrollHeight <= innerHeight + 1')
    assert page.evaluate('document.scrollingElement.scrollWidth <= innerWidth + 1')


def assert_selection(page, event_id):
    # Test geometry against the visible pane, not just a CSS selected class.
    for container, item in [('#event-list', '.card'), ('#timeline-scroll', '.bar')]:
        selected = page.locator(f'{container} {item}.selected')
        assert selected.count() == 1
        assert selected.get_attribute('data-id') == event_id
        assert selected.get_attribute('aria-pressed') == 'true'
        geometry = selected.evaluate('''(item, selector) => {
            const pane = document.querySelector(selector), p = pane.getBoundingClientRect(), r = item.getBoundingClientRect();
            const top = p.top + pane.clientTop;
            return {top: r.top, bottom: r.bottom, paneTop: top, paneBottom: top + pane.clientHeight,
                    height: r.height, paneHeight: pane.clientHeight,
                    centerError: Math.abs((r.top + r.bottom) / 2 - (top + pane.clientHeight / 2)),
                    scroll: pane.scrollTop, maxScroll: pane.scrollHeight - pane.clientHeight};
        }''', container)
        assert geometry['bottom'] > geometry['paneTop'] and geometry['top'] < geometry['paneBottom'], geometry
        if geometry['height'] <= geometry['paneHeight']:
            assert geometry['top'] >= geometry['paneTop'] - 1, geometry
            assert geometry['bottom'] <= geometry['paneBottom'] + 1, geometry
        if 1 < geometry['scroll'] < geometry['maxScroll'] - 1:
            assert geometry['centerError'] <= 1, geometry


def wheel(page, selector, delta):
    box = rect(page, selector)
    page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
    page.mouse.wheel(0, delta)
    page.wait_for_timeout(120)


def click_visible_item(page, container, item):
    # Choose an already-visible item after a real wheel scroll. No Playwright
    # automatic scrollIntoView or document positioning may hide a layout bug.
    target = page.locator(container).evaluate('''(pane, selector) => {
        const p = pane.getBoundingClientRect(), low = p.top + pane.clientHeight * .28;
        for (const item of pane.querySelectorAll(selector)) {
            const r = item.getBoundingClientRect();
            const left = Math.max(r.left, p.left), right = Math.min(r.right, p.right - 18);
            if (r.top >= low && r.top < p.top + pane.clientHeight - 8 && right > left) {
                return {id:item.dataset.id, x:(left+right)/2, y:(r.top+Math.min(r.bottom,p.top+pane.clientHeight-4))/2};
            }
        }
    }''', item)
    assert target, (container, item)
    page.mouse.click(target['x'], target['y'])
    assert_selection(page, target['id'])
    return target['id']


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
    checks = []
    for width, height in [(1440, 900), (1366, 768), (1280, 720), (390, 844)]:
        page = browser.new_page(viewport={'width': width, 'height': height})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:5001', wait_until='networkidle')
        page.wait_for_function('video.readyState >= 2 && video.clientHeight > 0')
        video_rect = rect(page, '#video')
        assert video_rect['y'] >= 0 and video_rect['y'] + video_rect['height'] <= height
        assert_fixed(page, video_rect)
        initial = page.locator('#event-list .selected').get_attribute('data-id')
        assert_selection(page, initial)
        selected_ids = []
        for delta in [1700, 3100, -2200]:
            before = page.locator('#event-list').evaluate('(n) => n.scrollTop')
            wheel(page, '#timeline-scroll', delta)
            assert page.locator('#event-list').evaluate('(n) => n.scrollTop') == before
            assert_fixed(page, video_rect)
            selected_ids.append(click_visible_item(page, '#timeline-scroll', '.bar'))
            assert_fixed(page, video_rect)

            before = page.locator('#timeline-scroll').evaluate('(n) => n.scrollTop')
            wheel(page, '#event-list', -delta)
            assert page.locator('#timeline-scroll').evaluate('(n) => n.scrollTop') == before
            assert_fixed(page, video_rect)
            selected_ids.append(click_visible_item(page, '#event-list', '.card'))
            assert_fixed(page, video_rect)

        # Scroll past each pane's boundaries; neither may pass scroll to the page.
        for selector in ['#timeline-scroll', '#event-list', '#video', 'header']:
            for delta in [100000, 100000, -100000, -100000]:
                wheel(page, selector, delta)
                assert_fixed(page, video_rect)

        # Supporting information also stays in the fixed sidebar; a timeline
        # click restores the event list and reveals its corresponding card.
        for tab in ['detail', 'frames', 'entities', 'records']:
            page.locator(f'#tab-{tab}').click()
            panel = page.locator(f'#tab-{tab}').get_attribute('aria-controls')
            assert page.locator(f'#{panel}').is_visible()
            wheel(page, f'#{panel}', 1000)
            assert_fixed(page, video_rect)
            wheel(page, '#timeline-scroll', 2000)
            click_visible_item(page, '#timeline-scroll', '.bar')
            assert page.locator('#events-panel').is_visible()
            assert_fixed(page, video_rect)

        # Changing scope and filtering keep the synchronized selection behavior.
        page.locator('#window-select').select_option(index=4)
        scope_first = page.locator('#event-list .selected').get_attribute('data-id')
        assert_selection(page, scope_first)
        wheel(page, '#timeline-scroll', 500)
        click_visible_item(page, '#timeline-scroll', '.bar')
        assert_fixed(page, video_rect)
        page.locator('#window-select').select_option('all')
        # Animation clips may contain no arm events. Search for an action
        # actually present in the current result, rather than a stage-video label.
        common_action = page.evaluate('''() => {
            const counts = new Map();
            for (const event of data.events) counts.set(event.action, (counts.get(event.action) || 0) + 1);
            return [...counts].sort((a,b)=>b[1]-a[1])[0][0];
        }''')
        page.locator('#search').fill(common_action)
        wheel(page, '#timeline-scroll', 900)
        click_visible_item(page, '#timeline-scroll', '.bar')
        assert_fixed(page, video_rect)
        page.locator('#search').fill('')
        wheel(page, '#timeline-scroll', 2000)
        click_visible_item(page, '#timeline-scroll', '.bar')
        assert_fixed(page, video_rect)

        # The fitted video and overlay retain the original source aspect ratio.
        aspect = page.evaluate('data.video.width / data.video.height')
        assert abs(video_rect['width'] / video_rect['height'] - aspect) < .005
        overlay = rect(page, '#video-boxes')
        assert all(abs(overlay[key] - video_rect[key]) < 1 for key in video_rect)
        assert not errors, errors
        page.screenshot(path=str(ROOT / f'outputs/fixed_viewer_{width}x{height}.png'))
        checks.append({'viewport': [width, height], 'video': video_rect, 'selected_events': selected_ids,
                       'page_scroll': 0, 'bidirectional_centering': True, 'browser_errors': errors})
        page.close()
    browser.close()
    result = {'status': 'passed', 'checks': checks}
    (ROOT / 'outputs/fixed_viewer_navigation_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
