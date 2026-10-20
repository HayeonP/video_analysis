"""With both Sintel viewers running, check every shot button in a real browser."""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
 for port in [5004,5006]:
  page=b.new_page();page.goto(f'http://127.0.0.1:{port}');page.wait_for_function("document.querySelector('#video').readyState>=2")
  errors=[]
  buttons=page.locator('#shots button')
  for i in range(buttons.count()):
   wanted=buttons.nth(i).get_attribute('data-shot');buttons.nth(i).click();page.wait_for_timeout(150)
   actual=page.locator('#shots .active').get_attribute('data-shot') if page.locator('#shots .active').count() else None
   if wanted!=actual:errors.append([wanted,actual,page.locator('#video').evaluate('(v)=>v.currentTime')])
  assert not errors, (port, errors)
  print(f"{port}: all {buttons.count()} shot buttons passed");page.close()
 b.close()
