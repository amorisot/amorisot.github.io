# Claude Code Guidelines for amorisot.github.io

## Project Structure
- `/static/` - Self-contained generative art pieces (HTML files with inline CSS/JS)
- `/static/style.css` - Global stylesheet for non-art pages
- Art pages are standalone HTML files that don't need external dependencies

## Generative Art Patterns
- Use `WanderingPoint` class for smooth, organic movement (see stickman.html)
- Animation loop: `requestAnimationFrame` with delta-time updates
- Canvas-based pieces should handle `window.resize` events
- Keep art pages self-contained (no external JS dependencies)

## Screenshots for Visual Iteration
When developing visual/generative art, use playwright for screenshots:

```bash
# Basic screenshot (wait for animations to settle)
npx playwright screenshot --wait-for-timeout=2000 static/myart.html /tmp/screenshot.png

# Interactive testing (e.g., clicking a toggle button)
cat > /tmp/test_interaction.js << 'EOF'
const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto('file:///full/path/to/static/myart.html');
    await page.waitForTimeout(1000);
    await page.click('#toggle');  // or other selector
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/result.png' });
    await browser.close();
})();
EOF
node /tmp/test_interaction.js
```

Then read the screenshot with the Read tool to analyze and iterate on the design.
