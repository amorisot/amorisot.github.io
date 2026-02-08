import subprocess
from datetime import datetime

import re

BLOG_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <link rel="icon" type="image/png" href="/favicon.png"/>
    <link rel="stylesheet" href="/static/style.css">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{description}">
    <meta property="og:title" content="{title} - Adrien Morisot">
    <meta property="og:description" content="{description}">
    <meta property="og:type" content="article">
    <meta property="og:url" content="{og_url}">
    <link rel="canonical" href="{og_url}">
    <title>Blog - {title}</title>
    {structured_data}
  </head>
  <body>
    <div id="menu">
      <li><a href="/about">🏡</a></li>
      <li><a href="/art">👨‍🎨</a></li>
      <li><a href="/ml">🤖</a></li>
      <li><a href="/books">📚</a></li>
      <li><a href="/blog">📝</a></li>
    </div>
    <div id="left">
      &nbsp;
    </div>
    <div id="content">
      {content}
    </div>
    <script data-goatcounter="https://amorisot.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
  </body>
</html>
""".strip()

# Each post has: content, published (required), updated (optional)
POSTS = {
  "hi": {
    "published": "2024-10-20",
    "content": """I am thinking a lot about reasoning in LLMs, and will probably write a blogpost about it soon.

Thanks for stopping by :)
""".strip(),
  },

  "why don't they make chickens without the salmonella?": {
    "published": "2025-07-01",
    "content": """
The question was originally facetious and rhetorical, but the answer is surprising.

I learnt most from this <a href="https://www.reddit.com/r/askscience/comments/ncfik/comment/c3816vb/">Reddit comment from more than a decade ago</a>.

<hr>

Turning whole, live chickens into pieces in a Colonel bucket or shrink-wrapped in a styro tray, is done in a very different way than segmenting range animals into consumer-friendly portions. The very specialized industrial process developed to deal specifically with poultry is actually the principal cause of systemic contamination of the meat itself.

Unlike cattle and hogs, chickens are inconveniently covered in (surprise!) feathers. After they are killed, the birds must be plucked before anything else can be done to them. In order for the plucking machines to work properly, the carcasses are dunked into a large vat of hot water, which relaxes the dermis and allows the feathers to be removed. Unfortunately, skin isn't the only organ the hot bath relaxes: the chickens, which have been killed mere seconds before, leak feces from their now well-relaxed bowels, right into the scald tank. The first bird of the day essentially turns the tank into a large pot of hot fecal soup; subsequent carcasses, usually tens of thousands per day, enrich the fecal contamination exponentially. All birds are submerged in this cauldron of concentrated bacterial broth for several minutes, before they continue down the process line.

Fresh out of the contaminated scalding tank, after a few seconds at the plucking station, the carcass is mechanically eviscerated. The residual fecal slop still clinging to the skin contaminates the gutting machine, which in turn inoculates the body cavity, splashing fluids and infecting the whole bird from the inside. The carcass is then washed again, usually in another (dirty) communal tank, but by now it is virtually impossible to remove all of the liquified, dilute fecal material from every nook, cranny, scrap, cut and fold of flesh. Some of the bacteria have a reproductive cycle of just a few minutes, so even after a thorough washing, just a few residual bacteria can re-contaminate the carcass in a very short time. (One surviving bacterium, reproducing every ~6 min = 1 million in 2 hr, 1 trillion in 4 hrs.)

The presence of of E. Coli, salmonella and camphylobacter in virtually ALL consumer chicken, including organic and farmer's markets, means it is best to keep raw chicken isolated (well-wrapped) and well refrigerated (bottom shelf). Prepare it as soon as possible, and cook it thoroughly. If you can't use it right away, immerse it fully in a strong brine or acidic marinade (zipper bags are good) to slow pathogen growth. Makes it moist and tasty, too!
""".strip(),
  },

  "10T vs 100T parameter sparse MoE's": {
    "published": "2025-09-11",
    "content": """As of September 2025, LLMs are spikily impressive: they outperform most people in mathematical and coding abilities, and they speak more languages fluently than any one person.

The depth and breadth of even small LLM's knowledge is staggering. However, it is important to remember that their competition is stiff, when it comes to performing tasks in the world.

Current generation LLMs --at most ~10T parameter sparse MoEs-- are competing against the human brain, which are 100T parameter, highly efficient, multimodal MoEs, pretrained on decades of experience, post-trained in university, RLVR'ed/RLHF'ed on the job. Humans should not be slept on.
""".strip(),
  },

  "a cute way of passing time while vibe coding": {
    "published": "2025-12-22",
    "content": """
While the model is chugging along, a pleasant and somewhat brain-dead way of passing the time is asking the model for a recreation in code of some work of art or biological phenomena.

It is low stakes, helps me refine my understanding of what models can and cannot do, and the output usually sparks joy --independent of whether or not the model did the task successfully!

I've started compiling the successful outputs of these efforts in the later parts of my <a href="/art">art (👨‍🎨)</a> page. See if you recognise them all!

An added benefit of this pastime is it allows me to start re-populating the art page, which I've ignored for too long!
""".strip(),
  },

  "micro-pmf": {
    "published": "2025-12-30",
    "content": """
The existence of <a href="https://en.wikipedia.org/wiki/Micromort">micromorts</a> and <a href="https://colah.github.io/personal/micromarriages/">micromarriages</a> implies the existence of micro-pmf's, associated with actions you take to increase the probability that your startup builds something that reaches product-market fit.

An incomplete (and largely facetious) list:

- Thinking about your product in the shower = 1 micro-pmf
- Talking to a customer = 2 micro-pmfs
- Listening to a customer = 10 micro-pmfs
- Hiring an engineer = 10 micro-pmfs
- Hiring a 10x engineer = 100 micro-pmfs
- Hiring a 100x engineer = 1 milli-pmf
- Closing a deal = 100 micro-pmfs
- Going to a conference to network = -10 micro-pmfs
- Having heated debates with your cofounders = 25 micro-pmfs

I'll update these, probably with more earnesty, as my co-proprietorship of my small business evolves.
""".strip(),
  },
  "on cohere's valuation": {
     "published": "2026-02-01",
    "content": """
When we were raising for Silmo in the valley, we got some snarky comments about Cohere: 'What happened to Cohere, at some point it was such a hot company, and now... nowhere to be seen. Completely eclipsed by OpenAI and Anthropic.'

At the time, I didn't have a good answer. But on contemplating the question, I realised something that I hadn't appreciated before: if you take the ratio of current valuation versus initial funding put into the company, Cohere performs better than the bigger labs.

These numbers aren't exact, but let's just say:

- OpenAI initial funding: ~1B. Current valuation: ~1T. Ratio: 1000x.
- Anthropic initial funding: ~500M. Current valuation: ~0.5T. Ratio: 1000x.
- Cohere initial funding: 2M. Current valuation: ~7B. Ratio: 3,500x!
- Mistral initial funding: 100M. Current valuation: ~10B. Ratio: 100x.

The others (gemini, llama, qwen, kimi, moonshot, deepseek) count less, since they are too new, or attached to hyperscalers that print money and have near-unlimited compute.

In short, Cohere did well, and is in some respects a bigger success than OpenAI!
""".strip(),
  },
  "A surprising pre-pmf startup feeling": {
    "published": "2026-02-07",
    "content": """
There are of course many feelings I've been feeling since starting the company, most of which are predictable: stress, excitement, impatience, satisfaction at making progress, are with me frequently.

But there's a particular feeling that I wasn't anticipating, that I end up spending lots of time with. It's hard to describe, but in broad strokes, I am trying to wrap my mind around a firm vision of the thing we are aspiring to build, both in the short term, the medium term, and the very long term, and in so doing I feel my mind stretching.

This is perhaps an unusual analogy, but it's how I imagine a model feels when it is trying to <a href="https://arxiv.org/pdf/2201.02177">grok</a> a dataset. It struggles and struggles and the loss is sometimes going down, but and sometimes stubbornly plateau-ing, and sometimes going up again. And the general trend is one of progress, fast at first, and then slower and slower as it tries to compress and model every last bit of information in the data.

I think it is my brain's attempt at trying to compress the vision for the company into maximally clear, truthful, simple statements and action plans (like a famous one <a href="https://www.tesla.com/en_ca/secret-master-plan">here</a>), and at the same time all the uncertainties in the vision and counterarguments to the assumptions rear their heads and in turn mold the thought process.

I wonder whether this feeling will fade or grow stronger once we find pmf.
  """.strip()
  }

}

def get_git_dates(filepath: str) -> tuple[str | None, str | None]:
    """
    Get the first commit date (published) and last commit date (updated) for a file.
    Returns (published_date, updated_date) as ISO format strings, or (None, None) if not in git.
    """
    try:
        # Get first commit date (oldest) - no --follow to avoid tracking renames
        first = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%aI", "--", filepath],
            capture_output=True, text=True, check=True
        )
        # Get last commit date (most recent)
        last = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", filepath],
            capture_output=True, text=True, check=True
        )

        first_date = first.stdout.strip().split('\n')[-1] if first.stdout.strip() else None
        last_date = last.stdout.strip().split('\n')[0] if last.stdout.strip() else None

        return (first_date, last_date)
    except subprocess.CalledProcessError:
        return (None, None)

def format_date_human(iso_date: str | None) -> str:
    """Convert ISO date to human-readable format like 'Dec 22, 2025'"""
    if not iso_date:
        return ""
    dt = datetime.fromisoformat(iso_date)
    return dt.strftime("%b %d, %Y")

def same_day(date1: str | None, date2: str | None) -> bool:
    """Check if two ISO dates are on the same day"""
    if not date1 or not date2:
        return True
    return datetime.fromisoformat(date1).date() == datetime.fromisoformat(date2).date()

def make_description(post: str, max_length: int = 160) -> str:
    """Extract a description from post content, stripping HTML and truncating."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', post)
    # Get first paragraph or sentence
    text = text.split('\n\n')[0].strip()
    # Truncate if needed
    if len(text) > max_length:
        text = text[:max_length-3].rsplit(' ', 1)[0] + '...'
    # Escape quotes for HTML attribute
    return text.replace('"', '&quot;')

def to_date_only(iso_date: str | None) -> str | None:
    """Convert ISO datetime to just YYYY-MM-DD, stripping timezone."""
    if not iso_date:
        return None
    return datetime.fromisoformat(iso_date).strftime("%Y-%m-%d")

def make_structured_data(title: str, published: str | None, updated: str | None) -> str:
    """Generate schema.org JSON-LD for SEO"""
    if not published:
        return ""
    pub_date = to_date_only(published)
    upd_date = to_date_only(updated)
    data = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "datePublished": pub_date,
    }
    if upd_date and pub_date != upd_date:
        data["dateModified"] = upd_date
    import json
    return f'<script type="application/ld+json">{json.dumps(data)}</script>'

def format_post(post: str, post_name: str, published: str | None = None, updated: str | None = None) -> str:
    final_post = f"{' '*4}<h3>" + post_name + "</h3>\n"

    # Add date info if available
    if published:
        date_html = f'{" "*6}<p class="post-dates"><small>Published: {format_date_human(published)}'
        if updated and not same_day(published, updated):
            date_html += f' · Updated: {format_date_human(updated)}'
        date_html += '</small></p>\n'
        final_post += date_html

    for block in post.split("\n\n"):
        # Check if this block is a list (lines starting with -)
        lines = block.strip().split("\n")
        if all(line.strip().startswith("- ") for line in lines if line.strip()):
            final_post += f"{' '*6}<ul>\n"
            for line in lines:
                if line.strip():
                    item = line.strip()[2:]  # Remove "- " prefix
                    final_post += f"{' '*8}<li>{item}</li>\n"
            final_post += f"{' '*6}</ul>\n"
        else:
            final_post += f"{' '*6}<p>{block}</p>\n"
    return final_post.strip()

def sanitize_filename(name: str) -> str:
    """
    Sanitize the filename by replacing spaces with underscores and removing special characters.
    """
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("?", "").replace("*", "_").replace("'", "")

def write_if_changed(filepath: str, content: str) -> bool:
    """Only write file if content differs. Returns True if written."""
    try:
        with open(filepath, "r") as f:
            if f.read() == content:
                return False
    except FileNotFoundError:
        pass
    with open(filepath, "w") as f:
        f.write(content)
    return True

def get_lastmod(filepath: str) -> str:
    """Get the last modified date (YYYY-MM-DD) for a file from git."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", filepath],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            return datetime.fromisoformat(result.stdout.strip()).strftime("%Y-%m-%d")
    except subprocess.CalledProcessError:
        pass
    return datetime.now().strftime("%Y-%m-%d")

def generate_sitemap(blog_posts: dict[str, str]):
    """Generate sitemap.xml with all pages. blog_posts is slug -> lastmod date."""
    # Static pages
    static_pages = [
        ("https://amorisot.github.io/", "index.html"),
        ("https://amorisot.github.io/about", "about.html"),
        ("https://amorisot.github.io/art", "art.html"),
        ("https://amorisot.github.io/ml", "ml.html"),
        ("https://amorisot.github.io/books", "books.html"),
        ("https://amorisot.github.io/blog", "blog.html"),
    ]

    urls = []
    for url, filepath in static_pages:
        lastmod = get_lastmod(filepath)
        urls.append(f"  <url>\n    <loc>{url}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    # Blog posts - use explicit dates
    for slug, lastmod in blog_posts.items():
        urls.append(f"  <url>\n    <loc>https://amorisot.github.io/blog/{slug}.html</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    sitemap = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    sitemap += "\n".join(urls)
    sitemap += "\n</urlset>\n"

    write_if_changed("sitemap.xml", sitemap)

def load_seedlings() -> str:
    """Load ideas from the markdown file and format as a seedlings post."""
    with open("blog/ideas_to_write_about_eventually.md", "r") as f:
        ideas = f.read().strip().split("\n")

    # Parse bullet points (lines starting with *)
    bullets = [line.lstrip("* ").strip() for line in ideas if line.startswith("*")]

    html = "<p>Ideas I'm mulling over, that may one day grow into full posts:</p>\n"
    html += "      <ul>\n"
    for bullet in bullets:
        html += f"        <li>{bullet}</li>\n"
    html += "      </ul>"
    return html

def main():
    blog_slugs = {}  # slug -> lastmod date for sitemap
    all_post_links = "<h1>'Blog'</h1>\n"

    # Add seedlings section at the top, demarcated
    seedlings_content = load_seedlings()
    seedlings_slug = "seedlings"
    filepath = f"blog/{seedlings_slug}.html"
    published, updated = get_git_dates("blog/ideas_to_write_about_eventually.md")
    blog_slugs[seedlings_slug] = to_date_only(updated or published)

    all_post_links += f"{' '*6}<li><a href='/blog/{seedlings_slug}.html'>🌱 seedlings</a></li>\n"
    all_post_links += f"{' '*6}<hr>\n"

    write_if_changed(filepath, BLOG_TEMPLATE.format(
        title="🌱 seedlings",
        og_url=f"https://amorisot.github.io/blog/{seedlings_slug}.html",
        description="Ideas I'm mulling over, that may one day grow into fuller posts.",
        content=f"    <h3>🌱 seedlings</h3>\n      {seedlings_content}",
        structured_data=make_structured_data("seedlings", published, updated)
    ))

    for post_name, post_data in POSTS.items():
        slug = sanitize_filename(post_name)
        filepath = f"blog/{slug}.html"
        published = post_data["published"]
        updated = post_data.get("updated")  # None if not specified
        content = post_data["content"]
        blog_slugs[slug] = updated or published  # Use updated if present, else published

        all_post_links += f"{' '*6}<li><a href='/blog/{slug}.html'>{post_name}</a></li>\n"
        write_if_changed(filepath, BLOG_TEMPLATE.format(
            title=post_name,
            og_url=f"https://amorisot.github.io/blog/{slug}.html",
            description=make_description(content),
            content=format_post(post=content, post_name=post_name, published=published, updated=updated),
            structured_data=make_structured_data(post_name, published, updated)
        ))

    write_if_changed("blog.html", BLOG_TEMPLATE.format(
        title="all",
        og_url="https://amorisot.github.io/blog",
        description="Blog posts by Adrien Morisot on ML, LLMs, and other topics.",
        content=all_post_links.strip(),
        structured_data=""
    ))

    generate_sitemap(blog_slugs)
    print(f"Generated {len(blog_slugs)} blog posts and updated sitemap.xml")

if __name__ == "__main__":
    main()
