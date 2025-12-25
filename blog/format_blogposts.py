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

POSTS = {
  "hi": """I am thinking a lot about reasoning in LLMs, and will probably write a blogpost about it soon.

Thanks for stopping by :)
""".strip(),

  "why don't they make chickens without the salmonella?": """
The question was originally facetious and rhetorical, but the answer is surprising.

I learnt most from this <a href="https://www.reddit.com/r/askscience/comments/ncfik/comment/c3816vb/">Reddit comment from more than a decade ago</a>.

<hr>

Turning whole, live chickens into pieces in a Colonel bucket or shrink-wrapped in a styro tray, is done in a very different way than segmenting range animals into consumer-friendly portions. The very specialized industrial process developed to deal specifically with poultry is actually the principal cause of systemic contamination of the meat itself.

Unlike cattle and hogs, chickens are inconveniently covered in (surprise!) feathers. After they are killed, the birds must be plucked before anything else can be done to them. In order for the plucking machines to work properly, the carcasses are dunked into a large vat of hot water, which relaxes the dermis and allows the feathers to be removed. Unfortunately, skin isn't the only organ the hot bath relaxes: the chickens, which have been killed mere seconds before, leak feces from their now well-relaxed bowels, right into the scald tank. The first bird of the day essentially turns the tank into a large pot of hot fecal soup; subsequent carcasses, usually tens of thousands per day, enrich the fecal contamination exponentially. All birds are submerged in this cauldron of concentrated bacterial broth for several minutes, before they continue down the process line.

Fresh out of the contaminated scalding tank, after a few seconds at the plucking station, the carcass is mechanically eviscerated. The residual fecal slop still clinging to the skin contaminates the gutting machine, which in turn inoculates the body cavity, splashing fluids and infecting the whole bird from the inside. The carcass is then washed again, usually in another (dirty) communal tank, but by now it is virtually impossible to remove all of the liquified, dilute fecal material from every nook, cranny, scrap, cut and fold of flesh. Some of the bacteria have a reproductive cycle of just a few minutes, so even after a thorough washing, just a few residual bacteria can re-contaminate the carcass in a very short time. (One surviving bacterium, reproducing every ~6 min = 1 million in 2 hr, 1 trillion in 4 hrs.)

The presence of of E. Coli, salmonella and camphylobacter in virtually ALL consumer chicken, including organic and farmer's markets, means it is best to keep raw chicken isolated (well-wrapped) and well refrigerated (bottom shelf). Prepare it as soon as possible, and cook it thoroughly. If you can't use it right away, immerse it fully in a strong brine or acidic marinade (zipper bags are good) to slow pathogen growth. Makes it moist and tasty, too!
""".strip(),
  "10T vs 100T parameter sparse MoE's":"""As of September 2025, LLMs are spikily impressive: they outperform most people in mathematical and coding abilities, and they speak more languages fluently than any one person.

The depth and breadth of even small LLM's knowledge is staggering. However, it is important to remember that their competition is stiff, when it comes to performing tasks in the world.

Current generation LLMs --at most ~10T parameter sparse MoEs-- are competing against the human brain, which are 100T parameter, highly efficient, multimodal MoEs, pretrained on decades of experience, post-trained in university, RLVR'ed/RLHF'ed on the job. Humans should not be slept on.
""".strip(),
  "a cute way of passing time while vibe coding": """
While the model is chugging along, a pleasant and somewhat brain-dead way of passing the time is asking the model for a recreation in code of some work of art or biological phenomena.

It is low stakes, helps me refine my understanding of what models can and cannot do, and the output usually sparks joy --independent of whether or not the model did the task successfully!

I've started compiling the successful outputs of these efforts in the later parts of my <a href="/art">art (👨‍🎨)</a> page. See if you recognise them all!

An added benefit of this pastime is it allows me to start re-populating the art page, which I've ignored for too long!
""".strip(),
  "list of crafts that are fun on dates, ranked": """
- Blowing glass: you learn a lot about the craft of glass blowing, but this is mostly too difficult or dangerous as an amateur, and so the instructor usually does most of the actual crafting. They'll let you blow into the tube (sometimes), and perhaps dip the hot glass into the colours, but not much beyond that. 3/10, would not recommend.
- Pottery: a blessed activity, difficult at the beginning, but you get better fast with practice, the results are immediately tangible (you can drink from them!), and having hands full of wet clay and the pressure to build fast enables pure focus. When doing pottery, flow state is easy to find. The studio at which I do it in Toronto, United Spirits Pottery, is great. Cathy and Ricky, the proprietors, are delightful. And Ricky has a storied past!
- Glass cutting: delightful. Best place in Toronto: Verbeek studios, in Leslieville. The proprietor, Lane, is lovely too.
- TODO: watercolour painting.
  """.strip()
#    "my career path, so far": """
# An extremely bare-bones summary of where I am at, and how I got here, listed as a series of narrative arcs.

# Arc 1: I did not know what I wanted to do in undergrad. I originally started studying philosophy and political science, as a guess as to what I was interested in.
# I did not resonate with political science. I was expecting modern theories of how to run states, and instead realised that much of the field still involves reading texts from centuries ago --Locke, Hobbes, Rousseau.
# Philosophy was more interesting. I gravitated particularly towards logic, the philosophy of mathematics, and ethics/metaethics. Unsurprisingly, given the former, I resonated far more with analytic than continental philosophy.
# After about a year, I switched to double majoring in philosophy and mathematics, and was much happier, pushing this to the end of my studies.

# Arc 2: 
# """.strip(),

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

def make_structured_data(title: str, published: str | None, updated: str | None) -> str:
    """Generate schema.org JSON-LD for SEO"""
    if not published:
        return ""
    data = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "datePublished": published,
    }
    if updated and not same_day(published, updated):
        data["dateModified"] = updated
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

    for line in post.split("\n\n"):
        final_post += f"{' '*6}<p>{line}</p>\n"
    return final_post.strip()

def sanitize_filename(name: str) -> str:
    """
    Sanitize the filename by replacing spaces with underscores and removing special characters.
    """
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("?", "").replace("*", "_").replace("'", "")

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

def generate_sitemap(blog_posts: list[str]):
    """Generate sitemap.xml with all pages."""
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

    # Blog posts
    for slug in blog_posts:
        filepath = f"blog/{slug}.html"
        lastmod = get_lastmod(filepath)
        urls.append(f"  <url>\n    <loc>https://amorisot.github.io/blog/{slug}.html</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")

    sitemap = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    sitemap += "\n".join(urls)
    sitemap += "\n</urlset>\n"

    with open("sitemap.xml", "w") as f:
        f.write(sitemap)

def main():
    blog_slugs = []
    with open(f"blog.html", "w") as f:
        all_post_links = "<h1>'Blog'</h1>\n"
        for post_name, post in POSTS.items():
            slug = sanitize_filename(post_name)
            blog_slugs.append(slug)
            filepath = f"blog/{slug}.html"
            published, updated = get_git_dates(filepath)

            all_post_links += f"{' '*6}<li><a href='/blog/{slug}.html'>{post_name}</a></li>\n"
            with open(filepath, "w") as f_blog:
                f_blog.write(BLOG_TEMPLATE.format(
                    title=post_name,
                    og_url=f"https://amorisot.github.io/blog/{slug}.html",
                    description=make_description(post),
                    content=format_post(post=post, post_name=post_name, published=published, updated=updated),
                    structured_data=make_structured_data(post_name, published, updated)
                ))

        f.write(BLOG_TEMPLATE.format(
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
