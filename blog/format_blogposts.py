BLOG_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <link rel="icon" type="image/png" href="favicon.png"/>
    <link rel="stylesheet" href="/static/style.css">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blog - {title}</title>
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
  </body>
</html>
""".strip()

POSTS = {
    "hi": """I am thinking a lot about reasoning in LLMs, and will probably write a blogpost about it soon.

Thanks for stopping by :)
""".strip(),

    "why don't they make chickens without the salmonella?": """
The question was originally facetious and rhetorical, but the answer is surprising.

I learnt most from this Reddit comment from more than a decade ago: https://www.reddit.com/r/askscience/comments/ncfik/comment/c3816vb/.

Turning whole, live chickens into pieces in a Colonel bucket or shrink-wrapped in a styro tray, is done in a very different way than segmenting range animals into consumer-friendly portions. The very specialized industrial process developed to deal specifically with poultry is actually the principal cause of systemic contamination of the meat itself.

Unlike cattle and hogs, chickens are inconveniently covered in (surprise!) feathers. After they are killed, the birds must be plucked before anything else can be done to them. In order for the plucking machines to work properly, the carcasses are dunked into a large vat of hot water, which relaxes the dermis and allows the feathers to be removed. Unfortunately, skin isn't the only organ the hot bath relaxes: the chickens, which have been killed mere seconds before, leak feces from their now well-relaxed bowels, right into the scald tank. The first bird of the day essentially turns the tank into a large pot of hot fecal soup; subsequent carcasses, usually tens of thousands per day, enrich the fecal contamination exponentially. All birds are submerged in this cauldron of concentrated bacterial broth for several minutes, before they continue down the process line.

Fresh out of the contaminated scalding tank, after a few seconds at the plucking station, the carcass is mechanically eviscerated. The residual fecal slop still clinging to the skin contaminates the gutting machine, which in turn inoculates the body cavity, splashing fluids and infecting the whole bird from the inside. The carcass is then washed again, usually in another (dirty) communal tank, but by now it is virtually impossible to remove all of the liquified, dilute fecal material from every nook, cranny, scrap, cut and fold of flesh. Some of the bacteria have a reproductive cycle of just a few minutes, so even after a thorough washing, just a few residual bacteria can re-contaminate the carcass in a very short time. (One surviving bacterium, reproducing every ~6 min = 1 million in 2 hr, 1 trillion in 4 hrs.)

The presence of of E. Coli, salmonella and camphylobacter in virtually ALL consumer chicken, including organic and farmer's markets, means it is best to keep raw chicken isolated (well-wrapped) and well refrigerated (bottom shelf). Prepare it as soon as possible, and cook it thoroughly. If you can't use it right away, immerse it fully in a strong brine or acidic marinade (zipper bags are good) to slow pathogen growth. Makes it moist and tasty, too!
""".strip()
}

def format_post(post: str) -> str:
    final_post = ""
    for line in post.split("\n\n"):
        final_post += f"{' '*6}<p>{line}</p>\n"
    return final_post.strip()

def sanitize_filename(name: str) -> str:
    """
    Sanitize the filename by replacing spaces with underscores and removing special characters.
    """
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("?", "").replace("*", "_").replace("'", "")

def main():
    with open(f"blog.html", "w") as f:
        all_post_links = ""
        for post_name, post in POSTS.items():
            all_post_links += f"{' '*6}<li><a href='/blog/{sanitize_filename(post_name)}.html'>{post_name}</a></li>\n"
            with open(f"blog/{sanitize_filename(post_name)}.html", "w") as f_blog:
                f_blog.write(BLOG_TEMPLATE.format(title=post_name, content=format_post(post)))
            
        f.write(BLOG_TEMPLATE.format(title="all", content=all_post_links.strip()))

if __name__ == "__main__":
    main()
