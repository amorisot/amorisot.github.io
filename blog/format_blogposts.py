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
    "hi": """Hello, world!

Today, we're talking about X.

We might also talk about Y, and Z.

Thanks!
""".strip(),
}

def format_post(post: str) -> str:
    final_post = ""
    for line in post.split("\n\n"):
        final_post += f"{' '*6}<p>{line}</p>\n"
    return final_post.strip()


def main():
    with open(f"blog.html", "w") as f:
        all_post_links = ""
        for post_name, post in POSTS.items():
            all_post_links += f"{' '*6}<li><a href='blog/{post_name}.html'>{post_name}</a></li>\n"
            with open(f"blog/{post_name}.html", "w") as f_blog:
                f_blog.write(BLOG_TEMPLATE.format(title=post_name, content=format_post(post)))
            
        f.write(BLOG_TEMPLATE.format(title="all", content=all_post_links.strip()))

if __name__ == "__main__":
    main()
