import markdown

body = markdown.markdown(open("README.md").read(), extensions=["extra"])

template = """<html><head><title>pypomp quant</title>
<style>
  body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; border: 1px solid #ddd; }
  code { background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: monospace; }
</style></head>
<body>%s</body></html>"""

open("index.html", "w").write(template % body)
print("Successfully generated index.html!")
