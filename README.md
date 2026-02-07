# Blog Post: Equivariant Neural Fields for PDE Forecasting

This repository contains a blog post for the Deep Learning for the Sciences seminar, covering the paper "Space-Time Continuous PDE Forecasting using Equivariant Neural Fields" by Knigge et al. (2024).

## About the Blog Post

This blog post explores how equivariant neural fields can be used to solve partial differential equations (PDEs) on complex geometries while respecting physical symmetries. The approach combines:
- Neural fields for continuous representations
- Equivariant architectures to encode symmetries
- Meta-learning for efficient inference

## Setup and Deployment

### Option 1: GitHub Pages (Recommended)

1. **Create a GitHub repository**
   - Go to GitHub and create a new repository
   - Name it something like `iclr-blog-2026` or `username.github.io` for a user site

2. **Push this code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: ICLR blog post"
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   git push -u origin main
   ```

3. **Enable GitHub Pages**
   - Go to your repository settings
   - Navigate to "Pages" in the left sidebar
   - Under "Source", select the `main` branch
   - Click Save
   - Your site will be published at `https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/`

### Option 2: Local Development

If you want to preview the blog locally before deploying:

1. **Install Ruby and Jekyll**
   ```bash
   # On macOS
   brew install ruby
   gem install bundler jekyll
   
   # On Ubuntu/Debian
   sudo apt-get install ruby-full build-essential
   gem install bundler jekyll
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run the local server**
   ```bash
   bundle exec jekyll serve
   ```
   
4. **View in browser**
   - Open `http://localhost:4000` in your web browser

## Repository Structure

```
iclr-blog-2026/
├── _config.yml           # Jekyll configuration
├── _layouts/             # Page layouts
│   ├── default.html      # Main layout with MathJax
│   └── post.html         # Blog post layout
├── _posts/               # Blog posts
│   └── 2026-02-03-equivariant-neural-fields.md
├── assets/
│   ├── css/
│   │   └── style.css     # Custom styling
│   └── bibliography/     # BibTeX references
├── index.html            # Homepage
├── Gemfile               # Ruby dependencies
└── README.md             # This file
```

## Key Features

- **MathJax Support**: Renders LaTeX equations beautifully
- **Responsive Design**: Works on desktop and mobile
- **Clean Layout**: Easy-to-read typography and spacing
- **GitHub Pages Ready**: Deploys automatically with no build step needed

## Customization

### Changing the Theme

Edit `_config.yml` to modify site settings:
```yaml
title: Your Blog Title
description: Your Description
```

### Adding Images

Place images in `assets/images/` and reference them in your post:
```markdown
![Alt text]({{ site.baseurl }}/assets/images/your-image.png)
```

### Modifying Styles

Edit `assets/css/style.css` to customize the appearance.

## Writing Additional Posts

Create new files in `_posts/` following the naming convention:
```
YYYY-MM-DD-post-title.md
```

Include front matter at the top:
```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD
categories: [category1, category2]
author: Your Name
description: Brief description
math: true
---
```

## Mathematical Equations

Use standard LaTeX syntax:
- Inline math: `$equation$` or `\(equation\)`
- Display math: `$$equation$$` or `\[equation\]`

Example:
```markdown
The heat equation is $$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$
```

## Troubleshooting

### Site not showing up on GitHub Pages
- Wait a few minutes after pushing (can take 5-10 minutes)
- Check Settings → Pages to ensure it's enabled
- Verify the branch is set to `main`

### Math not rendering
- Ensure MathJax is loaded (check browser console)
- Use `$$` for display math, not single `$`

### Styling issues
- Clear browser cache
- Check that `assets/css/style.css` is being loaded

## References

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [MathJax Documentation](https://www.mathjax.org/)

## License

This blog post is created for educational purposes as part of the Deep Learning for the Sciences seminar.

## Contact

For questions or feedback, please open an issue on this repository.
