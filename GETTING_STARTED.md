# Getting Your Blog Post Live - Complete Guide

This guide will walk you through deploying your ICLR blog post to GitHub Pages step by step.

## What You Have

Your blog post is ready to deploy! The repository contains:
- ✅ A complete Jekyll blog post with math support
- ✅ Professional styling and responsive design
- ✅ All necessary configuration files
- ✅ README and documentation

## Prerequisites

- A GitHub account (create one at https://github.com if needed)
- Basic command line familiarity
- This repository on your local machine

## Deployment Options

### Option A: Automated Deployment (Easiest)

We've created a script to automate most of the process:

```bash
cd /path/to/iclr-blog-2026
./deploy.sh
```

Follow the prompts and the script will guide you through:
1. Configuring git (if needed)
2. Connecting to your GitHub repository
3. Pushing the code

### Option B: Manual Deployment (Step-by-Step)

If you prefer to do it manually or the script doesn't work:

#### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `iclr-blog-2026` (or any name you like)
3. Description: "ICLR 2026 Blog Post - Equivariant Neural Fields"
4. **Public** repository
5. **Do NOT check** "Initialize this repository with a README"
6. Click "Create repository"

#### Step 2: Configure Git (First Time Only)

If this is your first time using git on this machine:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

#### Step 3: Connect and Push

Replace `YOUR-USERNAME` and `YOUR-REPO` with your actual values:

```bash
# Navigate to the project
cd /path/to/iclr-blog-2026

# Connect to GitHub
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git

# Push the code
git push -u origin main
```

**Authentication:**
- If you have Two-Factor Authentication enabled, you'll need a Personal Access Token
- Create one at: https://github.com/settings/tokens
- Click "Generate new token (classic)"
- Select scope: `repo`
- Copy the token and use it as your password when pushing

#### Step 4: Enable GitHub Pages

1. Go to your repository: `https://github.com/YOUR-USERNAME/YOUR-REPO`
2. Click "Settings" (top navigation bar)
3. Click "Pages" (left sidebar)
4. Under "Source":
   - Branch: `main`
   - Folder: `/ (root)`
5. Click "Save"
6. Wait 1-5 minutes

#### Step 5: View Your Blog

Your blog will be live at:
```
https://YOUR-USERNAME.github.io/YOUR-REPO/
```

## Troubleshooting Common Issues

### Problem: "Permission denied" when pushing

**Solution:**
```bash
# Make sure you're authenticated
# If using a Personal Access Token, use it as password
# Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### Problem: Site shows 404 error

**Solutions:**
1. Wait 5-10 minutes after enabling GitHub Pages
2. Check that Pages is enabled in Settings → Pages
3. Verify branch is set to `main`
4. Clear browser cache and try again

### Problem: CSS/styling not loading

**Solution:**
If your repo name is NOT `username.github.io`, update `_config.yml`:

```yaml
baseurl: "/YOUR-REPO-NAME"
url: "https://YOUR-USERNAME.github.io"
```

Then commit and push:
```bash
git add _config.yml
git commit -m "Fix baseurl for GitHub Pages"
git push
```

### Problem: Math equations not rendering

**Solutions:**
1. Wait for page to fully load (MathJax takes a moment)
2. Check browser console (F12) for errors
3. Try a different browser
4. Ensure you're using `$$` for display equations

### Problem: "fatal: remote origin already exists"

**Solution:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
```

## Making Updates to Your Blog

After initial deployment, to make changes:

1. Edit the files locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push
   ```
3. Wait 1-5 minutes for GitHub to rebuild

## File Structure Explanation

```
iclr-blog-2026/
├── _config.yml              # Site configuration
├── _layouts/                # HTML templates
│   ├── default.html         # Main page template (includes MathJax)
│   └── post.html           # Blog post template
├── _posts/                  # Your blog posts go here
│   └── 2026-02-03-equivariant-neural-fields.md
├── assets/
│   └── css/
│       └── style.css       # Custom styling
├── index.html              # Homepage
├── README.md               # Documentation
├── DEPLOYMENT.md           # Deployment guide
└── deploy.sh              # Automated deployment script
```

## Customization

### Change Site Title/Description

Edit `_config.yml`:
```yaml
title: Your Custom Title
description: Your custom description
```

### Add Images

1. Create folder: `assets/images/`
2. Add your images
3. Reference in markdown:
   ```markdown
   ![Alt text]({{ site.baseurl }}/assets/images/your-image.png)
   ```

### Modify Styling

Edit `assets/css/style.css` to change colors, fonts, spacing, etc.

### Add New Posts

Create a new file in `_posts/`:
- Filename: `YYYY-MM-DD-post-title.md`
- Include front matter:
  ```yaml
  ---
  layout: post
  title: "Your Title"
  date: YYYY-MM-DD
  author: Your Name
  math: true
  ---
  ```

## Testing Locally (Optional)

To preview your blog before deploying:

1. Install Ruby and Jekyll:
   ```bash
   # macOS
   brew install ruby
   gem install bundler jekyll
   
   # Linux
   sudo apt-get install ruby-full
   gem install bundler jekyll
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Run local server:
   ```bash
   bundle exec jekyll serve
   ```

4. Open http://localhost:4000

## Resources

- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Jekyll Docs](https://jekyllrb.com/docs/)
- [MathJax Docs](https://www.mathjax.org/)
- [Markdown Guide](https://www.markdownguide.org/)

## Getting Help

If you run into issues:

1. Check this guide and DEPLOYMENT.md
2. Search GitHub's documentation
3. Check the Jekyll documentation
4. Create an issue on the repository

## Success Checklist

Before considering deployment complete:

- [ ] Repository created on GitHub
- [ ] Code successfully pushed
- [ ] GitHub Pages enabled in Settings
- [ ] Site accessible at the GitHub Pages URL
- [ ] All pages load without errors
- [ ] Math equations render correctly
- [ ] Images display (if any)
- [ ] Links work
- [ ] Mobile-responsive design works
- [ ] CSS loads properly

## Next Steps After Deployment

1. **Share your blog**: Copy the URL and share it!
2. **Add content**: Write more blog posts
3. **Customize**: Make it your own with custom styling
4. **Monitor**: Check GitHub Actions for build status

## Quick Reference Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# Update site
git add .
git commit -m "Update message"
git push

# View remote URL
git remote -v

# Change remote URL
git remote set-url origin NEW-URL
```

Good luck with your blog post! 🚀
