# Step-by-Step Deployment Guide

Follow these steps to get your blog post live on GitHub Pages:

## Step 1: Prepare the Repository

The repository is already initialized and ready to go. You should see all the necessary files.

## Step 2: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" button in the top right corner
3. Select "New repository"
4. Choose a repository name:
   - For a project site: `iclr-blog-2026` (or any name you prefer)
   - For a user site: `YOUR-USERNAME.github.io`
5. Keep it **Public**
6. **Do NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 3: Connect and Push to GitHub

Run these commands in your terminal from the project directory:

```bash
# Navigate to the project directory
cd /path/to/iclr-blog-2026

# Add all files
git add .

# Commit the files
git commit -m "Initial commit: Equivariant Neural Fields blog post"

# Add your GitHub repository as remote (replace YOUR-USERNAME and YOUR-REPO)
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

# Push to GitHub
git push -u origin main
```

**Note**: You may need to authenticate with GitHub. If you have 2FA enabled, you'll need to use a Personal Access Token instead of your password.

### Creating a Personal Access Token (if needed):

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "Blog deployment")
4. Select scopes: check "repo"
5. Click "Generate token"
6. Copy the token (you won't see it again!)
7. Use this token as your password when pushing

## Step 4: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on "Settings" (top menu)
3. Scroll down to "Pages" in the left sidebar
4. Under "Source":
   - Select branch: `main`
   - Select folder: `/ (root)`
5. Click "Save"

## Step 5: Wait for Deployment

- GitHub will build and deploy your site (usually takes 1-5 minutes)
- You'll see a message: "Your site is ready to be published at..."
- Once ready, it will say: "Your site is published at..."

## Step 6: Access Your Blog

Your blog will be available at:
- **Project site**: `https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/`
- **User site**: `https://YOUR-USERNAME.github.io/`

## Troubleshooting

### Issue: Site shows 404 error
**Solution**: 
- Wait 5-10 minutes for initial deployment
- Check that GitHub Pages is enabled in Settings
- Verify the branch is set to `main`

### Issue: Formatting looks broken
**Solution**:
- If using a project site (not username.github.io), update `_config.yml`:
  ```yaml
  baseurl: "/YOUR-REPO-NAME"
  ```
- Commit and push the change

### Issue: Math equations not rendering
**Solution**:
- Check browser console for errors
- Ensure you're using `$$` for display math
- MathJax CDN may take a moment to load

### Issue: Authentication failed when pushing
**Solution**:
- Use a Personal Access Token instead of password
- Or set up SSH keys for GitHub

## Making Updates

To update your blog post:

```bash
# Make your changes to files
# Then:
git add .
git commit -m "Update blog post"
git push
```

GitHub Pages will automatically rebuild and deploy (takes 1-5 minutes).

## Quick Commands Summary

```bash
# Initial setup
git add .
git commit -m "Initial commit: blog post"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main

# Future updates
git add .
git commit -m "Description of changes"
git push
```

## Need Help?

- Check the main README.md for detailed documentation
- Visit [GitHub Pages Documentation](https://docs.github.com/en/pages)
- Open an issue on the repository if you encounter problems

## Success Checklist

- [ ] Repository created on GitHub
- [ ] Code pushed to GitHub
- [ ] GitHub Pages enabled in Settings
- [ ] Site is live and accessible
- [ ] Math equations rendering correctly
- [ ] All links working
- [ ] Responsive design works on mobile

Once all items are checked, your blog post is successfully deployed! 🎉
