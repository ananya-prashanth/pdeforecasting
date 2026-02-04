#!/bin/bash

# Quick Start Script for Deploying ICLR Blog Post
# This script helps you deploy your blog to GitHub Pages

echo "=========================================="
echo "ICLR Blog Post Deployment Helper"
echo "=========================================="
echo ""

# Check if git is configured
if ! git config user.name > /dev/null 2>&1; then
    echo "⚠️  Git user not configured. Let's set that up."
    echo ""
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    
    git config --global user.name "$git_name"
    git config --global user.email "$git_email"
    echo "✅ Git configured successfully!"
    echo ""
fi

echo "Now let's connect to GitHub:"
echo ""
echo "1. First, create a new repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: iclr-blog-2026 (or your choice)"
echo "   - Keep it PUBLIC"
echo "   - Do NOT initialize with README"
echo ""
read -p "Have you created the repository? (y/n): " created

if [ "$created" != "y" ]; then
    echo ""
    echo "Please create the repository first, then run this script again."
    exit 0
fi

echo ""
read -p "Enter your GitHub username: " username
read -p "Enter your repository name: " repo_name

echo ""
echo "Setting up remote repository..."
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/$username/$repo_name.git"

echo ""
echo "Attempting to push to GitHub..."
echo "Note: You may need to enter your GitHub credentials"
echo "      If you have 2FA, use a Personal Access Token as password"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Code pushed to GitHub"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Go to: https://github.com/$username/$repo_name/settings/pages"
    echo "2. Under 'Source', select branch 'main'"
    echo "3. Click 'Save'"
    echo "4. Wait 1-5 minutes for deployment"
    echo "5. Your site will be at: https://$username.github.io/$repo_name/"
    echo ""
    echo "📝 See DEPLOYMENT.md for detailed instructions"
    echo "=========================================="
else
    echo ""
    echo "⚠️  Push failed. Common issues:"
    echo ""
    echo "1. Authentication failed:"
    echo "   - If you have 2FA, create a Personal Access Token:"
    echo "     https://github.com/settings/tokens"
    echo "   - Use the token as your password when pushing"
    echo ""
    echo "2. Repository doesn't exist:"
    echo "   - Make sure you created it on GitHub"
    echo "   - Double-check the username and repo name"
    echo ""
    echo "See DEPLOYMENT.md for more help"
fi
