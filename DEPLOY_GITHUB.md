# Deploying VoltInfer to GitHub 🚀

The local Git repository for VoltInfer has been initialized, fully tested, packaged, and committed on the `main` branch.

---

## 1. Quick Push to GitHub

If you have already created the repository on GitHub under your account (`priyanshunegi1406-hub`):

```powershell
# Set the remote URL (replace if your repository has a different name/URL)
git remote add origin https://github.com/priyanshunegi1406-hub/VoltInfer.git

# Push the main branch to GitHub
git push -u origin main
```

*(If prompted, enter your GitHub Personal Access Token or sign in via the browser credential manager).*

---

## 2. If You Need to Create the Repository on GitHub First

1. Go to **[GitHub New Repository](https://github.com/new)**.
2. Set **Repository name**: `VoltInfer`
3. Description: `⚡ Energy-Aware, Hardware-in-the-Loop LLM Adaptive Runtime & Desktop Software`
4. Set to **Public** (or **Private**).
5. **Do NOT** initialize with README, .gitignore, or License (these are already generated and committed locally).
6. Click **Create repository**.
7. In your terminal at `C:\Users\priya\Downloads\VoltInfer`, run:
   ```powershell
   git remote add origin https://github.com/priyanshunegi1406-hub/VoltInfer.git
   git push -u origin main
   ```

---

## 3. Automated CI / GitHub Actions

Once pushed, GitHub Actions will automatically trigger `.github/workflows/ci.yml`:
- Runs all 22 tests across Python 3.10, 3.11, and 3.12 on both Windows and Ubuntu.
- Validates the test suite, packaging, and benchmark verification.
- Displays passing CI badges on your repository's README!
