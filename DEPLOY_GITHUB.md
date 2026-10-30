# Publishing VoltInfer builds

The project repository is [kathait-sarthak/VoltInfer](https://github.com/kathait-sarthak/VoltInfer). A Windows x64 portable executable is currently published on its Releases page. The Windows installer is built by the release workflow and will be attached to a release created from the updated workflow.

## Publish a Windows release

1. Commit and push the reviewed project changes to the repository.
2. Update the version in `pyproject.toml` and `setup.py` if this is a new release.
3. Create and push a matching version tag:

   ```powershell
   git tag v1.1.0
   git push origin v1.1.0
   ```

4. The `Build Windows App and Installer` workflow builds the standalone executable, smoke-tests it, packages `VoltInfer-Setup.exe` with Inno Setup, and attaches both files to the GitHub Release.
5. Confirm the workflow completed and both assets appear on the release page before sharing the download link.

The workflow can also be started manually from **GitHub → Actions → Build Windows App and Installer → Run workflow**. A manual run uploads artifacts for download from the workflow run, but does not publish a GitHub Release.

## Local Windows build

On Windows with Python 3.12 and Inno Setup 6 installed:

```powershell
python -m pip install pyinstaller psutil
pyinstaller --noconfirm --clean --onefile --console --name voltinfer-windows-x64 --add-data "dashboard;dashboard" --hidden-import voltinfer_gui voltinfer_launcher.py
$version = python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "/DAppVersion=$version" installer.iss
```

The build outputs `dist\voltinfer-windows-x64.exe` and `dist\VoltInfer-Setup.exe`. The installer packages VoltInfer only; users configure their own OpenAI-compatible LLM server and credentials.
