#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppId={{A217B7D2-19E1-4B32-9A1B-5B0F0D8D1174}
AppName=VoltInfer
AppVersion={#AppVersion}
AppPublisher=VoltInfer
DefaultDirName={localappdata}\Programs\VoltInfer
DefaultGroupName=VoltInfer
OutputDir=dist
OutputBaseFilename=VoltInfer-Setup
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\voltinfer-windows-x64.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "dist\voltinfer-windows-x64.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\VoltInfer Desktop"; Filename: "{app}\voltinfer-windows-x64.exe"; Parameters: "app"
Name: "{group}\VoltInfer Proxy"; Filename: "{app}\voltinfer-windows-x64.exe"; Parameters: "start"
Name: "{autodesktop}\VoltInfer"; Filename: "{app}\voltinfer-windows-x64.exe"; Parameters: "app"; Tasks: desktopicon

[Run]
Filename: "{app}\voltinfer-windows-x64.exe"; Parameters: "app"; Description: "Launch VoltInfer"; Flags: postinstall nowait skipifsilent
