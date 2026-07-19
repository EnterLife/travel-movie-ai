#define AppName "TravelMovieAI"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SourceRoot
  #define SourceRoot "..\dist\TravelMovieAI"
#endif
#ifndef InstallerOutput
  #define InstallerOutput "..\dist\installer"
#endif

[Setup]
AppId={{9BA2E17B-81F3-49C1-B099-B4046A22230E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=TravelMovieAI
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#InstallerOutput}
OutputBaseFilename=TravelMovieAI-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\TravelMovieAI.exe

[Files]
Source: "{#SourceRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{localappdata}\{#AppName}"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\TravelMovieAI.exe"; WorkingDir: "{localappdata}\{#AppName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\TravelMovieAI.exe"; WorkingDir: "{localappdata}\{#AppName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\TravelMovieAI.exe"; Description: "Launch {#AppName}"; WorkingDir: "{localappdata}\{#AppName}"; Flags: nowait postinstall skipifsilent
