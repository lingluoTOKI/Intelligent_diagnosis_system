# Builds the single graphical launcher used from the project root.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$compiler = Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$iconPath = Join-Path $projectRoot 'assets\brand\app.ico'
$launcherName = -join @(
    [char]0x773C,[char]0x79D1,[char]0x56FE,[char]0x50CF,[char]0x5206,[char]0x7C7B,
    [char]0x8F85,[char]0x52A9,[char]0x8BCA,[char]0x65AD,[char]0x7CFB,[char]0x7EDF
) + '.exe'
$outputPath = Join-Path $projectRoot $launcherName
$sourcePath = Join-Path $PSScriptRoot 'desktop_launcher.cs'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'Windows .NET Framework C# compiler is unavailable.' }
& $compiler /nologo /target:winexe /reference:System.Windows.Forms.dll "/win32icon:$iconPath" "/out:$outputPath" $sourcePath
if ($LASTEXITCODE -ne 0) { throw 'Desktop launcher build failed.' }
Unblock-File -LiteralPath $outputPath
Write-Output 'Desktop EXE launcher rebuilt.'
