[CmdletBinding()]
param(
    [string]$Source,
    [switch]$All,
    [ValidateRange(1, 100)]
    [int]$MaxPasses = 5,
    [switch]$Clean,
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LatexRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $LatexRoot "build"))
$RegisteredSources = [string[]]@(
    "thesis\main.tex",
    "slides\chaos-sync-presentation.tex",
    "notes\research-topic-qa.tex",
    "templates\research-presentation-template.tex",
    "seminars\b4-2026-05-27.tex"
)

function Write-UsageError {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Message)

    [Console]::Error.WriteLine("$Message`nUsage: build.ps1 (-Source <tex> | -All) [-MaxPasses 2..5] [-Clean] [-OutputDirectory <dir>]")
}

function Assert-BuildDirectory {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith($BuildRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "OutputDirectory must be under latex/build: $FullPath"
    }
}

function Clear-DocumentBuild {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    Assert-BuildDirectory -Path $Path
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return
    }

    # Why not: build 以外へ削除範囲が広がらないよう、検証済みの文書別フォルダだけを空にする。
    Get-ChildItem -LiteralPath $Path -Force | Remove-Item -Recurse -Force
}

function Invoke-ExternalCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Push-Location -LiteralPath $WorkingDirectory
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $CommandOutput = & $Command @Arguments 2>&1
        $ProcessExitCode = [int]$LASTEXITCODE
        $CommandOutput | ForEach-Object { Write-Host $_ }
        return $ProcessExitCode
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        Pop-Location
    }
}

function Test-RerunRequired {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$LogPath)

    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        return $false
    }
    $LogText = Get-Content -LiteralPath $LogPath -Raw -Encoding UTF8
    return $LogText -match "Rerun to get cross-references right|Label\(s\) may have changed|Please \(re\)run Biber|There were undefined references"
}

function Find-TexExecutable {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Name)

    $PathCommand = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $PathCommand) {
        return [string]$PathCommand.Source
    }

    $MiKTexDirectory = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
    $Candidate = Join-Path $MiKTexDirectory $Name
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        return [string]$Candidate
    }
    return ""
}

function Invoke-LatexBuild {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$BuildDirectory,
        [Parameter(Mandatory = $true)][int]$PassLimit,
        [Parameter(Mandatory = $true)][string]$LuaLatexCommand,
        [string]$BiberCommand,
        [string]$BibtexCommand
    )

    Assert-BuildDirectory -Path $BuildDirectory
    New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null

    $SourceItem = Get-Item -LiteralPath $SourcePath
    $JobName = $SourceItem.BaseName
    $LogPath = Join-Path $BuildDirectory "$JobName.log"
    $BibliographyProcessed = $false

    for ($Pass = 1; $Pass -le $PassLimit; $Pass++) {
        Write-Host "[$JobName] LuaLaTeX pass $Pass/$PassLimit"
        $LatexArguments = [string[]]@(
            "--enable-installer",
            "-synctex=1",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-directory=$BuildDirectory",
            $SourceItem.Name
        )
        $ExitCode = Invoke-ExternalCommand -Command $LuaLatexCommand -Arguments $LatexArguments -WorkingDirectory $SourceItem.DirectoryName
        if ($ExitCode -ne 0) {
            [Console]::Error.WriteLine("LuaLaTeX failed: $SourcePath (pass $Pass, exit $ExitCode)")
            return $false
        }

        if (-not $BibliographyProcessed) {
            $BcfPath = Join-Path $BuildDirectory "$JobName.bcf"
            $AuxPath = Join-Path $BuildDirectory "$JobName.aux"
            if (Test-Path -LiteralPath $BcfPath -PathType Leaf) {
                if ([string]::IsNullOrWhiteSpace($BiberCommand)) {
                    [Console]::Error.WriteLine("Biber is required but was not found: $BcfPath")
                    return $false
                }
                $BiberExitCode = Invoke-ExternalCommand -Command $BiberCommand -Arguments ([string[]]@($JobName)) -WorkingDirectory $BuildDirectory
                if ($BiberExitCode -ne 0) {
                    [Console]::Error.WriteLine("Biber failed: $JobName (exit $BiberExitCode)")
                    return $false
                }
                $BibliographyProcessed = $true
            }
            elseif ((Test-Path -LiteralPath $AuxPath -PathType Leaf) -and (Select-String -LiteralPath $AuxPath -Pattern "\\bibdata" -Quiet)) {
                if ([string]::IsNullOrWhiteSpace($BibtexCommand)) {
                    [Console]::Error.WriteLine("BibTeX is required but was not found: $AuxPath")
                    return $false
                }
                $BibtexExitCode = Invoke-ExternalCommand -Command $BibtexCommand -Arguments ([string[]]@($JobName)) -WorkingDirectory $BuildDirectory
                if ($BibtexExitCode -ne 0) {
                    [Console]::Error.WriteLine("BibTeX failed: $JobName (exit $BibtexExitCode)")
                    return $false
                }
                $BibliographyProcessed = $true
            }
        }

        if ($Pass -ge 2 -and -not (Test-RerunRequired -LogPath $LogPath)) {
            Write-Host "[$JobName] build complete: $BuildDirectory"
            return $true
        }
    }

    if (Test-RerunRequired -LogPath $LogPath) {
        [Console]::Error.WriteLine("Maximum passes reached while rerun was still requested: $SourcePath")
        return $false
    }
    Write-Host "[$JobName] build complete: $BuildDirectory"
    return $true
}

if (($All.IsPresent -and -not [string]::IsNullOrWhiteSpace($Source)) -or (-not $All.IsPresent -and [string]::IsNullOrWhiteSpace($Source))) {
    Write-UsageError -Message "Specify exactly one of -Source or -All."
    exit 2
}
if ($MaxPasses -lt 2 -or $MaxPasses -gt 5) {
    Write-UsageError -Message "MaxPasses must be between 2 and 5."
    exit 2
}
if ($All.IsPresent -and -not [string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Write-UsageError -Message "OutputDirectory can only be used with -Source."
    exit 2
}

$LuaLatex = Find-TexExecutable -Name "lualatex.exe"
if ([string]::IsNullOrWhiteSpace($LuaLatex)) {
    [Console]::Error.WriteLine("lualatex.exe was not found on PATH. Restart VS Code after installing MiKTeX.")
    exit 1
}
$Biber = Find-TexExecutable -Name "biber.exe"
$Bibtex = Find-TexExecutable -Name "bibtex.exe"

$Targets = [System.Collections.Generic.List[object]]::new()
if ($All.IsPresent) {
    foreach ($RelativeSource in $RegisteredSources) {
        $ResolvedSource = [System.IO.Path]::GetFullPath((Join-Path $LatexRoot $RelativeSource))
        $TargetOutput = Join-Path $BuildRoot ([System.IO.Path]::GetFileNameWithoutExtension($ResolvedSource))
        $Targets.Add([Tuple]::Create($ResolvedSource, $TargetOutput))
    }
}
else {
    $CandidateSource = if ([System.IO.Path]::IsPathRooted($Source)) { $Source } else { Join-Path (Get-Location) $Source }
    $ResolvedSource = [System.IO.Path]::GetFullPath($CandidateSource)
    $TargetOutput = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        Join-Path $BuildRoot ([System.IO.Path]::GetFileNameWithoutExtension($ResolvedSource))
    }
    elseif ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    }
    else {
        [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
    }
    $Targets.Add([Tuple]::Create($ResolvedSource, $TargetOutput))
}

$AnyFailure = $false
foreach ($Target in $Targets) {
    $TargetSource = [string]$Target.Item1
    $TargetBuild = [string]$Target.Item2
    if (-not (Test-Path -LiteralPath $TargetSource -PathType Leaf) -or [System.IO.Path]::GetExtension($TargetSource) -ne ".tex") {
        [Console]::Error.WriteLine("TeX source was not found: $TargetSource")
        $AnyFailure = $true
        continue
    }
    Assert-BuildDirectory -Path $TargetBuild
    if ($Clean.IsPresent) {
        Clear-DocumentBuild -Path $TargetBuild
    }
    $Succeeded = Invoke-LatexBuild -SourcePath $TargetSource -BuildDirectory $TargetBuild -PassLimit $MaxPasses -LuaLatexCommand $LuaLatex -BiberCommand $Biber -BibtexCommand $Bibtex
    if (-not $Succeeded) {
        $AnyFailure = $true
    }
}

if ($AnyFailure) {
    exit 1
}
exit 0
