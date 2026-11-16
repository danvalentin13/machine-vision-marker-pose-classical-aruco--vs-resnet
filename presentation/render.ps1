# render.ps1 -- exports every slide of a deck to PNG with PowerPoint itself (what the audience will see).
#   powershell -ExecutionPolicy Bypass -File presentation\render.ps1 [-Name deck.pptx] [-Out folder]
# Defaults: the deck Marker_Pose_Classical_vs_Learned.pptx -> presentation\render\slide_NN.png
param(
    [string]$Name = "Marker_Pose_Classical_vs_Learned.pptx",
    [string]$Out = "render"
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptx = Join-Path $here $Name
$dir = Join-Path $here $Out
New-Item -ItemType Directory -Force $dir | Out-Null
Get-ChildItem $dir -Filter "slide_*.png" | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($pptx, $true, $false, $false)
    $i = 1
    foreach ($s in $pres.Slides) {
        $s.Export((Join-Path $dir ("slide_{0:D2}.png" -f $i)), "PNG", 1600, 900)
        $i++
    }
    Write-Output ("exported " + ($i - 1) + " slides to " + $dir)
    $pres.Close()
}
finally {
    $app.Quit()
}
