# backprop_demo_video

to merge videos to one file 

1. On Windows
```
"file 'FFForwardPass.mp4'", "file 'FFBackwardPass.mp4'", "file 'LinearLayerDerivation.mp4'" | Out-File -Encoding ascii list.txt; ffmpeg -y -f concat -safe 0 -i list.txt -c copy LinearLayerFull.mp4; ii LinearLayerFull.mp4
```