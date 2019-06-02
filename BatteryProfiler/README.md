# Battery Profiler #

Outputs battery consumption of 360p, 480p, 720p videos by measuring phone battery in mAh at start and end of video playback.

## Code Implementation ##
 
There are two activities for the two screens of the app:
	1) Main Activity: Main screen with buttons that launch 360p, 480p, and 720p videos. Once a button is clicked, the ExoPlayer activity is started and the video is played. Once video playback is finished, the ExoPlayer passes the battery information back to the MainActivity by attaching the info to an Intent. The information received is handled in the onActivityResult callback method.
	2) ExoPlayer: Activity that prepares the video, renders the video, and does battery measurements before and after the video. 

## Extending the code ##

If you want to measure another video of a different resolution, make the following changes:
 	1) Main Activity: Create new button with onClickListener that calls startActivityForResult to launch the ExoPlayer activity. Pass a unique requestCode to startActivityForResult so that the MainActivity can handle the returned information depending on video quality in onActivityResult once video playback is finished. Also, modify onActivityResult to handle new requestCode. 
 	2) ExoPlayer: Modify prep_exo() so that it handles new video link. 