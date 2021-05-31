# Playing-NES-Games-with-NEAT
This is a Python 3.8 project that uses the NEAT algorithm to play Super Mario Bros. and The Legend of Zelda for the NES.
The folders are not super intuitively named, as I had to type in commands to change the directory manually due to how my Python environment was set up.
* g: Contains all the files for Mario, including checkpoints and finished .pkg files, which contain the neural networks that were the winners for their respective levels/conditions. I also have a backup copy of the Gym Retro Integration tool in a subfolder here. If you need it, just copy the .exe to %Python environment root%/Lib/site-packages/retro and it should run.
* z: Contains the Zelda files

The other two folders contain the Gym Retro integration files, which includes savestates and info files that the Gym Retro library uses to get information from the games' RAM and determine the done condition for Zelda (the one for Mario is overridden by my scripts, so that death is detected instantly). **The code will not work without these files being updated in your Python environment!**

Paste both folders into: %Python environment root%/Lib/site-packages/retro

If pasting them doesn't work, you should see an error when you try to run the scripts (Mario might appear to work without issue if you're just running the playback script, but Zelda will definitely break). Also, it goes without saying that you need the ROMs for these games for any of this to run. For obvious legal reasons, those aren't provided here.

A couple of other important notes about the files:
* The Mario folder is messier than the Zelda one, as I spent a lot of time trying different configurations, reward functions, levels, etc. You'll find several files without extensions with various configurations for the NEAT algorithm.
* There are single-core and multicore versions of the Mario and Zelda scripts. For Mario, they are m.py and mp.py; For Zelda, z.py and zp.py. The single-core ones were primarily used to make sure everything was working properly and may be missing some code found in the multicore versions.
* I saved a disassembly for Mario in the g folder and a RAM map of Zelda in the z folder. Neither of these were made by me and the credits and links are in the files themselves.
