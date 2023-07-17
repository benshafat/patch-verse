# patch-verse

This code was worked on as part of a home-exam, and implements the Adversarial Patch algorithm 
for attacking image classification models.

The code here is a deep refactoring of the code in https://github.com/jhayes14/adversarial-patch , which was insturmental in getting acquainted with the solution.

The code assumes you have an imagenet directory setup for image sampling. See the original project for details.

Once this is setup, the code is run from `demo.py`

Main changes over the original:

 * Config-file (json) to replace most arg-parse params
 * Lots of code-smell & bug fixes, cleaner design
 * Object oriented implementation
 * Logging
 * Initial defenses implemented

IMPORTANT: when I wrote the code, my laptop did not have NVIDIA GPU, so sadly this implementation runs on CPU - CUDA needs to be re-implemented (see original repo)
