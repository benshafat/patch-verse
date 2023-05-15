# patch-verse

This code was worked on as a side-project, trying to implement the Adversarial Patch algorithmfor attacking image classification models.

It was based off https://github.com/jhayes14/adversarial-patch

Code improvements:
	* Lots of code-smell fixes
	* Object oriented implementation
	* Prevented repeat calls to classifier on images we already know were misclassified in original

Bug fixes:
