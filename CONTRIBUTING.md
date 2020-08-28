<h1>Contributing to Primeclue</h1>
Everybody is welcome to contribute to Primeclue. Choose anything you want to work on, 
but before you do, drop me a line at lukasz.wojtow@gmail.com

Primeclue's branches:
1. ```dev``` - For initial checks like tests and warnings.
2. ```beta``` - For longer tests and more courageous users.
3. ```master``` - For changes that have been thoroughly tested.

To contribute, please do:

1. Fork the repo and write your changes against ```dev``` branch. Format your code with ```backend/scripts/format.sh```.
2. Run ```backend/scripts/build.sh```. Ensure there are no warnings and that all tests pass.
3. When finished, squash your work to a single commit. Add a description if needed.
4. Sync upstream changes to your repo. Don't try to rewrite branch's history.
5. Make a pull request to ```dev``` branch. I promise a quick review.
6. ```dev``` branch is merged to ```beta``` after short verification.
7. ```beta``` branch is manually tested. If everything works fine, changes are cherry-picked to ```master``` branch after a while.
8. ```master``` is released as a new version on regular basis.

Last but not least:
At the moment I don't see any need for ```unsafe``` code, and so I will not accept a PR with it, even for improved performance.
If you really think you have to use ```unsafe``` - talk to me first, please.
