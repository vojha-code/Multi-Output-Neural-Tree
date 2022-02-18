# Contributing
Fork, then clone the repo:

    git clone git@github.com:your-username/d3-save-svg.git

Set up your machine:

    npm install

If the tests pass, then the library should build in the `/build` folder.
Make your change. Add tests for your change. Make the tests pass:

    npm test

Push to your fork and [submit a pull request](https://github.com/edeno/d3-save-svg/compare/) to the develop branch.

### Additional Notes
Testing is via [Tape](https://github.com/substack/tape) and [jsdom](https://github.com/tmpvar/jsdom). Right now the tests are pretty rudimentary. Also `index.html` serves as a good check on whether things are working.

Development is done using the [git-flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow. Please merge changes into the `develop` branch.
