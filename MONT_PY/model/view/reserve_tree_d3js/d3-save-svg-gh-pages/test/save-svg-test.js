var tape = require('tape');
var d3SaveSvg = require('../');
var jsdom = require('jsdom');
var xmls = require('xmlserializer');
var Blob = require('w3c-blob');

var revokeObjectURLShim = function() {};

var createObjectURLShim = function() {};

var xmlsShim = function() {
  this.serializeToString = xmls.serializeToString;
};

tape('save() throws an error if element is not an svg element.', function(test) {
  jsdom.env({
    html: '<html><body><h1>hello</h1><svg></svg></body></html>',
    done: function(errs, window) {

      window.URL.createObjectURL = createObjectURLShim;
      window.URL.revokeObjectURL = revokeObjectURLShim;
      global.window = window;
      global.XMLSerializer = xmlsShim;
      global.Blob = Blob;
      global.document = window.document;
      var h1 = window.document.querySelector('h1');
      test.throws(function() {d3SaveSvg.save(h1);});
    },
  });

  test.end();
});

tape('save() accepts svg elements', function(test) {
  jsdom.env({
    html: '<html><body><h1>hello</h1><svg></svg></body></html>',
    done: function(errs, window) {

      window.URL.createObjectURL = createObjectURLShim;
      window.URL.revokeObjectURL = revokeObjectURLShim;
      global.window = window;
      global.XMLSerializer = xmlsShim;
      global.Blob = Blob;
      global.document = window.document;
      var svg = window.document.querySelector('svg');
      test.doesNotThrow(function() {d3SaveSvg.save(svg);});
    },
  });

  test.end();
});
