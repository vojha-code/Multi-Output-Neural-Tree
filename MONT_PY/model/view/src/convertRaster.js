export {converterEngine, getImageBase64, isDataURL};

function converterEngine(input) { // fn BLOB => Binary => Base64 ?
  var uInt8Array = new Uint8Array(input);
  var i = uInt8Array.length;
  var biStr = []; //new Array(i);
  while (i--) {
    biStr[i] = String.fromCharCode(uInt8Array[i]);
  }

  var base64 = window.btoa(biStr.join(''));
  return base64;
};

function getImageBase64(url, callback) {
  var xhr = new XMLHttpRequest(url);
  var img64;
  xhr.open('GET', url, true); // url is the url of a PNG/JPG image.
  xhr.responseType = 'arraybuffer';
  xhr.callback = callback;
  xhr.onload = function() {
    img64 = converterEngine(this.response); // convert BLOB to base64
    this.callback(null, img64); // callback : err, data
  };

  xhr.onerror = function() {
    callback('B64 ERROR', null);
  };

  xhr.send();
};

function isDataURL(str) {
  var uriPattern = /^\s*data:([a-z]+\/[a-z0-9\-]+(;[a-z\-]+\=[a-z\-]+)?)?(;base64)?,[a-z0-9\!\$\&\'\,\(\)\*\+\,\;\=\-\.\_\~\:\@\/\?\%\s]*\s*$/i;
  return !!str.match(uriPattern);
}
