import download from './download';
import preprocess from './preprocess';
import prefix from './namespaces';
import {converterEngine, getImageBase64, isDataURL} from './convertRaster';

export function save(svgElement, config) {
  if (svgElement.nodeName !== 'svg' || svgElement.nodeType !== 1) {
    throw 'Need an svg element input';
  }

  var config = config || {};
  var svgInfo = preprocess(svgElement, config);
  var defaultFileName = getDefaultFileName(svgInfo);
  var filename = config.filename || defaultFileName;
  var svgInfo = preprocess(svgElement);
  download(svgInfo, filename);
}

export function embedRasterImages(svg) {

  var images = svg.querySelectorAll('image');
  [].forEach.call(images, function(image) {
    var url = image.getAttribute('href');

    // Check if it is already a data URL
    if (!isDataURL(url)) {
      // convert to base64 image and embed.
      getImageBase64(url, function(err, d) {
        image.setAttributeNS(prefix.xlink, 'href', 'data:image/png;base64,' + d);
      });
    }

  });

}

function getDefaultFileName(svgInfo) {
  var defaultFileName = 'untitled';
  if (svgInfo.id) {
    defaultFileName = svgInfo.id;
  } else if (svgInfo.class) {
    defaultFileName = svgInfo.class;
  } else if (window.document.title) {
    defaultFileName = window.document.title.replace(/[^a-z0-9]/gi, '-').toLowerCase();
  }

  return defaultFileName;
}
