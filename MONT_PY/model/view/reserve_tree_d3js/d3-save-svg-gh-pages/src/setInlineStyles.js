import prefix from './namespaces';

export default function (svg) {

  // add empty svg element
  var emptySvg = window.document.createElementNS(prefix.svg, 'svg');
  window.document.body.appendChild(emptySvg);
  var emptySvgDeclarationComputed = window.getComputedStyle(emptySvg);

  // hardcode computed css styles inside svg
  var allElements = traverse(svg);
  var i = allElements.length;
  while (i--) {
    explicitlySetStyle(allElements[i]);
  }

  emptySvg.parentNode.removeChild(emptySvg);

  function explicitlySetStyle(element) {
    var cSSStyleDeclarationComputed = window.getComputedStyle(element);
    var i;
    var len;
    var key;
    var value;
    var computedStyleStr = '';

    for (i = 0, len = cSSStyleDeclarationComputed.length; i < len; i++) {
      key = cSSStyleDeclarationComputed[i];
      value = cSSStyleDeclarationComputed.getPropertyValue(key);
      if (value !== emptySvgDeclarationComputed.getPropertyValue(key)) {
        // Don't set computed style of width and height. Makes SVG elmements disappear.
        if ((key !== 'height') && (key !== 'width')) {
          computedStyleStr += key + ':' + value + ';';
        }

      }
    }

    element.setAttribute('style', computedStyleStr);
  }

  function traverse(obj) {
    var tree = [];
    tree.push(obj);
    visit(obj);
    function visit(node) {
      if (node && node.hasChildNodes()) {
        var child = node.firstChild;
        while (child) {
          if (child.nodeType === 1 && child.nodeName != 'SCRIPT') {
            tree.push(child);
            visit(child);
          }

          child = child.nextSibling;
        }
      }
    }

    return tree;
  }
}
