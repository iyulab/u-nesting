export function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
  triggerDownload(filename, URL.createObjectURL(blob));
}
export function downloadCanvasPng(filename, canvas) {
  triggerDownload(filename, canvas.toDataURL('image/png'));
}
function triggerDownload(filename, url) {
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
}
