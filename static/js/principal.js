// Variables para manejar la confirmación
var confirmCallback = null;
// Mostrar alerta
function showAlert(message, title) {
    if (title) document.querySelector('#customAlertOverlay .dialog-header').textContent = title;
    document.getElementById('customAlertBody').innerHTML = message;
    document.getElementById('customAlertOverlay').classList.add('active');
    // Establecer foco en el botón Aceptar después de un pequeño retraso para permitir que el diálogo se muestre
    setTimeout(function() {
        document.querySelector('#customAlertOverlay .dialog-btn-primary').focus();
    }, 50);
}
// Ocultar alerta
function hideAlert() {
    document.getElementById('customAlertOverlay').classList.remove('active');
}
// Mostrar confirmación
function showConfirm(message, callback, title) {
    if (title) document.querySelector('#customConfirmOverlay .dialog-header').textContent = title;
    document.getElementById('customConfirmBody').innerHTML = message;
    document.getElementById('customConfirmOverlay').classList.add('active');
    confirmCallback = callback;
    // Establecer foco en el botón Aceptar después de un pequeño retraso
    setTimeout(function() {
        document.querySelector('#customConfirmOverlay .dialog-btn-primary').focus();
    }, 50);
}
// Manejar respuesta de confirmación
function handleConfirm(response) {
    document.getElementById('customConfirmOverlay').classList.remove('active');
    if (confirmCallback) {
        confirmCallback(response);
        confirmCallback = null;
    }
}
// Reemplazos para alert() y confirm()
function customAlert(message, title) {
    showAlert(message, title || 'Alerta');
}
function customConfirm(message, callback, title) {
    showConfirm(message, callback, title || 'Confirmación');
}