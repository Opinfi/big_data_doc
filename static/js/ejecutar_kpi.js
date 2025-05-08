// JavaScript ejecutar_kpi...

// Verificación de Tippy
function initTippy() {
    if (typeof tippy === 'undefined') {
        console.error('Tippy.js no está disponible');
        return false;
    }
    return true;
}
// Función para inicializar tooltips 
function initTooltips() {
    // Destruye completamente todas las instancias previas
    tippy.hideAll();
    document.querySelectorAll('[data-tippy-root]').forEach(el => {
        el._tippy?.destroy();
    });
    const config = {
        allowHTML: true,
        theme: 'yellow-theme',
        placement: 'right',
        arrow: false,
        interactive: true,
        trigger: 'mouseenter focus',
        hideOnClick: true,
        appendTo: document.body, // Importante: adjunta al body
        zIndex: 10000, // Valor explícito
        onShow(instance) {
            tippy.hideAll({ duration: 0 });
            const content = instance.reference.getAttribute('data-tippy-content');
        },
        onHidden(instance) {
            instance.setContent('');
        },
        popperOptions: {
            modifiers: [
                {
                    name: 'preventOverflow',
                    options: {
                        boundary: document.body // Previene que se oculten
                    }
                }
            ]
        }
    };
    // Aplicación con verificación de elementos
    const elements = {
        label: document.querySelector('#labelExaminar'),
        input: document.querySelector('#inputUrl')
    };
    if (elements.label) {
        elements.label.setAttribute('data-tippy-content', 'Seleccione un archivo para cargar');
        tippy(elements.label, config);
    }    
    if (elements.input) {
        elements.input.setAttribute('data-tippy-content', 'Ingrese una URL válida del documento');
        tippy(elements.input, config);
    }
}
// Llamada inicial al cargar la página
document.addEventListener("DOMContentLoaded", function () { 
    initTooltips(); // Inicialización inicial
    document.getElementById('inputArchivo').addEventListener('change', function (e) {
        const fileName = e.target.files[0] ? e.target.files[0].name : "No se ha seleccionado ningún archivo";
        document.getElementById('file-name').textContent = fileName;
    });
    document.getElementById('listaDocumentos').addEventListener('change', function(e) {
        tippy.hideAll({ duration: 0 }); // Oculta todos los tooltips inmediatamente
        url = e.target.value;
        mostrarDocumento(url);
    });
    // Eventos para mouseleave en los elementos especificados
    const fileUploadContainer = document.querySelector('.custom-file-upload');
    const inputUrl = document.getElementById('inputUrl');
    if (fileUploadContainer) {
        fileUploadContainer.addEventListener('mouseenter', function() {
            const url = document.getElementById('listaDocumentos').value;
            if (url) mostrar_tooltips(url);
        });
    }
    if (inputUrl) {
        inputUrl.addEventListener('mouseenter', function() {
            const url = document.getElementById('listaDocumentos').value;
            if (url) mostrar_tooltips(url);
        });
    }
});
// Función para actualizar tooltips
function actualizarTooltips(contenidoHTML) {
    // Obtiene las instancias directamente del DOM
    const getInstance = (selector) => {
        const el = document.querySelector(selector);
        return el?._tippy || tippy(el);
    };
    const instances = [
        getInstance('#labelExaminar'),
        getInstance('#inputUrl')
    ].filter(Boolean);
    instances.forEach(instance => {
        instance.setProps({
            allowHTML: true,
            theme: 'yellow-theme'
        });        
        // Método de fuerza bruta para garantizar renderizado
        const container = instance.popper.querySelector('.tippy-content');
        if (container) {
            container.innerHTML = contenidoHTML;
            container.style.color = '#000000';
        }
    });
}
// Funcion para mostrar los tooltips del json actual
function mostrar_tooltips(url) {
    if (!url) return;
    fetch('/get_tooltips?archivo_json=' + encodeURIComponent(url))
        .then(response => {
            if (!response.ok) throw new Error('Error en la red');
            return response.json();
        })
        .then(data => {
            if (data.Status == "OK") {
                let tooltipHTML = data.tooltips.join('<br>');
                if (data.tooltips.length > 1) {
                    tooltipHTML += '<br>*<em>Debe enviar los archivos en un <strong>.zip</strong> o <strong>.rar</strong></em>';
                }
                // Actualiza los tooltips existentes
                const label = document.querySelector('#labelExaminar');
                const input = document.querySelector('#inputUrl');
                if (label && label._tippy) {
                    label._tippy.setContent(tooltipHTML);
                    label.setAttribute('data-tippy-content', tooltipHTML);
                }               
                if (input && input._tippy) {
                    input._tippy.setContent(tooltipHTML);
                    input.setAttribute('data-tippy-content', tooltipHTML);
                }
            }
        })
        .catch(error => {
            console.error('Error al cargar tooltips:', error);
            actualizarTooltips('Error al cargar información');
        });
}
// Funcion para mostrar el documento seleccionado
function mostrarDocumento(url) {
    const iframe = document.getElementById('visorDocumento');
    iframe.src = url;
    if (url.includes('.json')) {
        mostrar_tooltips(url);
    }
}
var url = "";
// Mostrar el primer documento al cargar la página
window.onload = function() {
    const select = document.getElementById('listaDocumentos');
    if (select.options.length > 0) {
        url = select.options[0].value;
        mostrarDocumento(url);
    }
};
// Funcion para mostrar el spinner
function mostrarSpinner() {
    document.getElementById('overlay').style.display = 'block';
}
// Funcion para ocultar el spinner
function ocultarSpinner() {
    document.getElementById('overlay').style.display = 'none';
}
// Variable global que retiene la url del doc Word generado
var url_informe_kpi_doc = null;
var url_informe_kpi_pdf = null;
function mostrarDocWord() {
    if (url_informe_kpi_doc != null)
        mostrarDocumento(url_informe_kpi_doc);
}
function mostrarDocPDF() {
    if (url_informe_kpi_pdf != null)
        mostrarDocumento(url_informe_kpi_pdf);
}
// Funcion para ejecutar el modelo
function ejecutarModelo() {
    const inputArchivo = document.getElementById('inputArchivo');
    const inputUrl = document.getElementById('inputUrl');
    let formData = new FormData();
    if (inputArchivo.files.length > 0) {
        // Si se cargó un archivo local
        formData.append('documento', inputArchivo.files[0]);
    } else if (inputUrl.value.trim() !== '') {
        // Si se proporcionó una URL
        formData.append('url', inputUrl.value.trim());
    } else {
        customAlert('Por favor, seleccione un archivo o introduzca una URL.');
        //alert('Por favor, seleccione un archivo o introduzca una URL.');
        return;
    }
    // Obtener la zona horaria del cliente (ej: "America/Caracas")
    const clientTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    formData.append('timezone', clientTimeZone);
    // Mostrar el spinner
    mostrarSpinner();
    // Enviar la solicitud al servidor para ejecutar el modelo
    fetch('/ejecutar_kpi', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.Status === 'OK') {
            customConfirm('Modelo ejecutado correctamente.', 'Éxito!');
            //alert('Modelo ejecutado correctamente.');
            // Actualizar el visor de documentos con el resultado
            url_informe_kpi_pdf = data.url_documento;
            mostrarDocumento(url_informe_kpi_pdf);
            // Guardamos la referencia al url_docx, para que opcionalmente 
            // el usuario quisiera ver el documento Word asociado...
            url_informe_kpi_doc = data.url_docx;
        } else {
            customAlert('Error al ejecutar el modelo: ' + data.Message, 'Error');
            //alert('Error al ejecutar el modelo: ' + data.Message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        customAlert('Error al ejecutar el modelo.','Error');
        //alert('Error al ejecutar el modelo.');
    })
    .finally(() => {
        // Ocultar el spinner
        ocultarSpinner();
    });
}