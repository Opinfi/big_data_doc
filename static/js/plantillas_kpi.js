// JavaScript Document Administracion de Plantillas KPI

// Funcion para mostrar el spinner
function mostrarSpinner() {
    document.getElementById('overlay').style.display = 'block';
}

// Funcion para ocultar el spinner
function ocultarSpinner() {
    document.getElementById('overlay').style.display = 'none';
}

async function Cargar_Plantilla(url_plantilla){
    // Enviar solicitud al servidor
    mostrarSpinner();
    const response = await fetch('/convertir_word_pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ruta_word: url_plantilla })
    });
    const data = await response.json();
    ocultarSpinner();
    if (data.error) {
        customAlert(data.error, "Error");
        return;
    }
    // Abrir el PDF en una nueva pestaña
    window.open(data.pdf_url, '_blank');
}

document.addEventListener("DOMContentLoaded", function () {

    document.getElementById('doc_plantilla').addEventListener('change', function (e) {
        // Obtiene el nombre del archivo seleccionado
        const fileName = e.target.files[0] ? e.target.files[0].name : "No se ha seleccionado ningún archivo";
        // Muestra el nombre del archivo en el span
        document.getElementById('file-name').textContent = fileName;
    });

    // Exponer la función al ámbito global
    window.Adicionar_Plantilla_Nueva = function() {
        const archivo = document.getElementById("doc_plantilla").files[0];
        if (archivo) {
            var formData = new FormData();
            formData.append("plantilla_kpi", archivo);
            var denominacion = document.getElementById("denominacion").value;
            formData.append("denominacion", denominacion);
            var clase = document.getElementById("clase").value;
            formData.append("clase", clase);
            if (denominacion=="") {
                customAlert("Debe introducir una Denominación o nombramiento para la nueva plantilla.");
                return;
            }
            if (clase=="") {
                customAlert("Debe introducir una clase o agrupamiento clasificado para la nueva plantilla.");
                return;
            }
            // Realizar una solicitud al servidor para chequear 
            // la existencia de campos de la plantilla seleccionada
            let xhr = new XMLHttpRequest();
            xhr.responseType = "json";
            xhr.open("POST", "/chequear_campos_plantilla", true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    const campos = xhr.response;
                    if (campos.length === 0) {
                        customAlert("La plantilla seleccionada no contiene campos con formato __CAMPO__.", "Error!");
                        return;
                    }
                    // Enviar la solicitud POST al servidor
                    let xhr2 = new XMLHttpRequest();
                    xhr2.responseType = "json";
                    xhr2.open("POST", "/plantillas_kpi", true);
                    xhr2.onload = function() {
                        if (xhr2.status === 200) {
                            resp = xhr2.response;
                            if (resp.Status == "OK") {
                                const tabla = document.getElementById('tabla_lista_plantillas_propias').getElementsByTagName('tbody')[0];
                                tabla.insertAdjacentHTML('beforeend', resp.Fila_nueva);
                                customConfirm("Plantilla KPI '"+denominacion+"' insertada correctamente.",null,"Éxito!");
                                document.getElementById("denominacion").value = "";
                                document.getElementById("clase").value = "";
                                document.getElementById("file-name").textContent = "No se ha seleccionado ningún archivo";
                            } else if (resp.Status == "ERROR"){
                                customAlert(resp.Message,"Error!");
                            }
                        }
                    }
                    xhr2.send(formData);
                }
            }
            xhr.send(formData);
        } else {
            customAlert("Debe insertar un documento que represente una Plantilla KPI.");
        }
    }
});