/* JavaScript Document

		Tratamiento de interfaz para desarrollar y editar los modelos de ejecucion de Informes KPI
*/
$(document).ready(function() {
	// Funcion para mostrar el spinner
	function mostrarSpinner() {
		document.getElementById('overlay').style.display = 'block';
	}
	// Funcion para ocultar el spinner
	function ocultarSpinner() {
		document.getElementById('overlay').style.display = 'none';
	}
	// El siguiente evento nos sirve para controlar que, si el usuario esta editando un modelo en su flujo de 
	// procesamiento y no ha guardado los cambios, se dispara el evento "beforeunload"
	window.addEventListener('beforeunload', function (e) {
		if (hayCambiosSinGuardar()) {
			e.preventDefault();
			e.returnValue = 'Esta accion puede hacer perder si edicion del modelo actual!';
		}
	});
	// Configurar el event listener
	window.addEventListener('change', function(e) {
		if (e.target.name === 'colTodas') {
			const container = e.target.closest('.contenedor_opt_columnas');
			const checkboxes = container.querySelectorAll('input[name="columnas"]');
			const isChecked = e.target.checked;			
			checkboxes.forEach(cb => {
				cb.checked = isChecked;
			});
			// Cambiar solo el texto del span
			const textSpan = e.target.nextElementSibling;
			if (textSpan && textSpan.classList.contains('select-all-text')) {
				textSpan.textContent = isChecked ? "Deseleccionar Todas" : "Seleccionar Todas";
			}
		}
	});
	let CambiosSinGuardar = true;
	function hayCambiosSinGuardar() {
		// Lógica para determinar si hay cambios sin guardar
		if (workflow_json.length > 0 || connections.length > 0) {
			if (!CambiosSinGuardar)
				return false;
			return true;
		}
		return false;
	}

	window.Salvar_modelo  = function() {
		// Aqui debe realizarse un evento XMLHttpRequest al servidor paa guardar la ultima version del
		// modelo nuevo o abierto para ser editado
		var formData = new FormData();
		//Hacemos una unica lista de dos listas, donde la primera contiene los objetos
		//y la segunda las conexiones entre ellos... Pero primero vamos a eliminar el atributo
		//"table" contenido en todas las tablas de los objetos...
		//Ademas de lo anterior, tambien se eliminan aquellos objetos en workflow_json que no
		//tiene conexiones con ningun objeto, partiendo del arreglo conections
		for(let i = 0; i < workflow_json.length; i++) {
			var obj = workflow_json[i];
			// averiguar si obj esta conectado hacia (to) o desde (from) otro objeto existente
			index = connections.findIndex(o => o.from === obj.id || o.to === obj.id);
			if (index == -1) {
				// nadie se conecta a obj !
				workflow_json.splice(i, 1); // Elimina 1 elemento en la posición i
				continue;
			}
			// se eliminaran todos los contenidos de filas y columnas de las tablas transitorias
			// en el flujo y solo dejar las que se utilizan en los obj tipo Documento				
			var tablas = [];
			if (["Limpieza","Conversion","Procesamiento","Grafico"].includes(obj.tipo)){
				tablas = obj.resultado;
			}
			for(let j = 0; j < tablas.length; j++) 
				tablas[j].table = {};
			// Actualizando left y top del objeto...
			let Element = $(`.element[data-id="${obj.id}"]`);
			obj.left = parseFloat(Element.css('left'));
			obj.top = parseFloat(Element.css('top'));
		}
		listas = JSON.stringify([ workflow_json, connections ]);
		formData.append("listas_json", listas);
		nombre_modelo = document.getElementById("nombre_modelo").value;
		formData.append("nombre_modelo", nombre_modelo);
		// Enviar la solicitud POST al servidor
		let xhr = new XMLHttpRequest();
		xhr.responseType = "json";
		xhr.open("POST", "/salvar_modelo", true);
		xhr.onload =  function() {
			if (xhr.status === 200) {
				resp = xhr.response;
				if (resp.Status=="OK") {
					CambiosSinGuardar = false;
					customConfirm("El flujo de procesos del actual modelo ha sido archivado y/o actualizado!.");
				} else if (resp.Status=="ERROR") {
					CambiosSinGuardar = true;
					customAlert(resp.Message);
				}
			};
		}
		xhr.send(formData);
	}

	// Variables principales en el manejo de los modelos
	let elements = {}; // Aqui se guardan todos los elementos de un modelo
	let connections = []; // Aqui se guardan las conexiones desde - hasta de entre dos elementos
	let workflow_json = []; // Aqui se guardan los atributos y propiedades y acciones por cada elemento
	let count_e = 0; // Contador para asegurar unicidad en los elementos
	let count_t = 0; // Contador para asegurar unicidad en las tablas
	const first_tooltipDoc = "Aquí puedes agregar tu documento que contenga una o más tablas, haciendo clic derecho.";
	// Etiquetas de los procesos asociados a su texto legible
	Procesos_Label2Descrip = {
		"Quitar_duplicados" : "Quitar duplicados",
		"Recuperar_num_corruptos" : "Recuperar n&uacute;meros corruptos",
		"Eliminar_filasfcond" : "Filtrar filas condicionalmente",
		"Agrupar_filas" : "Agrupar filas",
		"Ordenar_filas" : "Ordenar filas",
		"Estadistica_descriptiva" : "Estad&iacute;stica descriptiva",
		"Frecuencias_densidad_prob" : "Frecuencias (<em>Densidad Probabil&iacute;stica</em>)",
		"Prediccion_lineal" : "Predicci&oacute;n por regresi&oacute;n lineal",
		"Prediccion_nolineal" : "Predicci&oacute;n por regresi&oacute;n no lineal",
		"Prediccion_SeriesTemporales" : "Predicci&oacute;n por series temporales",
		"Clasif_Arbol_decision" : "Clasif. &Aacute;rbol de decisi&oacute;n",
		"Clasif_Random_forest" : "Clasif. Bosque aleatorio (<em>Random forest</em>)",
		"Clasif_Regresion_logistica" : "Clasif. Regresi&oacute;n log&iacute;stica",
		"Clasif_SVM" : "Clasif. M&aacute;quina de soporte vectorial (<em>SVM</em>)",
		"Generar_grafico_lineas" : "Gr&aacute;fico de l&iacute;neas",
		"Generar_grafico_barras" : "Gr&aacute;fico de barras",
		"Generar_grafico_dispersion" : "Gr&aacute;fico de dispersi&oacute;n",
		"Generar_grafico_pastel" : "Gr&aacute;fico de pastel",
		"Generar_grafico_histograma" : "Gr&aacute;fico de histograma"
	};
	// Funcion para mostrar el men� contextual correcto
	function showContextMenu(elementType, x, y) {
		// Muestra el menu correspondiente al tipo de elemento
		let menuId = "";
		switch (elementType) {
			case "Documento":
				menuId = "#context-menu-documento";
				break;
			case "Limpieza":
				menuId = "#context-menu-limpieza";
				break;
			case "Conversion":
				menuId = "#context-menu-conversion";
				break;
			case "Procesamiento":
				menuId = "#context-menu-procesamiento";
				break;
			case "Prediccion_futura":
				menuId = "#context-menu-prediccion";
				break;
			case "Clasificacion_predictiva":
				menuId = "#context-menu-clasificacion";
				break;
			case "Grafico":
				menuId = "#context-menu-grafico";
				break;
			case "InformeKPI":
				menuId = "#context-menu-informekpi";
				break;
			default:
				console.log("Tipo de elemento no reconocido");
				return;
		}		
		// Posiciona el men� contextual en las coordenadas del clic derecho
		$(menuId).css({
			top: y + "px",
			left: x + "px",
		}).show();
	}
	function ocultar_menues() {
		// remover la clase disabled-dialog (opacidad) de los menues contextuales
		$(".context-menu").filter(":visible").removeClass("disabled-dialog");
		$(".context-menu").hide(); // Ocultar todos los menúes contextuales
	}
	function drawConnection(fromId, toId, isNew) {  
		const fromElement = $(`.element[data-id="${fromId}"]`)[0];  
		const toElement = $(`.element[data-id="${toId}"]`)[0];  
		
		// chequear si existen ambos elementos
		if (! fromElement || ! toElement)
			return;
		
		const fromLeft = parseFloat(fromElement.style.left);
		const toLeft = parseFloat(toElement.style.left);
		
		const fromTop = parseFloat(fromElement.style.top);
		const toTop = parseFloat(toElement.style.top);

		const fromWidth = fromElement.offsetWidth; 
		const toWidth = toElement.offsetWidth; 
		// Obtener la altura del elemento from, mediante offsetHeight y el estilo de margen  
		var computedStyle = window.getComputedStyle(fromElement);  
		var marginTop = parseFloat(computedStyle.marginTop);  
		var marginBottom = parseFloat(computedStyle.marginBottom);  
		const fromHeight = fromElement.offsetHeight + marginTop + marginBottom;
		// Obtener la altura del elemento to, mediante offsetHeight y el estilo de margen  
		computedStyle = window.getComputedStyle(toElement);  
		marginTop = parseFloat(computedStyle.marginTop);  
		marginBottom = parseFloat(computedStyle.marginBottom);  
		const toHeight = toElement.offsetHeight + marginTop + marginBottom;
		
		const startX = Math.round(fromLeft + fromWidth); //(fromLeft + (fromWidth / 2));  
		const startY = Math.round(fromTop + (fromHeight / 2)); 
		// Se calcula donde mejor colocar el punto final, segun la ubicacion del elemento to con respecto al from...
		const pitch = 3; // denota una pequen~a separacion de la punta de la fecha al elemento to
		// Se supondra que los elementos estan mas o menos alineados, colocamos el punto final delante
		var endX = Math.round(toLeft - pitch);
		var endY = Math.round(toTop + (toHeight / 2));	
		if (fromTop - toTop > fromHeight) 
		{ // el from esta muy abajo y el to arriba, colocamos el punto debajo y medio del to
			endX = Math.round(toLeft + (toWidth / 2));
			endY = Math.round(toTop + toHeight + pitch);
		}
		// Calcular los puntos de control  
		const controlX1 = Math.round((startX + endX) / 2); // Control X1 es el medio en X  
		const controlY1 = Math.min(startY, endY) - 50; // Control Y1 un poco mas arriba  
		const controlX2 = controlX1; // Para una curva simetrica, es el mismo X  
		const controlY2 = Math.max(startY, endY) + 50; // Control Y2 mas abajo    
		
		// Crear la ruta de la curva, tipo Bezier
		const d = 'M ' + startX + ', ' + startY +   
				' C ' + controlX1 + ', ' + controlY1 +   
				' ' + controlX2 + ', ' + controlY2 +   
				' ' + endX + ', ' + endY;  
		
		// Adjuntar la flecha y el path al SVG  
		// Paso 1: Seleccionar el elemento padre  
		const conn_svg = document.getElementById('connection-svg');  
		// Flecha y su poligono de trazo como hijo
		const arrow = document.createElementNS("http://www.w3.org/2000/svg", "marker");
		arrow.setAttribute("id","arrow");
		arrow.setAttribute("markerWidth","7");
		arrow.setAttribute("markerHeight","5");		
		arrow.setAttribute("refX","0"); 
		arrow.setAttribute("refY","2.5");
		arrow.setAttribute("orient","auto");
		const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
		polygon.setAttribute("points","0 0, 7 2.5, 0 5");
		polygon.setAttribute("fill","gray");
		arrow.appendChild( polygon );
		// Curva
		const path = document.createElementNS("http://www.w3.org/2000/svg", "path"); 
		path.setAttribute("d", d);
		path.setAttribute("stroke", "gray");
		path.setAttribute("stroke-width", "2");
		path.setAttribute("fill", "none");
		path.setAttribute("marker-end", "url(#arrow)");
		
		conn_svg.appendChild( path )
		conn_svg.appendChild( arrow )

		// Guardar conexion si es nueva
		if ( isNew )
			connections.push({ from: fromId, to: toId });  
		// Al haber al menos una conexion entre dos elementos, activamos entonces la edicion 
		// del nombre del modelo...
		document.getElementById("nombre_modelo").disabled = false;
		// Si ya tiene escrito un nombre, habilitar boton de salvar cambios...
		if (document.getElementById("nombre_modelo").value != "")
			document.getElementById("btn_salvarmodelo").disabled = false;
	}  
	// Evento para mostrar el menu contextual al hacer clic derecho
	$(document).on('contextmenu', '.element', function(e) {
		e.preventDefault();
		const elementType = elements[$(this).data("id")].type; // Obtiene el tipo de elemento
		showContextMenu(elementType, e.pageX, e.pageY); // Muestra el men� correspondiente
		$(".context-menu").data("selected-id", $(this).data("id")); // Guarda el ID del elemento seleccionado
	});  	
	// Evento para ocultar el menu al hacer clic en cualquier parte
	$(document).click(function (e) {
		// Verificar si el clic ocurrió dentro de un menú contextual o dentro de un diálogo
		const isClickInsideMenu = $(e.target).closest(".context-menu").length > 0;
		const isClickInsideDialog = $(e.target).closest('[id^="dialogo-"]').length > 0;
		// Ocultar los menús solo si el clic ocurrió fuera de los menús y diálogos
		if (!isClickInsideMenu && !isClickInsideDialog) {
			ocultar_menues();
		}
	});
	function getFileNameFromUrl(url) {
		// Crear un objeto URL a partir de la cadena de la URL
		const urlObj = new URL(url);
		// Obtener la ruta del archivo desde la URL
		const pathname = urlObj.pathname;
		// Extraer el nombre del archivo con su extensión
		const fileNameWithExtension = pathname.split('/').pop();
		return fileNameWithExtension;
	}
	// Cerrar dialogos y menues
	function CerrarDialogosYMenues() {
		// Cerrar todos los diálogos abiertos
		$('.dialogo-container').remove(); // Cerrar el o los diálogos
		ocultar_menues(); // Ocultar menús contextuales
	}
	// Funcion general para el evento de un boton cancelar con id que comienza por "btnCancelar"
	$(document).on("click", '[id^="btnCancelar"]', function (e) {
		if ($(this).is('#btnCancelarEspecial')) {
			// Comportamiento especial
		} else {
			e.preventDefault();
			e.stopPropagation(); // Evitar que el evento se propague
			CerrarDialogosYMenues();
		}
	});
	// función genérica para generar diálogos
	var numIdTablas = 1; // Contador para los diálogos de tablas
	function crearDialogo(contenido) {
		const id = 'dialogo-' + numIdTablas++;
		const idCancelar = 'btnCancelar-' + id;
		const dialogo = $(`
			<div id="${id}" class="dialogo-container">
				${contenido}
				<button id="${idCancelar}" class="custom-button">Cancelar</button>
			</div>
		`);
		$('body').append(dialogo);
		return dialogo;
	}
	// Eventos para las opciones del menu
	$(document).on("click", ".context-menu li",  function ( e ) {
		e.preventDefault();
		// Evitar que el menu inicial se oculte
		e.stopPropagation();
		const id = $(".context-menu").data("selected-id"); // Obtiene el ID del elemento seleccionado
		const action = $(this).attr("id"); // Obtiene la acci�n seleccionada
		const id_icono = `tooltip-${id}`;
		switch (action) {
		case "Leer_pc":
			// Asignar el ID del tooltip al input de archivo
			$('#inputArchivo').attr('data-id-icono', id_icono);
			// Asignar el ID del elemento seleccionado (Documento) al input del archivo
			$('#inputArchivo').attr('data-id-doc', id);
			// Disparar el clic en el input de archivo
			$('#inputArchivo').click();
			break;
		case "Leer_url":
			let url = prompt("Por favor, Introduzca URL donde esta el documento:");
			if (url != ""){
				var formData = new FormData();
				formData.append("url", url);
				// Enviar la solicitud POST al servidor
				let xhr = new XMLHttpRequest();
				xhr.responseType = "json";
				xhr.open("POST", "/leer_doc_desde_url", true);
				xhr.onload =  function() {
					ocultarSpinner();
					if (xhr.status === 200) {
						const tooltip = document.getElementById(id_icono);
						tablas_vigentes = xhr.response;
						objDocumento = workflow_json.find( obj => obj.id === id );
						info_tablas = insertar_tablasObjDocumento(objDocumento, tablas_vigentes);
						archivo_name = getFileNameFromUrl(url);
						if (info_tablas != null && tooltip) {
							tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias l�neas
							str_tooltip = `<strong>${archivo_name}:</strong><ol>${info_tablas}</ol>`;
							tooltip.innerHTML = str_tooltip;
							objDocumento.tooltip = str_tooltip;
						}
						customConfirm("El documento '"+archivo_name+"' se ha cargado correctamente!");
					} else {
						customAlert("Error al subir el archivo desde la URL dada.");
					}
				};
				mostrarSpinner();
				xhr.send(formData);
			}
			break; 
		case "Exportar_tabla":
			// Obtener el ID del elemento de Limpieza
			var obj = workflow_json.find(o => o.id === id);
			if (!obj) {
				customAlert("No se encontró el objeto de Limpieza o de Conversion.");
				break;
			}
			tabla = obj.resultado[0];
			exportarTablaExistente(tabla, id);
			break;
		case "Quitar_duplicados":
		case "Recuperar_num_corruptos":
		case "Agrupar_filas":
		case "Ordenar_filas":
		case "Eliminar_filasfcond":
			// Obtener el ID del elemento de Limpieza
			const idLimConv = id;
			// Buscar el objeto de Limpieza en workflow_json
			const objLimpieza = workflow_json.find(obj => obj.id === idLimConv);
			if (!objLimpieza) {
				customAlert("No se encontró el objeto de Limpieza o de Conversion.");
				break;
			}
			// Buscar el objeto o los objetos conectado(s) a Limpieza
			var conexiones = connections.filter(conn => conn.to === idLimConv);
			if (!conexiones) {
				customAlert("Debe conectar este objeto de Limpieza a un objeto de Documento.");
				ocultar_menues();
				break;
			}
			var idObjs = conexiones.map(obj => obj.from);
			var tablasDisponibles = [];
			for(let i = 0; i < idObjs.length; i++)
			{
				const obj = workflow_json.find(obj => obj.id === idObjs[i]);
				if (!obj || (obj.tipo != "Documento" && obj.tipo != "Conversion" && obj.tipo != "Limpieza")) {
					customAlert("No se encontró objeto de Documento o Limpieza conectado.");
					// Ocultar todos los menús después de seleccionar una opción
					continue
				}
				if (obj.tipo == "Documento")
					tablasDisponibles.push(...obj.tablas);
				else
					tablasDisponibles.push(...obj.resultado);
			}
			// Mostrar un diálogo para seleccionar la tabla
			if (tablasDisponibles.length === 0) {
				customAlert("No hay tablas disponibles en el objeto de Documento.");
				ocultar_menues();
				break;
			}
			// Crear un diálogo con las tablas disponibles
			var dialogoTablas = `
					Seleccione una tabla:<br />
					<select id="selectTabla">
						${tablasDisponibles.map((tabla, index) => `
							<option value="${index}">Tabla ${index + 1} (${tabla.rows}x${tabla.cols})</option>
						`).join("")}
					</select>
					<br /><button id="btnSeleccionarTabla" class="custom-button">Seleccionar</button>
					&nbsp;&nbsp;`;
			crearDialogo(dialogoTablas);
			// Manejar la selección de la tabla
			$("#btnSeleccionarTabla").on("click", function () {
				const tablaIndex = $("#selectTabla").val();
				const tablaSeleccionada = tablasDisponibles[tablaIndex];
				// Mostrar un diálogo para seleccionar las columnas
				switch(action) {
					case "Agrupar_filas":
						mostrarDialogoAgruparXFilas(tablaSeleccionada, idLimConv);
						break;
					case "Ordenar_filas":
						mostrarDialogoOrdenar(tablaSeleccionada, idLimConv);
						break;
					case "Eliminar_filasfcond":
						mostrarDialogoEliminarFilas(tablaSeleccionada, idLimConv);
						break;
					default: // Acciones de Limpieza	
						mostrarDialogoColumnasLimpieza(tablaSeleccionada, idLimConv, action);
				}
				ocultar_menues();
			});
			break;
		case "delete":
			if (eliminar_Elemento( id )) {
				ocultar_menues();
				//chequeamos si al menos quedo una conexion entre dos elementos, para activar
				//el editor del nombre del modelo...
				if (connections.length > 0)
					document.getElementById("nombre_modelo").disabled = false;
				else
					document.getElementById("nombre_modelo").disabled = true;
			}
			else
				customAlert("Este objeto no se puede eliminar porque otros son dependientes.");
			break;
		case "Prediccion_futura":
		case "Clasificacion_predictiva":
			// Deshabilitar el menu de Procesamiento temporalmente
			$(".context-menu").filter(":visible").addClass("disabled-dialog");
			showContextMenu(action, e.pageX, e.pageY + 15); // Muestra el submenu correspondiente
			break;
		case "Clasif_Arbol_decision":
		case "Clasif_Random_forest":
		case "Clasif_Regresion_logistica":
		case "Clasif_SVM":
		case "Prediccion_lineal":
		case "Prediccion_nolineal":
		case "Prediccion_SeriesTemporales":
		case "Estadistica_descriptiva":
		case "Frecuencias_densidad_prob":
			// Aqui se toma el atributo "tablas" o del objeto Limpieza o
			// del obj Conersion, para de ahi seleccionar tabla y columna
			// De la columna (numerica), se elige el numero de estimaciones
			// futuras. Se realiza tal estimacion y se guardan en este obj
			// De ahi en adelante, los graficos y reportes solo ven los resultados
			// de eeste obj.
			// Obtener el ID del elemento de Procesamiento
			const idProc = id;
			// Buscar el objeto de Procesamiento en workflow_json
			const objProc = workflow_json.find(obj => obj.id === idProc);
			// obtener la lista de objetos conectados al obj Procesamiento...
			var conexiones = connections.filter(conn => conn.to === idProc);
			if (!conexiones) {
				customAlert("Debe conectar este objeto de Procesamiento a un objeto de Limpieza o Conversion.");
				ocultar_menues();
				return;
			}
			// dejamos en idObjs los id de obj conetados al Procesamiento seleccionado...
			var idObjs = conexiones.map(obj => obj.from);
			var tablasDisponibles_ = []; idTablas = [];
			for(let i = 0; i < idObjs.length; i++)
			{
				const obj = workflow_json.find(obj => obj.id === idObjs[i]);
				if (!obj || (obj.tipo !== "Limpieza" && obj.tipo !== "Conversion")) {
					customAlert("Se encontró un objeto que no es de Limpieza o Conversion.");
					ocultar_menues();
					return;
				}
				if (obj.resultado.length) {
					idTablas.push(...obj.resultado.map(o => o.idTable));
					tablasDisponibles_.push(...obj.resultado);
				} else {
					customAlert("El objeto tipo "+obj.tipo+" conectado no tiene tablas procesadas!");
					ocultar_menues();			
					return;
				}
			}
			//Filtramos para dejar solo las tablas limpiadas o con conversiones
			tablasDisponibles = tablasDisponibles_.filter(tab => idTablas.includes(tab.idTable));
			// Mostrar un diálogo para seleccionar la tabla
			if (tablasDisponibles.length === 0) {
				customAlert("No hay tablas disponibles limpiadas o con conversiones.");
				ocultar_menues();
				break;
			}
			// Crear un diálogo con las tablas procesadas en objs anteriores
			var dialogoTablas = `
					Seleccione una tabla:<br />
					<select id="selectTabla">
						${tablasDisponibles.map((tabla, index) => `
							<option value="${index}">Tabla ${index + 1} (${tabla.rows}x${tabla.cols})</option>
						`).join("")}
					</select>
					<br /><button id="btnSeleccionarTabla" class="custom-button">Seleccionar</button>
					&nbsp;&nbsp;`;
			// Mostrar el diálogo al lado del menú contextual
			crearDialogo(dialogoTablas);
			// Manejar la selección de la tabla
			$("#btnSeleccionarTabla").on("click", function () {
				const tablaIndex = $("#selectTabla").val();
				const tablaSeleccionada = tablasDisponibles[tablaIndex];
				// Mostrar un diálogo para seleccionar las columnas X y Y
				mostrarDialogoColumnasProc(tablaSeleccionada, idProc, action);
			});
			break;
		case "Generar_grafico_lineas":
		case "Generar_grafico_barras":
		case "Generar_grafico_dispersion":
		case "Generar_grafico_pastel":
		case "Generar_grafico_histograma":
			// Obtener el ID del elemento de Gráfico
			const idGrafico = id;
			// Buscar el objeto de Gráfico en workflow_json
			const objGrafico = workflow_json.find(obj => obj.id === idGrafico);
			if (!objGrafico) {
				customAlert("No se encontró el objeto de Gráfico.");
				break;
			}
			// Buscar el objeto o los objetos de Limpieza/Procesamiento conectado a Gráfico
			var conexiones = connections.filter(conn => conn.to === idGrafico);
			if (!conexiones) {
				customAlert("Debe conectar este objeto de Gráfico a un objeto de Limpieza o Procesamiento.");
				ocultar_menues();
				break;
			}
			const idLimProc = conexiones.map(obj => obj.from);
			var tablasDisponibles = [];
			var objDisponibles = [];
			for (let i = 0; i < idLimProc.length; i++) {
				const objLimProc = workflow_json.find(obj => obj.id === idLimProc[i]);
				if (!objLimProc || (objLimProc.tipo !== "Limpieza" && objLimProc.tipo !== "Conversion" && objLimProc.tipo !== "Procesamiento")) {
					customAlert("No se encontró el algún objeto de Limpieza o Procesamiento conectado.");
					continue;
				}
				tabla = objLimProc.resultado[0]; // Tomamos la unica tabla procesada, desde obj Limpieza, Conversion o Procesamiento
				tablasDisponibles.push( tabla );
				objDisponibles.push( objLimProc );
			}
			// Mostrar un diálogo para seleccionar la tabla
			if (tablasDisponibles.length === 0) {
				customAlert("No hay tablas disponibles en el objeto de Procesamiento.");
				ocultar_menues();
				break;
			}
			// Crear un diálogo con las tablas disponibles
			var dialogoTablas = `
					Seleccione una tabla:<br />
					<select id="selectTabla">
						${tablasDisponibles.map((tabla, index) => `
							<option value="${index}">Tabla ${index + 1} (${tabla.rows}x${tabla.cols})</option>
						`).join("")}
					</select>
					<br /><button id="btnSeleccionarTabla" class="custom-button">Seleccionar</button>
					&nbsp;&nbsp;`;
			// Mostrar el diálogo al lado del menú contextual
			crearDialogo(dialogoTablas);
			// Manejar la selección de la tabla
			$("#btnSeleccionarTabla").on("click", function () {
				const tablaIndex = $("#selectTabla").val();
				const tablaSeleccionada = tablasDisponibles[tablaIndex];
				// Aqui se averigua primero si el proceso, dentro de tablaSeleccionada, es de Prediccion_X 
				// Entonces se tendra en cuenta tambien la tabla de donde procede la seleccionada, para asi
				// graficar ambos pares de columnas del mismo tipo. Si no, se procede a graficar solo de las
				// columnas de la tabla seleccionada...
				objLimProc = objDisponibles[tablaIndex];
				if (objLimProc.tipo == "Procesamiento" && ["Prediccion_lineal","Prediccion_nolineal","Prediccion_SeriesTemporales"].includes(objLimProc.proceso)) {
					// Hallar obj conectado a objLimProc ...
					idx = connections.findIndex(o => o.to === objLimProc.id);
					from = connections[idx].from;
					objPrev = workflow_json.find(obj => obj.id === from);
					tablaAnterior = objPrev.resultado[0];
					mostrarDialogoColumnasGrafico2Pares(tablaAnterior, tablaSeleccionada, idGrafico, action);
				}
				else
					// Mostrar un diálogo para seleccionar solo las columnas X e Y de tablaSeleccionada
					mostrarDialogoColumnasGrafico(tablaSeleccionada, idGrafico, action);
			});
			break;
		case "Mostrar_grafico_actual":
			mostrarGraficoActual(id); // Mostrar el gráfico actual
			break;
		case "Seleccionar_mis_plantillasKPI":
			mostrarPlantillasKPI(id, "/obtener_misplantillaskpi");
		break;
		case "Seleccionar_otras_plantillasKPI":
			mostrarPlantillasKPI(id, "/obtener_otrasplantillaskpi");
		break;
		default:
			console.log("Acción no reconocida");
			customAlert("Acción no reconocida", "Error");
			ocultar_menues(); // Ocultar todos los menus despues de seleccionar una opcion
		}
	});
	let firstSelectedId = null;
	$(document).on('dblclick', '.element', function() {  
		const id = $(this).data('id');
		if (firstSelectedId === null) {  
			firstSelectedId = id;   
			$(this).css('border', '2px dashed blue');
		} else {  
			if (firstSelectedId !== id) {  
				// Comprobar si existe una conexion desde el primero al segundo elemento 
				const exists1 = connections.some(connection =>   
					connection.from === firstSelectedId && connection.to === id  
				);
				// y preguntar tambien si existe una conexion invertida, desde el segundo al primero
				const exists2 = connections.some(connection =>   
					connection.from === id && connection.to === firstSelectedId
				);
				if ( ! exists1 && ! exists2 )
					drawConnection(firstSelectedId, id, true); // Dibuja la conexion nueva al segundo elemento  
			}  
			$(`.element[data-id="${firstSelectedId}"]`).css('border', '');   
			firstSelectedId = null;   
		}  
	});  
	function redrawConnections() {
		$('#connection-svg').empty(); // Limpiar conexiones anteriores      
		connections.forEach(conn => {  
			drawConnection(conn.from, conn.to, false); // Redibujar conexiones existentes  
		});  
	}  
	$(document).on('click', '.tool',  function() {
		const lista_tiposNoTool = ["Nombre_modelo","Salvar_cambios"];
		const tooltip = $(this).find('.tooltiptext');
		tooltip.css('visibility', 'hidden').css('opacity', '0')
		const type = $(this).data('type');
		if (type == "InformeKPI" && workflow_json.find(obj => obj.tipo === type)) {
			// Si el objeto ya existe, no se agrega de nuevo
			customAlert("El objeto Informe KPI ya existe en el flujo de trabajo. Cada modelo debe tener un solo objeto Informe KPI.");
			return;
		}
		if (! lista_tiposNoTool.includes(type)) {
			addElement(type);
		}
	});	
	// Evento para mostrar el tooltip del objeto al pasar el mouse
	$(document).on('mouseenter', '.tooltip_absolute', function() {
		const tooltip = $(this).find('.tooltiptext_absolute');
		const text = tooltip.text().trim(); // Obtener el texto del tooltip
		const words = text.split(' '); // Dividir el texto en palabras
		// Ordenar las palabras por longitud (de mayor a menor)
		words.sort((a, b) => b.length - a.length);
		// Tomar las dos palabras más largas
		const longestWords = words.slice(0, 2);
		// Calcular el ancho en función de las dos palabras más largas
		const width = longestWords.reduce((acc, word) => acc + word.length, 0) * 8; // Ajusta el multiplicador según el tamaño de la fuente
		// Aplicar el ancho calculado
		tooltip.css({
			'visibility': 'visible',
			'opacity': '1',
			'width': `${width}px`, // Establecer el ancho calculado
			'white-space': 'normal', // Evitar que el texto se divida en varias líneas
			'word-wrap': 'break-word' // Asegurar que las palabras largas no se salgan del recuadro
		});
	});
	// Evento para ocultar el tooltip del objeto, al salir del elemento
	$(document).on('mouseleave', '.tooltip_absolute', function() {
		const tooltip = $(this).find('.tooltiptext_absolute');
		tooltip.css('visibility', 'hidden').css('opacity', '0');
	});
	// Evento para mostrar el tooltip del menu de herramientas, al pasar el mouse
	$(document).on('mouseenter', '.tooltip', function() {
		const tooltip = $(this).find('.tooltiptext');
		const text = tooltip.text().trim(); // Obtener el texto del tooltip
		const words = text.split(' '); // Dividir el texto en palabras
		// Ordenar las palabras por longitud (de mayor a menor)
		words.sort((a, b) => b.length - a.length);
		// Tomar las dos palabras más largas
		const longestWords = words.slice(0, 2);
		// Calcular el ancho en función de las dos palabras más largas
		const width = longestWords.reduce((acc, word) => acc + word.length, 0) * 8; // Ajusta el multiplicador según el tamaño de la fuente
		// Aplicar el ancho calculado
		tooltip.css({
			'visibility': 'visible',
			'opacity': '1',
			'width': `${width}px`, // Establecer el ancho calculado
			'white-space': 'normal', // Evitar que el texto se divida en varias líneas
			'word-wrap': 'break-word' // Asegurar que las palabras largas no se salgan del recuadro
		});
	});
	// Evento para ocultar el tooltip al salir del elemento
	$(document).on('mouseleave', '.tooltip', function() {
		const tooltip = $(this).find('.tooltiptext');
		tooltip.css('visibility', 'hidden').css('opacity', '0');
	});	
	function convertirTablas( tablas_vigentes ){
		// Verificar si el formato es 'split'
		if (tablas_vigentes.columns && tablas_vigentes.data) {
			// Convertir el formato 'split' a un objeto de tabla
			var tabla = {};
			tablas_vigentes.columns.forEach((col, i) => {
				tabla[col] = tablas_vigentes.data.map(row => row[i]);
			});
			tablas_vigentes.data = [tabla]; // Estructura esperada por el cliente
		}
		return tablas_vigentes;
	}
	 function insertar_tablasObjDocumento(objDocumento, tablas_vigentes) {
		let num_tablas = 0;
		if (! objDocumento) { // no se encontro el objeto Documento con identificador id_doc!!!
			customAlert("Elemento de Documento no hallado en la lista del flujo de trabajo!");
			return null;
		} else {
			// Actualizamos al objeto del Documento...
			tablas_vigentes = convertirTablas( tablas_vigentes );
			num_tablas = tablas_vigentes.length;
			objDocumento.tablas = Array();
			let info_tablas = ""; 
			for(let j = 0; j < num_tablas; j++) {
				var tabla = {};
				tabla.idTable = 'table-' + (performance.now() + count_t); ++count_t;
				//leyendo las propiedades (claves por columna) de la tabla j-esima ...
				claves = Object.keys(tablas_vigentes[j]);
				//rectificamos cada columna, si no es null, la an~adimos a tabla...
				tabla.table = {};
				for (let i = 0; i < claves.length; i++) {  
					if (tablas_vigentes[j][claves[i]] != null) {  
						tabla.table[claves[i]] = tablas_vigentes[j][claves[i]];
					}  
				}
				claves_no_null = Object.keys(tabla.table);
				tabla.cols = claves_no_null.length;
				if (typeof tabla.table[claves_no_null[0]] == "object")
					tabla.rows = Object.keys(tabla.table[claves_no_null[0]]).length;
				else
					tabla.rows = tabla.table[claves_no_null[0]].length;
				tabla.columnas = claves_no_null;
				objDocumento.tablas.push( tabla );
				info_tablas = info_tablas + '<li>Tabla ('+tabla.rows+'x'+tabla.cols+')</li>';
			}
			return info_tablas;
		}
	}
	// Evento para manejar la selección de archivo
	$('#inputArchivo').on('change', function(event) {
		const archivo = event.target.files[0]; // Obtiene el archivo seleccionado
		const id_icono = $(this).attr('data-id-icono'); // ID del tooltip
		const id_doc = $(this).attr('data-id-doc'); // ID del elemento Documento
		if (archivo) {
			// Crear un FormData para enviar el archivo
			const formData = new FormData();
			formData.append("documento", archivo);
			// Enviar la solicitud POST con fetch
			// Mostrar el spinner
			mostrarSpinner();
			fetch("/leer_doc_desde_pc", {
				method: "POST",
				body: formData
			})
			.then(response => {
				if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
				return response.text().then(text => {
					try {
						return JSON.parse(text);
					} catch (e) {
						console.error("Failed to parse JSON:", text.substring(0, 100) + "...");
						throw new Error("Invalid JSON response");
					}
				});
			})
			.then( data => {
				ocultarSpinner();
				const tooltip = document.getElementById(id_icono);
				tablas_vigentes = data; // Usamos directamente el JSON parseado
				objDocumento = workflow_json.find( obj => obj.id === id_doc );
				info_tablas = insertar_tablasObjDocumento(objDocumento, tablas_vigentes);
				if (info_tablas != null && tooltip) {
					// Actualizar el tooltip
					tooltip.style.whiteSpace = 'nowrap';
					str_tooltip = `<strong>${archivo.name}:</strong><ol>${info_tablas}</ol>`;
					tooltip.innerHTML = str_tooltip;
					objDocumento.tooltip = str_tooltip;
				}
				customConfirm("El documento '"+archivo.name+"' ha sido cargado correctamente!");
			})
			.catch( error => {
				console.error("Fetch error:", error);
				customAlert(`Error: ${error.message}`);
				ocultarSpinner();
			});
			// Limpiar el input para permitir nueva selección
			$(this).val('');
		}
	});
	function set_init_typeElement( type ) {
		// generar unico id
		const id = 'element-' + (performance.now() + count_e); ++count_e;
		switch (type) {
			case "Documento":
				return { 
					tipo : type,
					id : id,
					tooltip : first_tooltipDoc, // tooltip asociado
					left : 100+Math.random()*25, // ubicacion en el area grafica (left)
					top : Math.random()*200, // ubicacion en el area grafica (top)
					tablas : [] //lista de objetos de tablas, una o varias
						/*
							{
								idTable : "...",
								table : {
									"col 1" : [val11, val21, ...],
									"col 2" : [val21, val22, ...],
									:
								},
								cols : #, rows : #,
								columnas : {0:"col 1",1:"col 2", ...}
							}
						*/
				};
			case "Limpieza":
				return {
					tipo : type,
					id : id,
					tooltip : "(?)", // tooltip asociado
					left : 100+Math.random()*25, // ubicacion en el area grafica (left)
					top : Math.random()*200, // ubicacion en el area grafica (top)
					proceso : "",
					idtable : "", // debe ser un id de las tablas del obj conectado a esta
					cols : [], // columnas elegidas para el proceso
					resultado : [] // aqui se coloca una tabla resultante, cuyo idtable es nuevo
				};			
			case "Conversion":
				return { 
					tipo : type,
					id : id,
					tooltip : "(?)", // tooltip asociado
					left : 100+Math.random()*25, // ubicacion en el area grafica (left)
					top : Math.random()*200, // ubicacion en el area grafica (top)
					proceso : "",
					idtable : "", // debe ser un id de las tablas del obj conectado a esta
					cols : [], // columnas elegidas para aplicar la conversion
					resultado : [] // aqui se coloca una tabla resultante, cuyo idtable es nuevo
				};			
			case "Procesamiento":
				return { 
					tipo : type,
					id : id,
					tooltip : "(?)", // tooltip asociado
					left : 100+Math.random()*25, // ubicacion en el area grafica (left)
					top : Math.random()*200, // ubicacion en el area grafica (top)
					proceso : "", // Prediccion_Lineal, Prediccion_noLineal, Estadistica_descriptiva (media, mediana, max, min), Frecuencias
					idtable : "", // debe ser un id de las tablas del obj conectado a esta
					cols : [], // columnas a aplicar el proceso
					resultado : [] // aqui se coloca una tabla o texto resultantes, cuyo idtable es nuevo
					/*---- a partir de aqui pueden venir otros clave:valor, segun proceso
						Ejemplo: 
							proceso: "Prediccion_Lineal",
							:
							epocas: #,
							predicciones: [{x:#, y:#},{x:#, y:#},{x:#, y:#}, ...]
					*/
				};
			case "Grafico":
				return { 
					tipo : type,
					id : id,
					tooltip : "(?)", // tooltip asociado
					left : 100+Math.random()*25, // ubicacion en el area grafica (left)
					top : Math.random()*200, // ubicacion en el area grafica (top)
					type : "", // el tipo de grafico
					idtable : "", // debe ser un id de las tablas del obj conectado a esta
					cols : [], // columnas a utilizar, segun el type en la tabla referenciada
					resultado : [] // seria una especie de Blob que proviene del servidor 
				};
			case "InformeKPI":
				return { 
					tipo : type,
					id : id,
					tooltip : "(?)", // tooltip asociado
					left : 100+Math.random()*25, // ubicacion en el area grafica (left)
					top : Math.random()*200, // ubicacion en el area grafica (top)
					archivo : "", // una url con la plantilla kpi a aplicar: /templates/kpi/idUsuario_idPlantilla.docx
					asociaciones : {}, /* Diccionario de pares clave:valor, o sea:
						 CAMPO : { "tipo":<tipo>, "id":<id> }
						 	Donde CAMPO proviene del valor textual {{CAMPO}} proveniente de archivo
							donde <tipo> puede ser:
								- FECHA_HORA : Fecha y hora en que se genera el informe
								- NOM_APE_EJECUTOR : Nombre y Apellidos del que ejecuta la operacion
								- IMAGEN : Una imagen generada por el Servidor, de tipo PNG
								- TABLA : Una tabla de las resultante, se an~ade "idx":idx si hay mas de una tabla.
								- TEXTO : Un texto alfanumerico resultante
							Y <id> es el identificador del objeto de donde proviene el resultado. 
					*/
				};
			default:
			return {};
		}
	}
	const TiposDocumentos = {
		Documento: 'fa-regular fa-file',
		Limpieza: 'fa-regular fa-pen-to-square',
		Conversion: 'fa-solid fa-right-left',
		Procesamiento: 'fa-solid fa-gear',
		Grafico: 'fa-regular fa-chart-bar',
		InformeKPI: 'fa-regular fa-file-lines'
	};
	function addElement(type, obj_json = {}, conectar = true) {
		if (Object.keys(obj_json).length === 0) {
			// obtener la clase de objeto a mostrar, segun el tipo
			obj_json = set_init_typeElement( type );
		}
		const id = obj_json.id;
		if (obj_json.hasOwnProperty('tooltip'))
			var tooltip = obj_json.tooltip;
		else
			tooltip = '(?)';
		if (obj_json.hasOwnProperty('left'))
			var left = obj_json.left;
		else
			left = 100+Math.random()*25;
		if (obj_json.hasOwnProperty('top'))
			var top = obj_json.top;
		else
			top = Math.random()*200;
		workflow_json.push( obj_json )
		// Representacion grafica 
		const iconClass = TiposDocumentos[type];
		const new_element = $(`
			<div class="element tooltip_absolute" data-id="${id}" 
				style="left: ${left}px; top: ${top}px;">
			<i class="${iconClass}"></i>
			<span class="tooltiptext_absolute" id="tooltip-${id}">${tooltip}</span>
			</div>
		`);
		$('#workflow-container').append(new_element);  
		elements[id] = { type: type, connections: [] };  
		//console.log(`Elemento agregado: ${name} con ID: ${id}`);  
		new_element.draggable({  
			drag: function() {  
				redrawConnections();  
			}
		});
		if (! conectar)
			return;
		//Establecer nuevas conexiones automaticas para pares de elementos
		//que lo requieren normalmente y reubicar el nuevo elemento...
		let Element = $(`.element[data-id="${id}"]`);
		function reubicar_conectar( tipo, forward = 1 ) {
			objs = workflow_json.filter(obj => obj.tipo == tipo);
			if (objs.length) {
				// bucle para Buscar el elemento en el DOM
				objs.forEach(function(obj) {
					// Verificar que ese elemento no tenga conexion desde el hacia otro...
					idx = connections.findIndex(o => o.from === obj.id);
					if (idx == -1) {
						let fromElement = $(`.element[data-id="${obj.id}"]`);
						// Obtén la posición actual del elemento "fromElement"
						let fromLeft = parseFloat(fromElement.css('left')); // Obtiene el valor numérico de 'left'
						// Calcula la nueva posición para "Element"
						let newLeft = fromLeft + 150 + forward * 50*Math.random();
						if (newLeft <= 0)
							newLeft = 10;
						// Aplica la nueva posición a "Element"
						Element.css({
							left: newLeft + "px" // Nueva posición en el eje X
						});
						// Dibuja la conexión
						if (forward == 1)
							drawConnection(obj.id, id, true);
						else
							drawConnection(id, obj.id, true);
						return obj;
					}	
				});	
			}
			return null;
		}
		switch(type) {
			case "Documento":
				//Buscar si hay elementos Limpieza y conectar...
				reubicar_conectar( "Limpieza", forward = -1 );
				break;
			case "Limpieza":
				//Buscar si hay elementos de Documento y conectar...
				reubicar_conectar( "Documento" );
				break;
			case "Conversion":
				//Buscar si hay elementos de Limpieza y conectar...
				reubicar_conectar( "Limpieza" );
				break;
			case "Procesamiento":
				//Buscar si hay elementos de Conversion, sino el de Limpieza y conectar...
				obj = workflow_json.find(obj => obj.tipo == "Conversion");
				if (!reubicar_conectar( "Conversion" )) {
					reubicar_conectar( "Limpieza" );
				}
				break;
			case "Grafico":
				//Buscar si hay elementos de Procesamiento y conectar...
				reubicar_conectar( "Procesamiento" );
				break;
			case "InformeKPI":
				//Buscar si hay elementos de Grafico y conectar...
				reubicar_conectar( "Grafico" );
				break;
		}
		obj_json.left = parseFloat(Element.css('left'));
		obj_json.top = parseFloat(Element.css('top'));
	}
	// Variable para completar los dialogos de columnas seleccionadas con opcion
	const labelColTodas = `<label>
		<input type="checkbox" name="colTodas">
		<span class="select-all-text">Seleccionar Todas</span></label><br>`;

	function mostrarDialogoColumnasLimpieza(tabla, idLimpieza, action) {
		// Deshabilitar el menu y diálogo de selección de tablas
		$(".context-menu").filter(":visible").addClass("disabled-dialog");
		$('[id^="dialogo-"]').addClass("disabled-dialog");
		// Crear un dialogo con las columnas disponibles
		const dialogoColumnas = `
				Seleccione las columnas a combinar:<br />
				<div class="contenedor_opt_columnas">
				${labelColTodas}
				${tabla.columnas.map(columna => `
					<label>
						<input type="checkbox" name="columnas" value="${columna}">${columna}
					</label><br>
				`).join("")}
				</div>
				<button id="btnSeleccionarColumnas" class="custom-button">Seleccionar</button>
				&nbsp;&nbsp;`;
		// Mostrar el dialogo al lado del dialogo de tablas
		crearDialogo(dialogoColumnas);
		// Manejar la seleccion de las columnas
		$("#btnSeleccionarColumnas").on("click",  async function () {
			const columnasSeleccionadas = [];
			$("input[name='columnas']:checked").each(function () {
				columnasSeleccionadas.push($(this).val());
			});
			if (columnasSeleccionadas.length === 0) {
				customAlert("Debe seleccionar al menos una columna.");
				return;
			}
			// Aplicamos la accion pedida..
			switch(action) {
				case "Quitar_duplicados":
					// Aplicar la logica para eliminar duplicados
					const nueva_tabla = await aplicarQuitarDuplicados(tabla, columnasSeleccionadas);
					if (nueva_tabla != null) {
						// Fijar el proceso en el objeto de limpieza actual
						const objLimpieza = workflow_json.find(obj => obj.id === idLimpieza);
						objLimpieza.resultado = [nueva_tabla];
						objLimpieza.proceso = "Quitar_duplicados";
						objLimpieza.idtable = tabla.idTable;
						objLimpieza.cols = columnasSeleccionadas;
						// Actualizamos el tooltip correspondiente
						const tooltip = document.getElementById(`tooltip-${idLimpieza}`);
						if (tooltip) { // Actualiza el tooltip con los procesos de limpieza
							tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias lineas
							//formar la lista de columnas...
							str_cols = "";
							for(let i = 0; i < columnasSeleccionadas.length; i++)
								str_cols = str_cols + "<li>"+columnasSeleccionadas[i]+"</li>";
							str_tooltip = `<strong>Quitar duplicados:</strong><br />Tabla (${nueva_tabla.rows}x${nueva_tabla.cols}), columnas:<br/><ol>${str_cols}</ol>`;
							tooltip.innerHTML = str_tooltip;
							objLimpieza.tooltip = str_tooltip;
						}
					}
					break;
				case "Recuperar_num_corruptos":
					// Aplicar la logica para recuperar numeros corruptos
					const tabla_recup = await aplicarRecuperarNumCorruptos(tabla, columnasSeleccionadas);
					if (tabla_recup != null) {
						// Fijar el proceso en el objeto de limpieza actual
						const objLimpieza = workflow_json.find(obj => obj.id === idLimpieza);
						objLimpieza.resultado = [tabla_recup];
						objLimpieza.proceso = "Recuperar_num_corruptos";
						objLimpieza.idtable = tabla.idTable;
						objLimpieza.cols = columnasSeleccionadas;
						// Actualizamos el tooltip correspondiente
						const tooltip = document.getElementById(`tooltip-${idLimpieza}`);
						if (tooltip) { // Actualiza el tooltip con los procesos de limpieza
							tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias lineas
							//formar la lista de columnas...
							str_cols = "";
							for(let i = 0; i < columnasSeleccionadas.length; i++)
								str_cols = str_cols + "<li>"+columnasSeleccionadas[i]+"</li>";
							str_tooltip = `<strong>Recuperar n&uacute;meros corruptos:</strong><br/>Tabla (${tabla_recup.rows}x${tabla_recup.cols}), columnas:<br/><ol>${str_cols}</ol>`;
							tooltip.innerHTML = str_tooltip;
							objLimpieza.tooltip = str_tooltip;
						}
					}
					break;
				default:
					break;
			};
			CerrarDialogosYMenues();
		});
	}
	// Exportacion a un Documento Excel de la tabla seleccionada
	function exportarTablaExistente(tabla, idLimConv) {
		// Cargar la biblioteca SheetJS (xlsx) dinámicamente si no está disponible
		if (typeof XLSX === 'undefined') {
			const script = document.createElement('script');
			script.src = 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
			script.onload = () => generarExcel(tabla, idLimConv);
			document.head.appendChild(script);
		} else {
			generarExcel(tabla, idLimConv);
		}
	}
	// funcion auxiliar generarExcel para la exportacion...
	function generarExcel(tabla, idLimConv) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table) {
			customAlert("Tabla o columnas no válidas.", "Error");
			return null;
		}
		// Obtener las columnas (keys del objeto)
		const columnas = Object.keys(tabla.table);
		// Verificar si hay datos
		if (columnas.length === 0) {
			customAlert('La tabla no contiene datos para exportar', 'Error');
			return null;
		}
		// Obtener el número de filas (usamos la primera columna como referencia)
		const numFilas = tabla.table[columnas[0]].length;
		// Crear array de arrays con los datos
		const datos = [];
		// Añadir encabezados (primera fila)
		datos.push(columnas);
		// Añadir filas de datos
		for (let i = 0; i < numFilas; i++) {
			const fila = [];
			for (const columna of columnas) {
				fila.push(tabla.table[columna][i] || '0');
			}
			datos.push(fila);
		}
		// Crear hoja de cálculo
		const ws = XLSX.utils.aoa_to_sheet(datos);
		// Crear libro de trabajo
		const wb = XLSX.utils.book_new();
		XLSX.utils.book_append_sheet(wb, ws, "Datos");
		// Generar nombre de archivo por defecto
		const nombreDefault = `tabla_exportada_${idLimConv || new Date().toISOString().slice(0,10)}.xlsx`;
		// Crear diálogo personalizado para nombre y ubicación
		const dialogHTML = `
			<div id="exportDialog" class="dialog-overlay active">
				<div class="dialog-box" style="max-width: 500px;background-color: #90e0ef;">
					<div class="dialog-header" style="background: #0077b6;">Exportar a Excel</div>
					<div class="dialog-body">
						<div style="margin-bottom: 15px;">
							<label for="fileNameInput" style="display: block; margin-bottom: 5px;">Nombre del archivo:</label>
							<input type="text" id="fileNameInput" value="${nombreDefault}" style="width: 100%; padding: 8px; box-sizing: border-box;">
						</div>
						<div style="margin-bottom: 15px;">
							<button id="browseButton" class="dialog-btn dialog-btn-secondary" style="width: 100%;">Seleccionar carpeta destino</button>
							<div id="selectedFolder" style="margin-top: 5px; font-size: 0.9em; color: #666;"></div>
						</div>
					</div>
					<div class="dialog-footer" style="background: #00b4d8;">
						<button onclick="cancelExport()" class="dialog-btn dialog-btn-secondary">Cancelar</button>
						<button onclick="confirmExport()" class="dialog-btn dialog-btn-primary">Exportar</button>
					</div>
				</div>
			</div>
		`;
		// Añadir diálogo al DOM
		const dialogContainer = document.createElement('div');
		dialogContainer.innerHTML = dialogHTML;
		document.body.appendChild(dialogContainer);
		// Configurar el File System Access API (si está disponible)
		let fileHandle = null;
		document.getElementById('browseButton').addEventListener('click', async () => {
			try {
				// Intentar usar la API moderna
				fileHandle = await window.showDirectoryPicker();
				const folderName = fileHandle.name;
				document.getElementById('selectedFolder').innerHTML = `Carpeta seleccionada: <strong>${folderName}</strong>`;
			} catch (error) {
				// Fallback para navegadores que no soportan la API
				//console.log('File System Access API no soportada:', error);
				//document.getElementById('selectedFolder').textContent = 
				//	"Nota: Su navegador no soporta selección directa de carpeta. El archivo se descargará en la carpeta predeterminada.";
				confirmExport();
			}
		});
		// Funciones globales para los botones del diálogo
		window.cancelExport = function() {
			document.body.removeChild(dialogContainer);
			CerrarDialogosYMenues();
		};
		window.confirmExport = async function() {
			const fileName = document.getElementById('fileNameInput').value.trim() || nombreDefault;
			const finalFileName = fileName.endsWith('.xlsx') ? fileName : `${fileName}.xlsx`;	
			try {
				if (fileHandle) {
					// Guardar usando File System Access API
					const file = await fileHandle.getFileHandle(finalFileName, { create: true });
					const writable = await file.createWritable();
					const blob = new Blob([XLSX.write(wb, { bookType: 'xlsx', type: 'array' })]);
					await writable.write(blob);
					await writable.close();
				} else {
					// Fallback: Descarga tradicional
					XLSX.writeFile(wb, finalFileName);
				}
				customConfirm(`Tabla exportada correctamente como ${finalFileName}`, 'Éxito');
			} catch (error) {
				console.error('Error al exportar:', error);
				customAlert(`Error al exportar: ${error.message}`, 'Error');
			}
			document.body.removeChild(dialogContainer);
			CerrarDialogosYMenues();
		};
	}
	// Para validar si una columna es de categorias y si cada una se
	// repite mas de x veces
	function sonCategoriasValidas(lista, x) {
		// Si la lista no es de categorías, retornar false
		if (!lista.every(elemento => isNaN(Number(elemento)))) {
			return {"EsCategoria" : false, "Mensaje": "La columna no es de categorías, sino numérica."};
		}
		// Contar las repeticiones de cada categoría
		const conteo = {};
		for (const elemento of lista) {
			if (conteo[elemento]) {
				conteo[elemento]++;
			} else {
				conteo[elemento] = 1;
			}
		}
		// Verificar que todas las categorías se repitan más de x veces
		resp = Object.values(conteo).every(repeticiones => repeticiones > x);
		if (resp)
			return {"EsCategoria" : true, "Mensaje": "La columna es de categorías con "+x+" o más repeticiones por clase."};
		else
			return {"EsCategoria" : false, "Mensaje": "La columna es de categorías, pero algunas clases no tienen más de "+x+" repeticiones."};
	}
	function mostrarDialogoColumnasProc(tabla, idProc, action) {
		// Deshabilitar el menu y diálogo de selección de tablas
		$(".context-menu").filter(":visible").addClass("disabled-dialog");
		$('[id^="dialogo-"]').addClass("disabled-dialog");
		switch(action) {
			case "Prediccion_lineal":
			case "Prediccion_nolineal":
			case "Prediccion_SeriesTemporales":
				// Crear un diálogo con las columnas disponibles
				var dialogoColumnas = `
						Seleccione las columnas X y Y:
						<table width="100%">
							<tr>
								<td><strong>X</strong></td>
								<td><strong>Y</strong></td>
							</tr>
							<tr>
								<td><select id="colX">
								${tabla.columnas.map(columna => `
									<option value="${columna}">${columna}</options>
								`).join("")}
								</select></td>
								<td><select id="colY">
								${tabla.columnas.map(columna => `
									<option value="${columna}">${columna}</options>
								`).join("")}
								</select></td>
							</tr>
						</table>
						N&uacute;mero de &Eacute;pocas: <input type="text" size="5" id="epocas" onkeypress="return valideKeyOnlyNumbers(event)"/>
						<br /><button id="btnSeleccionarColumnas" class="custom-button">Seleccionar</button>
						&nbsp;&nbsp;`;
				// Mostrar el diálogo al lado del diálogo de tablas
				crearDialogo(dialogoColumnas);
				// Manejar la selección de las columnas
				$("#btnSeleccionarColumnas").on("click",  async function () {
					const epocas = Number.parseInt($("#epocas").val()); // # epocas de string a entero
					const colX = $("#colX").val();
					const colY = $("#colY").val();
					if (colX === colY) {
						customAlert("Debe seleccionar diferentes columnas X y Y.");
						return;
					}
					if (epocas === NaN) {
						customAlert("Debe introducir un numero positivo de epocas para predecir.");
						return;
					}
					const columnasSeleccionadas = [colX, colY];
					// Aplicar la logica para estimacion futura
					const nueva_tabla = await aplicarEstimacion_futura(tabla, columnasSeleccionadas, epocas, action);
					if (nueva_tabla != null) {
						// Fijar el proceso en el objeto de Procesamiento actual
						const objProc = workflow_json.find(obj => obj.id === idProc);
						objProc.proceso = action;
						objProc.idtable = tabla.idTable,
						objProc.cols = columnasSeleccionadas;
						objProc.resultado = [nueva_tabla]; // Aqui vienen solo las columnas elegidas con las predicciones
						objProc.epocas = epocas;
						// Actualizamos el tooltip correspondiente
						const tooltip = document.getElementById(`tooltip-${idProc}`);
						if (tooltip) { // Actualiza el tooltip con los Procesamientos
							tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias líneas
							str_tooltip = `<strong>${Procesos_Label2Descrip[action]}:</strong><br />Tabla (${tabla.rows}x${tabla.cols}) --> ${epocas} &eacute;pocas<br/>columnas: [${columnasSeleccionadas[0]},${columnasSeleccionadas[1]}]`;
							tooltip.innerHTML = str_tooltip;
							objProc.tooltip = str_tooltip;
						}
					}
					CerrarDialogosYMenues();
				});
				break;
			case "Estadistica_descriptiva":
				// Crear un dialogo con las columnas disponibles
				var dialogoColumnas = `
						Seleccione las columnas de referencia:<br />
						<div class="contenedor_opt_columnas">
						${labelColTodas}
						${tabla.columnas.map(columna => `
							<label>
								<input type="checkbox" name="columnas" value="${columna}"> ${columna}
							</label><br>
						`).join("")}
						</div>
						<button id="btnSeleccionarColumnas" class="custom-button">Seleccionar</button>
						&nbsp;&nbsp;`;
				// Mostrar el dialogo al lado del diaogo de tablas
				crearDialogo(dialogoColumnas);
				// Manejar la seleccion de las columnas
				$("#btnSeleccionarColumnas").on("click",  async function () {
					const columnasSeleccionadas = [];
					$("input[name='columnas']:checked").each(function () {
						columnasSeleccionadas.push($(this).val());
					});
					if (columnasSeleccionadas.length === 0) {
						customAlert("Debe seleccionar al menos una columna.");
						return;
					}
					// Verificar que todas las columnas seleccionadas sean de tipo numericas...
					for(let i = 0; i < columnasSeleccionadas.length; i++) {
						lista = tabla.table[columnasSeleccionadas[i]];
						if (!lista.every(elemento => isNaN(Number(elemento))))
							continue;
						else {
							customAlert(`Ha seleccionado la columna ${columnasSeleccionadas[i]}, la cual no es numérica!`)
							return;
						}
					}
					// Aplicamos la accion pedida..
					switch(action) {
						case "Estadistica_descriptiva":
							// Aplicar la logica para estadistica descriptiva
							const nueva_tabla = await aplicarEstadisticaDescriptiva(tabla, columnasSeleccionadas);
							if (nueva_tabla != null) {
								// Fijar el proceso en el objeto de Procesamiento actual
								const objProc = workflow_json.find(obj => obj.id === idProc);
								objProc.resultado = [nueva_tabla];
								objProc.proceso = action;
								objProc.idtable = tabla.idTable;
								objProc.cols = columnasSeleccionadas;
								// Actualizamos el tooltip correspondiente
								const tooltip = document.getElementById(`tooltip-${idProc}`);
								if (tooltip) { // Actualiza el tooltip con los procesos de limpieza
									tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias l�neas
									//formar la lista de columnas...
									str_cols = "";
									for(let i = 0; i < columnasSeleccionadas.length; i++)
										str_cols = str_cols + "<li>"+columnasSeleccionadas[i]+"</li>";
									str_tooltip = `<strong>Estad&iacute;stica descriptiva:</strong><br />Tabla (${nueva_tabla.rows}x${nueva_tabla.cols}), columnas:<br/><ol>${str_cols}</ol>`;
									tooltip.innerHTML = str_tooltip;
									objProc.tooltip = str_tooltip;
								}
							}
							break;
						case "Frecuencias_densidad_prob":
							break;
						default:
							break;
					};
					CerrarDialogosYMenues();
				});
				break;
			case "Clasif_Arbol_decision":
			case "Clasif_Random_forest":
			case "Clasif_Regresion_logistica":
			case "Clasif_SVM":
				// Crear un dialogo con las columnas a elegir y
				// la columna para predecir sus valores, que deben
				// ser categoricos
				var dialogoColumnas = `
						Seleccione las columnas para inferir a la de clasificaci&oacute;n:<br />
						<table width="100%">
							<tr>
								<td align="left"><strong>Columnas para inferir:</strong></td>
								<td align="left"><strong>Columna a clasificar:</strong></td>
							</tr>
							<tr>
								<td><div class="contenedor_opt_columnas">
								${labelColTodas}
								${tabla.columnas.map(columna => `
									<label>
										<input type="checkbox" name="columnas" value="${columna}"> ${columna}
									</label><br>
								`).join("")}
								</div></td>
								<td valign="top"><select id="colClasif">
								${tabla.columnas.map(columna => `
									<option value="${columna}">${columna}</options>
								`).join("")}
								</select>
								</td>
							</tr>
						</table>
						<button id="btnSeleccionarColumnas" class="custom-button">Seleccionar</button>
						&nbsp;&nbsp;`;
				// Mostrar el diálogo al lado del diálogo de tablas
				crearDialogo(dialogoColumnas);
				// Manejar la selección de las columnas
				$("#btnSeleccionarColumnas").on("click",  async function () {
					const colClasif = $("#colClasif").val();
					resp = sonCategoriasValidas(tabla.table[colClasif], 2);
					if (!resp.EsCategoria) {
						customAlert(resp.Mensaje);
						return;
					}
					var columnasSeleccionadas = [];
					$("input[name='columnas']:checked").each(function () {
						columnasSeleccionadas.push($(this).val());
					});
					if (columnasSeleccionadas.length < 2) {
						customAlert("Debe seleccionar al menos dos o más columnas de referencia.");
						return;
					}
					// Aplicar la logica para la clasificacion predictiva
					const resultados = await aplicarClasificacion_predictiva(tabla, columnasSeleccionadas, colClasif, action);
					if (resultados != null) {
						// Fijar el proceso en el objeto de Procesamiento actual
						const objProc = workflow_json.find(obj => obj.id === idProc);
						objProc.proceso = action;
						objProc.idtable = tabla.idTable,
						objProc.cols = columnasSeleccionadas;
						// Aqui viene una lista de 4 tablas:
						//	1. Matriz de confusion
						//	2. Metricas (recall, f1-score, precision, ...)
						//	3. Reporte de clasificacion por clase
						//	4. Metricas globales
						objProc.resultado = resultados;
						objProc.colClasif = colClasif;
						// Actualizamos el tooltip correspondiente
						const tooltip = document.getElementById(`tooltip-${idProc}`);
						if (tooltip) { // Actualiza el tooltip con los Procesamientos
							tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias líneas
							//formar la lista de columnas...
							str_cols = "";
							for(let i = 0; i < columnasSeleccionadas.length; i++)
								str_cols = str_cols + "<li>"+columnasSeleccionadas[i]+"</li>";
							str_tooltip = `
								<strong>${Procesos_Label2Descrip[action]}:</strong>
								<ul style="list-style-type: circle;">${str_cols}</ul>
								Predecir --> ${colClasif}`;
							tooltip.innerHTML = str_tooltip;
							objProc.tooltip = str_tooltip;
						}
					}
					CerrarDialogosYMenues();
				});
				break;
		}
	}
	 async function aplicarQuitarDuplicados(tabla, columnas) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table || !columnas || columnas.length === 0) {
			customAlert("Tabla o columnas no válidas.");
			return null;
		}
		// Generar una nueva copia a la tabla que se va a poder modificar...
		nueva_tabla = JSON.parse(JSON.stringify(tabla));
		// le damos un nuevo id...
		nueva_tabla.idTable = 'table-' + (performance.now() + count_t); ++count_t;
		// Llamada al servidor para realizar la operacion...
		const datos = {
			tabla: nueva_tabla.table,
			columnasReferencia: columnas
		};
		try {
			mostrarSpinner();
			const response = await fetch('/tabla_operaciones/Quitar_duplicados', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				throw new Error('Error en la respuesta del servidor');
			}
			const data = await response.json();
			// Actualizar la tabla original con los datos sin duplicados
			nueva_tabla.table = data.tablaModificada; //convertirTablas(JSON.parse(data.tablaModificada)).data[0];
			nueva_tabla.rows = nueva_tabla.table[columnas[0]].length;
			// Mostrar un mensaje de confirmación con el número de filas conservadas
			ocultarSpinner();
			customConfirm(`Se eliminaron duplicados por medio de combinar las columnas: [${columnas.join(", ")}]. Filas conservadas: ${nueva_tabla.rows}`);
			return nueva_tabla;
		} catch (error) {
			ocultarSpinner();
			customAlert(error);
			return null;
		}
	}
	 async function aplicarClasificacion_predictiva(tabla, columnasSeleccionadas, 
		colClasif, action) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table || !columnasSeleccionadas || columnasSeleccionadas.length === 0) {
			customAlert("Tabla o columnas no válidas.");
			return null;
		}
		// Llamada al servidor para realizar la operacion...
		const datos = {
			tabla: nueva_tabla.table,
			columnasReferencia: columnasSeleccionadas,
			colClasif: colClasif,
			proceso: action
		};
		try {
			// Mostrar el spinner
			mostrarSpinner();
			// Enviar solicitud al servidor para ejecutar modelo de clasificacion
			const response = await fetch('/tabla_operaciones/Clasificacion_predictiva', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				throw new Error('Error en la respuesta del servidor');
			}
			const data = await response.json();
			// en data viene una estructura, que son las cuatro tablas de evaluacion
			// o dos claves, donde el primero es mensaje, relativo a errores
			res = data["resultados"]
			customConfirm(data["mensaje"]);
			if (Object.keys(res).includes("matriz_confusion")) {
				resultados = [];
				// Iterar sobre las claves y valores de data...
				Object.entries(res).forEach(([clave, table]) => {
					tabla = new Object();
					tabla.table = table;
					tabla.idTable = 'table-' + (performance.now() + count_t); ++count_t;
					keys_table = Object.keys(table);
					tabla.cols = keys_table.length;
					tabla.rows = table[keys_table[0]].length;
					tabla.columnas = keys_table;
					// En estas tablas se agrega un atributo extra> la descripcion o nombre de la tabla
					tabla.descrip = clave;
					resultados.push( tabla );
				});
				return resultados;
			}
			else
				return null;
		} catch (error) {
			customAlert(error);
			return null;
		} finally {
			// Ocultar el spinner
			ocultarSpinner();
		}
	}
	 async function aplicarEstadisticaDescriptiva(tabla, columnas) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table || !columnas || columnas.length === 0) {
			customAlert("Tabla o columnas no válidas.");
			return null;
		}
		// Generar una nueva copia a la tabla que se va a poder modificar...
		nueva_tabla = JSON.parse(JSON.stringify(tabla));
		// le damos un nuevo id...
		nueva_tabla.idTable = 'table-' + (performance.now() + count_t); ++count_t;
		// Llamada al servidor para realizar la operacion...
		const datos = {
			tabla: nueva_tabla.table,
			columnasReferencia: columnas
		};
		try {
			mostrarSpinner();
			const response = await fetch('/tabla_operaciones/Estadistica_descriptiva', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				throw new Error('Error en la respuesta del servidor');
			}
			const data = await response.json();
			// Actualizar la tabla original con los datos sin duplicados
			nueva_tabla.table = data.tablaModificada;
			nueva_tabla.rows = nueva_tabla.table[columnas[0]].length;
			nueva_tabla.cols = Object.keys(nueva_tabla.table).length;
			// Mostrar un mensaje de confirmación con el número de filas conservadas
			ocultarSpinner();
			customConfirm(`Se calculó la Estadística descriptiva en las columnas: [${columnas.join(", ")}]`);
			return nueva_tabla;
		} catch (error) {
			ocultarSpinner();
			customAlert(error);
			return null;
		}
	}
	 async function aplicarRecuperarNumCorruptos(tabla, columnas) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table || !columnas || columnas.length === 0) {
			customAlert("Tabla o columnas no válidas.");
			return null;
		}
		// Generar una nueva copia a la tabla que se va a poder modificar...
		nueva_tabla = JSON.parse(JSON.stringify(tabla));
		// le damos un nuevo id...
		nueva_tabla.idTable = 'table-' + (performance.now() + count_t); ++count_t;
		// Llamada al servidor para realizar la operacion...
		const datos = {
			tabla: nueva_tabla.table,
			columnasReferencia: columnas
		};
		try {
			mostrarSpinner();
			const response = await fetch('/tabla_operaciones/Recuperar_num_corruptos', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				throw new Error('Error en la respuesta del servidor');
			}
			const data = await response.json();
			// Actualizar la tabla original con los datos sin duplicados
			nueva_tabla.table = data.tablaModificada; //convertirTablas(JSON.parse(data.tablaModificada)).data[0];
			nueva_tabla.rows = nueva_tabla.table[columnas[0]].length;
			ocultarSpinner();
			// Mostrar un mensaje de confirmación con el número de filas conservadas
			customConfirm(`Se recuperaron los números corruptos en las columnas: [${columnas.join(", ")}]`);
			return nueva_tabla;
		} catch (error) {
			ocultarSpinner();
			customAlert(error);
			return null;
		}
	}
	 async function aplicarEstimacion_futura(tabla, columnas, epocas, action) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table || !columnas || columnas.length !== 2) {
			customAlert("Tabla o columnas no válidas.");
			return false;
		}
		// Generar una nueva copia a la tabla que se va a poder modificar...
		nueva_tabla = JSON.parse(JSON.stringify(tabla));
		// le damos un nuevo id...
		nueva_tabla.idTable = 'table-' + (performance.now() + count_t); ++count_t;		
		// Función para eliminar columnas no deseadas
		function eliminarColumnas(tabla, columnas_a_conservar) {
			// Obtener todas las claves (nombres de columnas) del objeto
			const columnas = Object.keys(tabla);
			// Iterar sobre las columnas
			columnas.forEach(columna => {
				// Si la columna no está en la lista de columnas a conservar, eliminarla
				if (!columnas_a_conservar.includes(columna)) {
					delete tabla[columna]; // Eliminar la columna del objeto
				}
			});
			return tabla; // Devolver la tabla modificada
		}
		nueva_tabla.table = eliminarColumnas(nueva_tabla.table, columnas);
		nueva_tabla.cols = columnas.length;
		nueva_tabla.columnas = columnas;
		// Llamada al servidor para realizar la operacion...
		const datos = {
			tabla: nueva_tabla.table,
			columnasXY: columnas, // colX = columnas[0] colY = columnas[1]
			epocas: epocas,
			proceso: action
		};
		try {
			mostrarSpinner();
			const response = await fetch('/tabla_operaciones/Prediccion', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				throw new Error('Error en la respuesta del servidor');
			}
			let data = await response.json();
			// Actualizar la nueva tabla que tiene los datos historicos
			// mas las predicciones
			predicciones = data.predicciones;
			nueva_tabla.table = predicciones;
			columnas = Object.keys(predicciones);
			nueva_tabla.rows = predicciones[columnas[0]].length;
			// Mostrar un mensaje de confirmación con el número de filas conservadas
			ocultarSpinner();
			customConfirm(`Se realizaron ${epocas} predicciones futuras`);
			return nueva_tabla;
		} catch (error) {
			ocultarSpinner();
			customAlert(error);
			return null;
		}
	}
	// Función para mostrar el diálogo de selección de columnas para el gráfico de dispersión
	function mostrarDialogoColumnasGrafico(tabla, idGrafico, action) {
		// Deshabilitar el menú y diálogo de selección de tablas
		$(".context-menu").filter(":visible").addClass("disabled-dialog");
		$('[id^="dialogo-"]').addClass("disabled-dialog");
		// Crear un diálogo con las columnas disponibles
		const dialogoColumnas = `
				Seleccione las columnas X e Y:
				<table width="100%">
					<tr>
						<td><strong>X</strong></td>
						<td><strong>Y</strong></td>
					</tr>
					<tr>
						<td><select id="colX">
						${tabla.columnas.map(columna => `
							<option value="${columna}">${columna}</option>
						`).join("")}
						</select></td>
						<td><select id="colY">
						${tabla.columnas.map(columna => `
							<option value="${columna}">${columna}</option>
						`).join("")}
						</select></td>
					</tr>
				</table>
				<br /><button id="btnSeleccionarColumnas" class="custom-button">Seleccionar</button>
				&nbsp;&nbsp;`;
		// Mostrar el diálogo al lado del diálogo de tablas
		crearDialogo(dialogoColumnas);
		// Manejar la selección de las columnas
		$("#btnSeleccionarColumnas").on("click",  async function () {
			const colX = $("#colX").val();
			const colY = $("#colY").val();
			if (colX === colY) {
				customAlert("Debe seleccionar diferentes columnas X y Y.");
				return;
			}
			// Aplicar la lógica para generar el gráfico de dispersión
			const resp_gen = await generarGraficoX(tabla, colX, colY, idGrafico, action);
			if (resp_gen != null) {
				// Fijar el proceso en el objeto de Gráfico actual
				const objGrafico = workflow_json.find(obj => obj.id === idGrafico);
				objGrafico.idtable = tabla.idTable; // Esta tabla proviene del obj conectado al Grafico
				objGrafico.cols = [colX, colY];
				objGrafico.type = action;
				objGrafico.resultado = [resp_gen]; //colocar el id del contenedor del grafico generado
				// Actualizamos el tooltip correspondiente
				const tooltip = document.getElementById(`tooltip-${idGrafico}`);
				if (tooltip) {
					tooltip.style.whiteSpace = 'nowrap';
					str_tooltip = `
						<strong>${Procesos_Label2Descrip[action]}:</strong><br />
						Tabla (${tabla.rows}x${tabla.cols})
						<br/>columnas: [${colX}, ${colY}]`;
					tooltip.innerHTML = str_tooltip;
					objGrafico.tooltip = str_tooltip;
				}
			}
			CerrarDialogosYMenues();
		});
	}
	lista_tipoGraf = {
		"Generar_grafico_lineas" : "Gr&aacute;fico de l&iacute;neas",
		"Generar_grafico_barras" : "Gr&aacute;fico de barras",
		"Generar_grafico_dispersion" : "Gr&aacute;fico de dispersi&oacute;n",
		//"Generar_grafico_pastel" : "Gr&aacute;fico de pastel",
		//"Generar_grafico_histograma" : "Gr&aacute;fico de histograma"
	};
	function mostrarDialogoColumnasGrafico2Pares(tabla1, tabla2, idGrafico, action) {
		// Deshabilitar el menú y diálogo de selección de tablas
		$(".context-menu").filter(":visible").addClass("disabled-dialog");
		$('[id^="dialogo-"]').addClass("disabled-dialog");
		// Crear un diálogo con dos pares de columnas disponibles
		const dialogoColumnas = `
				Seleccione las columnas X e Y:
				<table width="100%">
					<tr>
						<td><strong>X</strong></td>
						<td><strong>Y</strong></td>
					</tr>
					<tr>
						<td><select id="colX">
						${tabla2.columnas.map(columna => `
							<option value="${columna}">${columna}</option>
						`).join("")}
						</select></td>
						<td><select id="colY">
						${tabla2.columnas.map(columna => `
							<option value="${columna}">${columna}</option>
						`).join("")}
						</select></td>
					</tr>
					<tr>
						<td><strong>Tipo Gráfico XY original</strong></td>
						<td><strong>Tipo Gráfico XY regresión</strong></td>
					</tr>
					<tr>
						<td><select id="tipoGraftabla1">
							${Object.entries(lista_tipoGraf).map(([key, value]) => `
								<option value="${key}">${value}</option>
							`).join("")}
						</select></td>
						<td><select id="tipoGraftabla2">
							${Object.entries(lista_tipoGraf).map(([key, value]) => `
								<option value="${key}" ${key === action ? 'selected' : ''}>${value}</option>
							`).join("")}
						</select></td>
					</tr>
				</table>
				<br /><button id="btnSeleccionarColumnas" class="custom-button">Seleccionar</button>
				&nbsp;&nbsp;`;
		// Mostrar el diálogo al lado del diálogo de tablas
		crearDialogo(dialogoColumnas);
		// Manejar la selección de las columnas
		$("#btnSeleccionarColumnas").on("click",  async function () {
			const colX = $("#colX").val();
			const colY = $("#colY").val();
			if (colX === colY) {
				customAlert("Debe seleccionar diferentes columnas X y Y.");
				return;
			};
			const tipoGtabla1 = $("#tipoGraftabla1").val();
			const tipoGtabla2 = $("#tipoGraftabla2").val();
			// Aplicar la lógica para generar el gráfico deseado en dos pares XY
			const resp_gen = await generarGraficoX2Pares(tabla1, tabla2, colX, colY, tipoGtabla1, tipoGtabla2, idGrafico);
			if (resp_gen != null) {
				// Fijar el proceso en el objeto de Gráfico actual
				const objGrafico = workflow_json.find(obj => obj.id === idGrafico);
				objGrafico.idtable = tabla2.idTable; // Esta tabla proviene del obj conectado al Grafico
				objGrafico.cols = [colX, colY];
				objGrafico.type = tipoGtabla2;
				objGrafico.resultado = [resp_gen]; //colocar el id del contenedor del grafico generado
				objGrafico.type_prev = tipoGtabla1;
				objGrafico.idtable_prev = tabla1.idTable;
				// Actualizamos el tooltip correspondiente
				const tooltip = document.getElementById(`tooltip-${idGrafico}`);
				if (tooltip) {
					tooltip.style.whiteSpace = 'nowrap';
					str_tooltip = `
						<strong>${Procesos_Label2Descrip[action]}:</strong><br />
						Tabla (${tabla.rows}x${tabla.cols})
						<br/>columnas: [${colX}, ${colY}]`;
					tooltip.innerHTML = str_tooltip;
					objGrafico.tooltip = str_tooltip;
				}
			}
			CerrarDialogosYMenues();
		});
	}
	 async function generarGraficoX(tabla, colX, colY, idGrafico, action) {
		// Verificar que la tabla y las columnas existan
		if (!tabla || !tabla.table || !colX || !colY) {
			customAlert("Tabla o columnas no válidas.");
			return null;
		}
		// Obtener los datos de las columnas X e Y
		const datosX = tabla.table[colX];
		const datosY = tabla.table[colY];
		// Verificar que los datos de X e Y tengan la misma longitud
		if (datosX.length !== datosY.length) {
			customAlert("Las columnas X e Y deben tener la misma longitud.");
			return null;
		}
		// Verificar si existen valores vacios o nulos
		if (datosX.some(valor => valor === null || valor === "") || datosY.some(valor => valor === null || valor === "")) {
			customAlert("Los datos de X o Y contienen valores vacíos o nulos.");
			return null;
		}
		// Obtener la posición del objeto Gráfico en el flujo de trabajo
		const elementoGrafico = document.querySelector(`.element[data-id="${idGrafico}"]`);
		if (!elementoGrafico) {
			customAlert("No se encontró el objeto Gráfico en el flujo de trabajo.");
			return null;
		}
		// Obtener las coordenadas del objeto Gráfico
		const rect = elementoGrafico.getBoundingClientRect();
		const offsetX = rect.left + window.scrollX; // Posición X absoluta
		const offsetY = rect.bottom + window.scrollY + 10; // Posición Y absoluta + margen
		// Verificar si ya existe un contenedor para este gráfico y eliminarlo
		const contenedorExistente = document.getElementById('grafico-' + idGrafico);
		if (contenedorExistente) {
			contenedorExistente.remove(); // Eliminar el contenedor anterior
		}
		// Crear el nuevo contenedor después de un pequeño retardo
		const graficoContainer = document.createElement('div');
		graficoContainer.id = 'grafico-' + idGrafico;
		graficoContainer.style.position = 'absolute'; // Posicionamiento absoluto
		graficoContainer.style.left = `${offsetX}px`; // Posición X
		graficoContainer.style.top = `${offsetY}px`; // Posición Y
		graficoContainer.style.width = '400px'; // Ancho del gráfico
		graficoContainer.style.height = '300px'; // Alto del gráfico
		graficoContainer.style.zIndex = '1000'; // Asegurar que esté por encima de otros elementos
		graficoContainer.style.backgroundColor = 'white'; // Fondo blanco
		graficoContainer.style.border = '1px solid #ccc'; // Borde para mejor visibilidad
		graficoContainer.style.boxShadow = '2px 2px 5px rgba(0, 0, 0, 0.2)'; // Sombra para destacar
		graficoContainer.style.resize = 'both'; // Permite redimensionar en ambas direcciones (horizontal y vertical)
		graficoContainer.style.overflow = 'hidden'; // Para que el contenido no se desborde 
		graficoContainer.style.minHeight = '100px';
		graficoContainer.style.minWidth = '100px';
		graficoContainer.style.display = 'none'; // Mostrar el gráfico inicialmente
		graficoContainer.style.cursor = 'grab'; // Cursor indicando que es arrastrable
		// Variables para controlar el arrastre
		let isDragging = false;
		let startX, startY, initialLeft, initialTop;
		// Evento para iniciar el arrastre (mousedown)
		graficoContainer.addEventListener('mousedown', (e) => {
			const rect = graficoContainer.getBoundingClientRect();
			const resizeHandleSize = 20;
			const isNearResizeCorner = 
				e.clientX > rect.right - resizeHandleSize && 
				e.clientY > rect.bottom - resizeHandleSize;
			// Si está en la esquina de redimensionamiento, ignorar arrastre
			if (isNearResizeCorner) {
				isDragging = false;
				return;
			}
			// Solo arrastrar si el clic es en el contenedor (no en la imagen)
			if (e.target === graficoContainer) {
				isDragging = true;
				startX = e.clientX;
				startY = e.clientY;
				initialLeft = parseInt(graficoContainer.style.left, 10);
				initialTop = parseInt(graficoContainer.style.top, 10);
				graficoContainer.style.cursor = 'grabbing';
				e.preventDefault();
			}
		});
		// Evento para mover el contenedor (mousemove)
		document.addEventListener('mousemove', (e) => {
			if (!isDragging) return;
			const dx = e.clientX - startX;
			const dy = e.clientY - startY;
			graficoContainer.style.left = `${initialLeft + dx}px`;
			graficoContainer.style.top = `${initialTop + dy}px`;
		});
		// Evento para finalizar el arrastre (mouseup)
		document.addEventListener('mouseup', (e) => {
			if (isDragging) {
				isDragging = false;
				graficoContainer.style.cursor = 'grab';
				// Verificar si fue un clic (sin movimiento significativo)
				const movedX = Math.abs(e.clientX - startX);
				const movedY = Math.abs(e.clientY - startY);	
				// Si el movimiento fue mínimo (< 5px), considerar clic y ocultar
				if (movedX < 5 && movedY < 5) {
					graficoContainer.style.display = 'none';
				}
			}
		});
		// Peticion al Servidor para el gráfico deseado
		const datos = {
			tabla: tabla.table,
			columnas: [colX, colY],
			type: action
		};
		try {
			mostrarSpinner();
			const response = await fetch('/generar_grafico', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				// Si la respuesta no es exitosa, leer el cuerpo de la respuesta como JSON
				ocultarSpinner();
				const errorData = await response.json();
				throw new Error(errorData.error || 'Error en la respuesta del servidor');
			}
			// Añadir el contenedor al cuerpo del documento
			document.body.appendChild(graficoContainer);
			// Convertir la respuesta a un blob (imagen)
			const blob = await response.blob();
			// Crear una URL para el blob
			const imageUrl = URL.createObjectURL(blob);
			// Crear una imagen y asignarle la URL del blob
			const img = document.createElement('img');
			img.src = imageUrl;
			img.style.width = '100%';
			img.style.height = '100%';
			img.style.pointerEvents = 'none'; // ¡Importante! Evita que la imagen bloquee eventos.
			// Añadir la imagen al contenedor del gráfico
			graficoContainer.appendChild(img);
			ocultarSpinner();
			// Mostrar el contenedor del gráfico
			mostrarGraficoActual(idGrafico);
			return graficoContainer.id;
		} catch (error) {
			ocultarSpinner();
			console.error("Error:", error.message);
			customAlert(error.message);  // Mostrar el mensaje de error al usuario
			return null;
		}
	}
	 async function generarGraficoX2Pares(tabla1, tabla2, colX, colY, tipoGtabla1, tipoGtabla2, idGrafico) {
		// Verificar que la tabla y las columnas existan
		if (!tabla1 || !tabla1.table || !tabla2 || !tabla2.table || !colX || !colY) {
			customAlert("Tabla o columnas no válidas.");
			return null;
		}
		// Obtener los datos de las columnas X e Y
		const datosX1 = tabla1.table[colX];
		const datosY1 = tabla1.table[colY];
		const datosX2 = tabla2.table[colX];
		const datosY2 = tabla2.table[colY];
		// Verificar que los datos de X e Y tengan la misma longitud
		if (datosX1.length !== datosY1.length) {
			customAlert("Las columnas X e Y de la tabla original deben tener la misma longitud.");
			return null;
		}
		if (datosX2.length !== datosY2.length) {
			customAlert("Las columnas X e Y de la tabla predicción deben tener la misma longitud.");
			return null;
		}
		// Verificar si existen valores vacios o nulos
		if (datosX1.some(valor => valor === null || valor === "") || datosY1.some(valor => valor === null || valor === "")) {
			customAlert("Los datos de X o Y de la tabla original contienen valores vacíos o nulos.");
			return null;
		}
		if (datosX2.some(valor => valor === null || valor === "") || datosY2.some(valor => valor === null || valor === "")) {
			customAlert("Los datos de X o Y de la tabla predicción contienen valores vacíos o nulos.");
			return null;
		}
		// Obtener la posición del objeto Gráfico en el flujo de trabajo
		const elementoGrafico = document.querySelector(`.element[data-id="${idGrafico}"]`);
		if (!elementoGrafico) {
			customAlert("No se encontró el objeto Gráfico en el flujo de trabajo.");
			return null;
		}
		// Obtener las coordenadas del objeto Gráfico
		const rect = elementoGrafico.getBoundingClientRect();
		const offsetX = rect.left + window.scrollX; // Posición X absoluta
		const offsetY = rect.bottom + window.scrollY + 10; // Posición Y absoluta + margen
		// Verificar si ya existe un contenedor para este gráfico y eliminarlo
		const contenedorExistente = document.getElementById('grafico-' + idGrafico);
		if (contenedorExistente) {
			contenedorExistente.remove(); // Eliminar el contenedor anterior
		}
		// Crear el nuevo contenedor después de un pequeño retardo
		const graficoContainer = document.createElement('div');
		graficoContainer.id = 'grafico-' + idGrafico;
		graficoContainer.style.position = 'absolute'; // Posicionamiento absoluto
		graficoContainer.style.left = `${offsetX}px`; // Posición X
		graficoContainer.style.top = `${offsetY}px`; // Posición Y
		graficoContainer.style.width = '400px'; // Ancho del gráfico
		graficoContainer.style.height = '300px'; // Alto del gráfico
		graficoContainer.style.zIndex = '1000'; // Asegurar que esté por encima de otros elementos
		graficoContainer.style.backgroundColor = 'white'; // Fondo blanco
		graficoContainer.style.border = '1px solid #ccc'; // Borde para mejor visibilidad
		graficoContainer.style.boxShadow = '2px 2px 5px rgba(0, 0, 0, 0.2)'; // Sombra para destacar
		graficoContainer.style.resize = 'both'; // Permite redimensionar en ambas direcciones (horizontal y vertical)
		graficoContainer.style.overflow = 'hidden'; // Para que el contenido no se desborde 
		graficoContainer.style.minHeight = '100px';
		graficoContainer.style.minWidth = '100px';
		graficoContainer.style.display = 'none'; // Mostrar el gráfico inicialmente
		graficoContainer.style.cursor = 'grab'; // Cursor indicando que es arrastrable
		// Variables para controlar el arrastre
		let isDragging = false;
		let startX, startY, initialLeft, initialTop;
		// Evento para iniciar el arrastre (mousedown)
		graficoContainer.addEventListener('mousedown', (e) => {
			const rect = graficoContainer.getBoundingClientRect();
			const resizeHandleSize = 20;
			const isNearResizeCorner = 
				e.clientX > rect.right - resizeHandleSize && 
				e.clientY > rect.bottom - resizeHandleSize;
			// Si está en la esquina de redimensionamiento, ignorar arrastre
			if (isNearResizeCorner) {
				isDragging = false;
				return;
			}
			// Solo arrastrar si el clic es en el contenedor (no en la imagen)
			if (e.target === graficoContainer) {
				isDragging = true;
				startX = e.clientX;
				startY = e.clientY;
				initialLeft = parseInt(graficoContainer.style.left, 10);
				initialTop = parseInt(graficoContainer.style.top, 10);
				graficoContainer.style.cursor = 'grabbing';
				e.preventDefault();
			}
		});
		// Evento para mover el contenedor (mousemove)
		document.addEventListener('mousemove', (e) => {
			if (!isDragging) return;
			const dx = e.clientX - startX;
			const dy = e.clientY - startY;
			graficoContainer.style.left = `${initialLeft + dx}px`;
			graficoContainer.style.top = `${initialTop + dy}px`;
		});
		// Evento para finalizar el arrastre (mouseup)
		document.addEventListener('mouseup', (e) => {
			if (isDragging) {
				isDragging = false;
				graficoContainer.style.cursor = 'grab';
				// Verificar si fue un clic (sin movimiento significativo)
				const movedX = Math.abs(e.clientX - startX);
				const movedY = Math.abs(e.clientY - startY);	
				// Si el movimiento fue mínimo (< 5px), considerar clic y ocultar
				if (movedX < 5 && movedY < 5) {
					graficoContainer.style.display = 'none';
				}
			}
		});
		// Peticion al Servidor para el gráfico deseado
		const datos = {
			tabla: tabla1.table,
			columnas: [colX, colY],
			type: tipoGtabla1,
			tabla_pred : tabla2.table,
			type_pred : tipoGtabla2
		};
		try {
			mostrarSpinner();
			const response = await fetch('/generar_grafico', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(datos)
			});
			if (!response.ok) {
				// Si la respuesta no es exitosa, leer el cuerpo de la respuesta como JSON
				const errorData = await response.json();
				throw new Error(errorData.error || 'Error en la respuesta del servidor');
			}
			// Añadir el contenedor al cuerpo del documento
			document.body.appendChild(graficoContainer);
			// Convertir la respuesta a un blob (imagen)
			const blob = await response.blob();
			// Crear una URL para el blob
			const imageUrl = URL.createObjectURL(blob);
			// Crear una imagen y asignarle la URL del blob
			const img = document.createElement('img');
			img.src = imageUrl;
			img.style.width = '100%';
			img.style.height = '100%';
			img.style.pointerEvents = 'none'; // ¡Importante! Evita que la imagen bloquee eventos.
			// Añadir la imagen al contenedor del gráfico
			graficoContainer.appendChild(img);
			ocultarSpinner();
			// Mostrar el contenedor del gráfico
			mostrarGraficoActual(idGrafico);
			return graficoContainer.id;
		} catch (error) {
			ocultarSpinner();
			console.error("Error:", error.message);
			customAlert(error.message);  // Mostrar el mensaje de error al usuario
			return null;
		}
	}
	// Mostrar el grafico actual del objeto grafico enfocado
	 function mostrarGraficoActual(idGrafico) {
		const graficoContainer = document.getElementById('grafico-' + idGrafico);
		if (!graficoContainer) {
			customAlert("No hay un gráfico generado para este objeto.");
			return;
		}
		graficoContainer.style.display = 'block'; // Solo mostrar, el clic para ocultar ya está en generarGraficoX
	}
	 async function mostrarPlantillasKPI(idPlantillaKPI, url_plantillas) {
		try {
			// Realizar una solicitud GET al servidor para obtener las plantillas segun corresponda
			const response = await fetch(url_plantillas, {
				method: "GET",  // Especificar el método GET
			});
			if (!response.ok) {
				customAlert("Error al obtener las plantillas KPI.");
			}
			const plantillas = await response.json();
			if (plantillas.length === 0) {
				customAlert("No tienes plantillas KPI.");
				return;
			}
			// Crear un diálogo para mostrar la lista de plantillas
			const dialogoPlantillas = `
					Seleccione una plantilla KPI:<br />
					<select id="selectPlantilla">
						${plantillas.map(plantilla => `
							<option value="${plantilla.archivo}">${plantilla.denominacion}</option>
						`).join("")}
					</select>
					<br /><button id="btnSeleccionarPlantilla" class="custom-button">Seleccionar</button>
					&nbsp;&nbsp;`;
			// Mostrar el diálogo al lado del menú contextual
			crearDialogo(dialogoPlantillas);
			// Manejar la selección de la plantilla
			$("#btnSeleccionarPlantilla").on("click",  async function () {
				const archivoPlantilla = $("#selectPlantilla").val();
				const denominacionPlantilla = $("#selectPlantilla option:selected").text(); // Corregido para obtener el texto de la opción seleccionada
				$("#dialogoPlantillas").remove();
				const campos = await leerCamposPlantilla(archivoPlantilla, idPlantillaKPI);
				if (campos != null) {
					const objPlantillaKPI = workflow_json.find(obj => obj.id == idPlantillaKPI);	
					mostrarAsociacionCampos(campos, idPlantillaKPI)
						.then(asociaciones => {
							let banner_tooltip = "";
							if (asociaciones) {
								objPlantillaKPI.archivo = archivoPlantilla;
								objPlantillaKPI.asociaciones = asociaciones;
								// Usar .map() y .join() para evitar tabulaciones no deseadas
								const asociacionesStr = Object.entries(asociaciones)
									.map(([campo, operacion]) => `${campo}: ${operacion}`)
									.join('<br/>');
								// Template literal sin indentación para evitar \t
								banner_tooltip = `<strong>Plantilla KPI:</strong><br/>(${denominacionPlantilla})<br/>${asociacionesStr}`;
							} else {
								objPlantillaKPI.archivo = archivoPlantilla;
								objPlantillaKPI.asociaciones = {};
								banner_tooltip = `<strong>Plantilla KPI:</strong><br/>Asociaciones de campos insuficiente.`;
							}
							// Mover esto DENTRO del .then()
							const tooltip = document.getElementById(`tooltip-${idPlantillaKPI}`);
							if (tooltip) {
								tooltip.innerHTML = banner_tooltip;
								tooltip.style.whiteSpace = 'nowrap';
							}
							objPlantillaKPI.tooltip = banner_tooltip;
						})
						.catch(error => {
							console.error("Error:", error);
						});
				}
			});
			ocultar_menues(); // Ocultar menús contextuales
		} catch (error) {
			customAlert(error.message);
		}
	}
	 async function leerCamposPlantilla(archivoPlantilla, idPlantillaKPI) {
		try {
			// Realizar una solicitud POST al servidor para leer los campos de la plantilla seleccionada
			mostrarSpinner();
			const response = await fetch("/leer_campos_plantilla", {
				method: "POST",
				headers: {
					"Content-Type": "application/x-www-form-urlencoded",
				},
				body: `archivo_plantilla=${encodeURIComponent(archivoPlantilla)}`,
			});
			if (!response.ok) {
				ocultarSpinner();
				customAlert("Error al leer los campos de la plantilla.");
				return null;
			}
			ocultarSpinner();
			const campos = await response.json();
			if (campos.length === 0) {
				customAlert("La plantilla seleccionada no contiene campos con formato {{CAMPO}}.");
				return null;
			}
			// Mostrar un diálogo para asociar los campos con los resultados del flujo de trabajo
			return campos;
		} catch (error) {
			ocultarSpinner();
			customAlert(error.message);
			return null;
		}
	}
	function mostrarAsociacionCampos(campos, idPlantillaKPI) {
		return new Promise((resolve, reject) => {
			// Crear un diálogo para mostrar los campos y permitir su asociación con los 
			// resultados del flujo de trabajo
			// Los campos se pueden asociar a:
			// - Campos de la Base de Datos (ejemplo> Nombre y Apellidos del ejecutor)
			// - Valores Automaticos del Sistema (ejemplo: Fecha y hora de emision)
			// - Graficos generados de Tipo, Tabla y Columnas y parametros de graficado
			// - Resultados de tipos de Procesamientos con resultados de tablas y Variables
	
			// Hallar objeto del Informe Plantillas KPI actual...
			const objPlantilla = workflow_json.find(obj => obj.id === idPlantillaKPI);
			// Filtrar las conexiones que van hacia el objeto Informe Plantilla KPI...
			const froms = connections
				.filter(conn => conn.to === idPlantillaKPI) // Filtra los objetos donde "to" es igual a idPlantillaKPI
				.map(conn => conn.from); // Extrae solo el atributo "from" de los objetos filtrados
			// Leer los tipos resultados enlazados al obj de InformeKPI
			const objs = workflow_json.filter(obj => ["Procesamiento", "Grafico"].includes(obj.tipo));
			let opciones = "";
			objs.forEach(obj => {
				if (froms.includes(obj.id)) {
					switch (obj.tipo) {
						case "Procesamiento":
							if (obj.resultado.length && obj.proceso){
								const action = Procesos_Label2Descrip[obj.proceso];
								const tablas = obj.resultado;
								for(let i = 0; i < tablas.length; i++) {
									tabla = tablas[i];
									// Aquí se usa JSON.stringify para convertir el objeto en una cadena JSON
									if (tabla.descrip !== undefined)
										s_extra = tabla.descrip;
									else
										s_extra = "";
									opciones += `<option value='${JSON.stringify({"tipo":"TABLA","id": obj.id,"idx":i})}'>Proceso: ${action}, Tabla(${tabla.rows}x${tabla.cols}), ${s_extra}</option>`;
								}
							}
							break;
						case "Grafico":
							if (obj.resultado.length && obj.type){
								const tipo_graph = Procesos_Label2Descrip[obj.type];
								opciones += `<option value='${JSON.stringify({"tipo":"IMAGEN","id":obj.id})}'>Grafico: ${tipo_graph}, Columnas: ${obj.cols}</option>`;
							}
					}
				}
			});
			const dialogoCampos = `
					<strong>Asociar Campos de la plantilla:</strong><br />
					<ul>
						${campos.map(campo => `
							<li>${campo}: 
								<select id="${campo}">
									<option value="0">Ninguno</option>
									<option value="FECHA_HORA">Fecha y hora de emisión</option>
									<option value="NOM_APE_EJECUTOR">Nombre y Apellidos del Ejecutor del Informe</option>
									${opciones}
								</select>
							</li>
						`).join("")}
					</ul>
					<br /><button id="btnAsociarCampos"  class="custom-button">Asociar Campos</button>
					&nbsp;&nbsp;`;
			// Mostrar el diálogo al lado del menú contextual
			crearDialogo(dialogoCampos);
			// Manejar la asociación de campos
			$("#btnAsociarCampos").on("click",  function () {
				const asociaciones = {};
				campos.forEach(campo => {
					const seleccion = $(`#${campo}`).val();
					if (seleccion !== "0") {
						// Aquí se usa JSON.parse para convertir la cadena JSON de vuelta a un objeto
						asociaciones[campo] = seleccion === "FECHA_HORA" || seleccion === "NOM_APE_EJECUTOR"
							? seleccion
							: JSON.parse(seleccion);
					}
				});
				const total_asoc = Object.keys(asociaciones).length;
				const total_minimo = Math.round(0.5 * campos.length);
				if (total_asoc >= total_minimo) {
					customConfirm("Se asociaron " + total_asoc + " campos de un total de " + campos.length + ".");
					// Resolver la promesa con el objeto asociaciones
					resolve(asociaciones);
				} else {
					customAlert("Debe asociar al menos el 50% del total de campos de la actual plantilla.");
					// Resolver la promesa con null
					resolve(null);
				}
				CerrarDialogosYMenues();
			});
		});
	}
	// Función para elimianr un elemento del área de trabajo
	function eliminar_Elemento( id ){
		//quitar el elemento de workflow_json y demas solo si no hay otro mas 
		//adelante dependiendo de 'el...
		function eliminar_obj( id ) {
			index = workflow_json.findIndex(obj => obj.id === id);
			workflow_json.splice(index, 1); // Elimina 1 elemento en la posición index
			// Eliminando todas las conexiones a ese elemento a eliminar...
			connections = connections.filter(conn => conn.from !== id && conn.to !== id);
			// eliminando objeto graficado
			$(`.element[data-id="${id}"]`).remove();
			// Redibujar las conexiones filtradas...
			redrawConnections();
			// Eliminando el elemento del mapa ...
			if (id in elements)
				delete elements[id];
			return true;
		}
		objelem = workflow_json.find(obj => obj.id === id);
		index = connections.findIndex(obj => obj.from === id);
		if (objelem && index != -1) {
			lista_tablas = [];
			if (objelem.tipo == "Documento") {
				lista_tablas = objelem.tablas;
			} else if (objelem.tipo != "InformeKPI") {
				lista_tablas = objelem.resultado;
			}
			if (lista_tablas.length > 0) {
				// buscamos si alguna de las tablas en la lista estan siendo
				// referenciada por algun otro elemento (idtable), conectado al actual
				var conexiones = connections.filter(conn => conn.from === id);
				for(let i = 0; i < conexiones.length; i++) {
					obj = workflow_json.find(o => o.id === conexiones[i].to);
					idx = lista_tablas.findIndex(tab => tab.idTable == obj.idtable);
					if (idx != -1) // Se encontro que uno de los obj conectados referencia 
					// a una de las tablas del obj actual
						return false;
				}
			}
		}
		// Si llegamos aqui es que el obj se puede eliminar
		return eliminar_obj( id );
	}
	// Evento para capturar opcion en la lista de modelos...
	$('#lista_modelos').on('change', function() {
		var sel = document.getElementById('lista_modelos').value;
		const toolElements = document.querySelectorAll(".tool");
		switch(sel) {
			case "-1": // ninguno
				toolElements.forEach(element => {  
					// Como se selecciono nuevo modelo, deben mostrarse todos los elememtos de clase tool 
					element.style.display = "none"; // Ocultar
				});
				// Primeramente, borremos todo lo que esta en 
				// workflow_json, conections y elements...
				while( workflow_json.length ) {
					for(let i = 0; i < workflow_json.length; i++) {
						id = workflow_json[i].id
						if (eliminar_Elemento( id ))
							break;
					}
				}
				// Asegurar las tres principales listas del modelo a
				// iniciarse...
				workflow_json = [];
				connections = [];
				elements = {};
				document.getElementById("nombre_modelo").value = "";
				break;
			default: // Otro caso: se selecciono un modelo existente!
				var formData = new FormData();
				formData.append("archivo_json", sel);
				// Enviar la solicitud POST al servidor
				let xhr = new XMLHttpRequest();
				xhr.responseType = "json";
				xhr.open("POST", "/leer_archivo_json", true);
				xhr.onload = function() {
					if (xhr.status === 200) {
						toolElements.forEach(element => {  
							// Como se selecciono nuevo modelo, deben mostrarse todos los elememtos de clase tool 
							element.style.display = "inline-block"; // Mostrar  
						});
						// Primeramente, borremos todo lo que esta en 
						// workflow_json, conections y elements...
						while( workflow_json.length ) {
							for(let i = 0; i < workflow_json.length; i++) {
								id = workflow_json[i].id
								if (eliminar_Elemento( id ))
									break;
							}
						}
						// Asegurar las tres principales listas del modelo a
						// iniciarse...
						workflow_json = [];
						connections = [];
						elements = {};
						// Ahora, cargamos el modelo seleccionado...
						resp = xhr.response;
						// Asignando a la lista correspondientes...
						_workflow_json = resp[0]; // elementos y ...
						_connections = resp[1]; // ... sus conexiones
						// Colocamos el nombre del modelo donde corresponde...
						textoSel = document.getElementById('lista_modelos').selectedOptions[0].text;
						document.getElementById("nombre_modelo").value = textoSel;
						// Y ahora mostramos cada elemento con sus conexiones 
						// en el espacio de trabajo...
						for(let i = 0; i < _workflow_json.length; i++) {
							obj = _workflow_json[i];
							type = obj.tipo;
							addElement(type, obj_json = obj, conectar = false);
						}
						for(i = 0; i < _connections.length; i++) {
							conn = _connections[i];
							drawConnection(conn.from, conn.to, true);
						}
					}
				}
				xhr.send(formData);
				break;
			case "0": // nuevo modelo?
				toolElements.forEach(element => {  
					// Como se selecciono nuevo modelo, deben mostrarse todos los elememtos de clase tool 
					element.style.display = "inline-block"; // Mostrar  
				});
				// Primeramente, borremos todo lo que esta en 
				// workflow_json, conections y elements...
				while( workflow_json.length ) {
					for(let i = 0; i < workflow_json.length; i++) {
						id = workflow_json[i].id
						if (eliminar_Elemento( id ))
							break;
					}
				}
				// Asegurar las tres principales listas del modelo a
				// iniciarse...
				workflow_json = [];
				connections = [];
				elements = {};
				document.getElementById("nombre_modelo").value = "";
				// Al hacer nuevo modelo, agregamos por defecto un elemento tipo documento
				addElement("Documento")
				break;
		}		
	});
	/*
	* Muestra un diálogo para configurar agrupaciones y operaciones en una tabla.
	* @param {Object} tableData - Datos de la tabla en formato {col1: [val1, val2,...], col2: [...]}.
	*/
	function mostrarDialogoAgruparXFilas(tableData, idConv) {
		table = tableData.table;
		// Crear el contenedor del diálogo
		const dialog = document.createElement("div");
		dialog.style.position = "fixed";
		dialog.style.top = "50%";
		dialog.style.left = "50%";
		dialog.style.transform = "translate(-50%, -50%)";
		dialog.style.backgroundColor = "#caf0f8";
		dialog.style.padding = "10px";
		dialog.style.border = "1px solid #000";
		dialog.style.zIndex = "1000";
		dialog.style.boxShadow = "0 4px 8px rgba(0,0,0,0.1)";
		dialog.style.maxWidth = "100%";
		// Columnas disponibles (primera lista)
		const columns = Object.keys(table);
		const availableList = document.createElement("div");
		availableList.innerHTML = "<h3>Columnas disponibles</h3>";
		const availableColumns = document.createElement("ul");
		availableColumns.style.listStyle = "none";
		availableColumns.style.padding = "0";
		availableColumns.style.maxHeight = "300px";
		availableColumns.style.overflowY = "auto";
		columns.forEach(col => {
			const li = document.createElement("li");
			li.textContent = col;
			li.style.padding = "5px";
			li.style.cursor = "pointer";
			li.style.borderBottom = "1px solid #eee";
			li.style.transition = "background-color 0.2s";
			li.addEventListener("click", () => {
				// Quitar selección previa
				if (selectedItem) {
					selectedItem.style.backgroundColor = "";
					selectedItem.style.fontWeight = "";
				}
				// Seleccionar nuevo elemento
				selectedItem = li;
				li.style.backgroundColor = "#90e0ef";
				li.style.fontWeight = "bold";
			});
			setupListItemClick(li, 'available'); // Identificamos que es de la lista disponible
			availableColumns.appendChild(li);
		});
		availableList.appendChild(availableColumns);
		// Columnas para agrupar (segunda lista)
		const groupByList = document.createElement("div");
		groupByList.innerHTML = "<h3>Agrupar por</h3>";
		const groupByColumns = document.createElement("ul");
		groupByColumns.style.listStyle = "none";
		groupByColumns.style.padding = "0";
		groupByColumns.style.maxHeight = "300px";
		groupByColumns.style.overflowY = "auto";
		groupByList.appendChild(groupByColumns);
		// Columnas para operar (tercera lista)
		const operationsList = document.createElement("div");
		operationsList.innerHTML = "<h3>Operaciones</h3>";
		const operationsColumns = document.createElement("ul");
		operationsColumns.style.listStyle = "none";
		operationsColumns.style.padding = "0";
		operationsColumns.style.maxHeight = "300px";
		operationsColumns.style.overflowY = "auto";
		operationsList.appendChild(operationsColumns);
		// Botones
		const buttonContainer = document.createElement("div");
		buttonContainer.style.display = "flex";
		buttonContainer.style.justifyContent = "space-between";
		buttonContainer.style.marginTop = "20px";
		const chooseGroupBtn = document.createElement("button");
		chooseGroupBtn.classList.add("custom-button");
		chooseGroupBtn.textContent = "Elegir como clave";
		chooseGroupBtn.addEventListener("click", () => {
			if (selectedItem && selectedList === 'available') {
				const newItem = selectedItem.cloneNode(true);
				setupListItemClick(newItem, 'groupBy'); // Marcamos que es de la lista de agrupamiento
				groupByColumns.appendChild(newItem);
			}
		});
		const chooseOperationBtn = document.createElement("button");
		chooseOperationBtn.classList.add("custom-button");
		chooseOperationBtn.textContent = "Elegir como operación";
		chooseOperationBtn.addEventListener("click", () => {
			const selected = selectedItem;
			if (selectedItem && selectedList === 'available') {
				const columnName = selectedItem.textContent;
				const isNumeric = isNumericColumn(table[columnName]);	
				// Permitir cualquier columna para count, pero solo numéricas para otras operaciones
				const li = document.createElement("li");
				li.style.padding = "5px";
				li.style.borderBottom = "1px solid #eee";
				const colName = document.createElement("span");
				colName.textContent = columnName;
				li.appendChild(colName);
				const select = document.createElement("select");
				const operations = isNumeric 
					? ["sum", "mean", "min", "max", "count"] 
					: ["count", "count_distinct", "count_categories"]; // Solo permitir count para columnas no numéricas
				operations.forEach(op => {
					const option = document.createElement("option");
					option.value = op;
					option.textContent = op;
					select.appendChild(option);
				});				
				li.appendChild(select);
				setupListItemClick(li, 'operations'); // Marcamos que es de la lista de operaciones
				operationsColumns.appendChild(li);
			}
		});
		const removeBtn = document.createElement("button");
		removeBtn.classList.add("custom-button");
		removeBtn.textContent = "Eliminar";
		removeBtn.addEventListener("click",  () => {
			if (selectedItem && selectedList !== 'available') {
				// Solo permitir eliminar de las listas de agrupamiento u operaciones
				selectedItem.remove();
				selectedItem = null;
				selectedList = null;
			} else if (selectedList === 'available') {
				customAlert("No puede eliminar columnas de la lista de disponibles. Seleccione un elemento de las listas de agrupamiento u operaciones.");
			}
		});
		const groupBtn = document.createElement("button");
		groupBtn.classList.add("custom-button");
		groupBtn.textContent = "Agrupar";
		groupBtn.addEventListener("click",  () => {
			const groupBy = Array.from(groupByColumns.children).map(li => li.textContent);
			if (groupBy.length == 0) {
				customAlert("Debe elegir al menos una columna como clave para agrupar.");
				return;
			}
			const operations = Array.from(operationsColumns.children).map(li => ({
				column: li.querySelector("span").textContent,
				operation: li.querySelector("select").value
			}));
			if (operations.length == 0) {
				customAlert("Debe elegir al menos una columna para operaciones en el proceso de agrupamiento.");
				return;
			}
			// Verificar que las operaciones sean válidas (solo count para no numéricas)
			const allValid = operations.every(op => {
				if (["count", "count_distinct", "count_categories"].includes(op.operation)) 
					return true;
				return isNumericColumn(table[op.column]);
			});
			if (!allValid) {
				customAlert("¡Operación no válida para columnas no numéricas! Solo se permite 'count'.'count_distinct' o 'count_categories'");
				return;
			}
			mostrarSpinner();
			fetch("/groupby", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ table, groupBy, operations })
			})
			.then(response => response.json())
			.then( nueva_tabla => {
				dialog.remove();
				CerrarDialogosYMenues();
				// Actualizar tu UI con el resultado
				const objConv = workflow_json.find(obj => obj.id === idConv);
				TablaX = new Object();
				TablaX.table = nueva_tabla;
				TablaX.idTable = 'table-' + (performance.now() + count_t); ++count_t;
				keys_table = Object.keys(nueva_tabla);
				TablaX.cols = keys_table.length;
				TablaX.rows = nueva_tabla[keys_table[0]].length;
				TablaX.columnas = keys_table;
				objConv.resultado = [TablaX];
				objConv.proceso = "Agrupar_filas";
				objConv.idtable = tableData.idTable;
				objConv.cols = groupBy;
				objConv.operations = operations;
				// Actualizamos el tooltip correspondiente
				const tooltip = document.getElementById(`tooltip-${idConv}`);
				if (tooltip) {
					str_cols = "";
					for(let i = 0; i < groupBy.length; i++)
						str_cols = str_cols + "<li>"+groupBy[i]+"</li>";
					str_tooltip = `<strong>Agrupar filas:</strong><br />Tabla (${tableData.rows}x${tableData.cols}), Agrupar por:<br/><ol>${str_cols}</ol>`;
					tooltip.innerHTML = str_tooltip;
					objConv.tooltip = str_tooltip;
				}
				ocultarSpinner();
				customConfirm("Se agruparon correctamente las filas!.")
			})
			.catch( error => {
				console.error("Error al agrupar:", error);
				ocultarSpinner();
				customAlert("Error al procesar la operación");
			});
		});
		const cancelBtn = document.createElement("button");
		cancelBtn.classList.add("custom-button");
		cancelBtn.textContent = "Cancelar";
		cancelBtn.addEventListener("click", () => {
			dialog.remove();
			CerrarDialogosYMenues();
		});
		buttonContainer.append(chooseGroupBtn, chooseOperationBtn, removeBtn, groupBtn, cancelBtn);
		// Diseño de 3 columnas
		const columnsContainer = document.createElement("div");
		columnsContainer.style.display = "grid";
		columnsContainer.style.gridTemplateColumns = "1fr 1fr 1fr";
		columnsContainer.style.gap = "10px";
		columnsContainer.append(availableList, groupByList, operationsList);
		dialog.append(columnsContainer, buttonContainer);
		document.body.appendChild(dialog);
		// Helper: Verificar si una columna es numérica
		function isNumericColumn(data) {
			if (!Array.isArray(data)) return false;
			const validValues = data.filter(val => 
				val !== null && val !== undefined && val !== '');
			if (validValues.length === 0) return false;
			const numericCount = validValues.filter(val => 
				typeof val === 'number' || (!isNaN(parseFloat(val)) && isFinite(val))
			).length;
			return (numericCount / validValues.length) >= 0.8;
		}
		let selectedItem = null;
		let selectedList = null; // Nueva variable para trackear de qué lista proviene el ítem seleccionado
		// Modificación en el evento click de los ítems
		function setupListItemClick(li, listType) {
			li.addEventListener("click", () => {
				// Quitar selección previa
				if (selectedItem) {
					selectedItem.style.backgroundColor = "";
					selectedItem.style.fontWeight = "";
				}
				// Seleccionar nuevo elemento
				selectedItem = li;
				selectedList = listType; // Guardamos de qué lista proviene
				li.style.backgroundColor = "#90e0ef";
				li.style.fontWeight = "bold";
			});
		}

	}
	// Mostrar dialogo para Filtrar filas ante una condicion
	function mostrarDialogoEliminarFilas(tableData, idLimpieza) {
		table = tableData.table;
		// Crear el contenedor del diálogo
		const dialog = document.createElement("div");
		dialog.style.position = "fixed";
		dialog.style.top = "50%";
		dialog.style.left = "50%";
		dialog.style.transform = "translate(-50%, -50%)";
		dialog.style.backgroundColor = "#caf0f8";
		dialog.style.padding = "20px";
		dialog.style.border = "1px solid #000";
		dialog.style.borderRadius = "8px";
		dialog.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
		dialog.style.zIndex = "1000";
		dialog.style.maxWidth = "500px";
		dialog.style.width = "90%";
	
		// Título
		const title = document.createElement("h2");
		title.textContent = "Filtrar filas condicionalmente";
		title.style.marginTop = "0";
		title.style.color = "#343a40";
		title.style.marginBottom = "15px";
		dialog.appendChild(title);
	
		// Contenedor principal
		const mainContainer = document.createElement("div");
		mainContainer.style.display = "flex";
		mainContainer.style.flexDirection = "column";
		mainContainer.style.gap = "15px";
	
		// Contenedor de controles
		const controls = document.createElement("div");
		controls.style.display = "flex";
		controls.style.gap = "10px";
		controls.style.marginBottom = "10px";
		controls.style.alignItems = "center";
		controls.style.flexWrap = "wrap";
	
		// Selector de columna
		const columnSelect = document.createElement("select");
		columnSelect.style.flex = "1";
		columnSelect.style.minWidth = "120px";
		columnSelect.style.padding = "8px";
		columnSelect.style.borderRadius = "4px";
		columnSelect.style.border = "1px solid #ced4da";
		Object.keys(table).forEach(col => {
			const option = document.createElement("option");
			option.value = col;
			option.textContent = col;
			columnSelect.appendChild(option);
		});
	
		// Selector de operador
		const operatorSelect = document.createElement("select");
		operatorSelect.style.minWidth = "120px";
		operatorSelect.style.padding = "8px";
		operatorSelect.style.borderRadius = "4px";
		operatorSelect.style.border = "1px solid #ced4da";
		const operators = [
			{ value: "==", text: "Igual a (==)" },
			{ value: "!=", text: "Distinto a (!=)" },
			{ value: ">", text: "Mayor que (>)" },
			{ value: "<", text: "Menor que (<)" },
			{ value: ">=", text: "Mayor o igual (>=)" },
			{ value: "<=", text: "Menor o igual (<=)" },
			{ value: "contains", text: "Contiene" },
			{ value: "not_contains", text: "No Contiene"}
		];
		operators.forEach(op => {
			const option = document.createElement("option");
			option.value = op.value;
			option.textContent = op.text;
			operatorSelect.appendChild(option);
		});
	
		// Input de valor
		const valueInput = document.createElement("input");
		valueInput.type = "text";
		valueInput.style.flex = "1";
		valueInput.style.minWidth = "120px";
		valueInput.style.padding = "8px";
		valueInput.style.borderRadius = "4px";
		valueInput.style.border = "1px solid #ced4da";
		valueInput.placeholder = "Valor a comparar";
	
		// Agregar controles al contenedor
		controls.appendChild(columnSelect);
		controls.appendChild(operatorSelect);
		controls.appendChild(valueInput);
		mainContainer.appendChild(controls);
	
		// Checkbox para mayúsculas/minúsculas
		const caseSensitiveContainer = document.createElement("div");
		caseSensitiveContainer.style.display = "flex";
		caseSensitiveContainer.style.alignItems = "center";
		caseSensitiveContainer.style.marginBottom = "15px";
		caseSensitiveContainer.style.gap = "8px";
	
		const caseSensitiveCheckbox = document.createElement("input");
		caseSensitiveCheckbox.type = "checkbox";
		caseSensitiveCheckbox.id = "case-sensitive-checkbox";
		caseSensitiveCheckbox.checked = false; // Por defecto no sensible a mayúsculas
	
		const caseSensitiveLabel = document.createElement("label");
		caseSensitiveLabel.htmlFor = "case-sensitive-checkbox";
		caseSensitiveLabel.textContent = "Respetar mayúsculas/minúsculas";
		caseSensitiveLabel.style.fontSize = "14px";
		caseSensitiveLabel.style.color = "#495057";
	
		caseSensitiveContainer.appendChild(caseSensitiveCheckbox);
		caseSensitiveContainer.appendChild(caseSensitiveLabel);
		mainContainer.appendChild(caseSensitiveContainer);
	
		dialog.appendChild(mainContainer);
	
		// Botones
		const buttonContainer = document.createElement("div");
		buttonContainer.style.display = "flex";
		buttonContainer.style.justifyContent = "flex-end";
		buttonContainer.style.gap = "10px";
	
		// Boton Cancelar
		const cancelBtn = document.createElement("button");
		cancelBtn.classList.add("custom-button");
		cancelBtn.textContent = "Cancelar";
		cancelBtn.addEventListener("click", () => {
			dialog.remove();
			CerrarDialogosYMenues();
		});
	
		// Boton Filtrar
		const filterBtn = document.createElement("button");
		filterBtn.classList.add("custom-button");
		filterBtn.textContent = "Filtrar";
		filterBtn.addEventListener("click",  () => {
			const column = columnSelect.value;
			const operator = operatorSelect.value;
			const value = valueInput.value;
			const caseSensitive = caseSensitiveCheckbox.checked;
	
			// Validación básica
			if (!column || !operator || value === "") {
				customAlert("Por favor complete todos los campos");
				return;
			}
	
			// Enviar al servidor
			mostrarSpinner();
			fetch("/filtrar_filas", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					table: table,
					column: column,
					operator: operator,
					value: value,
					case_sensitive: caseSensitive // Nuevo parámetro
				})
			})
			.then(response => response.json())
			.then( nueva_tabla => {
				dialog.remove();
				CerrarDialogosYMenues();
				// Actualizar tu UI con la tabla filtrada
				const objLimpieza = workflow_json.find(obj => obj.id === idLimpieza);
				TablaX = new Object();
				TablaX.table = nueva_tabla;
				TablaX.idTable = 'table-' + (performance.now() + count_t); ++count_t;
				keys_table = Object.keys(nueva_tabla);
				TablaX.cols = keys_table.length;
				TablaX.rows = nueva_tabla[keys_table[0]].length;
				TablaX.columnas = keys_table;
				objLimpieza.resultado = [TablaX];
				objLimpieza.proceso = "Eliminar_filasfcond";
				objLimpieza.idtable = tableData.idTable;
				objLimpieza.cols = column;
				objLimpieza.operator = operator;
				objLimpieza.value = value;
				objLimpieza.case_sensitive = caseSensitive; // Guardar en el workflow
				
				// Actualizamos el tooltip correspondiente
				const tooltip = document.getElementById(`tooltip-${idLimpieza}`);
				if (tooltip) {
					const caseText = caseSensitive ? " (sensible a mayúsculas/minúsculas)" : " (insensible a mayúsculas/minúsculas)";
					str_operator = operator;
					if (operator == 'contains') {
						str_operator = 'Contiene';
					} else if (operator == 'not_contains') {
						str_operator = 'No Contiene';
					}
					str_tooltip = `<strong>Filtrar filas condicionalmente:</strong><br />Desde Tabla (${tableData.rows}x${tableData.cols}), Filtrado por:<br/>- ${column} ${str_operator} "${value}"<br>${caseText}`;
					tooltip.innerHTML = str_tooltip;
					objLimpieza.tooltip = str_tooltip;
				}
				ocultarSpinner();
				customConfirm(`Se eliminaron las filas. Nueva tabla con ${nueva_tabla[column].length} filas`);
			})
			.catch( error => {
				ocultarSpinner();
				console.error("Error:", error);
				customAlert("Error al filtrar las filas");
			});
		});
	
		buttonContainer.appendChild(filterBtn);
		buttonContainer.appendChild(cancelBtn);
		dialog.appendChild(buttonContainer);
		document.body.appendChild(dialog);
	}
	// Dialogo para ordenar las filas segun columnas...
	function mostrarDialogoOrdenar(tableData, idConv) {
		table = tableData.table;
		const dialog = document.createElement("div");
		dialog.style.position = "fixed";
		dialog.style.top = "50%";
		dialog.style.left = "50%";
		dialog.style.transform = "translate(-50%, -50%)";
		dialog.style.backgroundColor = "#caf0f8";
		dialog.style.padding = "20px";
		dialog.style.borderRadius = "8px";
		dialog.style.boxShadow = "0 2px 10px rgba(0,0,0,0.1)";
		dialog.style.zIndex = "1000";
		// Selector de columna
		const columnSelect = document.createElement("select");
		columnSelect.style.width = "100%";
		columnSelect.style.padding = "8px";
		columnSelect.style.marginBottom = "10px";
		Object.keys(table).forEach(col => {
			const option = document.createElement("option");
			option.value = col;
			option.textContent = col;
			columnSelect.appendChild(option);
		});
		// Selector de orden (ascendente/descendente)
		const orderSelect = document.createElement("select");
		orderSelect.style.width = "100%";
		orderSelect.style.padding = "8px";
		const orders = [
			{ value: "true", text: "Ascendente (A-Z, 0-9)" },
			{ value: "false", text: "Descendente (Z-A, 9-0)" }
		];
		orders.forEach(order => {
			const option = document.createElement("option");
			option.value = order.value;
			option.textContent = order.text;
			orderSelect.appendChild(option);
		});
		// Botones
		const buttonContainer = document.createElement("div");
		buttonContainer.style.display = "flex";
		buttonContainer.style.justifyContent = "flex-end";
		buttonContainer.style.gap = "10px";
		buttonContainer.style.marginTop = "15px";
		// boton de cancelar
		const cancelBtn = document.createElement("button");
		cancelBtn.classList.add("custom-button")
		cancelBtn.textContent = "Cancelar";
		cancelBtn.onclick = () => {
			dialog.remove();
			CerrarDialogosYMenues();
		};
		// boton de ordenar
		const sortBtn = document.createElement("button");
		sortBtn.classList.add("custom-button")
		sortBtn.textContent = "Ordenar";
		sortBtn.style.backgroundColor = "#4CAF50";
		sortBtn.style.color = "white";
		sortBtn.onclick = () => {
			mostrarSpinner();
			fetch("/ordenar_tabla", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					table: table,
					sort_by: columnSelect.value,
					ascending: orderSelect.value === "true"
				})
			})
			.then(response => response.json())
			.then( nueva_tabla => {
				// Actualizar tu tabla en la UI aquí
				const objConv = workflow_json.find(obj => obj.id === idConv);
				TablaX = new Object();
				TablaX.table = nueva_tabla;
				TablaX.idTable = 'table-' + (performance.now() + count_t); ++count_t;
				keys_table = Object.keys(nueva_tabla);
				TablaX.cols = keys_table.length;
				TablaX.rows = nueva_tabla[keys_table[0]].length;
				TablaX.columnas = keys_table;
				objConv.resultado = [TablaX];
				objConv.proceso = "Ordenar_filas";
				objConv.idtable = tableData.idTable;
				objConv.cols = columnSelect.value;
				objConv.ascending = orderSelect.value === "true";
				// Actualizamos el tooltip correspondiente
				const tooltip = document.getElementById(`tooltip-${idConv}`);
				if (tooltip) { // Actualiza el tooltip con el proceso de Conversion
					//tooltip.style.whiteSpace = 'nowrap'; // Evitar que el texto se divida en varias lineas
					//formar la lista de columnas...
					str_tooltip = `<strong>Ordenar filas:</strong><br />Desde Tabla (${tableData.rows}x${tableData.cols}), <br>Ordenar por:<br/><ol>${columnSelect.value}</ol>`;
					tooltip.innerHTML = str_tooltip;
					objConv.tooltip = str_tooltip;
				}
				ocultarSpinner();
				dialog.remove();
				CerrarDialogosYMenues();
				customConfirm("Se realizó exitosamente el ordenamiento!")
			});
		};
		buttonContainer.appendChild(cancelBtn);
		buttonContainer.appendChild(sortBtn);
		// Agregar elementos al diálogo
		// Crear el elemento h3
		var h3 = document.createElement("h3");
		// Asignar el texto
		h3.textContent = "Ordenar tabla";
		// Añadir al diálogo
		dialog.appendChild(h3);
		dialog.appendChild(columnSelect);
		dialog.appendChild(orderSelect);
		dialog.appendChild(buttonContainer);
		document.body.appendChild(dialog);
	}
	//Poner deshabilitado el editor del nombre del modelo y el boton de salvar...
	document.getElementById("nombre_modelo").value = "";
	document.getElementById("nombre_modelo").disabled = true;
	document.getElementById("btn_salvarmodelo").disabled = true;
	//Disparamos el evento de cambio de seleccion de modelos para que
	//sea Ninguno o nuevo al principio.
	$('#lista_modelos').trigger('change');
});
// funcion para validar lo que se edita como nombre del modelo 
function valideKey(evt)
{
	// code es la representación ASCII de la tecla presionada.
	var code = (evt.which) ? evt.which : evt.keyCode;
	function chequear_nombre() {
		nombre = document.getElementById("nombre_modelo").value;
		boton = document.getElementById("btn_salvarmodelo");
		if (nombre.length < 10 || nombre[0]==" ") 
			//deshabilitar boton si largo menor que 10 o primer caracter es un espacio en blanco
			boton.disabled = true;
		else
			// Se esta escribiendo un nombre de modelo KPI correctamente, activamos el boton de salvar...
			boton.disabled = false;
	}
	if(code==8) // backspace.
	return chequear_nombre();
	if(code==127) // Delete.
	return chequear_nombre();
	if (code==0x20) //Space char?
	return chequear_nombre();
	if (code==0x2D || code==0x5F) //minus or underscore char?
	return chequear_nombre();
	if (code>=48 && code<=57) // is a number.
	return chequear_nombre();
	if (code>=0x61 && code<=0x7A) // is a lower letter?
	return chequear_nombre();
	if (code>=0x41 && code<=0x5A) // is a upper letter?
	return chequear_nombre();
	// other keys.
	return false;
}

function valideKeyOnlyNumbers(evt)
{
	// code is the decimal ASCII representation of the pressed key.
	var code = (evt.which) ? evt.which : evt.keyCode;
	
	if(code==8) // backspace.
	return true;
	if(code==127) // Delete.
	return true;
	if (code>=48 && code<=57) // is a number.
	return true;
	// other keys.
	return false;
}