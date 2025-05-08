// JavaScript Document

window.onload = function()
{
	document.getElementById("nombres").focus();
}

document.addEventListener("DOMContentLoaded", function () {

    document.getElementById('url_foto').addEventListener('change', function (e) {
        // Obtiene el nombre del archivo seleccionado
        const fileName = e.target.files[0] ? e.target.files[0].name : "No se ha seleccionado ningún archivo";
        // Muestra el nombre del archivo en el span
        document.getElementById('file-name').textContent = fileName;
		ChequearFoto();
    });
});

function chequear_form()
{
	var nombres = document.getElementById('nombres').value;
	var apellidos = document.getElementById('apellidos').value;
	var username = document.getElementById('username').value;
	var password = document.getElementById('password').value;
	var password2 = document.getElementById('password2').value;
	var email = document.getElementById('email').value;
	
	if (!(nombres.match(/^[A-Z,Á,É,Í,Ó,Ú,Ñ]/) && nombres.match(/[a-z,A-Z, ,Á,É,Í,Ó,Ú,Ñ,á,é,í,ó,ú,ñ]+$/)))
	{
		customAlert("Debe escribir su(s) nombre(s) correctamente: '"+nombres+"'. Primera letra en mayúscula de cada palabra, no usar simbolos ni caracteres extraños, solo letras");
		return
	}
	if (!(apellidos.match(/^[A-Z,Á,É,Í,Ó,Ú,Ñ]/) && apellidos.match(/[a-z,A-Z, ,Á,É,Í,Ó,Ú,Ñ,á,é,í,ó,ú,ñ]+$/)))
	{
		customAlert("Debe escribir su(s) apellidos(s) correctamente: '"+apellidos+"'. Primera letra en mayúscula de cada palabra, no usar simbolos ni caracteres extraños, solo letras.");
		return
	}
	if (!(username.match(/^[a-z]/) && username.match(/[a-z,_,0-9]+$/)))
	{
		customAlert("Debe escribir su nombre de usuario (username) correctamente: '"+username+"'. Solo usar letras minúsculas, números y guión bajo (_).");
		return
	}
	if (username.length < 5)
	{
		customAlert("Debe escribir su nombre de usuario (username) de al menos 5 caracteres: '"+username+"'.");
		return
	}
	if (password != password2) {
		customAlert("Asegúre escribir la misma contraseña (original y copia).");
		return
	}
	if (password.length < 7)
	{
		customAlert("Debe escribir su contraseña de al menos 7 caracteres.");
		return
	}
	// Define our regular expression.
	var validEmail =  /^\w+([.-_+]?\w+)*@\w+([.-]?\w+)*(\.\w{2,10})+$/;
	if (! validEmail.test(email) )
	{
		customAlert("No ha escrito correctamente el correo electrónico. Patrón: [uno o más nombres en minúsculas, separados por punto o guión bajo (_)]@[uno o más nombres en minúsculas, separados por punto]");
		return
	}
	document.forms["datos_usuario"].submit();
}

function ChequearFoto()
{
	var preview = document.getElementById('img_foto');
	var fi = document.getElementById('url_foto').files[0];
	var reader = new FileReader();
	
	reader.onloadend = function(){
		preview.src = reader.result;
	}
	if (fi){
		reader.readAsDataURL( fi );
	} else {
		preview.src = '';
	}
}
