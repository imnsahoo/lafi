console.log('js ...............')
$('input:radio').click(function() { 
    if($("#Total_hour").val == 'Total_hour') {
        $("#Role").prop("disabled",true);
    }
    else{
         $("#Role").prop("disabled",false);
    }
});