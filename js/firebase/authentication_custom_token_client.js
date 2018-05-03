// Initialize Firebase
var config = {
    apiKey: "XXXXXX",
    authDomain: "XXXXXX.firebaseapp.com",
    databaseURL: "https://XXXXXX.firebaseio.com",
    projectId: "XXXXXX",
    storageBucket: "",
    messagingSenderId: "XXXXXX"
};
firebase.initializeApp(config);

function signIn() {
	return new Promise(function (resolve, reject) {
			updateDisplayAppReady();	
			console.log("Calling cloud based function ...");
			$.ajax({
			  type: 'POST',
			  url: '/createUser',  
			  data: {ethAddress: currentAccount}, 			   //Use ethereum address as uid
			  success: function(data) {
        // Sucessfully signed in
			   console.log("Signed in.");
			   firebase.auth().signInWithCustomToken(data.token).catch(function(error) {
				  //Call promise error
				  reject(error);
				});
        //Call promise done
				resolve();
			  },
			  error: function() {
			   console.log("Error while sign-in!");
         //Call promise error
			   reject("Error while sign-in!");
			  }
			});
		});	
}
