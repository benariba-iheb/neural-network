let model = tf.sequential();
let canvas = document.createElement('canvas');
let ctx = canvas.getContext('2d');

canvas.width = 28;
canvas.height = 28;

let drawingCanvas = document.getElementById('canvas');
drawingCanvas.id = "dctx";
let dctx = drawingCanvas.getContext('2d');
drawingCanvas.width = 280;
drawingCanvas.height = 280;

let version = "0.1";
let counter = 0;

dctx.fillStyle = "black";
dctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
let imagesToLearn = 10;

const configHidden = {
    units: 784,
    inputShape: 784,
    activation: 'sigmoid',
};

const configOutput = {
    units: imagesToLearn,
    activation: 'sigmoid',
};

let current = 0;
const hidden = tf.layers.dense(configHidden);
const output = tf.layers.dense(configOutput);

//===================
//l'ajout des layers
//===================
   //couche d'entrée
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
}));

//couche de pooling
model.add(tf.layers.maxPooling2d({ //maxpooling va prendre la valeur la plus grande dans la matrice de 2x2 (le but est de réduire la taille de la matrice)
    poolSize: [2, 2],
    strides: [2, 2]
}));

//couche de convolution
model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
}));

//couche de pooling
model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
}));

//couche de convolution
model.add(tf.layers.flatten());     //rendre la matrice 2D sous forme de vecteur 1D
model.add(tf.layers.dense({         //couche de sortie (10 neurones)
    units: 10,
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax'
}));

//===================


const sgdOptimizer = tf.train.sgd(0.1); //optimiseur de descente de gradient stochastique
const config = { 
    optimizer: sgdOptimizer,
    loss: tf.losses.meanSquaredError
};
let arrIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
pickRandomIndex = () =>{ 
    if(arrIndex.length <= 0){
        for(let i =0; i < 10; i++){
            arrIndex[i] = i;
        }
        return arrIndex.splice(Math.floor(Math.random() * arrIndex.length), 1);
    }else {
        return arrIndex.splice(Math.floor(Math.random() * arrIndex.length), 1);
    }
};


async function start(){   //lancement du programme
    model = await tf.loadModel('https://sirdomin.github.io/DigitRecognition/model/my-model-1.json');
    model.compile(config);
}
async function getDataBytesView(imageData){  //récupération des données binaires de l'image
    let array = [];
    for(let j = 0; j < imageData.data.length / 4; j++){ 
        array[j] = imageData.data[j * 4] / 255; 
    }
    return array;
}

async function guessImage(){    //prédiction de l'image
    let imageToGuess = new Image();
    imageToGuess.onload = function(){   //dessin de l'image
         ctx.drawImage(imageToGuess,0 , 0, 28, 28);
            let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            getDataBytesView(imageData).then((dataBytesView)=>{  //récupération des données de l'image
                predictModel(dataBytesView)
            });
    };  
    imageToGuess.id = "imageToPredict";
    imageToGuess.width = 28;
    imageToGuess.height = 28;
    imageToGuess.src = drawingCanvas.toDataURL();
}

//xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);

async function predictModel(xs){    //prédiction du modèle
    let highestScore = [0, 0];  //meilleur score
    let input = tf.tensor2d([xs]);  
    let output = await model.predict(input.reshape([1, 28, 28, 1])); 
    let outputData = JSON.parse(output.toString().slice(13,150).split(',]')[0]);
    for(let i = 0; i < imagesToLearn; i++){   //recherche du meilleur score

        if(outputData[i] > highestScore[0])highestScore = [outputData[i], i];
    }
    for(let i = 0 ;i < 10; i++){    //affichage des résultats
        document.getElementsByClassName("result")[i].innerText = ` Guess: ${i} `;
        document.getElementsByClassName("result")[i].innerText += ` chance: ${Math.round(Math.round(outputData[i] * 1000000) / 10000)}%`;
        if(highestScore[1] === i) {
            document.getElementsByClassName("result")[i].style.color="lime";
            document.getElementsByClassName("result")[i].style.border = 'solid 1px lime'
        }
    }
}   

let drawingArray = [];
let test=false;
draw = () =>{
    if(drawing){    //dessin
        drawingArray.unshift({x: drawingX, y: drawingY});
        if(drawingArray.length > 1){
            dctx.beginPath();
            dctx.lineJoin="round";
            dctx.lineWidth = 12;
            dctx.moveTo(drawingArray[0].x, drawingArray[0].y);
            dctx.lineTo(drawingArray[1].x, drawingArray[1].y);
            dctx.strokeStyle='#ffffff';
            dctx.closePath();
            dctx.stroke();
            test = true;
        }
    }
    setTimeout(draw, 1000/60);
};
start().then(()=>{  //lancement du programme
    console.log("Load complete");
});
document.getElementById('btn').onclick = function(){    //bouton de prédiction
    
    for(let i = 0 ;i < 10; i++){    //réinitialisation des résultats
        document.getElementsByClassName("result")[i].style.color="#777777";
        document.getElementsByClassName("result")[i].style.border = 'solid 1px rgb(238,100,42)'
    }
  guessImage();     

};
