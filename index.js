const outputMessage = document.getElementById('output-area');
const textBox = document.getElementById('input-area');
const confidenceArea = document.getElementById('confidence-label');

let model = null;

tf.loadLayersModel('./model/model.json', compile=false).then(m => {
    model = m;
}).catch(e => {
    console.log(e);
});

const character_mappings = {
    'a': [0, 0, 0, 0, 1],
    'b': [0, 0, 0, 1, 0],
    'c': [0, 0, 0, 1, 1],
    'd': [0, 0, 1, 0, 0],
    'e': [0, 0, 1, 0, 1],
    'f': [0, 0, 1, 1, 0],
    'g': [0, 0, 1, 1, 1],
    'h': [0, 1, 0, 0, 0],
    'i': [0, 1, 0, 0, 1],
    'j': [0, 1, 0, 1, 0],
    'k': [0, 1, 0, 1, 1],
    'l': [0, 1, 1, 0, 0],
    'm': [0, 1, 1, 0, 1],
    'n': [0, 1, 1, 1, 0],
    'o': [0, 1, 1, 1, 1],
    'p': [1, 0, 0, 0, 0],
    'q': [1, 0, 0, 0, 1],
    'r': [1, 0, 0, 1, 0],
    's': [1, 0, 0, 1, 1],
    't': [1, 0, 1, 0, 0],
    'u': [1, 0, 1, 0, 1],
    'v': [1, 0, 1, 1, 0],
    'w': [1, 0, 1, 1, 1],
    'x': [1, 1, 0, 0, 0],
    'y': [1, 1, 0, 0, 1],
    'z': [1, 1, 0, 1, 0],
    ' ': [1, 1, 0, 1, 1],
}

function vectorize_message(msg) {
    const vec = []
    for(const c of msg){
        if(vec.length == 500) {
            return vec;
        }
        const char_vector = character_mappings[c];
        if(char_vector != undefined) {
            vec.push(...char_vector);
        }
    }
    while(vec.length < 500) {
        vec.push(0, 0, 0, 0, 0);
    }
    return vec;
}

textBox.addEventListener('keydown', async (e) => {
    if(model == null) return;
    const data = [];

    data.push(vectorize_message(textBox.value));

    const answer = (await model.predict(tf.tensor2d(data)).data())[0];

    let confidence = answer > 0.5 ? answer : 1 - answer;
    confidence = parseInt(confidence * 10000) / 100 + '%';

    if(answer > 0.5) {
        outputMessage.textContent = 'This message is dontasktoask worthy, confidence: ' + confidence;
    }else{
        outputMessage.textContent = 'This message is fine, confidence: ' + confidence;
    }
});