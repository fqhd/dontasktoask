const outputMessage = document.getElementById('output-area');
const textBox = document.getElementById('input-area');
const confidenceArea = document.getElementById('confidence-label');

let model = null;

tf.loadLayersModel('./model/model.json', compile=false).then(m => {
    model = m;
}).catch(e => {
    console.log(e);
});

function get_character_embeddings() {
    const embeddings = [];
    for(let i = 0; i < 128; i++) {
        const curr = [];
        for (let j = 0; j < 7; j++) {
            curr.push(parseInt(i / Math.pow(2, j)) % 2);
        }
        embeddings.push(curr);
    }
    return embeddings;
}

const asciiChars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ';
function get_character_mappings(){
    const embeddings = get_character_embeddings();
    const char_mappings = {};
    for(let i = 0; i < asciiChars.length; i++) {
        char_mappings[asciiChars[i]] = embeddings[i];
    }
    return char_mappings;
}

const character_mappings = get_character_mappings();

function vectorize_message(msg) {
    const vec = []
    for(const c of msg){
        if(vec.length == 7*50) {
            return vec;
        }
        const char_vector = character_mappings[c];
        if(char_vector != undefined) {
            vec.push(...char_vector);
        }
    }
    while(vec.length < 7*50) {
        for(let i = 0; i < 7; i++) {
            vec.push(Math.floor(Math.random() * 2));
        }
    }
    return vec;
}

textBox.addEventListener('input', async (e) => {
    if(model == null) return;
    const data = [];

    data.push(vectorize_message(e.target.value));

    const answer = (await model.predict(tf.tensor2d(data)).data())[0];

    let confidence = answer > 0.5 ? answer : 1 - answer;
    confidence = parseInt(confidence * 10000) / 100 + '%';

    if(answer > 0.5) {
        outputMessage.textContent = 'This message is dontasktoask worthy, confidence: ' + confidence;
    }else{
        outputMessage.textContent = 'This message is fine, confidence: ' + confidence;
    }
});