const brain = require('brain.js');
let net = new brain.NeuralNetwork();

const csv = require('csvtojson');

const SPLIT_POSITION = 0.98;
const ERROR_THRESHOLD = 0.0068;
const ITERATIONS = 10000;

(async () => {
    let data = await getWineData();
    let maxOutputValue = data.reduce((a, c) => Math.max(c['quality'], a), 0);
    let outputProperty = 'quality';

    //cut the train/test data at the split position
    let cutIndex = Math.floor(SPLIT_POSITION * data.length);

    //condition training data
    let trainingData = data.slice(0, cutIndex)
        .map(d => {
            let properties = Object.keys(d).map(k => ({ key: k, value: d[k] }));
            let result = { input: {}, output: {} };

            //split the training data into input and output
            properties.forEach(p => {
                if (p.key != outputProperty) result.input[p.key] = p.value;
                else result.output[p.key] = p.value / maxOutputValue;
            })

            return result;
        });

    //train
    net.train(trainingData, {
        iterations: ITERATIONS,
        errorThresh: ERROR_THRESHOLD,
        log: true
    });

    //condition test data
    let testData = data.slice(cutIndex);

    let c = 0, sum = 0;
    testData.forEach((d, i) => {
        let predictedQuality = net.run(d).quality;
        predictedQuality *= maxOutputValue;
        c++;
        sum += predictedQuality / d[outputProperty];
        console.log(`${i}: ${predictedQuality}/${d[outputProperty]} = ${predictedQuality / d[outputProperty]}`);
    });
    console.log(`average variance: ${Math.round(((sum / c) - 1) * 100)}%`)
})()

function getWineData() {
    return new Promise((resolve, reject) => {
        let result = [];
        csv({ delimiter: ';' })
            .fromFile('wine.csv')
            .on('json', o => result.push(o))
            .on('done', err => {
                if (err) reject(err);
                else resolve(result);
            });

    })
}